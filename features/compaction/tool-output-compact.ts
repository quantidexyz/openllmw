import { estimateBodyTokens } from "../../lib/canonical/token-estimate";

/**
 * Gateway-side TOOL-OUTPUT compaction — Plan B of the context-overflow
 * ladder (A: serve correct per-model context metadata so the client
 * compacts itself; B: THIS — compact tool outputs until the request fits
 * the target hop's budget; C: walk to a larger-context hop).
 *
 * Only tool OUTPUTS are ever rewritten — system/user/assistant text and
 * the tool-call structure are untouched. Two proven passes:
 *
 *   1. Codex's truncation policy (`codex-rs/utils/output-truncation`;
 *      `truncation_policy: { mode: "tokens", limit: 10000 }` in the
 *      models manifest): middle-truncate every oversized tool output to
 *      a per-output token cap — head + tail kept, elision marker in the
 *      middle, a "Warning: truncated output (original token count: N)"
 *      header on top.
 *   2. Claude Code's microcompact: when the per-output cap still doesn't
 *      fit the budget, CLEAR whole tool outputs oldest-first. The
 *      trailing round — the results answering the still-pending tool
 *      calls — is protected: it is only ever middle-truncated, never
 *      cleared.
 *
 * Pure + structural: operates on the parsed inbound body of any surface
 * (`messages` tool_result blocks, `chat_completions` tool messages,
 * `responses` function_call_output items), never mutates the input
 * (compaction rewrites a clone), and never throws on unexpected shapes —
 * unrecognized structures simply contribute no compactable slots.
 * Non-text payloads inside a tool output (images, documents, encrypted
 * content) are preserved verbatim, exactly like codex's own truncation.
 */

/** Codex's per-tool-output budget (`truncation_policy.limit`). */
export const PER_TOOL_OUTPUT_TOKEN_CAP = 10_000;

export type TCompactionSurface = "chat_completions" | "messages" | "responses";

export type TToolOutputCompaction = {
  /** The (possibly rewritten) body — reference-equal to the input when
   *  nothing changed. */
  readonly body: unknown;
  /** `estimateBodyTokens` of the returned body. */
  readonly estimatedTokens: number;
  readonly changed: boolean;
  /** `estimatedTokens <= budgetTokens` — false means even full
   *  compaction can't fit this hop (the caller falls to Plan C). */
  readonly fits: boolean;
};

const approxTokens = (s: string): number => Math.ceil(s.length / 4);

/** Codex-style middle truncation to a token budget: head + tail halves
 *  around an elision marker, original size in the warning header. */
const truncateMiddleToTokens = (text: string, budgetTokens: number): string => {
  const originalTokens = approxTokens(text);
  // Reserve room for the header + elision marker inside the budget.
  const keepChars = Math.max(0, budgetTokens * 4 - 200);
  const head = text.slice(0, Math.ceil(keepChars / 2));
  const tail = text.slice(text.length - Math.floor(keepChars / 2));
  const omitted = Math.max(
    0,
    originalTokens - approxTokens(head) - approxTokens(tail),
  );
  return `Warning: truncated output (original token count: ${originalTokens})\n\n${head}\n…[~${omitted} tokens truncated]…\n${tail}`;
};

const clearedMarker = (originalTokens: number): string =>
  `[tool output cleared by the gateway to fit the model's context window — original token count: ${originalTokens}]`;

type TJsonObject = Record<string, unknown>;

const isObj = (v: unknown): v is TJsonObject =>
  v !== null && typeof v === "object" && !Array.isArray(v);

/** One rewritable tool output. `read` returns only its TEXT payload;
 *  `write` replaces the text while preserving any non-text blocks. */
type TSlot = {
  /** Trailing round (answers to the still-pending tool calls) — never
   *  cleared, only middle-truncated. */
  readonly isProtected: boolean;
  readonly read: () => string;
  readonly write: (next: string) => void;
};

const isTextBlock = (b: unknown): b is TJsonObject & { text: string } =>
  isObj(b) &&
  typeof b.text === "string" &&
  (b.type === "text" || b.type === "output_text" || b.type === "input_text");

/** Slot over a `string | block[]` content value living at `holder[key]`.
 *  Writes collapse the text blocks into one and keep non-text blocks. */
const contentSlot = (
  holder: TJsonObject,
  key: string,
  isProtected: boolean,
): TSlot | null => {
  const value = holder[key];
  if (typeof value === "string") {
    return {
      isProtected,
      read: () => holder[key] as string,
      write: (next) => {
        holder[key] = next;
      },
    };
  }
  if (Array.isArray(value) && value.some(isTextBlock)) {
    return {
      isProtected,
      read: () =>
        (holder[key] as unknown[])
          .filter(isTextBlock)
          .map((b) => b.text)
          .join("\n"),
      // The FIRST text block takes the compacted text in place (its own
      // type and any sibling fields preserved); later text blocks drop
      // (their content is folded into the compacted value); non-text
      // blocks keep their original positions — an image-first array
      // stays image-first.
      write: (next) => {
        let replaced = false;
        holder[key] = (holder[key] as unknown[]).flatMap((b) => {
          if (!isTextBlock(b)) return [b];
          if (replaced) return [];
          replaced = true;
          return [{ ...b, text: next }];
        });
      },
    };
  }
  return null;
};

/** Anthropic `messages` surface: `tool_result` blocks inside user turns.
 *  Protected = blocks after the LAST assistant message (the pending
 *  round); earlier rounds are already answered by later assistant turns. */
const messagesSlots = (body: unknown): TSlot[] => {
  if (!isObj(body) || !Array.isArray(body.messages)) return [];
  const messages = body.messages;
  let lastAssistant = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (
      isObj(messages[i]) &&
      (messages[i] as TJsonObject).role === "assistant"
    ) {
      lastAssistant = i;
      break;
    }
  }
  const slots: TSlot[] = [];
  for (const [i, m] of messages.entries()) {
    if (!isObj(m) || m.role !== "user" || !Array.isArray(m.content)) continue;
    for (const block of m.content) {
      if (!isObj(block) || block.type !== "tool_result") continue;
      const slot = contentSlot(block, "content", i > lastAssistant);
      if (slot !== null) slots.push(slot);
    }
  }
  return slots;
};

/** Canonical `chat_completions` surface: `role: "tool"` messages.
 *  Protected = tool messages after the LAST assistant message (the
 *  pending round) — contiguity is NOT required: the client may inject
 *  context (e.g. loaded Skill instructions as a user message) after the
 *  pending result, and that must not demote it to clearable. */
const chatSlots = (body: unknown): TSlot[] => {
  if (!isObj(body) || !Array.isArray(body.messages)) return [];
  const messages = body.messages;
  let lastAssistant = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (
      isObj(messages[i]) &&
      (messages[i] as TJsonObject).role === "assistant"
    ) {
      lastAssistant = i;
      break;
    }
  }
  const slots: TSlot[] = [];
  for (const [i, m] of messages.entries()) {
    if (!isObj(m) || m.role !== "tool") continue;
    const slot = contentSlot(m, "content", i > lastAssistant);
    if (slot !== null) slots.push(slot);
  }
  return slots;
};

/** `responses` surface: `function_call_output` input items. Protected =
 *  outputs after the LAST model-emitted item (a `function_call` or an
 *  assistant `message`) — the pending round, whether or not later
 *  client-injected context items follow it. */
const responsesSlots = (body: unknown): TSlot[] => {
  if (!isObj(body) || !Array.isArray(body.input)) return [];
  const input = body.input;
  let lastModelItem = -1;
  for (let i = input.length - 1; i >= 0; i--) {
    const item = input[i];
    if (!isObj(item)) continue;
    if (
      item.type === "function_call" ||
      (item.type === "message" && item.role === "assistant")
    ) {
      lastModelItem = i;
      break;
    }
  }
  const slots: TSlot[] = [];
  for (const [i, item] of input.entries()) {
    if (!isObj(item) || item.type !== "function_call_output") continue;
    const slot = contentSlot(item, "output", i > lastModelItem);
    if (slot !== null) slots.push(slot);
  }
  return slots;
};

const collectSlots = (surface: TCompactionSurface, body: unknown): TSlot[] =>
  surface === "messages"
    ? messagesSlots(body)
    : surface === "responses"
      ? responsesSlots(body)
      : chatSlots(body);

/**
 * Rewrite `rawBody`'s tool outputs until its estimate fits
 * `budgetTokens`. Returns the input untouched when it already fits.
 */
export const compactToolOutputsToBudget = (
  surface: TCompactionSurface,
  rawBody: unknown,
  budgetTokens: number,
): TToolOutputCompaction => {
  const before = estimateBodyTokens(rawBody);
  if (before <= budgetTokens) {
    return {
      body: rawBody,
      estimatedTokens: before,
      changed: false,
      fits: true,
    };
  }
  const body = structuredClone(rawBody);
  const slots = collectSlots(surface, body);
  let estimate = before;
  let changed = false;
  // Pass 1 — codex per-output cap, oldest first.
  for (const slot of slots) {
    if (estimate <= budgetTokens) break;
    const text = slot.read();
    const tokens = approxTokens(text);
    if (tokens <= PER_TOOL_OUTPUT_TOKEN_CAP) continue;
    const next = truncateMiddleToTokens(text, PER_TOOL_OUTPUT_TOKEN_CAP);
    slot.write(next);
    estimate -= tokens - approxTokens(next);
    changed = true;
  }
  // Pass 2 — clear whole outputs oldest first; the trailing round is
  // protected (only ever truncated by pass 1).
  for (const slot of slots) {
    if (estimate <= budgetTokens) break;
    if (slot.isProtected) continue;
    const text = slot.read();
    const tokens = approxTokens(text);
    const next = clearedMarker(tokens);
    if (approxTokens(next) >= tokens) continue;
    slot.write(next);
    estimate -= tokens - approxTokens(next);
    changed = true;
  }
  if (!changed) {
    return {
      body: rawBody,
      estimatedTokens: before,
      changed: false,
      fits: false,
    };
  }
  const estimatedTokens = estimateBodyTokens(body);
  return {
    body,
    estimatedTokens,
    changed,
    fits: estimatedTokens <= budgetTokens,
  };
};

/** Compaction targets this fraction of the hop's input budget — the
 *  chars/4 estimator can UNDER-count real BPE tokens (code runs ~3.2
 *  chars/token), so landing at 80% keeps the compacted request safely
 *  inside the window the vendor actually enforces. */
export const COMPACT_TARGET_FACTOR = 0.8;

/**
 * Confidence multiplier before a hop is abandoned over context. The
 * estimator can also OVER-count 30–100% on repetitive content where BPE
 * compresses heavily, so a hop is only skipped (request-time) or dropped
 * (plan-time) when the estimate CLEARLY exceeds its limit — borderline
 * cases go to the real upstream tokenizer, which gets the final word.
 * Shared by `fitRequestToHopBudget` and the cloud's plan gate
 * (`dropContextOversizedHops`) so the two thresholds can't drift.
 */
export const CONTEXT_SKIP_CONFIDENCE_FACTOR = 1.5;

export type THopContextFit =
  | {
      readonly kind: "serve";
      /** The body to dispatch — reference-equal to the input when no
       *  compaction was needed. */
      readonly body: unknown;
      readonly changed: boolean;
    }
  | { readonly kind: "skip" };

/**
 * The ONE per-hop context-ladder decision, shared verbatim by the cloud
 * dispatch chain (BYOK) and the daemon walker (device) so the two paths
 * cannot drift: a request inside the hop's budget serves as-is; over
 * budget, tool outputs are compacted toward the target (Plan B); the hop
 * SKIPS — so the chain walks to a larger-context hop (Plan C) — only
 * when the POST-compaction estimate still clearly exceeds the limit (the
 * shared 1.5× confidence factor: estimator noise must not abandon a hop
 * the real tokenizer might accept, the pre-ladder bias preserved). The
 * FINAL hop never skips — it serves the best-effort compacted body
 * (never-drop-all). An unknown limit always serves untouched.
 */
export const fitRequestToHopBudget = (params: {
  readonly surface: TCompactionSurface;
  readonly body: unknown;
  readonly estimatedTokens: number;
  readonly inputTokenLimit: number | null;
  readonly finalHop: boolean;
}): THopContextFit => {
  const limit = params.inputTokenLimit;
  if (limit === null || params.estimatedTokens <= limit) {
    return { kind: "serve", body: params.body, changed: false };
  }
  const compacted = compactToolOutputsToBudget(
    params.surface,
    params.body,
    Math.floor(limit * COMPACT_TARGET_FACTOR),
  );
  if (
    !params.finalHop &&
    compacted.estimatedTokens > limit * CONTEXT_SKIP_CONFIDENCE_FACTOR
  ) {
    return { kind: "skip" };
  }
  return { kind: "serve", body: compacted.body, changed: compacted.changed };
};

/**
 * The lowest estimate tool-output compaction could reach for this body —
 * a read-only computation (no clone, no rewrite). The cloud's plan-time
 * context gate uses it so a hop that compaction could still save is not
 * pre-dropped: Plan C (dropping/walking) must never preempt Plan B.
 */
export const compactionFloorTokens = (
  surface: TCompactionSurface,
  rawBody: unknown,
): number => {
  let floor = estimateBodyTokens(rawBody);
  for (const slot of collectSlots(surface, rawBody)) {
    const text = slot.read();
    const tokens = approxTokens(text);
    // Protected slots keep their ACTUAL truncated representation (header
    // + elision marker included), not the nominal cap — the floor must
    // never claim less than compaction can really achieve.
    const kept = slot.isProtected
      ? tokens > PER_TOOL_OUTPUT_TOKEN_CAP
        ? approxTokens(truncateMiddleToTokens(text, PER_TOOL_OUTPUT_TOKEN_CAP))
        : tokens
      : Math.min(tokens, approxTokens(clearedMarker(tokens)));
    floor -= tokens - kept;
  }
  return floor;
};
