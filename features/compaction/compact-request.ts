/**
 * LAST-RESORT request compaction — shrink an oversized request body just enough
 * to fit a target context budget, so a request that is larger than EVERY hop's
 * window is served (degraded) instead of hard-failing with a 400/502.
 *
 * This deliberately reverses, IN THE FAILURE CORNER ONLY, the decision recorded
 * in `../context-skip.ts`: gateway-side request rewriting was removed because
 * PROACTIVE, estimator-driven rewrites destabilise the prompt-cache prefix and
 * degrade requests the vendor would have accepted. This compactor is NOT
 * proactive — callers invoke it exclusively after the whole fallback chain is
 * exhausted, where the alternative is a guaranteed failure and a prompt-cache
 * miss is strictly the better outcome. It is also cache-aware: it preserves the
 * leading `cache_control`-bearing prefix, compacting from the middle-old region
 * forward so the cached prefix and the recent tail survive where the budget
 * allows.
 *
 * Pure and immutable — never mutates the input; returns a fresh body. Reuses the
 * ruler-backed {@link estimateBodyTokens} for the fit check, so on a warm isolate
 * the "does it fit now?" loop is a real tokenizer, not `chars/4`.
 */
import type { TTokenEncoding } from "../../lib/canonical/encoding-select";
import { estimateBodyTokens } from "../../lib/canonical/token-estimate";
import { COMPACTION_MIN_VISIBLE_TEXT_CHARS } from "./compaction-text";

/**
 * The elision marker dropped into the middle of a truncated tool output. Kept
 * short and unambiguous so a model reading the transcript understands the gap.
 */
const ELISION = "\n\n…[openllm: output truncated to fit context]…\n\n";

/**
 * Keep the head and tail of `text`, replacing the middle with {@link ELISION},
 * so the result is approximately `maxChars` long. Head-heavy (2/3 head, 1/3
 * tail): the start of a tool output — the command, the first rows — usually
 * carries more signal than the end. Never returns something below
 * `COMPACTION_MIN_VISIBLE_TEXT_CHARS`; a `maxChars` at or above the input length
 * returns the input untouched.
 *
 * Works in characters, not tokens: the caller drives the token-fit loop and this
 * is the cheap per-string knob. A char budget is a safe proxy — fewer chars is
 * always fewer-or-equal tokens for any BPE.
 */
export const truncateMiddleToChars = (
  text: string,
  maxChars: number,
): string => {
  const cap = Math.max(COMPACTION_MIN_VISIBLE_TEXT_CHARS, maxChars);
  if (text.length <= cap) return text;
  const budget = cap - ELISION.length;
  if (budget <= COMPACTION_MIN_VISIBLE_TEXT_CHARS) {
    // No room for both ends around the marker — keep a short head only.
    return text.slice(0, cap - ELISION.length > 0 ? cap - ELISION.length : cap);
  }
  const headLen = Math.ceil((budget * 2) / 3);
  const tailLen = budget - headLen;
  const head = text.slice(0, headLen);
  const tail = tailLen > 0 ? text.slice(text.length - tailLen) : "";
  return `${head}${ELISION}${tail}`;
};

type TRecord = Record<string, unknown>;

const isRecord = (v: unknown): v is TRecord =>
  v !== null && typeof v === "object" && !Array.isArray(v);

/** True when a message / content block carries a `cache_control` breakpoint. */
const hasCacheControl = (v: unknown): boolean => {
  if (!isRecord(v)) return false;
  if (v.cache_control != null) return true;
  const content = v.content;
  if (Array.isArray(content)) return content.some(hasCacheControl);
  return false;
};

/**
 * The result of a compaction attempt. `compacted` is false when the body was
 * already within budget OR could not be shrunk to fit (caller then surfaces the
 * original terminal error). `estimatedTokens` is the post-attempt estimate.
 */
export type TCompactionResult = {
  readonly body: unknown;
  readonly compacted: boolean;
  readonly estimatedTokens: number;
};

/**
 * The surfaces a body can arrive on. `messages` is Anthropic-shaped;
 * `chat_completions` / `responses` are canonical OpenAI-shaped (the cloud path
 * canonicalises `responses` before dispatch).
 */
export type TCompactionSurface = "messages" | "chat_completions" | "responses";

const isAnthropic = (surface: TCompactionSurface): boolean =>
  surface === "messages";

// ─── Anthropic-shaped truncation ──────────────────────────────────────────

/**
 * Rewrite the text inside a single Anthropic `tool_result` block, capping its
 * serialized size. `content` is `string | block[]`; string form truncates
 * directly, block-array form truncates each `text` block in place (order + type
 * + sibling fields preserved — image/document blocks untouched).
 */
const truncateAnthropicToolResult = (
  block: TRecord,
  maxChars: number,
): TRecord => {
  const content = block.content;
  if (typeof content === "string") {
    return { ...block, content: truncateMiddleToChars(content, maxChars) };
  }
  if (Array.isArray(content)) {
    const next = content.map((inner) =>
      isRecord(inner) && inner.type === "text" && typeof inner.text === "string"
        ? { ...inner, text: truncateMiddleToChars(inner.text, maxChars) }
        : inner,
    );
    return { ...block, content: next };
  }
  return block;
};

/**
 * Truncate every `tool_result` block in an Anthropic messages array to
 * `maxChars`, walking oldest→newest. Returns a fresh messages array. The last
 * user turn is left untouched even if it carries a tool_result, so the live
 * query is never degraded — and a `cache_control`-bearing message is left
 * byte-identical so the cached prefix (the ~10x-cost breakpoint) survives.
 */
const truncateAnthropicToolOutputs = (
  messages: ReadonlyArray<unknown>,
  maxChars: number,
  lastUserIndex: number,
): unknown[] =>
  messages.map((msg, i) => {
    if (
      i === lastUserIndex ||
      !isRecord(msg) ||
      !Array.isArray(msg.content) ||
      hasCacheControl(msg)
    ) {
      return msg;
    }
    const content = msg.content.map((block) =>
      isRecord(block) && block.type === "tool_result"
        ? truncateAnthropicToolResult(block, maxChars)
        : block,
    );
    return { ...msg, content };
  });

// ─── Canonical (OpenAI) truncation ────────────────────────────────────────

/**
 * Truncate the output of every `role:"tool"` message to `maxChars`, oldest→
 * newest, skipping the last user turn. `content` is `string | part[]`; string
 * truncates directly, part-array truncates `text` parts in place. A
 * `cache_control`-bearing message is left byte-identical so the cached prefix
 * survives (dropping/altering a breakpoint silently ~10x's token cost).
 */
const truncateCanonicalToolOutputs = (
  messages: ReadonlyArray<unknown>,
  maxChars: number,
  lastUserIndex: number,
): unknown[] =>
  messages.map((msg, i) => {
    if (
      i === lastUserIndex ||
      !isRecord(msg) ||
      msg.role !== "tool" ||
      hasCacheControl(msg)
    )
      return msg;
    const content = msg.content;
    if (typeof content === "string") {
      return { ...msg, content: truncateMiddleToChars(content, maxChars) };
    }
    if (Array.isArray(content)) {
      const next = content.map((part) =>
        isRecord(part) && part.type === "text" && typeof part.text === "string"
          ? { ...part, text: truncateMiddleToChars(part.text, maxChars) }
          : part,
      );
      return { ...msg, content: next };
    }
    return msg;
  });

// ─── Turn dropping (with pairing + cache preservation) ─────────────────────

/**
 * Indices of messages that are safe to drop, oldest first. NEVER drops: the
 * system prompt (canonical only — Anthropic `system` is a top-level field), the
 * last user turn, any message carrying a `cache_control` breakpoint (the cached
 * prefix), and — for canonical — a `tool` result whose paired assistant
 * `tool_call` we'd be keeping. Pairing is preserved by dropping the assistant
 * turn and its tool results together — canonical tool messages by
 * `tool_call_id`, Anthropic tool_result blocks by `tool_use_id` in a following
 * user message — so neither shape is left with an orphaned tool result.
 */
const droppableTurnStart = (messages: ReadonlyArray<unknown>): number => {
  // Skip the leading cache_control-bearing prefix and any system messages.
  let start = 0;
  while (start < messages.length) {
    const m = messages[start];
    if (isRecord(m) && m.role === "system") {
      start++;
      continue;
    }
    if (hasCacheControl(m)) {
      start++;
      continue;
    }
    break;
  }
  return start;
};

/**
 * Collect the ids that a canonical `tool` message pairs with, so dropping an
 * assistant turn also drops its orphaned tool results. For Anthropic the pairing
 * lives inside a single message's block array, so dropping the message is
 * self-consistent and this is unused.
 */
const canonicalToolCallIds = (msg: unknown): ReadonlySet<string> => {
  const ids = new Set<string>();
  if (isRecord(msg) && Array.isArray(msg.tool_calls)) {
    for (const tc of msg.tool_calls) {
      if (isRecord(tc) && typeof tc.id === "string") ids.add(tc.id);
    }
  }
  return ids;
};

/**
 * The `tool_use` block ids inside an Anthropic assistant message. Unlike the
 * canonical shape (tool calls on the message), Anthropic tool_use lives in the
 * `content` block array — and its paired `tool_result` sits in a SEPARATE later
 * user message (`tool_use.id` ↔ `tool_result.tool_use_id`). So dropping an
 * assistant message is NOT self-contained: its tool_results must go too, or the
 * transcript is left with an orphaned tool_result the vendor rejects.
 */
const anthropicToolUseIds = (msg: unknown): ReadonlySet<string> => {
  const ids = new Set<string>();
  if (isRecord(msg) && Array.isArray(msg.content)) {
    for (const block of msg.content) {
      if (
        isRecord(block) &&
        block.type === "tool_use" &&
        typeof block.id === "string"
      ) {
        ids.add(block.id);
      }
    }
  }
  return ids;
};

/**
 * Remove the `tool_result` blocks that reference `ids` from a following user
 * message, returning the rewritten message — or `null` when that empties the
 * message (caller then drops it whole, mirroring the adapter's empty-turn rule).
 * Non-tool_result blocks and unrelated tool_results are preserved in place.
 */
const stripPairedToolResults = (
  msg: unknown,
  ids: ReadonlySet<string>,
): unknown | null => {
  if (!isRecord(msg) || !Array.isArray(msg.content)) return msg;
  const kept = msg.content.filter(
    (block) =>
      !(
        isRecord(block) &&
        block.type === "tool_result" &&
        typeof block.tool_use_id === "string" &&
        ids.has(block.tool_use_id)
      ),
  );
  if (kept.length === 0) return null;
  return { ...msg, content: kept };
};

/**
 * Drop the oldest droppable messages until the body fits or nothing more can go.
 * Preserves: system prompt, cache prefix, last user turn, and tool-call pairing
 * (a dropped assistant turn takes its `tool` results with it; a dropped
 * Anthropic message is self-contained). Returns the surviving messages.
 */
const dropOldestTurns = (
  messages: ReadonlyArray<unknown>,
  anthropic: boolean,
  fits: (msgs: ReadonlyArray<unknown>) => boolean,
): unknown[] => {
  const survivors = [...messages];
  const start = droppableTurnStart(messages);
  // Recompute the protected tail index within `survivors` as we splice.
  let i = start;
  while (i < survivors.length) {
    if (fits(survivors)) break;
    // Never drop at or past the last user turn.
    const lastUser = survivors.reduceRight<number>(
      (acc, m, idx) =>
        acc === -1 && isRecord(m) && m.role === "user" ? idx : acc,
      -1,
    );
    if (i >= lastUser) break;
    const msg = survivors[i];
    if (isRecord(msg) && (msg.role === "system" || hasCacheControl(msg))) {
      i++;
      continue;
    }
    if (!anthropic && isRecord(msg) && msg.role === "assistant") {
      // Canonical: drop the assistant turn AND its paired tool results together.
      const callIds = canonicalToolCallIds(msg);
      survivors.splice(i, 1);
      if (callIds.size > 0) {
        for (let j = i; j < survivors.length; ) {
          const t = survivors[j];
          if (
            isRecord(t) &&
            t.role === "tool" &&
            typeof t.tool_call_id === "string" &&
            callIds.has(t.tool_call_id)
          ) {
            survivors.splice(j, 1);
          } else {
            j++;
          }
        }
      }
      continue;
    }
    if (anthropic && isRecord(msg) && msg.role === "assistant") {
      // Anthropic: an assistant `tool_use` pairs with a `tool_result` in a
      // SEPARATE later user message. Drop the assistant turn AND strip its
      // paired tool_results (dropping a now-empty user turn), preserving the
      // last-user turn (guarded by the `i >= lastUser` break above).
      const useIds = anthropicToolUseIds(msg);
      survivors.splice(i, 1);
      if (useIds.size > 0) {
        for (let j = i; j < survivors.length; ) {
          const stripped = stripPairedToolResults(survivors[j], useIds);
          if (stripped === null) {
            survivors.splice(j, 1);
          } else {
            survivors[j] = stripped;
            j++;
          }
        }
      }
      continue;
    }
    survivors.splice(i, 1);
  }
  return survivors;
};

// ─── The public entry point ────────────────────────────────────────────────

/** Char budget per tool output at each truncation pass, tightening each round. */
const TRUNCATION_PASSES = [8000, 2000, 500] as const;

const lastUserTurnIndex = (messages: ReadonlyArray<unknown>): number =>
  messages.reduceRight<number>(
    (acc, m, idx) =>
      acc === -1 && isRecord(m) && m.role === "user" ? idx : acc,
    -1,
  );

/**
 * Shrink `body` until its estimated size is `<= targetTokens`, or report that it
 * can't. The ladder, cheapest-and-least-destructive first:
 *   1. Truncate tool outputs (3 tightening passes) — bulky, low-signal, and NOT
 *      part of the cached prefix in practice.
 *   2. Drop the oldest non-cached, non-system, non-last-user turns.
 * Recomputes the ruler estimate after each step and stops the moment it fits.
 *
 * `encoding` selects the ruler family for the fit check (defaults inside
 * `estimateBodyTokens`). Returns `compacted:false` when the body already fit or
 * could not be shrunk to target — the caller then surfaces its original error.
 */
export const compactRequestToFit = (
  body: unknown,
  surface: TCompactionSurface,
  targetTokens: number,
  encoding?: TTokenEncoding,
): TCompactionResult => {
  const estimate = (b: unknown): number => estimateBodyTokens(b, encoding);

  const initial = estimate(body);
  if (initial <= targetTokens) {
    return { body, compacted: true, estimatedTokens: initial };
  }
  if (!isRecord(body) || !Array.isArray(body.messages)) {
    // Nothing structured to shrink.
    return { body, compacted: false, estimatedTokens: initial };
  }

  const anthropic = isAnthropic(surface);
  let messages: unknown[] = [...body.messages];
  const withMessages = (msgs: unknown[]): TRecord => ({
    ...body,
    messages: msgs,
  });

  // Pass 1: truncate tool outputs, tightening the cap each round.
  for (const cap of TRUNCATION_PASSES) {
    const lastUser = lastUserTurnIndex(messages);
    messages = anthropic
      ? truncateAnthropicToolOutputs(messages, cap, lastUser)
      : truncateCanonicalToolOutputs(messages, cap, lastUser);
    const est = estimate(withMessages(messages));
    if (est <= targetTokens) {
      return {
        body: withMessages(messages),
        compacted: true,
        estimatedTokens: est,
      };
    }
  }

  // Pass 2: drop oldest droppable turns.
  messages = dropOldestTurns(
    messages,
    anthropic,
    (msgs) => estimate(withMessages([...msgs])) <= targetTokens,
  );

  const finalBody = withMessages(messages);
  const finalEst = estimate(finalBody);
  return {
    body: finalBody,
    compacted: finalEst <= targetTokens,
    estimatedTokens: finalEst,
  };
};
