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
 * The char length of a text value that is either a bare string or a block-array
 * of `{ type, text }` parts (Anthropic tool_result / canonical tool / Responses
 * output contents all take one of these two shapes).
 */
const textLen = (content: unknown): number => {
  if (typeof content === "string") return content.length;
  if (Array.isArray(content)) {
    let n = 0;
    for (const part of content) {
      if (isRecord(part) && typeof part.text === "string")
        n += part.text.length;
    }
    return n;
  }
  return 0;
};

/**
 * Total chars of truncatable tool output across the body, per surface: Anthropic
 * `tool_result` contents, canonical `tool` message contents, Responses
 * `function_call_output` outputs. Used to solve the first-pass truncation cap
 * from the actual deficit rather than a fixed ladder.
 */
const toolOutputChars = (
  items: ReadonlyArray<unknown>,
  surface: TCompactionSurface,
): number => {
  let n = 0;
  for (const item of items) {
    if (!isRecord(item)) continue;
    if (surface === "messages") {
      if (Array.isArray(item.content)) {
        for (const block of item.content) {
          if (isRecord(block) && block.type === "tool_result") {
            n += textLen(block.content);
          }
        }
      }
    } else if (surface === "responses") {
      if (item.type === "function_call_output") n += textLen(item.output);
    } else if (item.role === "tool") {
      n += textLen(item.content);
    }
  }
  return n;
};

/**
 * The largest tool-output char cap that trims ~only what the deficit needs.
 * Truncating every tool output to `cap` removes roughly `total − outputs×cap`
 * chars; we want to remove `deficitChars ≈ (estimatedTokens − targetTokens) ×
 * CHARS_PER_TOKEN`. Capping the result at 0.5× the average tool output keeps a
 * meaningful portion of each output rather than over-shrinking on a small
 * deficit (the live-test report: a ~9% deficit was destroying ~95% of content).
 * Returns `null` when there is no tool output to truncate.
 */
const CHARS_PER_TOKEN = 4;
const firstPassCap = (
  items: ReadonlyArray<unknown>,
  surface: TCompactionSurface,
  deficitTokens: number,
): number | null => {
  const total = toolOutputChars(items, surface);
  if (total === 0) return null;
  const outputs = items.reduce<number>((count, item) => {
    if (!isRecord(item)) return count;
    if (surface === "messages")
      return (
        count +
        (Array.isArray(item.content)
          ? (item.content as unknown[]).filter(
              (b) => isRecord(b) && b.type === "tool_result",
            ).length
          : 0)
      );
    if (surface === "responses")
      return item.type === "function_call_output" ? count + 1 : count;
    return item.role === "tool" ? count + 1 : count;
  }, 0);
  if (outputs === 0) return null;
  const deficitChars = deficitTokens * CHARS_PER_TOKEN;
  const keepChars = Math.max(0, total - deficitChars);
  const cap = Math.ceil(keepChars / outputs);
  const floorCap = Math.ceil((total / outputs) * 0.5);
  return Math.max(cap, floorCap, COMPACTION_MIN_VISIBLE_TEXT_CHARS);
};

/**
 * The result of a compaction attempt. `compacted` is true when the body fits the
 * target budget — either it already did (returned untouched) or it was shrunk to
 * fit. It is false ONLY when compaction could not get it under the target (the
 * caller then surfaces the original terminal error). `estimatedTokens` is the
 * post-attempt estimate.
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

// ─── Responses (ChatGPT / Codex) truncation ───────────────────────────────
//
// A Responses body is `{ input: string | item[] }` — NOT a `messages` array.
// Items are `message` (role + `input_text`/`output_text`/image/file parts),
// `function_call` (`call_id`, `arguments`), `function_call_output` (`call_id`,
// bulky `output`), and `reasoning`. Tool pairing is `function_call.call_id` ↔
// `function_call_output.call_id`, as SEPARATE top-level items. There is no
// per-item `cache_control` (Responses caches via the top-level
// `prompt_cache_key`, which compaction never touches).

/**
 * Truncate the `output` of every `function_call_output` item to `maxChars`,
 * oldest→newest, skipping the last user turn. `output` is `string | part[]`;
 * string truncates directly, part-array truncates `input_text`/`output_text`
 * parts in place.
 */
const truncateResponsesToolOutputs = (
  input: ReadonlyArray<unknown>,
  maxChars: number,
  lastUserIndex: number,
): unknown[] =>
  input.map((item, i) => {
    if (
      i === lastUserIndex ||
      !isRecord(item) ||
      item.type !== "function_call_output"
    ) {
      return item;
    }
    const output = item.output;
    if (typeof output === "string") {
      return { ...item, output: truncateMiddleToChars(output, maxChars) };
    }
    if (Array.isArray(output)) {
      const next = output.map((part) =>
        isRecord(part) &&
        (part.type === "input_text" || part.type === "output_text") &&
        typeof part.text === "string"
          ? { ...part, text: truncateMiddleToChars(part.text, maxChars) }
          : part,
      );
      return { ...item, output: next };
    }
    return item;
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

/** True when `msg` carries a `tool_result` block referencing any id in `ids`. */
const toolResultsReference = (
  msg: unknown,
  ids: ReadonlySet<string>,
): boolean =>
  isRecord(msg) &&
  Array.isArray(msg.content) &&
  msg.content.some(
    (block) =>
      isRecord(block) &&
      block.type === "tool_result" &&
      typeof block.tool_use_id === "string" &&
      ids.has(block.tool_use_id),
  );

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
  // The tool_call ids belonging to PROTECTED assistant turns (system or
  // cache_control — kept by the loop below). A `tool` message paired to one of
  // these must NOT be dropped, or the protected assistant is left with an
  // orphaned tool_call. Computed once up front because protection doesn't change
  // as we splice.
  const protectedToolCallIds = new Set<string>();
  if (!anthropic) {
    for (const m of messages) {
      if (isRecord(m) && m.role === "assistant" && hasCacheControl(m)) {
        for (const id of canonicalToolCallIds(m)) protectedToolCallIds.add(id);
      }
    }
  }
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
    // A canonical `tool` message paired to a PROTECTED assistant must survive.
    if (
      !anthropic &&
      isRecord(msg) &&
      msg.role === "tool" &&
      typeof msg.tool_call_id === "string" &&
      protectedToolCallIds.has(msg.tool_call_id)
    ) {
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
      // paired tool_results (dropping a now-empty user turn).
      const useIds = anthropicToolUseIds(msg);
      // But NOT if any of those tool_uses pair into the PROTECTED last user turn
      // — dropping the assistant would orphan (or, via stripping, degrade) the
      // live query. Keep this assistant and move on.
      if (
        useIds.size > 0 &&
        toolResultsReference(survivors[lastUser], useIds)
      ) {
        i++;
        continue;
      }
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

/** The index of the last Responses `message` item with `role:"user"`. */
const lastResponsesUserIndex = (input: ReadonlyArray<unknown>): number =>
  input.reduceRight<number>(
    (acc, item, idx) =>
      acc === -1 &&
      isRecord(item) &&
      item.type === "message" &&
      item.role === "user"
        ? idx
        : acc,
    -1,
  );

/**
 * Drop the oldest droppable Responses `input` items until the body fits or
 * nothing more can go. Preserves the last user `message` (the live query) and
 * `function_call` ↔ `function_call_output` pairing (dropping a `function_call`
 * takes its matching `function_call_output` with it, and vice-versa), so the
 * upstream never sees an orphaned call or output. Leading `message` items with
 * `role:"system"`/`"developer"` (the instructions prefix) are never dropped.
 */
const dropOldestResponsesItems = (
  input: ReadonlyArray<unknown>,
  fits: (items: ReadonlyArray<unknown>) => boolean,
): unknown[] => {
  const survivors = [...input];
  // Skip a leading system/developer instructions prefix.
  let start = 0;
  while (start < survivors.length) {
    const it = survivors[start];
    if (
      isRecord(it) &&
      it.type === "message" &&
      (it.role === "system" || it.role === "developer")
    ) {
      start++;
      continue;
    }
    break;
  }
  let i = start;
  while (i < survivors.length) {
    if (fits(survivors)) break;
    const lastUser = lastResponsesUserIndex(survivors);
    if (i >= lastUser) break;
    const item = survivors[i];
    if (
      isRecord(item) &&
      item.type === "message" &&
      (item.role === "system" || item.role === "developer")
    ) {
      i++;
      continue;
    }
    // Dropping a function_call / function_call_output must take its pair too.
    if (
      isRecord(item) &&
      (item.type === "function_call" || item.type === "function_call_output") &&
      typeof item.call_id === "string"
    ) {
      const callId = item.call_id;
      survivors.splice(i, 1);
      for (let j = 0; j < survivors.length; ) {
        const other = survivors[j];
        if (
          isRecord(other) &&
          (other.type === "function_call" ||
            other.type === "function_call_output") &&
          other.call_id === callId
        ) {
          survivors.splice(j, 1);
          if (j < i) i--; // a removal before the cursor shifts it left
        } else {
          j++;
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
 * `estimateBodyTokens`). `compacted` is true when the body fits the target
 * (already-fit bodies are returned untouched); it is false only when the body
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
  if (!isRecord(body)) {
    return { body, compacted: false, estimatedTokens: initial };
  }

  // Responses (ChatGPT / Codex): shape is `{ input: item[] }`, handled by its
  // own item-walk rather than the `messages`-array path.
  if (surface === "responses" && Array.isArray(body.input)) {
    return compactResponsesBody(
      body,
      body.input,
      targetTokens,
      initial,
      estimate,
    );
  }

  if (!Array.isArray(body.messages)) {
    // Nothing structured to shrink (e.g. a bare-string Responses `input`, or an
    // unrecognised body).
    return { body, compacted: false, estimatedTokens: initial };
  }

  const anthropic = isAnthropic(surface);
  let messages: unknown[] = [...body.messages];
  const withMessages = (msgs: unknown[]): TRecord => ({
    ...body,
    messages: msgs,
  });

  // Pass 1: truncate tool outputs, trying a deficit-sized cap FIRST (trim ~only
  // what's needed), then the fixed ladder as a coarser fallback. The deficit is
  // solved against target × 0.95 so the per-string ceil() rounding + non-tool
  // overhead in the estimate can't leave the result a hair over the window (the
  // exact-deficit solve landed 128014 on a 128000 target → fell to the coarse
  // ladder and over-shrank).
  const firstCap = firstPassCap(
    messages,
    surface,
    initial - targetTokens * 0.95,
  );
  const caps =
    firstCap === null ? TRUNCATION_PASSES : [firstCap, ...TRUNCATION_PASSES];
  for (const cap of caps) {
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

/**
 * The `responses`-surface arm of {@link compactRequestToFit}: same two-pass
 * ladder (truncate `function_call_output`s → drop oldest items) over the
 * Responses `input[]` array, preserving the last user message and
 * function_call ↔ function_call_output pairing.
 */
const compactResponsesBody = (
  body: TRecord,
  inputItems: ReadonlyArray<unknown>,
  targetTokens: number,
  initialEstimate: number,
  estimate: (b: unknown) => number,
): TCompactionResult => {
  let input: unknown[] = [...inputItems];
  const withInput = (items: unknown[]): TRecord => ({ ...body, input: items });

  // Pass 1: truncate tool outputs, deficit-sized cap first (solved against
  // target × 0.95 — see the messages arm), then the fixed ladder.
  const firstCap = firstPassCap(
    input,
    "responses",
    initialEstimate - targetTokens * 0.95,
  );
  const caps =
    firstCap === null ? TRUNCATION_PASSES : [firstCap, ...TRUNCATION_PASSES];
  for (const cap of caps) {
    const lastUser = lastResponsesUserIndex(input);
    input = truncateResponsesToolOutputs(input, cap, lastUser);
    const est = estimate(withInput(input));
    if (est <= targetTokens) {
      return { body: withInput(input), compacted: true, estimatedTokens: est };
    }
  }

  // Pass 2: drop oldest droppable items (pairing preserved).
  input = dropOldestResponsesItems(
    input,
    (items) => estimate(withInput([...items])) <= targetTokens,
  );

  const finalBody = withInput(input);
  const finalEst = estimate(finalBody);
  return {
    body: finalBody,
    compacted: finalEst <= targetTokens,
    estimatedTokens: finalEst,
  };
};
