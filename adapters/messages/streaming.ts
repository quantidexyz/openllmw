import type {
  TAnthropicStreamEvent,
  TChatCompletionChunk,
} from "@openllmsh/protocol";
import { ensureCompactionSafeVisibleText } from "../../features/compaction/compaction-text";
import { encodeSseEvent } from "../../lib/streaming/sse";
import { upstreamErrorFrom } from "../../lib/streaming/upstream-error";
import { plainTextFromReasoningItems } from "./reasoning-from-items";
import {
  encodeReasoningSignature,
  reasoningItemsFromUnknown,
} from "./reasoning-signature";

/**
 * Shown as the (collapsed) thinking text when an upstream reasoning
 * item carries resumable `encrypted_content` but no human summary. The
 * block still needs *some* text for clients that reject an empty
 * `thinking` — the value that matters is the `signature` we attach.
 */
const REASONING_PLACEHOLDER_TEXT = "[reasoning]";

const stopReasonFor = (
  finish: TChatCompletionChunk["choices"][number]["finish_reason"],
): "end_turn" | "max_tokens" | "tool_use" | "refusal" | null => {
  if (finish === null || finish === undefined) return null;
  switch (finish) {
    case "stop":
      return "end_turn";
    case "length":
      return "max_tokens";
    case "tool_calls":
    case "function_call":
      return "tool_use";
    case "content_filter":
      return "refusal";
  }
};

// runtime-only: stateful translation buffer.
export type TMessagesStreamState = {
  startEmitted: boolean;
  messageId: string;
  model: string;
  inputTokens: number;
  cacheReadTokens: number;
  cacheCreationTokens: number;
  /** Anthropic content_index for the text block. null = not yet opened. */
  textBlockIndex: number | null;
  /** True while the text block is open (between content_block_start and stop). */
  textBlockOpen: boolean;
  /** Responses `reasoning_summary_text.delta` → Anthropic thinking block. */
  thinkingBlockIndex: number | null;
  thinkingBlockOpen: boolean;
  /** OpenAI tool_calls[i].index → Anthropic content_index. */
  toolCallToContentIndex: Map<number, number>;
  /**
   * Tool calls seen but not yet OPENED because their `name` is still empty.
   * Anthropic requires `tool_use.name` at `content_block_start`, and a client
   * (Claude Code) drops a `tool_use` whose name is `""` — so it never runs the
   * tool. The OpenAI Responses wire (grok/chatgpt) can split a tool call's
   * `name` / `id` / argument fragments across separate events, so the first
   * fragment may carry no name. We buffer `id`/`name`/`args` per OpenAI index
   * here and open the block the moment a non-empty name arrives (flushing the
   * buffered args), or belatedly at finish with a synthesized id. Mirrors
   * CLIProxyAPI's accumulate-until-(name)+belated-open (`openai_claude_response.go`).
   */
  pendingToolCalls: Map<number, { id: string; name: string; args: string }>;
  /** True once any tool_use block has been OPENED (announced) for this message. */
  emittedToolUse: boolean;
  /** Anthropic content_index → still-open flag. */
  openToolContentIndexes: Set<number>;
  /** Next free Anthropic content_index. */
  nextContentIndex: number;
  finalStopReason: ReturnType<typeof stopReasonFor>;
  /**
   * Concatenation of all `reasoning_content` / thinking deltas. On
   * terminal chunk, if there was no non-empty `text_delta`, we mirror
   * this into a synthetic `text` block so Claude Code `/compact` sees
   * valid user-visible text (same rule as `toAnthropicMessagesResponse`).
   */
  thinkingAccumulated: string;
  /** True after at least one non-empty `content` delta became `text_delta`. */
  emittedNonemptyTextDelta: boolean;
  /** Set when `message_stop` is emitted — detects truncated upstream streams. */
  messageStopEmitted: boolean;
  /** Concatenation of `delta.content` text deltas (for compaction min-length padding). */
  textAccumulated: string;
  /**
   * How many leading chars of `thinkingAccumulated` were already emitted as
   * `thinking_delta` events. Keeps streaming aligned when `reasoning_items`
   * arrives as a snapshot (final chunk only).
   */
  thinkingDeltaEmittedLen: number;
  /**
   * Encoded `thinking.signature` carrying the upstream's `reasoning`
   * item(s) (Codex/Responses `encrypted_content`). Set when a chunk
   * carries `reasoning_items`; flushed onto the thinking block right
   * before it closes so Claude Code replays it next turn. Without this
   * the model loses chain-of-thought state and loops forever.
   */
  pendingReasoningSignature: string | null;
  /** True once `signature_delta` has been emitted (emit exactly once). */
  reasoningSignatureEmitted: boolean;
  /** Provider-executed hosted searches emitted so far — drives the terminal
   *  `message_delta.usage.server_tool_use.web_search_requests`. */
  serverSearchCount: number;
};

export const newMessagesStreamState = (): TMessagesStreamState => ({
  startEmitted: false,
  messageId: "",
  model: "",
  inputTokens: 0,
  cacheReadTokens: 0,
  cacheCreationTokens: 0,
  textBlockIndex: null,
  textBlockOpen: false,
  thinkingBlockIndex: null,
  thinkingBlockOpen: false,
  toolCallToContentIndex: new Map(),
  pendingToolCalls: new Map(),
  emittedToolUse: false,
  openToolContentIndexes: new Set(),
  nextContentIndex: 0,
  finalStopReason: null,
  thinkingAccumulated: "",
  emittedNonemptyTextDelta: false,
  messageStopEmitted: false,
  textAccumulated: "",
  thinkingDeltaEmittedLen: 0,
  pendingReasoningSignature: null,
  reasoningSignatureEmitted: false,
  serverSearchCount: 0,
});

/**
 * Emit the reasoning `signature_delta` onto the (still-open) thinking
 * block so Claude Code replays it verbatim next turn. Anthropic
 * requires `signature_delta` to be the thinking block's LAST delta,
 * before `content_block_stop` and before any tool_use block opens — so
 * this runs the moment `reasoning_items` arrive (Codex sends them on
 * the reasoning item's `output_item.done`, ahead of the function call).
 */
const emitReasoningSignature = (
  state: TMessagesStreamState,
  out: TAnthropicStreamEvent[],
): void => {
  if (
    state.pendingReasoningSignature === null ||
    state.reasoningSignatureEmitted ||
    state.openToolContentIndexes.size > 0
  ) {
    return;
  }
  if (!state.thinkingBlockOpen) {
    // The signed thinking block carries ONLY the Codex/Responses
    // chain-of-thought STATE (opaque signature). Human-readable
    // reasoning is streamed separately as visible `text`. Close any
    // open text / tool block first so content blocks stay strictly
    // sequential (Anthropic rejects overlapping blocks).
    closeTextBlock(state, out);
    closeAllToolBlocks(state, out);
    const idx = openThinkingBlock(state, out);
    out.push({
      type: "content_block_delta",
      index: idx,
      delta: { type: "thinking_delta", thinking: REASONING_PLACEHOLDER_TEXT },
    });
  }
  if (state.thinkingBlockIndex !== null) {
    out.push({
      type: "content_block_delta",
      index: state.thinkingBlockIndex,
      delta: {
        type: "signature_delta",
        signature: state.pendingReasoningSignature,
      },
    });
    state.reasoningSignatureEmitted = true;
  }
};

const closeThinkingBlock = (
  state: TMessagesStreamState,
  out: TAnthropicStreamEvent[],
): void => {
  if (state.thinkingBlockOpen && state.thinkingBlockIndex !== null) {
    out.push({
      type: "content_block_stop",
      index: state.thinkingBlockIndex,
    });
    state.thinkingBlockOpen = false;
  }
};

const openThinkingBlock = (
  state: TMessagesStreamState,
  out: TAnthropicStreamEvent[],
): number => {
  if (state.thinkingBlockIndex === null) {
    state.thinkingBlockIndex = state.nextContentIndex;
    state.nextContentIndex += 1;
  }
  if (!state.thinkingBlockOpen) {
    out.push({
      type: "content_block_start",
      index: state.thinkingBlockIndex,
      content_block: { type: "thinking", thinking: "" },
    });
    state.thinkingBlockOpen = true;
  }
  return state.thinkingBlockIndex;
};

const closeTextBlock = (
  state: TMessagesStreamState,
  out: TAnthropicStreamEvent[],
): void => {
  if (state.textBlockOpen && state.textBlockIndex !== null) {
    out.push({
      type: "content_block_stop",
      index: state.textBlockIndex,
    });
    state.textBlockOpen = false;
  }
};

const closeAllToolBlocks = (
  state: TMessagesStreamState,
  out: TAnthropicStreamEvent[],
): void => {
  for (const idx of state.openToolContentIndexes) {
    out.push({ type: "content_block_stop", index: idx });
  }
  state.openToolContentIndexes.clear();
  // Fresh tool_call deltas must open new content blocks. If reasoning or
  // another branch stops blocks mid-stream but leaves stale tc.index →
  // Anthropic index mappings, later `input_json_delta` targets a block we
  // already emitted `content_block_stop` for — Claude Code sees prose +
  // truncated pseudo-tools (e.g. literal `<tool_call>` tail).
  state.toolCallToContentIndex.clear();
};

/**
 * Open a tool_use content block for a now-named buffered tool call and flush
 * any argument fragments accumulated while its name was still empty. The block
 * stays open (subsequent fragments stream straight through). `id` is
 * synthesized when the upstream never supplied one, so the block is always
 * executable. Anthropic content blocks are strictly sequential, so any open
 * text / tool / thinking block is closed first.
 */
const openBufferedToolBlock = (
  state: TMessagesStreamState,
  out: TAnthropicStreamEvent[],
  openAiIndex: number,
  pending: { id: string; name: string; args: string },
): number => {
  closeTextBlock(state, out);
  closeAllToolBlocks(state, out);
  closeThinkingBlock(state, out);
  const contentIndex = state.nextContentIndex;
  state.nextContentIndex += 1;
  state.toolCallToContentIndex.set(openAiIndex, contentIndex);
  state.pendingToolCalls.delete(openAiIndex);
  state.emittedToolUse = true;
  state.openToolContentIndexes.add(contentIndex);
  out.push({
    type: "content_block_start",
    index: contentIndex,
    content_block: {
      type: "tool_use",
      id:
        pending.id !== ""
          ? pending.id
          : `toolu_${state.messageId}_${openAiIndex}`,
      name: pending.name,
      input: {},
    },
  });
  if (pending.args.length > 0) {
    out.push({
      type: "content_block_delta",
      index: contentIndex,
      delta: { type: "input_json_delta", partial_json: pending.args },
    });
  }
  return contentIndex;
};

/**
 * Belatedly open every still-pending tool call that accumulated a name but was
 * never opened (its `content_block_start` awaited a non-empty name that only
 * arrived on the terminal event, or the block was reset mid-stream). Called at
 * finish so a named-but-unopened tool still reaches the client. Pending calls
 * that never got a name are dropped (an unnamed tool is unexecutable) — mirrors
 * CLIProxyAPI's belated-emit that skips `accumulator.Name == ""`.
 */
const flushPendingToolBlocks = (
  state: TMessagesStreamState,
  out: TAnthropicStreamEvent[],
): void => {
  const indexes = [...state.pendingToolCalls.keys()].sort((a, b) => a - b);
  for (const idx of indexes) {
    const pending = state.pendingToolCalls.get(idx);
    if (pending === undefined || pending.name === "") {
      state.pendingToolCalls.delete(idx);
      continue;
    }
    const contentIndex = openBufferedToolBlock(state, out, idx, pending);
    out.push({ type: "content_block_stop", index: contentIndex });
    state.openToolContentIndexes.delete(contentIndex);
  }
};

const openTextBlock = (
  state: TMessagesStreamState,
  out: TAnthropicStreamEvent[],
): number => {
  closeThinkingBlock(state, out);
  // A previously-stopped content index can never be reopened (Anthropic
  // streaming indices are unique + monotonic). If the text block was
  // closed — e.g. a signed thinking block or a tool block was emitted
  // in between — allocate a FRESH index instead of resurrecting the
  // stopped one.
  if (state.textBlockIndex === null || !state.textBlockOpen) {
    state.textBlockIndex = state.nextContentIndex;
    state.nextContentIndex += 1;
  }
  if (!state.textBlockOpen) {
    out.push({
      type: "content_block_start",
      index: state.textBlockIndex,
      content_block: { type: "text", text: "" },
    });
    state.textBlockOpen = true;
  }
  return state.textBlockIndex;
};

/**
 * Translate one OpenAI ChatCompletion chunk into zero or more
 * Anthropic SSE events. Handles both text deltas and tool_call deltas:
 * each new tool_call opens a new content_block(tool_use); subsequent
 * deltas for that tool_call emit input_json_delta events.
 */
export const chunkToMessagesEvents = (
  chunk: TChatCompletionChunk,
  state: TMessagesStreamState,
): TAnthropicStreamEvent[] => {
  const out: TAnthropicStreamEvent[] = [];

  if (!state.startEmitted) {
    state.startEmitted = true;
    state.messageId = chunk.id;
    state.model = chunk.model;
    out.push({
      type: "message_start",
      message: {
        id: state.messageId,
        type: "message",
        role: "assistant",
        model: state.model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    });
  }

  const choice = chunk.choices[0];
  const deltaReasoning = choice?.delta.reasoning_content ?? null;
  const deltaText = choice?.delta.content ?? null;
  const deltaToolCalls = choice?.delta.tool_calls ?? null;

  if (
    deltaReasoning !== null &&
    deltaReasoning !== undefined &&
    deltaReasoning.length > 0
  ) {
    // Reasoning text WITHOUT a replay-safe signature must be visible
    // `text`, never a `thinking` block. Anthropic hard-rejects a
    // signature-less thinking block the moment Claude Code replays the
    // assistant turn (`thinking.signature: Field required`), which
    // triggers a client retry storm and breaks prompt-cache reuse —
    // draining the user's subscription. The Codex/Responses
    // chain-of-thought STATE still rides the signed thinking block
    // opened by `emitReasoningSignature`. `thinkingAccumulated` /
    // `thinkingDeltaEmittedLen` stay as the reasoning-dedup ledger
    // shared with the `reasoning_items` snapshot path below.
    state.thinkingAccumulated += deltaReasoning;
    const reasoningTail = state.thinkingAccumulated.slice(
      state.thinkingDeltaEmittedLen,
    );
    if (reasoningTail.length > 0 && state.openToolContentIndexes.size === 0) {
      const idx = openTextBlock(state, out);
      out.push({
        type: "content_block_delta",
        index: idx,
        delta: { type: "text_delta", text: reasoningTail },
      });
      state.thinkingDeltaEmittedLen = state.thinkingAccumulated.length;
      state.emittedNonemptyTextDelta = true;
      state.textAccumulated += reasoningTail;
    }
  }

  const reasoningItems = reasoningItemsFromUnknown(
    choice?.delta.reasoning_items,
  );
  if (reasoningItems.length > 0) {
    const sig = encodeReasoningSignature(reasoningItems);
    if (sig !== null) state.pendingReasoningSignature = sig;
  }

  const fromReasoningItems = plainTextFromReasoningItems(
    choice?.delta.reasoning_items,
  );
  // The `reasoning_items` snapshot carries the same human-readable
  // summary as the `reasoning_content` deltas, just delivered as a
  // growing snapshot. Emit only the clean tail beyond what was already
  // streamed (as visible `text`, same rule as above). A divergent
  // snapshot (not a superset of what we already sent) is ignored rather
  // than re-emitted — we cannot retract already-streamed text, and the
  // `reasoning_content` deltas already conveyed it.
  if (
    fromReasoningItems.length > state.thinkingAccumulated.length &&
    fromReasoningItems.startsWith(state.thinkingAccumulated)
  ) {
    state.thinkingAccumulated = fromReasoningItems;
    const itemsTail = state.thinkingAccumulated.slice(
      state.thinkingDeltaEmittedLen,
    );
    if (itemsTail.length > 0 && state.openToolContentIndexes.size === 0) {
      const idx = openTextBlock(state, out);
      out.push({
        type: "content_block_delta",
        index: idx,
        delta: { type: "text_delta", text: itemsTail },
      });
      state.thinkingDeltaEmittedLen = state.thinkingAccumulated.length;
      state.emittedNonemptyTextDelta = true;
      state.textAccumulated += itemsTail;
    }
  }

  // Reasoning state arrived (Codex `reasoning` item completed) — seal it
  // onto the thinking block now, while it is still open and before any
  // tool_use block, so Claude Code replays the `signature` next turn.
  emitReasoningSignature(state, out);

  // Provider-executed hosted searches (Codex `webSearch` items on a chatgpt
  // hop) → self-contained server_tool_use + web_search_tool_result block
  // pairs. Blocks must stay strictly sequential, so anything open closes
  // first; codex interleaves commentary text → search → answer text, and
  // `openTextBlock` allocates a fresh index for the post-search text. The
  // result content is an empty list — Codex never exposes result items; the
  // findings ride the grounded answer text (see the JSON adapter's note).
  const deltaSearches = choice?.delta.server_search_calls ?? null;
  if (deltaSearches !== null && deltaSearches !== undefined) {
    for (const search of deltaSearches) {
      closeTextBlock(state, out);
      closeAllToolBlocks(state, out);
      closeThinkingBlock(state, out);
      const useIndex = state.nextContentIndex;
      state.nextContentIndex += 1;
      out.push({
        type: "content_block_start",
        index: useIndex,
        content_block: {
          type: "server_tool_use",
          id: search.id,
          name: "web_search",
          input: {},
        },
      });
      out.push({
        type: "content_block_delta",
        index: useIndex,
        delta: {
          type: "input_json_delta",
          partial_json: JSON.stringify({ query: search.query }),
        },
      });
      out.push({ type: "content_block_stop", index: useIndex });
      const resultIndex = state.nextContentIndex;
      state.nextContentIndex += 1;
      out.push({
        type: "content_block_start",
        index: resultIndex,
        content_block: {
          type: "web_search_tool_result",
          tool_use_id: search.id,
          content: (search.results ?? []).map((r) => ({
            type: "web_search_result",
            url: r.url,
            title: r.title ?? r.url,
          })),
        },
      });
      out.push({ type: "content_block_stop", index: resultIndex });
      state.serverSearchCount += 1;
    }
  }

  // Text delta → open text block (if not already) + content_block_delta.
  if (deltaText !== null && deltaText !== undefined && deltaText.length > 0) {
    const idx = openTextBlock(state, out);
    out.push({
      type: "content_block_delta",
      index: idx,
      delta: { type: "text_delta", text: deltaText },
    });
    state.emittedNonemptyTextDelta = true;
    state.textAccumulated += deltaText;
  }

  // Tool-call deltas → open tool_use blocks + input_json_delta events.
  if (deltaToolCalls !== null && deltaToolCalls !== undefined) {
    for (const tc of deltaToolCalls) {
      const contentIndex = state.toolCallToContentIndex.get(tc.index);
      if (contentIndex !== undefined) {
        // Block already opened — stream this argument fragment straight through.
        const argFragment = tc.function?.arguments ?? "";
        if (argFragment.length > 0) {
          out.push({
            type: "content_block_delta",
            index: contentIndex,
            delta: { type: "input_json_delta", partial_json: argFragment },
          });
        }
        continue;
      }
      // Not opened yet — accumulate id / name / args in the pending buffer.
      // We CANNOT open a tool_use block until we have a non-empty `name`
      // (Anthropic requires it at `content_block_start`, and Claude Code
      // silently drops a nameless tool_use — the sub-agent never spawns). The
      // Responses wire may deliver name/id/args across separate events, so the
      // first fragment can be nameless.
      const pending = state.pendingToolCalls.get(tc.index) ?? {
        id: "",
        name: "",
        args: "",
      };
      if (tc.id != null && tc.id !== "") pending.id = tc.id;
      const fragmentName = tc.function?.name;
      if (fragmentName != null && fragmentName !== "") {
        pending.name = fragmentName;
      }
      pending.args += tc.function?.arguments ?? "";
      state.pendingToolCalls.set(tc.index, pending);
      // The moment we know the name, open the block and flush buffered args.
      if (pending.name !== "") {
        openBufferedToolBlock(state, out, tc.index, pending);
      }
    }
  }

  if (chunk.usage != null) {
    state.inputTokens = chunk.usage.prompt_tokens;
    state.cacheReadTokens =
      chunk.usage.prompt_tokens_details?.cached_tokens ?? 0;
    state.cacheCreationTokens =
      chunk.usage.prompt_tokens_details?.cache_creation_tokens ?? 0;
  }

  const turnEnds =
    choice?.finish_reason !== null && choice?.finish_reason !== undefined;

  // Live token feed: a usage-bearing chunk that does NOT end the turn —
  // the running estimate synthesized by `withLiveUsageEstimate`, or an
  // upstream that reports incremental usage — emits a standalone
  // `message_delta` so a CLI's token counter climbs mid-stream instead
  // of staying at zero until completion. No `stop_reason`, no
  // `message_stop`: the turn is still open and the terminal
  // `message_delta` below reconciles to the provider's exact totals.
  if (
    chunk.usage != null &&
    !turnEnds &&
    state.startEmitted &&
    !state.messageStopEmitted
  ) {
    out.push({
      type: "message_delta",
      delta: { stop_reason: null, stop_sequence: null },
      usage: {
        output_tokens: chunk.usage.completion_tokens,
        input_tokens: state.inputTokens,
        ...(state.cacheCreationTokens > 0
          ? { cache_creation_input_tokens: state.cacheCreationTokens }
          : {}),
        ...(state.cacheReadTokens > 0
          ? { cache_read_input_tokens: state.cacheReadTokens }
          : {}),
      },
    });
  }

  if (choice?.finish_reason !== null && choice?.finish_reason !== undefined) {
    let finalStopReason = stopReasonFor(choice.finish_reason);
    // A tool call whose name only arrived on the terminal event (or never
    // streamed a `content_block_start` because its name stayed empty until
    // now) is opened + closed here so a named-but-unopened `Task`/`Agent`
    // call still reaches the client. Runs BEFORE the stop-reason override
    // below (it sets `emittedToolUse` when it opens a block).
    flushPendingToolBlocks(state, out);
    // If the upstream emitted tool_call deltas during the stream but
    // wrongly settled on `finish_reason: "stop"`, override to
    // `tool_use`. Without this, Claude Code receives a
    // `stop_reason: "end_turn"` alongside `tool_use` blocks and never
    // calls the tool back — observed on chatgpt.com Responses API.
    if (
      state.emittedToolUse &&
      (finalStopReason === null || finalStopReason === "end_turn")
    ) {
      finalStopReason = "tool_use";
    }
    state.finalStopReason = finalStopReason;
    closeThinkingBlock(state, out);
    if (state.textBlockOpen && state.textBlockIndex !== null) {
      const trimmed = state.textAccumulated.trim();
      if (trimmed.length > 0) {
        const safe = ensureCompactionSafeVisibleText(state.textAccumulated);
        if (safe.length > trimmed.length && safe.startsWith(trimmed)) {
          out.push({
            type: "content_block_delta",
            index: state.textBlockIndex,
            delta: { type: "text_delta", text: safe.slice(trimmed.length) },
          });
        }
      }
    }
    closeTextBlock(state, out);
    closeAllToolBlocks(state, out);
    // A signature received while a tool was open is deferred with its
    // reasoning text, then emitted only after the tool block is sealed.
    emitReasoningSignature(state, out);
    closeThinkingBlock(state, out);
    const deferredReasoning = state.thinkingAccumulated.slice(
      state.thinkingDeltaEmittedLen,
    );
    if (deferredReasoning.length > 0) {
      const idx = openTextBlock(state, out);
      out.push({
        type: "content_block_delta",
        index: idx,
        delta: { type: "text_delta", text: deferredReasoning },
      });
      state.thinkingDeltaEmittedLen = state.thinkingAccumulated.length;
      state.emittedNonemptyTextDelta = true;
      state.textAccumulated += deferredReasoning;
      closeTextBlock(state, out);
    }
    if (
      !state.emittedNonemptyTextDelta &&
      state.thinkingAccumulated.length > 0
    ) {
      const idx = openTextBlock(state, out);
      out.push({
        type: "content_block_delta",
        index: idx,
        delta: {
          type: "text_delta",
          text: ensureCompactionSafeVisibleText(state.thinkingAccumulated),
        },
      });
      closeTextBlock(state, out);
    }

    const outputTokens = chunk.usage?.completion_tokens ?? 0;
    out.push({
      type: "message_delta",
      delta: {
        stop_reason: state.finalStopReason,
        stop_sequence: null,
      },
      usage: {
        output_tokens: outputTokens,
        input_tokens: state.inputTokens,
        ...(state.cacheCreationTokens > 0
          ? { cache_creation_input_tokens: state.cacheCreationTokens }
          : {}),
        ...(state.cacheReadTokens > 0
          ? { cache_read_input_tokens: state.cacheReadTokens }
          : {}),
        ...(state.serverSearchCount > 0
          ? {
              server_tool_use: {
                web_search_requests: state.serverSearchCount,
              },
            }
          : {}),
      },
    });
    out.push({ type: "message_stop" });
    state.messageStopEmitted = true;
  }
  return out;
};

/**
 * Encode an Anthropic stream event as SSE bytes. Anthropic uses an
 * `event: <name>\n` prefix line (not just `data:` like OpenAI), so we
 * can't reuse `encodeSseEvent` directly.
 */
export const encodeAnthropicSseEvent = (
  event: TAnthropicStreamEvent,
): Uint8Array => {
  const lines = `event: ${event.type}\n`;
  const body = encodeSseEvent(event);
  const prefix = new TextEncoder().encode(lines);
  const out = new Uint8Array(prefix.byteLength + body.byteLength);
  out.set(prefix, 0);
  out.set(body, prefix.byteLength);
  return out;
};

/**
 * Pipe a stream of OpenAI ChatCompletion chunks into Anthropic-format
 * SSE bytes. Used by the `/v1/messages` handler when the runner
 * returned a streaming outcome.
 */
export const chunksToMessagesSseBytes = (
  chunks: ReadableStream<TChatCompletionChunk>,
): ReadableStream<Uint8Array> => {
  const reader = chunks.getReader();
  const state = newMessagesStreamState();
  const buffer: Uint8Array[] = [];
  // One-chunk lookahead. The OpenAI streaming spec delivers token
  // counts in a SEPARATE trailing chunk (`choices: []`, `usage` set,
  // emitted under `stream_options.include_usage`) that arrives AFTER
  // the `finish_reason` chunk. The terminal `message_delta` is built
  // from the finish chunk, so without folding that trailing usage in
  // first the CLI feed gets `usage:{input_tokens:0,output_tokens:0}` —
  // i.e. no token counter once you go through the proxy. Providers
  // that already put usage on the finish chunk are untouched.
  let pending: TChatCompletionChunk | null = null;
  const readChunk = async (): Promise<
    { value: TChatCompletionChunk; done: false } | { done: true }
  > => {
    if (pending !== null) {
      const v = pending;
      pending = null;
      return { value: v, done: false };
    }
    const r = await reader.read();
    return r.done ? { done: true } : { value: r.value, done: false };
  };
  const isUsageOnly = (c: TChatCompletionChunk): boolean =>
    c.choices.length === 0 && c.usage != null;
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        for (;;) {
          if (buffer.length > 0) {
            const next = buffer.shift();
            if (next !== undefined) controller.enqueue(next);
            return;
          }
          const read = await readChunk();
          if (read.done) {
            if (state.startEmitted && !state.messageStopEmitted) {
              // Upstream ended WITHOUT a finish_reason — the stream was
              // cut (Vercel maxDuration hard-kill, provider drop,
              // client/network abort). This terminal is synthetic: we
              // do NOT know the turn completed. It must signal
              // truncation (`length` → Anthropic `max_tokens`), never
              // `stop`. `stop` maps to `end_turn`, which the
              // tool_use override (chunkToMessagesEvents) promotes to
              // `stop_reason: "tool_use"` whenever a tool block was
              // opened — promising Claude Code an executable tool whose
              // `input_json_delta` is a truncated, unparseable JSON
              // fragment. Claude Code then blocks forever trying to
              // JSON.parse it ("announced an action then froze"). With
              // `length` the override (null/end_turn only) does not
              // fire, the client sees an honest cut turn and re-prompts
              // to resume — same contract as `withStreamDeadline`.
              const tail = chunkToMessagesEvents(
                {
                  id:
                    state.messageId !== ""
                      ? state.messageId
                      : "chatcmpl-truncated",
                  object: "chat.completion.chunk",
                  created: Math.floor(Date.now() / 1000),
                  model: state.model !== "" ? state.model : "unknown",
                  choices: [
                    {
                      index: 0,
                      delta: {},
                      finish_reason: "length",
                    },
                  ],
                },
                state,
              );
              for (const e of tail) {
                buffer.push(encodeAnthropicSseEvent(e));
              }
            }
            while (buffer.length > 0) {
              const next = buffer.shift();
              if (next !== undefined) controller.enqueue(next);
            }
            controller.close();
            return;
          }
          let value = read.value;
          // If this chunk ends the turn but has no usage yet, peek the
          // next one: a trailing usage-only chunk gets folded in so the
          // terminal `message_delta` carries real input/output/cache
          // tokens. Anything else is stashed and processed next.
          const endsTurn = value.choices.some(
            (c) => c.finish_reason != null && c.finish_reason !== undefined,
          );
          if (endsTurn && value.usage == null) {
            const la = await readChunk();
            if (!la.done) {
              if (isUsageOnly(la.value)) {
                value = { ...value, usage: la.value.usage };
              } else {
                pending = la.value;
              }
            }
          }
          const events = chunkToMessagesEvents(value, state);
          for (const e of events) buffer.push(encodeAnthropicSseEvent(e));
        }
      } catch (err) {
        // Upstream errored mid-stream — for example Anthropic's
        // `event: error` (overloaded_error/api_error). Surface as an
        // Anthropic-format error event so Claude Code sees a clean
        // failure rather than an abruptly closed stream (which is
        // exactly what manifested as "compaction fails at 20%").
        const { type, message } = upstreamErrorFrom(err);
        buffer.push(
          encodeAnthropicSseEvent({
            type: "error",
            error: { type, message },
          }),
        );
        const next = buffer.shift();
        if (next !== undefined) controller.enqueue(next);
        controller.close();
      }
    },
    cancel(reason) {
      reader.cancel(reason).catch(() => {});
    },
  });
};
