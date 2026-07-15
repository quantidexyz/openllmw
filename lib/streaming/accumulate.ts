import type {
  TChatCompletionChunk,
  TChatCompletionResponse,
  TToolCall,
} from "@quantidexyz/openllmp";

import { plainTextFromReasoningItems } from "../../adapters/messages/reasoning-from-items";

// runtime-only: builder state while we walk a chunk stream. Tool calls
// arrive as `{index, id?, function:{name?, arguments?}}` deltas that
// must be reassembled into the final `tool_calls` array.
type TToolCallBuilder = {
  id: string;
  name: string;
  arguments: string;
};

/**
 * Drain a stream of ChatCompletion chunks into a single non-streaming
 * response. Used for providers that ONLY stream upstream (chatgpt.com
 * `/backend-api/codex/responses`) so non-streaming callers still get a
 * `chat.completion` JSON envelope.
 *
 * Concatenates `delta.content` into a single message string, rebuilds
 * `tool_calls` from indexed deltas, and uses the final chunk's
 * `finish_reason` / `usage` if present.
 */
export const accumulateChunksToResponse = async (
  chunks: ReadableStream<TChatCompletionChunk>,
  providerModelId: string,
): Promise<TChatCompletionResponse> => {
  const reader = chunks.getReader();
  let content = "";
  let finishReason: TChatCompletionResponse["choices"][number]["finish_reason"] =
    null;
  let usage: TChatCompletionResponse["usage"] = {
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
  };
  let id = "";
  let created = Math.floor(Date.now() / 1000);
  const toolCalls = new Map<number, TToolCallBuilder>();
  let reasoningContent = "";
  let reasoningItems: ReadonlyArray<unknown> | undefined;

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value.id !== "") id = value.id;
      if (value.created !== 0) created = value.created;
      const choice = value.choices[0];
      if (choice === undefined) {
        if (value.usage !== undefined && value.usage !== null) {
          usage = value.usage;
        }
        continue;
      }
      const delta = choice.delta;
      if (delta !== undefined) {
        if (typeof delta.content === "string") content += delta.content;
        if (typeof delta.reasoning_content === "string") {
          reasoningContent += delta.reasoning_content;
        }
        if (
          Array.isArray(delta.reasoning_items) &&
          delta.reasoning_items.length > 0
        ) {
          reasoningItems = delta.reasoning_items;
        }
        if (Array.isArray(delta.tool_calls)) {
          for (const tc of delta.tool_calls) {
            const existing = toolCalls.get(tc.index) ?? {
              id: "",
              name: "",
              arguments: "",
            };
            const next: TToolCallBuilder = {
              id: tc.id ?? existing.id,
              name: tc.function?.name ?? existing.name,
              arguments: existing.arguments + (tc.function?.arguments ?? ""),
            };
            toolCalls.set(tc.index, next);
          }
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
        finishReason = choice.finish_reason;
      }
      if (value.usage !== undefined && value.usage !== null) {
        usage = value.usage;
      }
    }
  } catch (err) {
    // A stream failure AFTER the terminal finish_reason chunk (a trailing
    // upstream error event, a connection reset while draining the tail)
    // cannot change the already-complete answer — salvage it instead of
    // discarding the whole turn (issue #274). A failure BEFORE the terminal
    // chunk still rejects: the answer is incomplete and the error is the
    // only truthful outcome.
    if (finishReason === null) throw err;
    // Same debug gate as provider-decode's dropped-chunk warning: keep the
    // salvage observable without adding a logger dep to the pure wire layer.
    if (process.env.OPENLLM_DEBUG_STREAM === "1") {
      console.warn(
        "[accumulateChunksToResponse] salvaged completed answer after trailing stream error:",
        err instanceof Error ? err.message : String(err),
      );
    }
  } finally {
    // Release the underlying chunk source promptly when the caller
    // bails (deadline cutoff, response abort, …) so the upstream
    // `fetch` body doesn't stay open waiting for our next read.
    reader.cancel().catch(() => {});
  }

  const finalToolCalls: TToolCall[] = [];
  const indices = [...toolCalls.keys()].sort((a, b) => a - b);
  for (const idx of indices) {
    const tc = toolCalls.get(idx);
    if (tc === undefined) continue;
    finalToolCalls.push({
      id: tc.id,
      type: "function",
      function: { name: tc.name, arguments: tc.arguments },
    });
  }

  // If the upstream forgot to set `finish_reason: "tool_calls"` while
  // emitting tool_call deltas (observed on chatgpt.com `/codex/responses`
  // for `gpt-5.x-codex`), force it. Without this, the messages adapter
  // maps `stop` → `end_turn`, which tells Claude Code the turn is over
  // and the tool_use block goes unused.
  const effectiveFinishReason: TChatCompletionResponse["choices"][number]["finish_reason"] =
    finalToolCalls.length > 0 &&
    (finishReason === null || finishReason === "stop")
      ? "tool_calls"
      : (finishReason ?? "stop");

  const reasoningFromItems = plainTextFromReasoningItems(reasoningItems);
  /** OpenLLM: surface ref `Delta.reasoning_items` text as `reasoning_content` when deltas omitted. */
  const effectiveReasoningContent =
    reasoningContent.length > 0 ? reasoningContent : reasoningFromItems;

  return {
    id: id !== "" ? id : `chatcmpl-${crypto.randomUUID()}`,
    object: "chat.completion",
    created,
    model: providerModelId,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content,
          ...(finalToolCalls.length > 0 ? { tool_calls: finalToolCalls } : {}),
          ...(effectiveReasoningContent.length > 0
            ? { reasoning_content: effectiveReasoningContent }
            : {}),
          ...(reasoningItems !== undefined
            ? { reasoning_items: [...reasoningItems] }
            : {}),
        },
        finish_reason: effectiveFinishReason,
      },
    ],
    usage,
  };
};
