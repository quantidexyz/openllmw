import type {
  TAnthropicProviderOptions,
  TAnthropicStopReason,
  TAnthropicStreamEvent,
  TChatCompletionChunk,
  TUsage,
} from "@openllmsh/protocol";
import { AnthropicStreamEvent } from "@openllmsh/protocol";
import { decodeProviderEventStream } from "../../lib/streaming/provider-decode";
import { UpstreamStreamError } from "../../lib/streaming/upstream-error";

const compactionPayloadToText = (raw: unknown): string => {
  if (raw === null || raw === undefined) return "";
  if (typeof raw === "string") return raw;
  if (typeof raw === "number" || typeof raw === "boolean") return String(raw);
  try {
    return JSON.stringify(raw);
  } catch {
    return "";
  }
};

const finishReasonFor = (
  stop: TAnthropicStopReason | null,
): TChatCompletionChunk["choices"][number]["finish_reason"] => {
  if (stop === null) return null;
  switch (stop) {
    case "end_turn":
    case "stop_sequence":
    case "pause_turn":
      return "stop";
    case "max_tokens":
      return "length";
    case "tool_use":
      return "tool_calls";
    case "refusal":
      return "content_filter";
  }
};

// runtime-only: mutable streaming state held across calls.
export type TAnthropicStreamState = {
  id: string | null;
  model: string;
  inputTokens: number;
  cacheCreationTokens: number;
  cacheReadTokens: number;
  created: number;
  /** Map of Anthropic content-block index → OpenAI tool_calls index. */
  toolCallIndexFor: Map<number, number>;
  /** Next OpenAI tool_calls index to assign. */
  nextToolCallIndex: number;
};

export const newAnthropicStreamState = (
  options: TAnthropicProviderOptions,
): TAnthropicStreamState => ({
  id: null,
  model: options.providerModelId,
  inputTokens: 0,
  cacheCreationTokens: 0,
  cacheReadTokens: 0,
  created: Math.floor(Date.now() / 1000),
  toolCallIndexFor: new Map(),
  nextToolCallIndex: 0,
});

export const fromAnthropicStreamEvent = (
  event: TAnthropicStreamEvent,
  state: TAnthropicStreamState,
  _options: TAnthropicProviderOptions,
): TChatCompletionChunk | null => {
  if (event.type === "message_start") {
    state.id = event.message.id;
    state.inputTokens = event.message.usage.input_tokens;
    state.cacheCreationTokens =
      event.message.usage.cache_creation_input_tokens ?? 0;
    state.cacheReadTokens = event.message.usage.cache_read_input_tokens ?? 0;
    return {
      id: state.id,
      object: "chat.completion.chunk",
      created: state.created,
      model: state.model,
      choices: [
        {
          index: 0,
          delta: { role: "assistant", content: "" },
          finish_reason: null,
        },
      ],
    };
  }

  if (event.type === "content_block_start") {
    if (event.content_block.type === "tool_use") {
      const toolCallIndex = state.nextToolCallIndex;
      state.nextToolCallIndex += 1;
      state.toolCallIndexFor.set(event.index, toolCallIndex);
      return {
        id: state.id ?? "",
        object: "chat.completion.chunk",
        created: state.created,
        model: state.model,
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: toolCallIndex,
                  id: event.content_block.id,
                  type: "function",
                  function: { name: event.content_block.name, arguments: "" },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      };
    }
    if (event.content_block.type === "compaction") {
      const text = compactionPayloadToText(event.content_block.content);
      if (text.length === 0) return null;
      return {
        id: state.id ?? "",
        object: "chat.completion.chunk",
        created: state.created,
        model: state.model,
        choices: [
          {
            index: 0,
            delta: { content: text },
            finish_reason: null,
          },
        ],
      };
    }
    return null;
  }

  if (event.type === "content_block_delta") {
    if (event.delta.type === "text_delta") {
      return {
        id: state.id ?? "",
        object: "chat.completion.chunk",
        created: state.created,
        model: state.model,
        choices: [
          {
            index: 0,
            delta: { content: event.delta.text },
            finish_reason: null,
          },
        ],
      };
    }
    if (event.delta.type === "input_json_delta") {
      const toolCallIndex = state.toolCallIndexFor.get(event.index);
      if (toolCallIndex === undefined) return null;
      return {
        id: state.id ?? "",
        object: "chat.completion.chunk",
        created: state.created,
        model: state.model,
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: toolCallIndex,
                  function: { arguments: event.delta.partial_json },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      };
    }
    if (event.delta.type === "compaction_delta") {
      const text = compactionPayloadToText(event.delta.content);
      if (text.length === 0) return null;
      return {
        id: state.id ?? "",
        object: "chat.completion.chunk",
        created: state.created,
        model: state.model,
        choices: [
          {
            index: 0,
            delta: { content: text },
            finish_reason: null,
          },
        ],
      };
    }
    return null;
  }

  if (event.type === "error") {
    // Anthropic emits `event: error` mid-stream for overloaded_error,
    // api_error, etc. Throwing here puts the canonical chunk stream into
    // an errored state; the surface-specific encoder (chunksToSseBytes /
    // chunksToMessagesSseBytes) catches the rejection and emits a proper
    // trailing error frame instead of silently truncating.
    throw new UpstreamStreamError(event.error.type, event.error.message);
  }

  if (event.type === "message_delta") {
    const finish = finishReasonFor(event.delta.stop_reason);
    // `state.inputTokens` holds Anthropic's `input_tokens`, which EXCLUDES the
    // cache fields. Canonical `prompt_tokens` includes them — see `usageFor`
    // in ./response.ts.
    const promptTokens =
      state.inputTokens + state.cacheReadTokens + state.cacheCreationTokens;
    const usage: TUsage = {
      prompt_tokens: promptTokens,
      completion_tokens: event.usage.output_tokens,
      total_tokens: promptTokens + event.usage.output_tokens,
      ...(state.cacheCreationTokens > 0 || state.cacheReadTokens > 0
        ? {
            prompt_tokens_details: {
              cached_tokens: state.cacheReadTokens,
              cache_creation_tokens: state.cacheCreationTokens,
            },
          }
        : {}),
    };
    return {
      id: state.id ?? "",
      object: "chat.completion.chunk",
      created: state.created,
      model: state.model,
      choices: [
        {
          index: 0,
          delta: {},
          finish_reason: finish,
        },
      ],
      usage,
    };
  }

  return null;
};

/**
 * Decode a raw Anthropic SSE byte stream into canonical chunks — the ONE
 * place the `(AnthropicStreamEvent, newAnthropicStreamState,
 * fromAnthropicStreamEvent)` decode triple is wired together, shared by
 * the daemon walker's upstream decode and the cloud dispatch chain's
 * passthrough peek (the core provider SPEC carries the same members as
 * data for the schema-driven runner).
 */
export const decodeAnthropicEventStream = (
  raw: ReadableStream<Uint8Array>,
  providerModelId: string,
): ReadableStream<TChatCompletionChunk> =>
  decodeProviderEventStream(
    raw,
    {
      eventSchema: AnthropicStreamEvent,
      initialState: newAnthropicStreamState,
      eventToChunk: fromAnthropicStreamEvent,
    },
    { providerModelId },
  );
