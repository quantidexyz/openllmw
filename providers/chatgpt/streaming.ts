import type {
  TChatCompletionChunk,
  TChatGptProviderOptions,
} from "@quantidexyz/openllmp";
import {
  buildReasoningItem,
  type TReasoningItem,
} from "../../adapters/messages/reasoning-signature";
import { UpstreamStreamError } from "../../lib/streaming/upstream-error";

// runtime-only: Responses API events arrive as freeform JSON dicts. We
// hand-discriminate on the `type` field. A typed Schema would have to
// enumerate every Responses API event the chatgpt.com endpoint emits,
// which drifts often — Schema.Unknown + structural checks is more
// resilient and matches LiteLLM's `chunk_parser` strategy.
export type TChatGptStreamEvent = Record<string, unknown>;

// runtime-only: per-stream state. We track whether we have observed
// any tool-call output item so that the final `response.completed`
// event can choose `finish_reason: "tool_calls"` even when the
// terminal event's `response.output` payload omits the tool-call
// entries (which we have seen in practice on `gpt-5.x-codex`).
//
// `reasoningItems` collects every `reasoning` output item (keyed by its
// `id`, insertion-ordered) as it streams. Codex emits these with
// `encrypted_content` when the request set
// `include: ["reasoning.encrypted_content"]`; they MUST be echoed back
// next turn or the model loops forever.
//
// Codex output order is `reasoning` → `function_call`, and the reasoning
// item's full `encrypted_content` lands on its `response.output_item.done`
// BEFORE the function call streams. We emit `delta.reasoning_items` right
// there so the Anthropic adapter can attach the `signature` to the
// still-open thinking block (Anthropic requires `signature_delta` to be
// the thinking block's last delta, before any tool_use block opens).
// `response.completed` re-folds `response.output[]` as a fallback for
// responses that omit per-item `.done`. `emittedReasoningIds` dedupes so
// a given item round-trips exactly once. Mirrors litellm
// `transformation.py:1321-1356`.
export type TChatGptStreamState = {
  hasToolCall: boolean;
  reasoningItems: Map<string, TReasoningItem>;
  emittedReasoningIds: Set<string>;
  /**
   * `output_index`es that received ≥1 `function_call_arguments.delta`.
   * Some Codex backends (the "codex-spark" case) emit the completed
   * arguments ONLY via `function_call_arguments.done` /
   * `output_item.done` with NO `.delta` events. Without finalizing
   * from `.done`, the tool call reaches the model with empty `{}`
   * arguments → tool fails → the model re-issues it → loop. We must
   * emit the `.done` arguments, but ONLY when no `.delta` already
   * streamed them (else the accumulator double-concatenates and the
   * JSON corrupts). Ref: litellm Responses transformation; langchainjs#8049.
   */
  argsStreamedIndexes: Set<number>;
  /** `output_index`es whose authoritative `.done` args were emitted. */
  argsFinalizedIndexes: Set<number>;
};

export const newChatGptStreamState = (
  _options: TChatGptProviderOptions,
): TChatGptStreamState => ({
  hasToolCall: false,
  reasoningItems: new Map(),
  emittedReasoningIds: new Set(),
  argsStreamedIndexes: new Set(),
  argsFinalizedIndexes: new Set(),
});

/**
 * Emit the authoritative complete `arguments` for a function call that
 * streamed no `.delta`s. Returns null when the args already streamed
 * (or were already finalized, or are empty) so we never double-count.
 */
const finalizeToolArgs = (
  state: TChatGptStreamState,
  outputIndex: number,
  args: string,
  options: TChatGptProviderOptions,
): TChatCompletionChunk | null => {
  if (
    args.length === 0 ||
    state.argsStreamedIndexes.has(outputIndex) ||
    state.argsFinalizedIndexes.has(outputIndex)
  ) {
    return null;
  }
  state.argsFinalizedIndexes.add(outputIndex);
  state.hasToolCall = true;
  return {
    ...baseChunk(options),
    choices: [
      {
        index: 0,
        delta: {
          tool_calls: [
            {
              index: outputIndex,
              type: "function",
              function: { arguments: args },
            },
          ],
        },
        finish_reason: null,
      },
    ],
  };
};

const captureReasoningItem = (
  state: TChatGptStreamState,
  item: Record<string, unknown>,
): TReasoningItem | null => {
  if (stringField(item, "type") !== "reasoning") return null;
  const id = stringField(item, "id") ?? "";
  const built = buildReasoningItem(
    id,
    stringField(item, "encrypted_content") ?? null,
    item.summary,
  );
  // `output_item.added` carries no `encrypted_content`; `.done` /
  // `response.completed` do. Last write wins so the final (complete)
  // item — the one with the resumable blob — is what we round-trip.
  state.reasoningItems.set(id, built);
  return built;
};

/** Reasoning items captured but not yet emitted, in insertion order. */
const drainUnemittedReasoning = (
  state: TChatGptStreamState,
): TReasoningItem[] => {
  const out: TReasoningItem[] = [];
  for (const [id, item] of state.reasoningItems) {
    if (state.emittedReasoningIds.has(id)) continue;
    state.emittedReasoningIds.add(id);
    out.push(item);
  }
  return out;
};

const baseChunk = (
  options: TChatGptProviderOptions,
): Pick<TChatCompletionChunk, "id" | "object" | "created" | "model"> => ({
  id: `chatcmpl-${crypto.randomUUID()}`,
  object: "chat.completion.chunk",
  created: Math.floor(Date.now() / 1000),
  model: options.providerModelId,
});

const stringField = (
  obj: Record<string, unknown>,
  key: string,
): string | undefined => {
  const v = obj[key];
  return typeof v === "string" ? v : undefined;
};

const numberField = (
  obj: Record<string, unknown>,
  key: string,
): number | undefined => {
  const v = obj[key];
  return typeof v === "number" ? v : undefined;
};

const objectField = (
  obj: Record<string, unknown>,
  key: string,
): Record<string, unknown> | undefined => {
  const v = obj[key];
  return v !== null && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : undefined;
};

const APPLY_PATCH_ITEM_TYPES = new Set([
  "apply_patch",
  "apply_patch_call",
  "custom_tool_call",
]);

const stringifyJson = (value: unknown): string => {
  if (typeof value === "string") return value;
  return JSON.stringify(value ?? {});
};

const applyPatchOperation = (
  item: Record<string, unknown>,
): unknown | undefined =>
  item.operation ?? item.input ?? item.arguments ?? item.action;

const isApplyPatchItem = (item: Record<string, unknown>): boolean => {
  const type = stringField(item, "type");
  if (type !== undefined && APPLY_PATCH_ITEM_TYPES.has(type)) return true;
  const name = stringField(item, "name");
  return name === "apply_patch" && applyPatchOperation(item) !== undefined;
};

const isToolCallItem = (item: Record<string, unknown>): boolean =>
  stringField(item, "type") === "function_call" || isApplyPatchItem(item);

const toolCallId = (item: Record<string, unknown>): string | undefined =>
  stringField(item, "call_id") ?? stringField(item, "id");

const toolCallName = (item: Record<string, unknown>): string | undefined =>
  isApplyPatchItem(item) ? "apply_patch" : stringField(item, "name");

const toolCallArguments = (item: Record<string, unknown>): string => {
  if (isApplyPatchItem(item)) return stringifyJson(applyPatchOperation(item));
  return stringField(item, "arguments") ?? "";
};

/**
 * Translate one Responses API streaming event into one ChatCompletion
 * chunk. Returns null for events we ignore (heartbeats, `response.created`
 * once we've already emitted role, etc.).
 *
 * Mirrors `OpenAiResponsesToChatCompletionStreamIterator
 * .translate_responses_chunk_to_openai_stream` from
 * `completion_extras/litellm_responses_transformation/transformation.py:1090-1378`.
 */
export const chatGptEventToChunk = (
  event: TChatGptStreamEvent,
  state: TChatGptStreamState,
  options: TChatGptProviderOptions,
): TChatCompletionChunk | null => {
  const type = stringField(event, "type");
  if (type === undefined) return null;

  if (type === "response.created") {
    return {
      ...baseChunk(options),
      choices: [
        {
          index: 0,
          delta: { role: "assistant", content: "" },
          finish_reason: null,
        },
      ],
    };
  }

  if (type === "response.output_text.delta") {
    const delta = stringField(event, "delta");
    if (delta === undefined) return null;
    return {
      ...baseChunk(options),
      choices: [{ index: 0, delta: { content: delta }, finish_reason: null }],
    };
  }

  if (type === "response.reasoning_summary_text.delta") {
    const delta = stringField(event, "delta");
    if (delta === undefined || delta.length === 0) return null;
    return {
      ...baseChunk(options),
      choices: [
        {
          index: 0,
          delta: { reasoning_content: delta },
          finish_reason: null,
        },
      ],
    };
  }

  if (type === "response.output_item.done") {
    const item = objectField(event, "item");
    if (item !== undefined) captureReasoningItem(state, item);
    const drained = drainUnemittedReasoning(state);
    if (drained.length > 0) {
      return {
        ...baseChunk(options),
        choices: [
          {
            index: 0,
            delta: { reasoning_items: drained },
            finish_reason: null,
          },
        ],
      };
    }
    // Codex-spark: the completed function call lands here with its full
    // `arguments` and never sent a `.delta`. Finalize so the tool isn't
    // invoked with empty input (→ tool error → re-issue → loop).
    if (item !== undefined && isToolCallItem(item) && !isApplyPatchItem(item)) {
      const outputIndex = numberField(event, "output_index") ?? 0;
      return finalizeToolArgs(
        state,
        outputIndex,
        stringField(item, "arguments") ?? "",
        options,
      );
    }
    return null;
  }

  if (type === "response.function_call_arguments.done") {
    const outputIndex = numberField(event, "output_index") ?? 0;
    return finalizeToolArgs(
      state,
      outputIndex,
      stringField(event, "arguments") ?? "",
      options,
    );
  }

  if (type === "response.output_item.added") {
    const item = objectField(event, "item");
    if (item === undefined) return null;
    captureReasoningItem(state, item);
    if (!isToolCallItem(item)) return null;
    state.hasToolCall = true;
    const callId = toolCallId(item);
    const name = toolCallName(item);
    const outputIndex = numberField(event, "output_index") ?? 0;
    return {
      ...baseChunk(options),
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: outputIndex,
                ...(callId !== undefined ? { id: callId } : {}),
                type: "function",
                function: {
                  ...(name !== undefined ? { name } : {}),
                  arguments: isApplyPatchItem(item)
                    ? toolCallArguments(item)
                    : "",
                },
              },
            ],
          },
          finish_reason: null,
        },
      ],
    };
  }

  if (type === "response.function_call_arguments.delta") {
    const delta = stringField(event, "delta");
    if (delta === undefined) return null;
    state.hasToolCall = true;
    const outputIndex = numberField(event, "output_index") ?? 0;
    state.argsStreamedIndexes.add(outputIndex);
    return {
      ...baseChunk(options),
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: outputIndex,
                type: "function",
                function: { arguments: delta },
              },
            ],
          },
          finish_reason: null,
        },
      ],
    };
  }

  // `response.failed` / `response.incomplete` / `response.error` are
  // chatgpt.com's way of saying the upstream gave up mid-stream. Without
  // surfacing these we silently close the stream and the proxy
  // accumulator builds an empty content array, which Claude Code
  // rejects as "no valid text content" (observed on `/compact`).
  // Throw so the runner converts it into an SSE `event: error` for
  // streaming clients and a 502 envelope for non-streaming clients.
  if (
    type === "response.failed" ||
    type === "response.incomplete" ||
    type === "error"
  ) {
    const response = objectField(event, "response");
    const errorObj =
      objectField(event, "error") ??
      (response !== undefined ? objectField(response, "error") : undefined);
    // `error` events carry top-level `code`/`message` (Responses spec);
    // `response.incomplete` carries `response.incomplete_details.reason`
    // (e.g. "max_output_tokens"). Both were previously dropped, leaving
    // only "upstream chatgpt <type>" — the one diagnostic the client gets.
    // An EMPTY string field is treated as absent so the next fallback runs
    // (a `""` message/code would otherwise win the chain and blank the
    // diagnostic).
    const nonEmpty = (s: string | undefined): string | undefined =>
      s !== undefined && s.length > 0 ? s : undefined;
    const incomplete =
      response !== undefined
        ? objectField(response, "incomplete_details")
        : undefined;
    const message =
      (errorObj !== undefined
        ? nonEmpty(stringField(errorObj, "message"))
        : undefined) ??
      nonEmpty(stringField(event, "message")) ??
      (incomplete !== undefined
        ? nonEmpty(stringField(incomplete, "reason"))
        : undefined) ??
      `upstream chatgpt ${type}`;
    const code =
      (errorObj !== undefined
        ? nonEmpty(stringField(errorObj, "type"))
        : undefined) ??
      (errorObj !== undefined
        ? nonEmpty(stringField(errorObj, "code"))
        : undefined) ??
      nonEmpty(stringField(event, "code")) ??
      type;
    // Typed so downstream error handling can tell "the vendor reported an
    // error" apart from "we could not decode the stream" (issue #274 —
    // the web-search accumulate path used to flatten both into a generic
    // 502 that discarded the vendor's reason).
    throw new UpstreamStreamError(code, `${code}: ${message}`);
  }

  if (type === "response.completed") {
    const response = objectField(event, "response");
    // Prefer per-stream state: we already counted every
    // `response.output_item.added` tool item and every
    // `response.function_call_arguments.delta` as we walked the
    // stream, so we don't depend on whatever shape the terminal
    // event chooses to ship `response.output[]` in. Fall back to
    // a structural check on `response.output[]` so single-event
    // unit tests (which call this without first feeding the added
    // event) still observe the correct finish_reason.
    let hasToolCall = state.hasToolCall;
    if (!hasToolCall) {
      const output = response !== undefined ? response.output : undefined;
      if (Array.isArray(output)) {
        for (const item of output) {
          if (
            item !== null &&
            typeof item === "object" &&
            isToolCallItem(item as Record<string, unknown>)
          ) {
            hasToolCall = true;
            break;
          }
        }
      }
    }
    const finishReason = hasToolCall ? "tool_calls" : "stop";

    // Fold any reasoning items that only appeared in the terminal
    // `response.output[]` snapshot (some gpt-5.x-codex responses omit
    // the per-item `.done` event). Mirrors litellm 1321-1356.
    const completedOutput =
      response !== undefined ? response.output : undefined;
    if (Array.isArray(completedOutput)) {
      for (const item of completedOutput) {
        if (item !== null && typeof item === "object" && !Array.isArray(item)) {
          captureReasoningItem(state, item as Record<string, unknown>);
        }
      }
    }
    const reasoningItems = drainUnemittedReasoning(state);

    let usage: TChatCompletionChunk["usage"] | undefined;
    const usageRaw =
      response !== undefined ? objectField(response, "usage") : undefined;
    if (usageRaw !== undefined) {
      const inTok = numberField(usageRaw, "input_tokens") ?? 0;
      const outTok = numberField(usageRaw, "output_tokens") ?? 0;
      const inDetails = objectField(usageRaw, "input_tokens_details");
      const outDetails = objectField(usageRaw, "output_tokens_details");
      const cached =
        inDetails !== undefined
          ? numberField(inDetails, "cached_tokens")
          : undefined;
      const reasoning =
        outDetails !== undefined
          ? numberField(outDetails, "reasoning_tokens")
          : undefined;
      usage = {
        prompt_tokens: inTok,
        completion_tokens: outTok,
        total_tokens: inTok + outTok,
        ...(cached !== undefined
          ? { prompt_tokens_details: { cached_tokens: cached } }
          : {}),
        ...(reasoning !== undefined
          ? { completion_tokens_details: { reasoning_tokens: reasoning } }
          : {}),
      };
    }

    return {
      ...baseChunk(options),
      choices: [
        {
          index: 0,
          delta:
            reasoningItems.length > 0
              ? { reasoning_items: reasoningItems }
              : {},
          finish_reason: finishReason,
        },
      ],
      ...(usage !== undefined ? { usage } : {}),
    };
  }

  // response.in_progress / response.content_part.* / response.created
  // tail / etc — ignore.
  return null;
};
