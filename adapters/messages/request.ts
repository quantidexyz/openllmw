import type {
  TAnthropicContentBlock,
  TAnthropicMessage,
  TAnthropicRequest,
  TAnthropicTool,
  TChatCompletionRequest,
  TChatMessage,
  TToolCall,
} from "@quantidexyz/openllmp";
import { decodeReasoningSignature } from "./reasoning-signature";

const extractTextFromBlocks = (
  blocks: ReadonlyArray<TAnthropicContentBlock>,
): string =>
  blocks
    .filter(
      (b): b is { type: "text"; text: string } & TAnthropicContentBlock =>
        b.type === "text",
    )
    .map((b) => b.text)
    .join("\n");

type TCanonicalContentPart =
  | { readonly type: "text"; readonly text: string }
  | {
      readonly type: "image_url";
      readonly image_url: { readonly url: string };
    };

const blockToCanonicalPart = (
  block: TAnthropicContentBlock,
): TCanonicalContentPart | null => {
  if (block.type === "text") {
    return { type: "text", text: block.text };
  }
  if (block.type === "image") {
    if (block.source.type === "url") {
      return { type: "image_url", image_url: { url: block.source.url } };
    }
    if (block.source.type === "base64") {
      return {
        type: "image_url",
        image_url: {
          url: `data:${block.source.media_type};base64,${block.source.data}`,
        },
      };
    }
  }
  return null;
};

const blocksToCanonicalContent = (
  blocks: ReadonlyArray<TAnthropicContentBlock>,
): string | ReadonlyArray<TCanonicalContentPart> => {
  const parts: TCanonicalContentPart[] = [];
  for (const b of blocks) {
    const p = blockToCanonicalPart(b);
    if (p !== null) parts.push(p);
  }
  if (parts.length === 0) return "";
  if (parts.length === 1 && parts[0]?.type === "text") return parts[0].text;
  // If every part is text, collapse to a single string for simpler
  // downstream handling.
  if (parts.every((p) => p.type === "text")) {
    return parts.map((p) => (p.type === "text" ? p.text : "")).join("\n");
  }
  return parts;
};

const extractSystemText = (
  system: TAnthropicRequest["system"] | undefined,
): string => {
  if (system === undefined) return "";
  if (typeof system === "string") return system;
  return system.map((b) => b.text).join("\n\n");
};

const toolUseToOpenAIToolCall = (block: {
  readonly id: string;
  readonly name: string;
  readonly input: unknown;
}): TToolCall => ({
  id: block.id,
  type: "function",
  function: {
    name: block.name,
    arguments: JSON.stringify(block.input ?? {}),
  },
});

const toolResultContentToString = (
  content: Extract<TAnthropicContentBlock, { type: "tool_result" }>["content"],
): string => {
  if (typeof content === "string") return content;
  // A tool_result may carry image blocks alongside text — the canonical
  // OpenAI `role: "tool"` message is text-only, so keep the text and
  // annotate the images rather than leaking `undefined` into the join.
  return content
    .map((b) =>
      b.type === "text" ? b.text : "[image content omitted from tool result]",
    )
    .join("\n");
};

/**
 * Default `parameters` schema for Anthropic native server tools that
 * don't carry an `input_schema` of their own (`web_search_*`,
 * `web_fetch_*`, …). Without this, a non-Anthropic provider sees a
 * function tool with no arg shape and calls it with empty input — the
 * proxy then can't pull a query out, search runs against `""`, and the
 * whole WebSearch flow returns an empty result.
 *
 * Today we know only `web_search_*` needs this. Adding more native
 * types is a one-line extension below.
 */
const NATIVE_TOOL_DEFAULT_PARAMETERS = (
  toolType: string | undefined,
): unknown => {
  if (typeof toolType !== "string") return undefined;
  if (toolType.startsWith("web_search_")) {
    return {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "The web search query to run.",
        },
      },
      required: ["query"],
      additionalProperties: false,
    };
  }
  return undefined;
};

const anthropicToolToOpenAI = (
  tool: TAnthropicTool,
): NonNullable<TChatCompletionRequest["tools"]>[number] => {
  const parameters =
    tool.input_schema ?? NATIVE_TOOL_DEFAULT_PARAMETERS(tool.type);
  return {
    type: "function",
    function: {
      name: tool.name,
      ...(tool.description !== undefined
        ? { description: tool.description }
        : {}),
      ...(parameters !== undefined ? { parameters } : {}),
    },
  };
};

const anthropicToolChoiceToOpenAI = (
  choice: NonNullable<TAnthropicRequest["tool_choice"]>,
): NonNullable<TChatCompletionRequest["tool_choice"]> | undefined => {
  switch (choice.type) {
    case "auto":
      return "auto";
    case "any":
      return "required";
    case "none":
      return "none";
    case "tool":
      return { type: "function", function: { name: choice.name } };
  }
};

/**
 * Walk an Anthropic-style message and emit the canonical OpenAI
 * message(s). An assistant turn with mixed text + tool_use blocks
 * yields ONE assistant message carrying both `content` (text) and
 * `tool_calls`. A user turn with one or more tool_result blocks
 * yields ONE OpenAI `role: "tool"` message PER block.
 */
const splitAnthropicMessage = (m: TAnthropicMessage): TChatMessage[] => {
  // A `role: "system"` turn inlined in `messages[]` (Claude Code
  // v2.1.157+ under `mid-conversation-system-2026-04-07`) maps to the
  // canonical `system` role. Handle it BEFORE the generic
  // string-content branch below, which would otherwise mis-bucket a
  // string-content system turn as an assistant message.
  if (m.role === "system") {
    const content =
      typeof m.content === "string"
        ? m.content
        : extractTextFromBlocks(m.content);
    return [{ role: "system", content }];
  }

  if (typeof m.content === "string") {
    return m.role === "user"
      ? [{ role: "user", content: m.content }]
      : [{ role: "assistant", content: m.content }];
  }

  if (m.role === "user") {
    const toolResults = m.content.filter(
      (b): b is Extract<TAnthropicContentBlock, { type: "tool_result" }> =>
        b.type === "tool_result",
    );
    if (toolResults.length > 0) {
      return toolResults.map<TChatMessage>((b) => ({
        role: "tool",
        content: toolResultContentToString(b.content),
        tool_call_id: b.tool_use_id,
      }));
    }
    // Preserve image blocks so vision flows through to the chosen
    // upstream provider (works regardless of which provider the user
    // selected — OpenAI keeps image_url native, Anthropic re-emits via
    // contentToBlocks).
    const content = blocksToCanonicalContent(m.content);
    return [{ role: "user", content }];
  }

  // assistant
  const text = extractTextFromBlocks(m.content);
  const toolCalls = m.content
    .filter(
      (b): b is Extract<TAnthropicContentBlock, { type: "tool_use" }> =>
        b.type === "tool_use",
    )
    .map(toolUseToOpenAIToolCall);
  // Carry prior extended-thinking forward as canonical
  // `reasoning_content`. The Anthropic→Anthropic case is now handled by
  // the native pass-through (which keeps `thinking` + `signature`
  // verbatim), so this only fires for a cross-provider fallback — and
  // there a non-Anthropic model can't use the opaque `signature`
  // anyway, but DOES benefit from the reasoning text as context.
  // Previously these blocks were silently dropped (request.ts only
  // handled text + tool_use), truncating multi-turn reasoning history.
  const thinkingBlocks = m.content.filter(
    (b): b is Extract<TAnthropicContentBlock, { type: "thinking" }> =>
      b.type === "thinking",
  );
  const thinking = thinkingBlocks
    .map((b) => b.thinking)
    .filter((t) => t.length > 0)
    .join("\n\n");
  // Recover the upstream `reasoning` item(s) we smuggled out through
  // the thinking `signature`. Echoing them back lets a reasoning
  // upstream (Codex/Responses) resume its chain-of-thought instead of
  // restarting — and re-issuing the same tool call — every turn.
  // Genuine Anthropic signatures decode to null and are ignored here
  // (that path is the native pass-through, not this adapter).
  const reasoningItems = thinkingBlocks.flatMap(
    (b) => decodeReasoningSignature(b.signature) ?? [],
  );
  const assistantMsg: Extract<TChatMessage, { role: "assistant" }> = {
    role: "assistant",
    content: text,
    ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
    ...(thinking.length > 0 ? { reasoning_content: thinking } : {}),
    ...(reasoningItems.length > 0 ? { reasoning_items: reasoningItems } : {}),
  };
  return [assistantMsg];
};

/**
 * Adapt an inbound `/v1/messages` (Anthropic-format) request to the
 * internal canonical `TChatCompletionRequest` (OpenAI shape).
 *
 * Used for the genuine cross-provider path (Anthropic-format client →
 * a non-Anthropic upstream). Anthropic→Anthropic no longer flows
 * through here — it takes the native pass-through, which keeps
 * `cache_control`, `thinking`+`signature`, `top_k` and structured
 * `system` verbatim. Here we map text, images, tools/tool_choice,
 * tool_use↔tool_calls, tool_result→tool messages, and prior
 * `thinking`→`reasoning_content`. `cache_control` is intentionally
 * not propagated: no non-Anthropic upstream honours it.
 *
 * Known follow-up: OpenAI rejects tool names >64 chars; LiteLLM
 * truncates with a reverse map. Deferred — long tool names in practice
 * flow Anthropic→Anthropic (pass-through, no limit).
 */
export const fromAnthropicMessagesRequest = (
  req: TAnthropicRequest,
): TChatCompletionRequest => {
  const messages: TChatMessage[] = [];

  const systemText = extractSystemText(req.system);
  if (systemText.length > 0) {
    messages.push({ role: "system", content: systemText });
  }

  for (const m of req.messages) {
    for (const expanded of splitAnthropicMessage(m)) {
      // Drop empty user messages so a turn that was purely tool_result
      // doesn't leak through as an empty text turn.
      if (
        expanded.role === "user" &&
        typeof expanded.content === "string" &&
        expanded.content.length === 0
      ) {
        continue;
      }
      messages.push(expanded);
    }
  }

  const tools = req.tools?.map(anthropicToolToOpenAI);
  const toolChoice =
    req.tool_choice !== undefined
      ? anthropicToolChoiceToOpenAI(req.tool_choice)
      : undefined;

  return {
    model: req.model,
    messages,
    max_completion_tokens: req.max_tokens,
    ...(req.temperature !== undefined ? { temperature: req.temperature } : {}),
    ...(req.top_p !== undefined ? { top_p: req.top_p } : {}),
    ...(req.stop_sequences !== undefined
      ? { stop: req.stop_sequences as readonly string[] as string[] }
      : {}),
    ...(req.stream !== undefined ? { stream: req.stream } : {}),
    ...(tools !== undefined && tools.length > 0 ? { tools } : {}),
    ...(toolChoice !== undefined ? { tool_choice: toolChoice } : {}),
    ...(req.metadata?.user_id !== undefined
      ? { user: req.metadata.user_id }
      : {}),
  };
};
