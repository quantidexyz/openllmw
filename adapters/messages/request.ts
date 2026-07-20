import type {
  TAnthropicContentBlock,
  TAnthropicMessage,
  TAnthropicRequest,
  TAnthropicTool,
  TChatCompletionRequest,
  TChatMessage,
  TToolCall,
} from "@openllmsh/protocol";
import type { TCanonicalContentPart } from "../../lib/canonical/content-part";
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

/** Readable text degradation for a citable search_result block. */
const renderSearchResult = (
  block: Extract<TAnthropicContentBlock, { type: "search_result" }>,
): string => {
  const body = block.content.map((b) => b.text).join("\n");
  return `[search result] ${block.title} (${block.source})\n${body}`;
};

/** Pull the text out of an opaque custom-content document source. */
const extractTextFromUnknownContent = (content: unknown): string => {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter(
      (b): b is { type: "text"; text: string } =>
        typeof b === "object" &&
        b !== null &&
        (b as { type?: unknown }).type === "text" &&
        typeof (b as { text?: unknown }).text === "string",
    )
    .map((b) => b.text)
    .join("\n");
};

const documentToCanonicalPart = (
  block: Extract<TAnthropicContentBlock, { type: "document" }>,
): TCanonicalContentPart => {
  const src = block.source;
  if (src.type === "base64") {
    return {
      type: "file",
      file: {
        file_data: `data:${src.media_type};base64,${src.data}`,
        ...(block.title != null ? { filename: block.title } : {}),
      },
    };
  }
  if (src.type === "file") {
    return {
      type: "file",
      file: {
        file_id: src.file_id,
        ...(block.title != null ? { filename: block.title } : {}),
      },
    };
  }
  if (src.type === "text") {
    // Plain-text documents ARE text — carry the content directly.
    return { type: "text", text: src.data };
  }
  if (src.type === "content") {
    const text = extractTextFromUnknownContent(src.content);
    return {
      type: "text",
      text:
        text.length > 0
          ? text
          : "[document omitted — custom-content document had no extractable text]",
    };
  }
  // url source — the wire layer is pure (no fetch) and OpenAI file parts
  // have no URL form, so degrade to an annotation. Anthropic→Anthropic
  // passthrough keeps URL documents working; only cross-provider hops
  // land here.
  return {
    type: "text",
    text: `[document omitted — URL documents are not supported by this provider: ${src.url}]`,
  };
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
    // file_id image source — OpenAI image_url has no file-reference
    // form, so annotate rather than silently dropping the image.
    return {
      type: "text",
      text: "[image omitted — file_id image sources are not supported by this provider]",
    };
  }
  if (block.type === "document") {
    return documentToCanonicalPart(block);
  }
  if (block.type === "search_result") {
    // No canonical carrier for citable search results — degrade to a
    // readable text rendering (citations are dropped cross-provider;
    // Anthropic→Anthropic passthrough keeps them intact).
    return { type: "text", text: renderSearchResult(block) };
  }
  if (block.type === "container_upload") {
    return {
      type: "text",
      text: "[container upload omitted — code-execution file references are not supported by this provider]",
    };
  }
  return null;
};

const blocksToCanonicalContent = (
  blocks: ReadonlyArray<TAnthropicContentBlock>,
): string | ReadonlyArray<TCanonicalContentPart> => {
  const parts: TCanonicalContentPart[] = [];
  for (const b of blocks) {
    // A document's `context` is retrieval metadata the model should
    // still see — surface it as a preceding text part so it isn't lost
    // on cross-provider hops.
    if (b.type === "document" && b.context != null && b.context.length > 0) {
      parts.push({ type: "text", text: b.context });
    }
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
  // A tool_result may carry image / document / search_result blocks
  // alongside text — the canonical OpenAI `role: "tool"` message is
  // text-only, so keep the text and annotate the rest rather than
  // leaking `undefined` into the join.
  return content
    .map((b) => {
      if (b.type === "text") return b.text;
      if (b.type === "document") {
        if (b.source.type === "text") return b.source.data;
        if (b.source.type === "content") {
          const text = extractTextFromUnknownContent(b.source.content);
          if (text.length > 0) return text;
        }
        return "[document content omitted from tool result]";
      }
      if (b.type === "search_result") return renderSearchResult(b);
      return "[image content omitted from tool result]";
    })
    .join("\n");
};

/**
 * Does an inbound Anthropic-wire body declare the `web_search_*` SERVER
 * tool? The typed descriptor is the client's explicit request for
 * platform-executed search (Claude Code's WebSearch). Consumers switch a
 * capable non-Anthropic hop onto its provider-NATIVE search (grok
 * web/x_search, kimi builtin `$web_search`) — never armed by a bare
 * `web_search`-NAMED function tool, which stays an ordinary client tool.
 */
export const declaresAnthropicServerSearchTool = (
  rawBody: unknown,
): boolean => {
  if (rawBody === null || typeof rawBody !== "object") return false;
  const tools = (rawBody as { readonly tools?: unknown }).tools;
  return (
    Array.isArray(tools) &&
    tools.some(
      (tool) =>
        tool !== null &&
        typeof tool === "object" &&
        typeof (tool as { readonly type?: unknown }).type === "string" &&
        (tool as { readonly type: string }).type.startsWith("web_search_"),
    )
  );
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
    if (m.content.some((b) => b.type === "tool_result")) {
      // A tool_result turn may carry SIBLING blocks — Claude Code delivers
      // loaded-skill instructions as a text block alongside the Skill
      // tool_result. Emit a `tool` message per tool_result AND a `user`
      // message per contiguous run of other blocks, preserving block order;
      // returning only the tool messages silently severed that injected
      // context from the conversation on every cross-provider hop.
      const out: TChatMessage[] = [];
      let siblings: TAnthropicContentBlock[] = [];
      const flushSiblings = (): void => {
        if (siblings.length === 0) return;
        const content = blocksToCanonicalContent(siblings);
        siblings = [];
        if (typeof content === "string" && content.length === 0) return;
        out.push({ role: "user", content });
      };
      for (const b of m.content) {
        if (b.type === "tool_result") {
          flushSiblings();
          out.push({
            role: "tool",
            content: toolResultContentToString(b.content),
            tool_call_id: b.tool_use_id,
          });
        } else {
          siblings.push(b);
        }
      }
      flushSiblings();
      return out;
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
    ...(thinking.length > 0
      ? { reasoning_content: thinking }
      : toolCalls.length > 0
        ? { reasoning_content: "" }
        : {}),
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
