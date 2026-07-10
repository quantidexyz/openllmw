import type {
  TAnthropicContentBlock,
  TAnthropicMessage,
  TAnthropicProviderOptions,
  TAnthropicRequest,
  TAnthropicTool,
  TChatCompletionRequest,
  TChatMessage,
  TFilePart,
  TToolCall,
} from "@quantidexyz/openllmp";
import {
  extractMessageText,
  parseToolArguments,
} from "../../lib/canonical/message";
import { mapReasoningEffortToAnthropic } from "./adaptive-thinking";

const DEFAULT_MAX_TOKENS = 4096;

/** Anthropic only has the one cache tier on the wire. */
type TCacheControl = { readonly type: "ephemeral" };

type TContentPart = Exclude<
  NonNullable<TChatMessage["content"]>,
  string
>[number];
type TTextPart = Extract<TContentPart, { type: "text" }>;

/**
 * Read a prompt-cache breakpoint off any object that may carry one
 * (content part, message, tool). LiteLLM treats a present, dict-shaped
 * `cache_control` as "cache up to and including this element"
 * (`add_cache_control_to_content`); a `null` clears it. We normalise to
 * Anthropic's single ephemeral tier.
 */
const readCacheControl = (v: {
  readonly cache_control?: { readonly type: "ephemeral" } | null;
}): TCacheControl | undefined =>
  v.cache_control != null ? { type: "ephemeral" } : undefined;

const withCacheControl = <T extends TAnthropicContentBlock>(
  block: T,
  cc: TCacheControl | undefined,
): T => (cc === undefined ? block : { ...block, cache_control: cc });

/**
 * Anthropic accepts `cache_control` on text / image / document /
 * search_result / tool_use / tool_result blocks — never on thinking. A
 * message-level breakpoint applies to the message's LAST cache-eligible
 * block (LiteLLM folds `message["cache_control"]` onto the final
 * content element). Don't clobber a block that already carries its own
 * part-level breakpoint.
 */
const applyMessageCacheControl = (
  blocks: TAnthropicContentBlock[],
  cc: TCacheControl | undefined,
): void => {
  if (cc === undefined || blocks.length === 0) return;
  for (let i = blocks.length - 1; i >= 0; i--) {
    const b = blocks[i];
    if (b === undefined) continue;
    if (
      b.type === "text" ||
      b.type === "image" ||
      b.type === "document" ||
      b.type === "search_result" ||
      b.type === "tool_use" ||
      b.type === "tool_result"
    ) {
      if (b.cache_control == null) blocks[i] = { ...b, cache_control: cc };
      return;
    }
  }
};

const DATA_URL_RE = /^data:([^;,]+);base64,(.*)$/;

/** Runtime-agnostic base64 → UTF-8 (browser, node, bun). */
const decodeBase64Utf8 = (b64: string): string => {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder().decode(bytes);
};

/**
 * Map a base64 payload + media type to the Anthropic block that can
 * legally carry it: images → image block, PDFs → base64 document
 * source, text/plain → text document source, anything else → text
 * annotation. (Anthropic only accepts `application/pdf` for base64
 * documents and `text/plain` for text documents.)
 */
const base64ToDocumentBlock = (
  mediaType: string,
  data: string,
  title: string | undefined,
): TAnthropicContentBlock => {
  if (mediaType.startsWith("image/")) {
    return {
      type: "image",
      source: { type: "base64", media_type: mediaType, data },
    };
  }
  if (mediaType === "application/pdf") {
    return {
      type: "document",
      source: { type: "base64", media_type: mediaType, data },
      ...(title !== undefined ? { title } : {}),
    };
  }
  if (mediaType === "text/plain") {
    // Malformed base64 must degrade like any other unusable payload —
    // a throw here would 400 the whole request at the transform layer.
    try {
      return {
        type: "document",
        source: {
          type: "text",
          media_type: "text/plain",
          data: decodeBase64Utf8(data),
        },
        ...(title !== undefined ? { title } : {}),
      };
    } catch {
      return {
        type: "text",
        text: "[file content omitted — file_data was not valid base64]",
      };
    }
  }
  return {
    type: "text",
    text: `[file content omitted — unsupported media type ${mediaType}]`,
  };
};

const imageUrlToAnthropicBlock = (
  url: string,
  cc: TCacheControl | undefined,
): TAnthropicContentBlock => {
  const m = DATA_URL_RE.exec(url);
  if (m !== null) {
    const mediaType = m[1] ?? "image/png";
    const data = m[2] ?? "";
    // Some OpenAI-format clients (LiteLLM among them) smuggle PDFs and
    // other files through `image_url` data URLs. A non-image media type
    // would make an invalid Anthropic image block — route it to the
    // block type that can carry it instead.
    if (!mediaType.startsWith("image/")) {
      return withCacheControl(
        base64ToDocumentBlock(mediaType, data, undefined),
        cc,
      );
    }
    return withCacheControl(
      {
        type: "image",
        source: { type: "base64", media_type: mediaType, data },
      },
      cc,
    );
  }
  return withCacheControl({ type: "image", source: { type: "url", url } }, cc);
};

const filePartToAnthropicBlock = (
  part: Pick<TFilePart, "file">,
): TAnthropicContentBlock => {
  const { file_data, file_id, filename } = part.file;
  if (file_data !== undefined) {
    const m = DATA_URL_RE.exec(file_data);
    if (m !== null) {
      return base64ToDocumentBlock(
        m[1] ?? "application/octet-stream",
        m[2] ?? "",
        filename,
      );
    }
    return {
      type: "text",
      text: "[file content omitted — file_data was not a base64 data URL]",
    };
  }
  if (file_id !== undefined) {
    // Anthropic Files API source (beta) — pass the reference through.
    return {
      type: "document",
      source: { type: "file", file_id },
      ...(filename !== undefined ? { title: filename } : {}),
    };
  }
  return {
    type: "text",
    text: "[file content omitted — file part carried neither file_data nor file_id]",
  };
};

const contentToBlocks = (
  content: TChatMessage["content"] | null | undefined,
): TAnthropicContentBlock[] => {
  if (content == null) return [];
  if (typeof content === "string") {
    if (content.length === 0) return [];
    return [{ type: "text", text: content }];
  }
  const out: TAnthropicContentBlock[] = [];
  for (const part of content) {
    if (part.type === "text") {
      if (part.text.length > 0) {
        out.push(
          withCacheControl(
            { type: "text", text: part.text },
            readCacheControl(part),
          ),
        );
      }
      continue;
    }
    if (part.type === "image_url") {
      out.push(
        imageUrlToAnthropicBlock(part.image_url.url, readCacheControl(part)),
      );
      continue;
    }
    if (part.type === "file") {
      out.push(
        withCacheControl(
          filePartToAnthropicBlock(part),
          readCacheControl(part),
        ),
      );
      continue;
    }
    // input_audio: Anthropic doesn't accept audio on chat; drop with a
    // text annotation so the model is at least aware something was
    // attached.
    out.push({
      type: "text",
      text: "[audio content omitted — Anthropic does not accept audio input]",
    });
  }
  return out;
};

const isSystem = (
  m: TChatMessage,
): m is Extract<TChatMessage, { role: "system" }> => m.role === "system";

const isUser = (
  m: TChatMessage,
): m is Extract<TChatMessage, { role: "user" }> => m.role === "user";

const isAssistant = (
  m: TChatMessage,
): m is Extract<TChatMessage, { role: "assistant" }> => m.role === "assistant";

const isTool = (
  m: TChatMessage,
): m is Extract<TChatMessage, { role: "tool" }> => m.role === "tool";

const toolCallToUseBlock = (call: TToolCall): TAnthropicContentBlock => ({
  type: "tool_use",
  id: call.id,
  name: call.function.name,
  input: parseToolArguments(call.function.arguments),
});

const assistantToBlocks = (
  m: Extract<TChatMessage, { role: "assistant" }>,
): TAnthropicContentBlock[] => {
  const blocks: TAnthropicContentBlock[] = [];
  // Assistant turns currently only carry text — vision/audio output is
  // not a thing on this side of the wire — so flatten to a single text
  // block. Tool calls follow.
  const text = extractMessageText(m.content);
  if (text.length > 0) blocks.push({ type: "text", text });
  for (const call of m.tool_calls ?? []) {
    blocks.push(toolCallToUseBlock(call));
  }
  return blocks;
};

const toolMessageToResultBlock = (
  m: Extract<TChatMessage, { role: "tool" }>,
): TAnthropicContentBlock =>
  withCacheControl(
    {
      type: "tool_result",
      tool_use_id: m.tool_call_id,
      content: extractMessageText(m.content),
    },
    readCacheControl(m),
  );

/**
 * Walk the OpenAI messages array and produce Anthropic messages.
 *
 * Anthropic requires alternating user/assistant turns. Multiple OpenAI
 * `role: "tool"` messages in a row collapse into a single Anthropic
 * `role: "user"` turn with multiple `tool_result` content blocks.
 *
 * `cache_control` is preserved end-to-end: part-level breakpoints ride
 * their block, message-level breakpoints land on the message's last
 * cache-eligible block (LiteLLM `anthropic_messages_pt` semantics).
 * Without this, Claude Code / opencode / aider lose every breakpoint
 * and Anthropic re-bills the whole prefix each turn (~10x).
 */
const buildTurnMessages = (
  messages: ReadonlyArray<TChatMessage>,
): TAnthropicMessage[] => {
  const out: TAnthropicMessage[] = [];
  let pendingToolResults: TAnthropicContentBlock[] = [];

  const flushToolResults = (): void => {
    if (pendingToolResults.length === 0) return;
    out.push({ role: "user", content: pendingToolResults });
    pendingToolResults = [];
  };

  for (const m of messages) {
    if (isSystem(m)) continue;
    if (isTool(m)) {
      pendingToolResults.push(toolMessageToResultBlock(m));
      continue;
    }
    flushToolResults();
    if (isUser(m)) {
      const blocks = contentToBlocks(m.content);
      if (blocks.length === 0) continue;
      applyMessageCacheControl(blocks, readCacheControl(m));
      // Collapse single bare-text block to a plain string for cleaner
      // wire payloads — but ONLY when no breakpoint rides it (a string
      // can't carry `cache_control`).
      const only = blocks[0];
      if (
        blocks.length === 1 &&
        only !== undefined &&
        only.type === "text" &&
        only.cache_control == null
      ) {
        out.push({ role: "user", content: only.text });
      } else {
        out.push({ role: "user", content: blocks });
      }
      continue;
    }
    if (isAssistant(m)) {
      const blocks = assistantToBlocks(m);
      if (blocks.length === 0) continue;
      applyMessageCacheControl(blocks, readCacheControl(m));
      out.push({ role: "assistant", content: blocks });
    }
  }
  flushToolResults();
  return out;
};

const toAnthropicTool = (
  tool: NonNullable<TChatCompletionRequest["tools"]>[number],
): TAnthropicTool => {
  const cc = readCacheControl(tool);
  return {
    name: tool.function.name,
    ...(tool.function.description !== undefined
      ? { description: tool.function.description }
      : {}),
    input_schema: tool.function.parameters ?? {
      type: "object",
      properties: {},
    },
    ...(cc !== undefined ? { cache_control: cc } : {}),
  };
};

const toAnthropicToolChoice = (
  choice: NonNullable<TChatCompletionRequest["tool_choice"]>,
): TAnthropicRequest["tool_choice"] => {
  if (choice === "auto") return { type: "auto" };
  if (choice === "required") return { type: "any" };
  if (choice === "none") return { type: "none" };
  return { type: "tool", name: choice.function.name };
};

type TSystemTextBlock = {
  readonly type: "text";
  readonly text: string;
  readonly cache_control?: TCacheControl;
};

/**
 * Build the Anthropic `system` field. The flat-string form cannot carry
 * a breakpoint, so the moment ANY system message/part is marked we emit
 * the structured array form with `cache_control` on the right block —
 * exactly LiteLLM's `_transform_system_message` behaviour. Otherwise we
 * keep the legacy joined string for clean wire payloads / back-compat.
 */
const buildSystem = (
  messages: ReadonlyArray<TChatMessage>,
): string | ReadonlyArray<TSystemTextBlock> | undefined => {
  const blocks: TSystemTextBlock[] = [];
  const push = (text: string, cc: TCacheControl | undefined): void => {
    blocks.push(
      cc !== undefined
        ? { type: "text", text, cache_control: cc }
        : { type: "text", text },
    );
  };

  for (const m of messages) {
    if (!isSystem(m)) continue;
    const msgCc = readCacheControl(m);
    if (typeof m.content === "string") {
      if (m.content.length > 0) push(m.content, msgCc);
      continue;
    }
    const textParts = m.content.filter(
      (p): p is TTextPart => p.type === "text" && p.text.length > 0,
    );
    textParts.forEach((p, idx) => {
      // A message-level breakpoint folds onto the message's last block.
      const isLast = idx === textParts.length - 1;
      push(p.text, readCacheControl(p) ?? (isLast ? msgCc : undefined));
    });
  }

  if (blocks.length === 0) return undefined;
  if (blocks.some((b) => b.cache_control !== undefined)) return blocks;
  return blocks.map((b) => b.text).join("\n\n");
};

export const toAnthropicRequest = (
  req: TChatCompletionRequest,
  options: TAnthropicProviderOptions,
): TAnthropicRequest => {
  const system = buildSystem(req.messages);

  const turnMessages = buildTurnMessages(req.messages);

  const max_tokens =
    req.max_completion_tokens ??
    req.max_tokens ??
    options.defaultMaxTokens ??
    DEFAULT_MAX_TOKENS;

  const stop_sequences =
    typeof req.stop === "string"
      ? [req.stop]
      : Array.isArray(req.stop)
        ? req.stop
        : undefined;

  const tools = req.tools?.map(toAnthropicTool);
  const toolChoice =
    req.tool_choice !== undefined
      ? toAnthropicToolChoice(req.tool_choice)
      : undefined;

  // `reasoning_effort` → Anthropic extended thinking. Single source of
  // truth in `./adaptive-thinking.ts`; this is the only call site on
  // the canonical (`/v1/chat/completions` → Anthropic) path.
  const effortMapping =
    req.reasoning_effort !== undefined
      ? mapReasoningEffortToAnthropic(
          req.reasoning_effort,
          options.providerModelId,
          max_tokens,
        )
      : null;

  return {
    model: options.providerModelId,
    messages: turnMessages,
    max_tokens,
    ...(system !== undefined ? { system } : {}),
    ...(req.temperature !== undefined ? { temperature: req.temperature } : {}),
    ...(req.top_p !== undefined ? { top_p: req.top_p } : {}),
    ...(stop_sequences !== undefined ? { stop_sequences } : {}),
    ...(req.stream !== undefined ? { stream: req.stream } : {}),
    ...(tools !== undefined && tools.length > 0 ? { tools } : {}),
    ...(toolChoice !== undefined ? { tool_choice: toolChoice } : {}),
    ...(req.user !== undefined ? { metadata: { user_id: req.user } } : {}),
    ...(effortMapping !== null ? { thinking: effortMapping.thinking } : {}),
    ...(effortMapping?.output_config !== undefined
      ? { output_config: effortMapping.output_config }
      : {}),
  };
};
