import type {
  TChatCompletionRequest,
  TChatGptProviderOptions,
  TChatMessage,
} from "@quantidexyz/openllmp";
import {
  reasoningItemsFromUnknown,
  reasoningItemToResponsesInput,
  type TReasoningResponsesInput,
} from "../../adapters/messages/reasoning-signature";
import { extractMessageText } from "../../lib/canonical/message";
import { CHATGPT_DEFAULT_INSTRUCTIONS } from "./common";

const CHATGPT_NAME_RE = /^[a-zA-Z0-9_-]+$/;
const CHATGPT_NAME_SUB_RE = /[^a-zA-Z0-9_-]/g;
const COLLAPSE_UNDERSCORE_RE = /_+/g;

/**
 * A stable, deterministic 64-bit hash (two FNV-1a passes with distinct offset
 * bases → 16 hex chars). Non-cryptographic — a prompt-cache key is a routing
 * hint, not a security boundary — but sync + env-agnostic (no `node:crypto` /
 * async WebCrypto), which the pure wire layer requires.
 */
const stableHash = (s: string): string => {
  let h1 = 0x811c9dc5; // FNV offset basis
  let h2 = 0x1000193; // FNV prime, reused as a second, independent seed
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 0x01000193) >>> 0;
    h2 = Math.imul(h2 ^ c, 0x85ebca77) >>> 0;
  }
  return h1.toString(16).padStart(8, "0") + h2.toString(16).padStart(8, "0");
};

/**
 * Derive a prompt-cache key that is STABLE across every turn of one
 * conversation and DISTINCT across conversations. Codex uses its own
 * `thread_id`; the stateless gateway has none, so we key off the immutable
 * conversation prefix — the `instructions` (Codex preamble + any user system
 * text) plus the first user turn — which never changes as the conversation
 * grows. Without this, OpenAI's automatic prefix-hash routing collides every
 * conversation onto the same cache lane (our Codex preamble prefix is
 * byte-identical across users), so distinct conversations evict each other and
 * cache-hit rate collapses — burning subscription quota. See the reference
 * `codex-rs/core/src/client.rs::prompt_cache_key`.
 */
const derivePromptCacheKey = (
  instructions: string,
  conversation: ReadonlyArray<TChatMessage>,
): string => {
  const firstUser = conversation.find((m) => m.role === "user");
  // Hash a canonical structured value (JSON), not a delimited text concat:
  // the COMPLETE first-user content keeps conversations distinct when they
  // differ only by non-text parts (images/files), and JSON encoding removes
  // delimiter ambiguity.
  const firstUserContent = firstUser !== undefined ? firstUser.content : null;
  return `openllm-${stableHash(
    JSON.stringify({ firstUserContent, instructions }),
  )}`;
};

/** Max `prompt_cache_key` length the OpenAI Responses backend accepts. A
 *  longer key 400s (`prompt_cache_key` too long); the partner client clamps to
 *  the same bound. Our synthesized key is 24 chars — this only bites a
 *  client's forwarded key. */
const PROMPT_CACHE_KEY_MAX = 64;

/** Clamp by Unicode code point (not UTF-16 unit) so a multi-byte key isn't
 *  split mid-character. */
const clampPromptCacheKey = (key: string): string => {
  const cps = Array.from(key);
  return cps.length <= PROMPT_CACHE_KEY_MAX
    ? key
    : cps.slice(0, PROMPT_CACHE_KEY_MAX).join("");
};

/**
 * Coerce a `name` field to match `^[a-zA-Z0-9_-]+$`. ChatGPT 400s on
 * any other character with a `pattern` error which triggers a retry
 * spiral. Mirrors `chat/transformation.py:64-79`.
 */
const sanitizeName = (name: string): string => {
  if (name === "" || CHATGPT_NAME_RE.test(name)) return name;
  const cleaned = name
    .replace(CHATGPT_NAME_SUB_RE, "_")
    .replace(COLLAPSE_UNDERSCORE_RE, "_")
    .replace(/^_+|_+$/g, "");
  return cleaned.length > 0 ? cleaned : "tool";
};

// runtime-only: a single Responses API content part. The input/output
// distinction matters — the chatgpt.com endpoint rejects an
// `output_text` part on a `user` message and vice versa.
type TResponsesContentPart =
  | { readonly type: "input_text"; readonly text: string }
  | { readonly type: "output_text"; readonly text: string }
  | {
      readonly type: "input_image";
      readonly image_url: string;
      readonly detail?: "auto" | "low" | "high";
    }
  | {
      readonly type: "input_file";
      readonly filename?: string;
      readonly file_data?: string;
      readonly file_id?: string;
    };

// runtime-only: a single item in the Responses API `input` array.
// Mirrors the union from openai-python's `ResponseInputItem`.
type TResponsesInputItem =
  | {
      readonly type: "message";
      readonly role: "user" | "assistant" | "system" | "developer";
      readonly content: ReadonlyArray<TResponsesContentPart>;
    }
  | {
      readonly type: "function_call";
      readonly call_id: string;
      readonly name: string;
      readonly arguments: string;
    }
  | {
      readonly type: "function_call_output";
      readonly call_id: string;
      readonly output: string;
    }
  | TReasoningResponsesInput;

const contentToInputParts = (
  content: TChatMessage["content"] | null | undefined,
): TResponsesContentPart[] => {
  if (content == null) return [{ type: "input_text", text: "" }];
  if (typeof content === "string") {
    return [{ type: "input_text", text: content }];
  }
  const parts: TResponsesContentPart[] = [];
  for (const block of content) {
    if (block.type === "text") {
      parts.push({ type: "input_text", text: block.text });
    } else if (block.type === "image_url") {
      parts.push({
        type: "input_image",
        image_url: block.image_url.url,
        ...(block.image_url.detail !== undefined
          ? { detail: block.image_url.detail }
          : {}),
      });
    } else if (block.type === "file") {
      // Responses `input_file.file_data` is a data URL — same encoding
      // as the canonical file part, so it copies through directly.
      parts.push({
        type: "input_file",
        ...(block.file.filename !== undefined
          ? { filename: block.file.filename }
          : {}),
        ...(block.file.file_data !== undefined
          ? { file_data: block.file.file_data }
          : {}),
        ...(block.file.file_id !== undefined
          ? { file_id: block.file.file_id }
          : {}),
      });
    }
  }
  if (parts.length === 0) parts.push({ type: "input_text", text: "" });
  return parts;
};

const contentToOutputParts = (
  content: TChatMessage["content"] | null | undefined,
): TResponsesContentPart[] => {
  if (content == null) return [{ type: "output_text", text: "" }];
  if (typeof content === "string") {
    return [{ type: "output_text", text: content }];
  }
  const parts: TResponsesContentPart[] = [];
  for (const block of content) {
    if (block.type === "text") {
      parts.push({ type: "output_text", text: block.text });
    }
  }
  if (parts.length === 0) parts.push({ type: "output_text", text: "" });
  return parts;
};

/**
 * Pull every `role: "system"` message out of the array, return both the
 * trimmed message list and the concatenated text. ChatGPT's
 * `/backend-api/codex/responses` endpoint rejects system turns on the
 * wire — they must ride in the top-level `instructions` field.
 *
 * Mirrors `_merge_system_and_developer_into_instruction_text` from
 * `chat/transformation.py:41-61`.
 */
const extractSystemInstructions = (
  messages: ReadonlyArray<TChatMessage>,
): { conversation: TChatMessage[]; instructions: string } => {
  const parts: string[] = [];
  const conversation: TChatMessage[] = [];
  for (const msg of messages) {
    if (msg.role === "system") {
      const text = extractMessageText(msg.content);
      if (text.trim().length > 0) parts.push(text);
      continue;
    }
    conversation.push(msg);
  }
  return {
    conversation,
    instructions: parts.filter((p) => p.trim().length > 0).join("\n\n"),
  };
};

const TOOL_RESULT_IMAGE_REPLAY_TEXT = "Attached image(s) from tool result:";

/**
 * Convert one canonical tool-result message into Responses input items.
 * `function_call_output.output` is a STRING — the shape Codex itself and
 * litellm's reference transformation send, and the one field xAI's partner
 * client (openclaw) force-coerces for Grok before every request (see
 * docs/audit/2026-07-14-grok-upstream-wire-openclaw-comparison.md §F1).
 * Non-text parts can't ride in the string: images are replayed as an
 * IMMEDIATELY-FOLLOWING `user` message (adjacent, not end-of-input, so the
 * conversation prefix stays byte-stable across turns for prompt caching);
 * media without a text sibling leaves a placeholder so the model knows the
 * tool returned something.
 */
const toolResultToItems = (
  msg: Extract<TChatMessage, { readonly role: "tool" }>,
): TResponsesInputItem[] => {
  const content = msg.content;
  const text = extractMessageText(content);
  const images: TResponsesContentPart[] = [];
  let hasOtherMedia = false;
  if (content != null && typeof content !== "string") {
    for (const block of content) {
      if (block.type === "image_url") {
        images.push({
          type: "input_image",
          image_url: block.image_url.url,
          ...(block.image_url.detail !== undefined
            ? { detail: block.image_url.detail }
            : {}),
        });
      } else if (block.type !== "text") {
        hasOtherMedia = true;
      }
    }
  }
  const output =
    text.trim().length > 0
      ? text
      : hasOtherMedia
        ? "(see attached media)"
        : images.length > 0
          ? "(see attached image)"
          : "";
  const items: TResponsesInputItem[] = [
    { type: "function_call_output", call_id: msg.tool_call_id, output },
  ];
  if (images.length > 0) {
    items.push({
      type: "message",
      role: "user",
      content: [
        { type: "input_text", text: TOOL_RESULT_IMAGE_REPLAY_TEXT },
        ...images,
      ],
    });
  }
  return items;
};

/**
 * Convert the canonical OpenAI ChatCompletion message array into
 * Responses API input items. Mirrors
 * `convert_chat_completion_messages_to_responses_api` from
 * `completion_extras/litellm_responses_transformation/transformation.py:203-289`.
 *
 * - `user` / `system` content -> `input_text` parts (system already
 *   pulled into `instructions` upstream of this call).
 * - `assistant` text content  -> `output_text` parts.
 * - `assistant.tool_calls`    -> one `function_call` item per call.
 * - `tool` (tool result)      -> `function_call_output` (string) via
 *   {@link toolResultToItems}, plus an image-replay user message.
 */
const messagesToInputItems = (
  messages: ReadonlyArray<TChatMessage>,
): TResponsesInputItem[] => {
  const items: TResponsesInputItem[] = [];
  for (const msg of messages) {
    if (msg.role === "user") {
      items.push({
        type: "message",
        role: "user",
        content: contentToInputParts(msg.content),
      });
      continue;
    }
    if (msg.role === "tool") {
      items.push(...toolResultToItems(msg));
      continue;
    }
    if (msg.role === "assistant") {
      // Echo prior `reasoning` item(s) back, in order, immediately
      // before the assistant's tool calls / content. The Responses API
      // requires this for reasoning models (`store: false`); dropping
      // it makes the model restart reasoning and loop. Mirrors litellm
      // `transformation.py:261-262, 279-280`.
      const reasoningItems = reasoningItemsFromUnknown(msg.reasoning_items);
      for (const r of reasoningItems) {
        items.push(reasoningItemToResponsesInput(r));
      }
      const toolCalls = msg.tool_calls;
      if (toolCalls !== undefined && toolCalls.length > 0) {
        for (const call of toolCalls) {
          items.push({
            type: "function_call",
            call_id: call.id,
            name: sanitizeName(call.function.name),
            arguments: call.function.arguments,
          });
        }
        // Assistant text alongside tool_calls is rare, but allowed —
        // emit a separate message item if present.
        if (msg.content != null) {
          const text = extractMessageText(msg.content);
          if (text.trim().length > 0) {
            items.push({
              type: "message",
              role: "assistant",
              content: contentToOutputParts(msg.content),
            });
          }
        }
        continue;
      }
      if (msg.content != null) {
        items.push({
          type: "message",
          role: "assistant",
          content: contentToOutputParts(msg.content),
        });
      }
    }
    // system messages are filtered out before this call.
  }
  return items;
};

// runtime-only: a single tool definition in the Responses API. Note
// the FLAT shape — the chat-completions tool wrapper
// (`{type:"function", function:{name,...}}`) is not accepted here.
type TResponsesToolDef = {
  readonly type: "function";
  readonly name: string;
  readonly description?: string;
  readonly parameters?: unknown;
  readonly strict?: boolean;
};

// runtime-only: a Codex built-in / non-function tool carried verbatim from the
// inbound Responses request (`custom` apply_patch, `web_search`,
// `image_generation`, `tool_search`). Opaque — re-emitted as-is to the chatgpt
// upstream, which is the same endpoint Codex sends them to natively.
type TResponsesPassthroughToolDef = {
  readonly type: string;
  readonly [key: string]: unknown;
};

const toolsToResponses = (
  tools: NonNullable<TChatCompletionRequest["tools"]>,
): TResponsesToolDef[] =>
  tools.map((tool) => ({
    type: "function",
    name: sanitizeName(tool.function.name),
    ...(tool.function.description !== undefined
      ? { description: tool.function.description }
      : {}),
    ...(tool.function.parameters !== undefined
      ? { parameters: tool.function.parameters }
      : {}),
    ...(tool.function.strict !== undefined
      ? { strict: tool.function.strict }
      : {}),
  }));

// runtime-only: tool_choice in the Responses API. Mirrors the tools
// shape — FLAT `{type:"function", name}`, NOT the chat-completions
// `{type:"function", function:{name}}` wrapper. Forwarding the chat
// shape verbatim 400s with `Unknown parameter: 'tool_choice.function'.`
type TResponsesToolChoice =
  | "auto"
  | "none"
  | "required"
  | { readonly type: "function"; readonly name: string };

const toResponsesToolChoice = (
  choice: NonNullable<TChatCompletionRequest["tool_choice"]>,
): TResponsesToolChoice =>
  choice === "auto" || choice === "none" || choice === "required"
    ? choice
    : { type: "function", name: sanitizeName(choice.function.name) };

// runtime-only: payload sent to `/backend-api/codex/responses`. Strictly
// the keys allowed by `ChatGPTResponsesAPIConfig.transform_responses_api_request`
// (`responses/transformation.py:215-227`), plus `max_output_tokens`. Anything
// outside this list is dropped to avoid `Unsupported parameter` 400s.
//
// Notably ABSENT: `temperature`, `top_p`, `frequency_penalty`,
// `presence_penalty`, `seed`, `response_format`, `metadata`, `user`.
// `temperature`/`top_p` are ACCEPTED by the Grok chat proxy (verified live
// 2026-07-14) but stay off this wire pending a chatgpt.com probe — the Codex
// allowed-list doesn't include them. `response_format`/`text.format` is
// deliberately dropped: the Grok proxy accepts the field but does NOT
// enforce it and derails into a runaway generation (a 16k-token garbage
// completion, verified live — audit 2026-07-14 §5).
export type TChatGptRequestBody = {
  readonly model: string;
  readonly input: ReadonlyArray<TResponsesInputItem>;
  readonly instructions: string;
  readonly stream: true;
  readonly store: false;
  readonly include: ReadonlyArray<string>;
  // The client's token cap — emitted ONLY on the non-Codex Responses
  // variant (`codexInstructions: false`, i.e. grok), where the chat proxy
  // honors it as a hard cap (verified live 2026-07-14). The chatgpt.com
  // Codex endpoint rejects it (`Unsupported parameter: max_output_tokens`),
  // so Codex hops keep dropping it.
  readonly max_output_tokens?: number;
  readonly tools?: ReadonlyArray<
    TResponsesToolDef | TResponsesPassthroughToolDef
  >;
  readonly tool_choice?: TResponsesToolChoice;
  // `summary: "auto"` rides with every effort — Codex itself sends it, the
  // Grok proxy accepts it (verified live), and the stream decoder already
  // maps `response.reasoning_summary_text.delta` → `reasoning_content`.
  readonly reasoning?: {
    readonly effort: "low" | "medium" | "high";
    readonly summary: "auto";
  };
  readonly previous_response_id?: string;
  readonly truncation?: "auto" | "disabled";
  // Stable per-conversation prompt-cache routing hint (preserved from the
  // inbound Codex request or synthesized from the prefix). Codex ALWAYS sends
  // this; omitting it collapses cache-hit rate. See `derivePromptCacheKey`.
  readonly prompt_cache_key?: string;
};

/**
 * Convert canonical OpenAI ChatCompletion → ChatGPT/Codex Responses API body.
 *
 * 1. Pull system messages into `instructions`.
 * 2. Prepend the Codex preamble if not already present (required by
 *    gpt-5.x or the server returns empty `output`).
 * 3. Convert `messages` -> `input` items.
 * 4. Sanitize every tool name + assistant tool_call name.
 * 5. Force `stream: true`, `store: false`,
 *    `include: ["reasoning.encrypted_content"]`.
 * 6. Map `max_tokens` / `max_completion_tokens` -> `max_output_tokens` —
 *    non-Codex upstreams only (`codexInstructions: false`).
 * 7. Map `reasoning_effort` -> `reasoning.effort` (+ `summary: "auto"`).
 * 8. DROP every other key — only the allowed-list is forwarded.
 *
 * Mirrors `transform_request` in `chat/transformation.py:212-248` plus
 * the Responses-API allowed-list filter in
 * `responses/transformation.py:215-229`.
 */
export const toChatGptRequest = (
  req: TChatCompletionRequest,
  options: TChatGptProviderOptions,
): TChatGptRequestBody => {
  const { conversation, instructions: fromSystem } = extractSystemInstructions(
    req.messages,
  );

  // The Codex preamble is a Codex IDENTITY required by the ChatGPT backend, but
  // wrong for other providers on the same Responses wire (xAI Grok). Inject it
  // only when the caller wants it — `codexInstructions !== false` (undefined =
  // inject, preserving every existing chatgpt caller). When suppressed, only the
  // user's OWN system messages (`fromSystem`) ride in `instructions`.
  let instructions = fromSystem;
  if (
    options.codexInstructions !== false &&
    !instructions.includes(CHATGPT_DEFAULT_INSTRUCTIONS)
  ) {
    instructions =
      instructions.length > 0
        ? `${CHATGPT_DEFAULT_INSTRUCTIONS}\n\n${instructions}`
        : CHATGPT_DEFAULT_INSTRUCTIONS;
  }

  const input = messagesToInputItems(conversation);
  // Prefer the verbatim `responses_tools` passthrough (Codex's full original
  // tool set, function + non-function) — it round-trips apply_patch /
  // web_search / image_generation / tool_search intact to the same endpoint
  // Codex speaks to natively. Fall back to the function-only canonical tools
  // (a cross-wire client, or a non-Codex caller that set `tools`).
  const responsesTools:
    | ReadonlyArray<TResponsesToolDef | TResponsesPassthroughToolDef>
    | undefined =
    req.responses_tools !== undefined && req.responses_tools.length > 0
      ? (req.responses_tools as ReadonlyArray<TResponsesPassthroughToolDef>)
      : req.tools !== undefined && req.tools.length > 0
        ? toolsToResponses(req.tools)
        : undefined;

  return {
    model: options.providerModelId,
    input,
    instructions,
    stream: true,
    store: false,
    include: ["reasoning.encrypted_content"],
    // Preserve the caller's key when present (a genuine Codex request already
    // carries a stable per-thread one); otherwise synthesize one off the
    // conversation prefix so turns of the same conversation share a cache lane.
    prompt_cache_key: clampPromptCacheKey(
      req.prompt_cache_key !== undefined && req.prompt_cache_key.length > 0
        ? req.prompt_cache_key
        : derivePromptCacheKey(instructions, conversation),
    ),
    ...(responsesTools !== undefined ? { tools: responsesTools } : {}),
    ...(req.tool_choice !== undefined
      ? { tool_choice: toResponsesToolChoice(req.tool_choice) }
      : {}),
    ...(() => {
      // Non-Codex Responses upstreams only (grok): the Codex backend 400s
      // on the field, the Grok chat proxy honors it (see the type comment).
      if (options.codexInstructions !== false) return {};
      const cap = req.max_completion_tokens ?? req.max_tokens;
      return cap !== undefined ? { max_output_tokens: cap } : {};
    })(),
    ...(() => {
      // ChatGPT's Responses API only accepts `low | medium | high`.
      // Map the wider canonical enum (`minimal/xhigh/max/none`) down to
      // the closest supported neighbour: `minimal` → low, `xhigh`/`max`
      // → high, `none` → reasoning omitted entirely.
      const e = req.reasoning_effort;
      if (e === undefined || e === "none") return {};
      const effort: "low" | "medium" | "high" =
        e === "minimal" || e === "low"
          ? "low"
          : e === "medium"
            ? "medium"
            : "high";
      return { reasoning: { effort, summary: "auto" as const } };
    })(),
  };
};
