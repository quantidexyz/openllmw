import type {
  TChatCompletionRequest,
  TChatMessage,
  TResponsesInputItem,
  TResponsesRequest,
  TToolCall,
} from "@quantidexyz/openllmp";
import type { TCanonicalContentPart } from "../../lib/canonical/content-part";

/**
 * Inbound adapter: OpenAI **Responses API** request → canonical
 * ChatCompletion request. The inverse of `toChatGptRequest`
 * (`@quantidexyz/openllmw/providers/chatgpt/request.ts`), so a Codex client (which
 * speaks only the Responses API) can drive the gateway's normal pipeline.
 *
 *  - `instructions`            → a leading `system` message.
 *  - `input` (string)          → a single `user` message.
 *  - `input` item `message`    → user/assistant/system message (parts →
 *                                canonical text/image content).
 *  - `input` item `function_call` → an `assistant` message with one
 *                                `tool_calls` entry.
 *  - `input` item `function_call_output` → a `tool` message.
 *  - `input` item `reasoning`  → carried onto the NEXT assistant message's
 *                                `reasoning_items` so it round-trips.
 *  - flat `tools`/`tool_choice` → canonical wrapped shapes.
 *  - `reasoning.effort` → `reasoning_effort`; `max_output_tokens` → `max_tokens`.
 */

type TResponsesContentPart = Extract<
  TResponsesInputItem,
  { type: "message" }
>["content"];

/** Responses content (string | parts) → canonical content (string | parts). */
const toCanonicalContent = (
  content: TResponsesContentPart,
): string | TCanonicalContentPart[] => {
  if (typeof content === "string") return content;
  const parts: TCanonicalContentPart[] = [];
  for (const p of content) {
    if (p.type === "input_text" || p.type === "output_text") {
      parts.push({ type: "text", text: p.text });
    } else if (p.type === "input_image") {
      parts.push({
        type: "image_url",
        image_url: {
          url: p.image_url,
          ...(p.detail !== undefined ? { detail: p.detail } : {}),
        },
      });
    } else if (p.type === "input_file") {
      parts.push({
        type: "file",
        file: {
          ...(p.file_data !== undefined ? { file_data: p.file_data } : {}),
          ...(p.file_id != null ? { file_id: p.file_id } : {}),
          ...(p.filename !== undefined ? { filename: p.filename } : {}),
        },
      });
    }
  }
  return parts;
};

/** Flatten Responses content to plain text (for `tool` message bodies). */
const contentToText = (content: TResponsesContentPart): string => {
  if (typeof content === "string") return content;
  return content
    .map((p) =>
      p.type === "input_text" || p.type === "output_text" ? p.text : "",
    )
    .join("");
};

/**
 * Canonical tools — `function` tools ONLY. Codex's non-function built-ins
 * (`custom` apply_patch, `web_search`, `image_generation`, `tool_search`) have
 * no canonical representation, so they're dropped here; they survive for the
 * chatgpt upstream via the verbatim `responses_tools` passthrough instead (see
 * `fromResponsesRequest`). For a cross-wire upstream (anthropic / openai-chat)
 * the non-function tools can't be honoured anyway, so dropping them is correct.
 */
const mapTools = (
  tools: TResponsesRequest["tools"],
): TChatCompletionRequest["tools"] => {
  if (tools == null) return undefined;
  const fns = tools.filter(
    (t) => t.type === "function" && typeof t.name === "string",
  );
  if (fns.length === 0) return undefined;
  return fns.map((t) => ({
    type: "function" as const,
    function: {
      name: t.name as string,
      ...(t.description != null ? { description: t.description } : {}),
      ...(t.parameters !== undefined ? { parameters: t.parameters } : {}),
      ...(t.strict != null ? { strict: t.strict } : {}),
    },
  }));
};

const mapToolChoice = (
  choice: TResponsesRequest["tool_choice"],
): TChatCompletionRequest["tool_choice"] =>
  choice == null || typeof choice === "string"
    ? (choice ?? undefined)
    : { type: "function", function: { name: choice.name } };

/** Codex sends `minimal`; the canonical enum has it, so pass effort through. */
const mapEffort = (
  reasoning: TResponsesRequest["reasoning"],
): TChatCompletionRequest["reasoning_effort"] => {
  const e = reasoning?.effort;
  return e === undefined || e === null ? undefined : e;
};

export const fromResponsesRequest = (
  req: TResponsesRequest,
): TChatCompletionRequest => {
  const messages: TChatMessage[] = [];

  const instr = req.instructions;
  if (typeof instr === "string" && instr.trim().length > 0) {
    messages.push({ role: "system", content: instr });
  }

  // Buffer reasoning items until the next assistant-role output, which they
  // precede in the Responses `input` ordering (mirrors `toChatGptRequest`).
  let pendingReasoning: unknown[] = [];
  const takeReasoning = (): { reasoning_items?: ReadonlyArray<unknown> } => {
    if (pendingReasoning.length === 0) return {};
    const items = pendingReasoning;
    pendingReasoning = [];
    return { reasoning_items: items };
  };

  if (typeof req.input === "string") {
    messages.push({ role: "user", content: req.input });
  } else {
    for (const item of req.input) {
      if (item.type === "reasoning") {
        pendingReasoning.push(item);
      } else if (item.type === "function_call") {
        const toolCall: TToolCall = {
          id: item.call_id,
          type: "function",
          function: { name: item.name, arguments: item.arguments },
        };
        messages.push({
          role: "assistant",
          content: null,
          tool_calls: [toolCall],
          ...takeReasoning(),
        });
      } else if (item.type === "function_call_output") {
        messages.push({
          role: "tool",
          tool_call_id: item.call_id,
          content: contentToText(item.output),
        });
      } else {
        // message item
        const role = item.role === "developer" ? "system" : item.role;
        const content = toCanonicalContent(item.content);
        if (role === "assistant") {
          messages.push({ role: "assistant", content, ...takeReasoning() });
        } else {
          messages.push({ role, content });
        }
      }
    }
    // Trailing reasoning with no assistant output to attach to.
    if (pendingReasoning.length > 0) {
      messages.push({ role: "assistant", content: null, ...takeReasoning() });
    }
  }

  const tools = mapTools(req.tools);
  const toolChoice = mapToolChoice(req.tool_choice);
  const effort = mapEffort(req.reasoning);

  return {
    model: req.model,
    messages,
    stream: req.stream === true,
    ...(tools !== undefined ? { tools } : {}),
    // Carry the ORIGINAL Responses tools verbatim (function + non-function) so
    // a chatgpt upstream re-emits Codex's apply_patch / web_search / … intact.
    // Stripped before every openai-family upstream. See the schema field doc.
    ...(req.tools != null && req.tools.length > 0
      ? { responses_tools: req.tools }
      : {}),
    ...(toolChoice !== undefined ? { tool_choice: toolChoice } : {}),
    ...(effort !== undefined ? { reasoning_effort: effort } : {}),
    ...(req.max_output_tokens != null
      ? { max_tokens: req.max_output_tokens }
      : {}),
    ...(req.temperature != null ? { temperature: req.temperature } : {}),
    ...(req.top_p != null ? { top_p: req.top_p } : {}),
    ...(req.parallel_tool_calls != null
      ? { parallel_tool_calls: req.parallel_tool_calls }
      : {}),
    // Preserve Codex's per-conversation prompt-cache key so the re-encoded
    // upstream keeps routing to the same cache (rather than synthesizing a new
    // one). Absent → `toChatGptRequest` derives a stable key from the prefix.
    ...(req.prompt_cache_key != null
      ? { prompt_cache_key: req.prompt_cache_key }
      : {}),
  };
};
