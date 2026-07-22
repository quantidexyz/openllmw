import type {
  TChatCompletionChunk,
  TChatCompletionResponse,
} from "@openllmsh/protocol";

/**
 * ─── Structured refusal detection (provider-independent) ───────────────
 *
 * Some providers answer `200 OK` with a body that IS a refusal rather
 * than a transport error — an Anthropic policy/safety block
 * (`stop_reason: "refusal"`), an OpenAI `finish_reason: "content_filter"`,
 * a Gemini `promptFeedback.blockReason`, etc. Every provider adapter
 * normalises these to the SAME canonical shape: a choice whose
 * `finish_reason === "content_filter"`. So detection is a single
 * canonical-field check — NEVER a match on model ids, request ids,
 * provider names, links, or human-readable policy prose.
 *
 * With a multi-provider fallback chain, "the model refused" is the wrong
 * terminal default: another candidate may not refuse the same canonical
 * request. These predicates let a caller treat a pre-commit structured
 * refusal like any other pre-output failure — walk to the next hop —
 * while a refusal that arrives AFTER real output stays committed (the
 * caller decides based on WHEN it sees the refusal, not just whether).
 *
 * Shared by the cloud dispatch chain (`@openllm/core`) and the coreless
 * daemon walker so the two paths cannot drift — the same rule the shared
 * streaming peek follows.
 */

/** A single decoded chunk that carries a canonical structured refusal. */
export const isRefusalChunk = (chunk: TChatCompletionChunk): boolean =>
  chunk.choices.some((choice) => choice.finish_reason === "content_filter");

/** A fully-decoded non-streaming response that IS a canonical refusal. */
export const isCanonicalRefusal = (
  response: TChatCompletionResponse,
): boolean =>
  response.choices.some((choice) => choice.finish_reason === "content_filter");
