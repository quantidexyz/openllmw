/**
 * The TWO input-token estimators, together in one module so the choice between
 * them is visible at the point of use. They are not interchangeable — each is
 * wrong for the other's job:
 *
 *   - {@link estimateBodyTokens} — the ROUTING gate. Every string in the body
 *     counts, so it OVER-estimates (discriminators, tool names, a base64 image
 *     payload). That is the point: it decides whether a hop's context window can
 *     hold the request, where over-shooting costs one extra chain step and
 *     under-shooting ships a request that 400s.
 *   - {@link estimateAnthropicInputTokens} — the CLIENT-FACING preflight
 *     (`/v1/messages/count_tokens`). Counts only what the model actually reads,
 *     so a client's context-window indicator isn't inflated by transport noise.
 *     Never use the routing estimator here: base64 image data alone would make
 *     Claude Code believe a fresh session was nearly full.
 */

/**
 * Rough input-token estimator over a request body, format-agnostic.
 *
 * Walks the JSON value summing string lengths (everything else
 * contributes zero) and divides by 4. Works for both canonical
 * OpenAI `ChatCompletionRequest` and raw Anthropic Messages bodies
 * because both encode all user-visible text inside string values.
 *
 * Used pre-fetch by the chain orchestrator to skip a hop whose
 * provider model can't fit the request — preventing the otherwise
 * inevitable 400 + waste of the chain entry. The estimate is
 * intentionally conservative: in borderline cases we'd rather burn
 * an extra chain step than ship an oversized request that 400s.
 */

const stringCharsFromAny = (v: unknown): number => {
  if (typeof v === "string") return v.length;
  if (Array.isArray(v)) {
    let n = 0;
    for (const x of v) n += stringCharsFromAny(x);
    return n;
  }
  if (v !== null && typeof v === "object") {
    let n = 0;
    for (const k of Object.keys(v as Record<string, unknown>)) {
      n += stringCharsFromAny((v as Record<string, unknown>)[k]);
    }
    return n;
  }
  return 0;
};

export const estimateBodyTokens = (body: unknown): number =>
  Math.ceil(stringCharsFromAny(body) / 4);

// ~4 chars per token — within ~15% for English prose, which is all a preflight
// needs (model-availability check + a context-window indicator). `max(1, …)`
// so any non-empty text counts as at least one token.
const textTokens = (text: string): number =>
  text.length === 0 ? 0 : Math.max(1, Math.ceil(text.length / 4));

/**
 * The text an Anthropic content value actually puts in front of the model:
 * `text` blocks, a `tool_use`'s serialized input, a `tool_result`'s body
 * (recursively — it may itself be a block array), and prior `thinking`. Block
 * types the model does not read as text (image/document sources) contribute
 * nothing, which is exactly why this is not `estimateBodyTokens`.
 */
const anthropicContentText = (content: unknown): string => {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (block === null || typeof block !== "object") continue;
    const b = block as Record<string, unknown>;
    if (b.type === "text" && typeof b.text === "string") parts.push(b.text);
    else if (b.type === "tool_use") {
      // The name is billed alongside the arguments — both are in the transcript
      // the model reads back.
      if (typeof b.name === "string") parts.push(b.name);
      parts.push(JSON.stringify(b.input ?? {}));
    } else if (b.type === "tool_result") {
      const inner = b.content;
      if (typeof inner === "string") parts.push(inner);
      else if (Array.isArray(inner)) parts.push(anthropicContentText(inner));
    } else if (b.type === "thinking" && typeof b.thinking === "string") {
      parts.push(b.thinking);
    }
  }
  return parts.join("\n");
};

/**
 * Estimated `input_tokens` for an Anthropic-shaped Messages body — the answer
 * `/v1/messages/count_tokens` serves ONLY as a fallback, when the vendor's own
 * endpoint isn't reachable (no Anthropic-wire hop at the head of the chain, no
 * local credential, or an upstream refusal). Whenever the vendor can be asked,
 * its exact count is relayed instead and this function is not consulted.
 *
 * Single-sourced here because BOTH preflight paths need the same number: the
 * cloud handler (`@openllm/api/handlers/count-tokens`) and the daemon's local
 * surface (`runCountTokens` in the walker). They used to carry separate copies.
 *
 * KNOWN LIMIT — text only. Images and documents contribute nothing, so an
 * image-heavy (or image-only) request UNDER-counts, materially. This is
 * deliberate: Anthropic prices an image at roughly `(w × h) / 750` tokens, and
 * neither dimension is recoverable from a base64 blob or a URL without decoding
 * or fetching it — so any number we put there would be fabricated, and a
 * fabricated count is worse than a knowingly-low one for a fallback whose
 * consumers use it as a context-window indicator. Counting the blob's own length
 * (what {@link estimateBodyTokens} does) is not a substitute: it tracks encoding
 * size, not token cost, and overstates by orders of magnitude.
 */
export const estimateAnthropicInputTokens = (body: unknown): number => {
  if (body === null || typeof body !== "object") return 0;
  const b = body as Record<string, unknown>;
  let total = 0;
  const system = b.system;
  if (typeof system === "string") total += textTokens(system);
  else if (Array.isArray(system)) {
    total += textTokens(anthropicContentText(system));
  }
  if (Array.isArray(b.messages)) {
    for (const m of b.messages) {
      if (m === null || typeof m !== "object") continue;
      total += textTokens(
        anthropicContentText((m as Record<string, unknown>).content),
      );
    }
  }
  // Tool schemas are billed as input too, and they are not small.
  if (Array.isArray(b.tools) && b.tools.length > 0) {
    total += textTokens(JSON.stringify(b.tools));
  }
  return total;
};
