/**
 * Recovery for a replayed `reasoning.encrypted_content` the upstream can't
 * decrypt. The OpenAI Responses backend (ChatGPT/Codex) and the Grok chat
 * proxy only decrypt encrypted reasoning THEY minted for that
 * account+model+session. On a fallback that switches account/model, the prior
 * hop's encrypted reasoning replays into a hop that can't decrypt it and the
 * backend 400s. Stripping the resumable state and retrying makes the request
 * succeed (the model just re-reasons) instead of surfacing a hard error — the
 * exact recovery the official partner client performs
 * (audit 2026-07-14-codex-upstream-wire §F2).
 */

/** Does a 400 body signal an undecryptable replayed reasoning item? Matched
 *  loosely across the OpenAI + xAI phrasings (`invalid_encrypted_content`,
 *  `thinking_signature_invalid`, "could not decrypt", "failed to decrypt"). */
const DECRYPT_FAILURE_RE =
  /encrypted[_\s-]?content|thinking[_\s-]?signature|(?:could|failed|unable)[^.]*?decrypt|decrypt[^.]*?fail/i;

export const isEncryptedContentError = (rawBody: string): boolean =>
  rawBody.length > 0 && DECRYPT_FAILURE_RE.test(rawBody);

/** True when a built Responses body carries a `reasoning` input item with
 *  non-empty `encrypted_content` — i.e. stripping it would change the request
 *  (so a retry is worth attempting). */
export const responsesBodyHasEncryptedContent = (body: unknown): boolean => {
  if (body === null || typeof body !== "object") return false;
  const input = (body as { input?: unknown }).input;
  if (!Array.isArray(input)) return false;
  return input.some(
    (item) =>
      item !== null &&
      typeof item === "object" &&
      (item as { type?: unknown }).type === "reasoning" &&
      typeof (item as { encrypted_content?: unknown }).encrypted_content ===
        "string" &&
      ((item as { encrypted_content: string }).encrypted_content.length ?? 0) >
        0,
  );
};

/** Return a copy of a built Responses body with `encrypted_content` removed
 *  from every `reasoning` input item (the item itself stays — its `summary`
 *  keeps the visible chain — only the resumable ciphertext is dropped). */
export const stripResponsesEncryptedContent = (body: unknown): unknown => {
  if (body === null || typeof body !== "object") return body;
  const b = body as { input?: unknown };
  if (!Array.isArray(b.input)) return body;
  const input = b.input.map((item) => {
    if (
      item === null ||
      typeof item !== "object" ||
      (item as { type?: unknown }).type !== "reasoning"
    ) {
      return item;
    }
    const { encrypted_content: _dropped, ...rest } = item as Record<
      string,
      unknown
    >;
    return rest;
  });
  return { ...(body as Record<string, unknown>), input };
};
