/**
 * Choose the next chain hop that can accommodate an input of the required
 * size. Shared by the cloud dispatch chain and the coreless daemon walker so
 * context-overflow demotion cannot drift between the two paths.
 */
export const nextLargerContextModel = (
  chainModels: ReadonlyArray<string>,
  currentModel: string,
  requiredTokens: number,
  windowOf: (modelId: string) => number | null,
): string | null => {
  const currentIndex = chainModels.indexOf(currentModel);
  if (currentIndex === -1) return null;
  for (const modelId of chainModels.slice(currentIndex + 1)) {
    const window = windowOf(modelId);
    if (window !== null && window > requiredTokens) return modelId;
  }
  return null;
};

/**
 * Extract the upstream tokenizer's authoritative request size from an overflow
 * diagnostic. Recognises the three vendor shapes observed live:
 *   - vLLM-style:   "maximum prompt length is N ... contains M tokens" → M
 *   - Anthropic:    "prompt is too long: N tokens > M maximum" → N
 *   - Kimi/DeepSeek:"... token limit: X (requested: N)" → N
 * Returns the REQUIRED size (how big the request actually is), not the window.
 */
export const contextOverflowRequiredTokens = (raw: string): number | null => {
  const patterns = [
    // vLLM-style: the "contains M tokens" count.
    /maximum (?:prompt|context) length is\s+[\d,]+.*?contains\s+([\d,]+)\s+tokens/i,
    // Anthropic: "prompt is too long: N tokens > M maximum".
    /prompt is too long:\s*([\d,]+)\s*tokens/i,
    // Kimi/DeepSeek: "token limit: X (requested: N)". The "token limit" prefix
    // is required so an unrelated "(requested: N)" (e.g. a quota / validation
    // error) is NOT misread as a context overflow.
    /token limit:?\s*[\d,]+\s*\(requested:\s*([\d,]+)\)/i,
  ];
  for (const pattern of patterns) {
    const match = pattern.exec(raw);
    if (match === null) continue;
    const required = Number((match[1] ?? "").replaceAll(",", ""));
    if (Number.isSafeInteger(required) && required > 0) return required;
  }
  return null;
};
