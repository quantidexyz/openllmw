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
 * Extract the upstream tokenizer's authoritative request size from the common
 * "maximum prompt length is N ... contains M tokens" overflow diagnostic.
 */
export const contextOverflowRequiredTokens = (raw: string): number | null => {
  const match = /maximum (?:prompt|context) length is\s+([\d,]+).*?contains\s+([\d,]+)\s+tokens/i.exec(
    raw,
  );
  if (match === null) return null;
  const required = Number((match[2] ?? "").replaceAll(",", ""));
  return Number.isSafeInteger(required) && required > 0 ? required : null;
};
