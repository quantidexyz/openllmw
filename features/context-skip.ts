/**
 * Context-skip gate — the per-hop context decision, shared verbatim by
 * the cloud dispatch chain (BYOK) and the daemon walker (device) so the
 * two paths cannot drift.
 *
 * The context "ladder" is deliberately thin (reference-proxy parity —
 * ref/CLIProxyAPI does no gateway-side rewriting either):
 *
 *   Plan A — serve correct per-model context metadata (and pass through
 *            Codex's own `/responses/compact`) so the CLIENT compacts
 *            itself with a STABLE prefix — the only compaction that
 *            preserves prompt-cache affinity;
 *   gate   — THIS module: skip a hop only when the estimate CLEARLY
 *            exceeds its window (saves the doomed round trip + prefill
 *            wait);
 *   walk   — the pre-commit first-event peek
 *            (`lib/streaming/peek.ts`): the real upstream tokenizer gets
 *            the final word, and a pre-output rejection walks the chain
 *            with zero double-spend.
 *
 * Gateway-side tool-output rewriting (the former "Plan B") was removed
 * on purpose: estimator-driven per-request rewrites destabilise the
 * conversation prefix (prompt-cache misses → quota drain) and silently
 * degrade requests the vendor would have accepted.
 */

/**
 * Confidence multiplier before a hop is abandoned over context. The
 * routing estimator is intentionally conservative (it counts every string
 * value), so an estimate above the catalogued input budget is enough to skip
 * a non-final hop: that model has already advertised it cannot accept the
 * request. Final and unknown-limit hops still reach the real tokenizer.
 *
 * A factor above 1 let known-over-budget requests through to an inevitable
 * upstream context rejection. Keep this at one unless the estimator contract
 * itself changes.
 */
export const CONTEXT_SKIP_CONFIDENCE_FACTOR = 1;

/**
 * Per-provider ruler→vendor-token calibration factor for last-resort compaction.
 * The local `estimateBodyTokens` ruler (o200k / claude encodings) counts FEWER
 * tokens than the vendor's own tokenizer for these providers, so a body
 * "compacted to fit window W" on our ruler can be W × factor real tokens and
 * still overflow upstream (making the retry fail too — and the one-shot latch
 * forbids a second attempt). Dividing the compaction target by the factor makes
 * the compacted body fit the VENDOR's tokenizer, not just ours.
 *
 * Seeded from live measurement (docs/audit/2026-07-23-compaction-live-test.md §4):
 *   - claude_code (Anthropic): vendor counted 299434 vs our 216246 → ×1.385
 *   - kimi_code (Kimi):        vendor counted 317227 vs our 283969 → ×1.117
 *   - chatgpt/openai (o200k):  o200k is OpenAI's own encoding → ~1.0
 * This is the static seed for the catalog-driven Layer-2 calibration the spike
 * recommended; it should eventually be derived from logged `tokens_in` instead.
 */
export const PROVIDER_TOKEN_ESTIMATE_FACTOR: Readonly<Record<string, number>> =
  {
    anthropic: 1.4,
    claude_code: 1.4,
    kimi_code: 1.12,
    // OpenAI-family rulers are (near-)exact; no inflation.
    openai: 1,
    chatgpt: 1,
    grok: 1,
  };

/**
 * The compaction target for a hop: its input window shrunk by the provider's
 * calibration factor so a body compacted to this size fits the vendor's real
 * tokenizer. Unknown providers use 1 (no inflation — fail open).
 */
export const compactionTargetFor = (
  provider: string,
  window: number,
): number => {
  const factor = PROVIDER_TOKEN_ESTIMATE_FACTOR[provider] ?? 1;
  return Math.floor(window / factor);
};

/**
 * Should this hop be skipped for context? True only when a later hop
 * remains (the FINAL hop always serves — the real tokenizer must get the
 * last word, never the heuristic estimator), the model's input budget is
 * known, and the conservative routing estimate exceeds it. An unknown
 * limit always serves.
 */
export const shouldSkipHopForContext = (params: {
  readonly estimatedTokens: number;
  readonly inputTokenLimit: number | null;
  readonly finalHop: boolean;
}): boolean =>
  !params.finalHop &&
  params.inputTokenLimit !== null &&
  params.estimatedTokens >
    params.inputTokenLimit * CONTEXT_SKIP_CONFIDENCE_FACTOR;
