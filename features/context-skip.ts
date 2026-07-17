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
 * chars/4 estimator can over-count 30–100% on repetitive content where
 * BPE compresses heavily, so a hop is only skipped (request-time) or
 * dropped (plan-time) when the estimate CLEARLY exceeds its limit —
 * borderline cases go to the real upstream tokenizer, which gets the
 * final word via the pre-commit peek walk.
 *
 * Bias note: false negatives (under-skip) degrade to "request hits
 * upstream → rejection → peek walks the chain" — a wasted round trip,
 * nothing more. False positives (over-skip) could strand a request on a
 * worse hop, so we err conservative.
 */
export const CONTEXT_SKIP_CONFIDENCE_FACTOR = 1.5;

/**
 * Should this hop be skipped for context? True only when a later hop
 * remains (the FINAL hop always serves — the real tokenizer must get the
 * last word, never the heuristic estimator), the model's input budget is
 * known, and the estimate clearly exceeds it (the shared confidence
 * factor). An unknown limit always serves.
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
