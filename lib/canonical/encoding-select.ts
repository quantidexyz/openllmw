/**
 * The token-encoding ruler behind the estimators in `./token-estimate.ts`.
 *
 * A token count is only defined relative to ONE model's vocabulary, so the
 * estimators need to know which family they're counting for. `wire` can't
 * depend on the catalog (it's `schema`-only), so callers resolve their hop's
 * provider to one of these coarse families and pass it in; the map from
 * provider → family lives with the catalog in `packages/api`.
 *
 * The ruler itself is our own count-only BPE core over vendored vocab data —
 * see `../token-ruler`. It replaced the `ai-tokenizer` dependency: same exact
 * counts, but a ~15× smaller resident footprint (o200k: +47 MB heap / +42 MB
 * external → +0.1 MB / +4.8 MB) and no native/WASM glue, so it compiles into
 * the Bun daemon binary unchanged. This module is a thin family ↔ surface
 * adapter over that core; the heavy vocab is imported lazily inside the ruler,
 * so an estimate that never runs never pays the parse cost.
 */
import type { TRulerFamily } from "../token-ruler";
import { __resetRulersForTest, getRuler, peekRuler } from "../token-ruler";

/**
 * The coarse tokenizer families we select between. NOT provider ids — several
 * providers share a ruler:
 *   - `claude`  — Anthropic's reverse-engineered BPE (Claude 3/4.x).
 *   - `o200k`   — OpenAI o200k_base; also the best local stand-in for the
 *                 tiktoken-derived vocabs (Qwen / Kimi / DeepSeek / GLM), which
 *                 land within a few %. The DEFAULT when a caller has no better
 *                 information.
 *
 * This is exactly the ruler's {@link TRulerFamily}; the alias keeps the
 * estimator-facing name stable.
 */
export type TTokenEncoding = TRulerFamily;

export const DEFAULT_ENCODING: TTokenEncoding = "o200k";

/**
 * Pick the ruler family for a request surface. `messages` is Anthropic-wire, so
 * the Claude ruler is the accurate one; `chat_completions` and `responses` are
 * OpenAI-family (o200k also being the best local stand-in for the tiktoken-
 * derived Qwen/Kimi/DeepSeek vocabs). Shared by the cloud handler and the daemon
 * so the same body is measured with the same ruler on both paths.
 */
export const encodingForSurface = (
  surface: "messages" | "chat_completions" | "responses",
): TTokenEncoding => (surface === "messages" ? "claude" : "o200k");

/**
 * A minimal counter — just the `count(text)` we need — so the rest of the
 * module doesn't couple to the ruler's class shape.
 */
export type TTokenCounter = {
  readonly count: (text: string) => number;
};

/**
 * Resolve (and cache) the counter for a family. Async because the vocab data is
 * imported lazily — the first call per family builds a ~65k–200k-entry lookup;
 * every call after is a cache hit. Concurrent first-calls are deduped inside the
 * ruler.
 */
export const getTokenCounter = (
  encoding: TTokenEncoding,
): Promise<TTokenCounter> => getRuler(encoding);

/**
 * Synchronously return an already-loaded counter, or `null` if the family
 * hasn't been warmed yet. Lets the (synchronous) estimators use the real ruler
 * when it's hot and fall back to the char heuristic on a cold isolate, without
 * forcing every caller onto an async signature. Pair with `getTokenCounter` at
 * a warmup point (e.g. request entry) to make the hot path the common one.
 */
export const peekTokenCounter = (
  encoding: TTokenEncoding,
): TTokenCounter | null => peekRuler(encoding);

/**
 * TEST-ONLY: drop the per-isolate warm-counter cache so the estimators fall back
 * to `chars/4` again. The counter cache is module-global (in the ruler), so a
 * test that warms a family (via {@link getTokenCounter}) would otherwise leak
 * the warm ruler into every later suite in the same process. Call in `afterAll`.
 */
export const __resetTokenCountersForTest = (): void => {
  __resetRulersForTest();
};
