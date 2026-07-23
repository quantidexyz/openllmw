/**
 * The token-encoding ruler behind the estimators in `./token-estimate.ts`.
 *
 * A token count is only defined relative to ONE model's vocabulary, so the
 * estimators need to know which family they're counting for. `wire` can't
 * depend on the catalog (it's `schema`-only), so callers resolve their hop's
 * provider to one of these coarse families and pass it in; the map from
 * provider → family lives with the catalog in `packages/api`.
 *
 * Encodings come from `ai-tokenizer` (pure-JS, no WASM — runs unchanged in the
 * Bun-compiled daemon binary, on Vercel, and in the browser). Each encoding
 * file is 2–8 MB uncompressed, so:
 *   - import each submodule SELECTIVELY (`ai-tokenizer/encoding/<name>`), never
 *     the `./encoding` barrel (the barrel pulls every encoding and 4×'s the
 *     bundle — measured 8.1 MB vs 1.9 MB, see the spike FINDINGS);
 *   - load it LAZILY and cache the constructed tokenizer once per isolate, so an
 *     estimate that never runs never pays the parse cost.
 */
import { Tokenizer } from "ai-tokenizer";

/**
 * The coarse tokenizer families we select between. NOT provider ids — several
 * providers share a ruler:
 *   - `claude`  — Anthropic's reverse-engineered BPE (Claude 3/4.x).
 *   - `o200k`   — OpenAI o200k_base; also the best local stand-in for the
 *                 tiktoken-derived vocabs (Qwen / Kimi / DeepSeek / GLM), which
 *                 land within a few %. The DEFAULT when a caller has no better
 *                 information.
 */
export type TTokenEncoding = "claude" | "o200k";

export const DEFAULT_ENCODING: TTokenEncoding = "o200k";

/**
 * A minimal counter — just the `count(text)` we need — so the rest of the
 * module doesn't couple to `ai-tokenizer`'s class shape.
 */
export type TTokenCounter = {
  readonly count: (text: string) => number;
};

// Per-isolate cache of constructed tokenizers, keyed by family. The dynamic
// import of the (heavy) encoding data happens once per family, on first use.
const counters = new Map<TTokenEncoding, TTokenCounter>();
const loading = new Map<TTokenEncoding, Promise<TTokenCounter>>();

const loadEncoding = async (
  encoding: TTokenEncoding,
): Promise<TTokenCounter> => {
  const mod =
    encoding === "claude"
      ? await import("ai-tokenizer/encoding/claude")
      : await import("ai-tokenizer/encoding/o200k_base");
  return new Tokenizer(mod);
};

/**
 * Resolve (and cache) the counter for a family. Async because the encoding data
 * is imported lazily — the first call per family loads a 2–8 MB module, every
 * call after is a Map hit. The in-flight `loading` promise dedupes concurrent
 * first-calls so two requests racing on a cold family don't both parse it.
 */
export const getTokenCounter = async (
  encoding: TTokenEncoding,
): Promise<TTokenCounter> => {
  const cached = counters.get(encoding);
  if (cached !== undefined) return cached;
  const inflight = loading.get(encoding);
  if (inflight !== undefined) return inflight;
  const p = loadEncoding(encoding)
    .then((c) => {
      counters.set(encoding, c);
      return c;
    })
    // Clear the in-flight entry on BOTH fulfilment and rejection — otherwise a
    // failed dynamic import (e.g. a transient module-load error) leaves a
    // rejected promise cached here forever and every later call re-throws it
    // instead of retrying.
    .finally(() => {
      loading.delete(encoding);
    });
  loading.set(encoding, p);
  return p;
};

/**
 * Synchronously return an already-loaded counter, or `null` if the family
 * hasn't been warmed yet. Lets the (synchronous) estimators use the real ruler
 * when it's hot and fall back to the char heuristic on a cold isolate, without
 * forcing every caller onto an async signature. Pair with `getTokenCounter` at
 * a warmup point (e.g. request entry) to make the hot path the common one.
 */
export const peekTokenCounter = (
  encoding: TTokenEncoding,
): TTokenCounter | null => counters.get(encoding) ?? null;
