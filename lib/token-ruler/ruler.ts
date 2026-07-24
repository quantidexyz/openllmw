/**
 * The token ruler: one count-only BPE core ({@link BpeCounter}) over two
 * vocabs, lazily constructed and cached per family. This is what
 * `../canonical/encoding-select.ts` warms and peeks; callers there resolve a
 * request surface to a family and never touch this module directly.
 *
 * Vocab data is imported LAZILY (dynamic `import()`), so a family that is never
 * counted never pays to build its ~65k–200k-entry lookup. Construction is the
 * only cost (~160 ms for o200k, less for Claude); every `count` after is a
 * warm, allocation-light merge.
 */
import type { BpeCounter } from "./bpe-core";

/**
 * The coarse tokenizer families the ruler counts for. NOT provider ids — many
 * providers share a ruler (see `../canonical/encoding-select.ts`).
 */
export type TRulerFamily = "claude" | "o200k";

const counters = new Map<TRulerFamily, BpeCounter>();
const loading = new Map<TRulerFamily, Promise<BpeCounter>>();

const build = async (family: TRulerFamily): Promise<BpeCounter> => {
  const { BpeCounter: Counter } = await import("./bpe-core");
  const vocab =
    family === "claude"
      ? (await import("./vocab/claude")).claudeVocab
      : (await import("./vocab/o200k")).o200kVocab;
  return new Counter(vocab);
};

/**
 * Resolve (and cache) the counter for a family, building it on first use. The
 * in-flight promise dedupes concurrent first-calls; the `loading` entry is
 * cleared on both fulfilment and rejection so a transient import failure can be
 * retried rather than caching a rejected promise forever.
 */
export const getRuler = (family: TRulerFamily): Promise<BpeCounter> => {
  const cached = counters.get(family);
  if (cached !== undefined) return Promise.resolve(cached);
  const inflight = loading.get(family);
  if (inflight !== undefined) return inflight;
  const p = build(family)
    .then((counter) => {
      counters.set(family, counter);
      return counter;
    })
    .finally(() => {
      loading.delete(family);
    });
  loading.set(family, p);
  return p;
};

/** Synchronously return an already-built counter, or `null` if not warmed. */
export const peekRuler = (family: TRulerFamily): BpeCounter | null =>
  counters.get(family) ?? null;

/** TEST-ONLY: drop the per-isolate counter cache. */
export const __resetRulersForTest = (): void => {
  counters.clear();
  loading.clear();
};
