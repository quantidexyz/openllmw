/**
 * Count-only byte-pair-encoding core.
 *
 * Vendored and reduced from `gpt-tokenizer` (Bazyli Brzoska, MIT) — see
 * NOTICE below. We keep ONLY the token-COUNT path: the regex pre-split, the
 * string/binary rank lookups, and the `bytePairMerge` that the count walks.
 * Everything `gpt-tokenizer` carries for the full round-trip — the `decoder`,
 * `encode`/`decode`, the LRU merge cache, chat/cost helpers — is dropped,
 * because the gateway never needs the token ids, only how many there are.
 *
 * Why vendor instead of `import`: `ai-tokenizer` (the prior dep) materialised
 * a ~200k-key JS object per encoding (+47 MB heap / +42 MB external for o200k,
 * ~320 ms cold parse). This core over the same vocab, stored as a rank-indexed
 * array, measures +0.1 MB heap / +4.8 MB external / ~160 ms — a ~15× resident
 * cut for byte-identical counts — and being plain `.ts` it compiles straight
 * into the Bun daemon binary with no WASM / native-addon glue. One core serves
 * BOTH the o200k and the (reverse-engineered) Claude vocab; only the rank table
 * differs. See `./vocab/*`.
 *
 * NOTICE — portions derived from gpt-tokenizer:
 *   MIT License · Copyright (c) 2023-2024 Bazyli Brzoska
 *   https://github.com/niieani/gpt-tokenizer
 */

// The BPE merge + byte-comparison inner loops index arrays whose bounds are
// guaranteed by the surrounding loop conditions; non-null assertions there are
// deliberate and hot-path (avoiding a redundant undefined-check per byte).
// biome-ignore-all lint/style/noNonNullAssertion: bounds guaranteed by loop conditions in hot BPE paths

/**
 * The vocab as a rank-indexed array: `ranks[i]` is the token whose BPE rank is
 * `i`. UTF-8-safe tokens are stored as a `string`; tokens that aren't valid
 * UTF-8 are stored as their raw `number[]` bytes. Holes are allowed (an unused
 * rank). This is exactly `gpt-tokenizer`'s on-disk shape, so its generated
 * `bpeRanks/*` data drops in unchanged and our Claude table is generated to
 * match.
 */
export type TRawBytePairRanks = readonly (string | readonly number[] | null)[];

export type TBpeVocab = {
  /** Rank-indexed token table (see {@link TRawBytePairRanks}). */
  readonly bytePairRankDecoder: TRawBytePairRanks;
  /** The encoding's pre-tokenisation regex (global + unicode flags). */
  readonly tokenSplitRegex: RegExp;
};

const textEncoder = new TextEncoder();
const utf8Decoder = new TextDecoder("utf8", { fatal: false });

/**
 * True when `bytes` is a well-formed UTF-8 sequence — so we know whether a
 * byte pair can be looked up in the string map or must go to the binary map.
 */
const isValidUtf8 = (bytes: Uint8Array): boolean => {
  let i = 0;
  while (i < bytes.length) {
    const byte1 = bytes[i]!;
    let numBytes = 0;
    let codePoint = 0;
    if (byte1 <= 0x7f) {
      numBytes = 1;
      codePoint = byte1;
    } else if ((byte1 & 0xe0) === 0xc0) {
      numBytes = 2;
      codePoint = byte1 & 0x1f;
      if (byte1 <= 0xc1) return false; // overlong
    } else if ((byte1 & 0xf0) === 0xe0) {
      numBytes = 3;
      codePoint = byte1 & 0x0f;
    } else if ((byte1 & 0xf8) === 0xf0) {
      numBytes = 4;
      codePoint = byte1 & 0x07;
      if (byte1 > 0xf4) return false; // > U+10FFFF
    } else {
      return false;
    }
    if (i + numBytes > bytes.length) return false;
    for (let j = 1; j < numBytes; j++) {
      const byte = bytes[i + j];
      if (byte === undefined || (byte & 0xc0) !== 0x80) return false;
      codePoint = (codePoint << 6) | (byte & 0x3f);
    }
    if (numBytes === 2 && codePoint < 0x80) return false;
    if (numBytes === 3 && codePoint < 0x800) return false;
    if (numBytes === 4 && codePoint < 0x10000) return false;
    if (codePoint >= 0xd800 && codePoint <= 0xdfff) return false; // surrogate
    if (codePoint > 0x10ffff) return false;
    i += numBytes;
  }
  return true;
};

const tryConvertToString = (arr: Uint8Array): string | undefined =>
  isValidUtf8(arr) ? utf8Decoder.decode(arr) : undefined;

const compareUint8Arrays = (a: Uint8Array, b: Uint8Array): number => {
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    if (a[i] !== b[i]) return a[i]! - b[i]!;
  }
  return a.length - b.length;
};

/**
 * A prepared counter over one vocab. Construction builds the reverse lookups
 * once (string→rank map + sorted binary table); `count` is then allocation-
 * light and never touches a decoder or cache.
 */
/**
 * Cap on the per-piece merge-count cache. Real prompts are dominated by a small
 * recurring vocabulary of pieces, so a few thousand entries captures nearly all
 * the reuse; the cap is what keeps a pathological input (every piece distinct)
 * from growing the map without bound. On overflow we clear rather than evict
 * one-by-one — no LRU bookkeeping on the hot path, and the cache simply refills.
 */
const MERGE_COUNT_CACHE_MAX = 8192;

export class BpeCounter {
  private readonly stringRankEncoder = new Map<string, number>();
  private readonly binarySortedEncoder: [Uint8Array, number][] = [];
  private readonly tokenSplitRegex: RegExp;
  /**
   * piece → its merged token count. The vendoring note above records that
   * `gpt-tokenizer`'s LRU merge cache was dropped; that turned out to be the
   * single largest cost in this core, because `count` re-ran the full merge for
   * EVERY occurrence of a piece. Prompts repeat pieces heavily (indentation,
   * common words, punctuation runs), so counting a realistic body did the same
   * merge thousands of times. A piece's count is a pure function of the piece
   * and the vocab, so caching it is exact — the counts are unchanged.
   */
  private readonly mergeCountCache = new Map<string, number>();

  constructor(vocab: TBpeVocab) {
    // forEach skips array holes (unused ranks); we also skip explicit
    // null/undefined, which the generated Claude table uses for unused ranks
    // (JSON can't serialise a hole). Matches tiktoken semantics.
    vocab.bytePairRankDecoder.forEach((value, rank) => {
      if (value == null) return;
      if (typeof value === "string") {
        this.stringRankEncoder.set(value, rank);
        return;
      }
      this.binarySortedEncoder.push([new Uint8Array(value), rank]);
    });
    this.binarySortedEncoder.sort((a, b) => compareUint8Arrays(a[0], b[0]));

    this.tokenSplitRegex = vocab.tokenSplitRegex;
  }

  /**
   * Number of tokens `text` encodes to under this vocab.
   *
   * All input is treated as LITERAL text — special-token strings (e.g.
   * `<|endoftext|>`) are counted as their constituent byte tokens, never as a
   * single control token. That is the correct behaviour for a request-size
   * estimator: user content is never actually a control token, and tiktoken
   * itself refuses to encode special strings from untrusted text unless
   * explicitly allowed. It also keeps the count an upper-bound-safe estimate.
   */
  count(text: string): number {
    let tokensCount = 0;
    for (const [match] of text.matchAll(this.tokenSplitRegex)) {
      const direct = this.stringRankEncoder.get(match);
      if (direct !== undefined) {
        tokensCount++;
        continue;
      }
      const cached = this.mergeCountCache.get(match);
      if (cached !== undefined) {
        tokensCount += cached;
        continue;
      }
      const merged = this.bytePairMergeCount(textEncoder.encode(match));
      if (this.mergeCountCache.size >= MERGE_COUNT_CACHE_MAX) {
        this.mergeCountCache.clear();
      }
      this.mergeCountCache.set(match, merged);
      tokensCount += merged;
    }
    return tokensCount;
  }

  private rankOfBytes(key: Uint8Array): number | undefined {
    const asString = tryConvertToString(key);
    if (asString !== undefined) return this.stringRankEncoder.get(asString);
    const index = this.binarySearch(key);
    return index === -1 ? undefined : this.binarySortedEncoder[index]![1];
  }

  private binarySearch(key: Uint8Array): number {
    let low = 0;
    let high = this.binarySortedEncoder.length - 1;
    while (low <= high) {
      const mid = (low + high) >>> 1;
      const cmp = compareUint8Arrays(this.binarySortedEncoder[mid]![0], key);
      if (cmp === 0) return mid;
      if (cmp < 0) low = mid + 1;
      else high = mid - 1;
    }
    return -1;
  }

  /**
   * The BPE merge loop, returning only the final partition COUNT (never the
   * token ids). Identical merge order to tiktoken/`gpt-tokenizer`, so the count
   * is exact — we just skip building the output id array.
   */
  private bytePairMergeCount(piece: Uint8Array): number {
    if (piece.length === 1) return 1;

    const starts: number[] = [];
    const ranks: number[] = [];
    // pairStart/pairEnd default to the `starts`-relative span, but callers can
    // override them — the init loop MUST pass explicit byte indices because
    // `starts` isn't fully built yet (reading starts[i+2] there is undefined).
    const rankAt = (
      i: number,
      pairStart = starts[i],
      pairEnd = starts[i + 2],
    ): number => {
      if (pairEnd === undefined) return Number.POSITIVE_INFINITY;
      const rank = this.rankOfBytes(piece.subarray(pairStart, pairEnd));
      return rank ?? Number.POSITIVE_INFINITY;
    };

    for (let i = 0; i <= piece.length; i++) {
      starts.push(i);
      ranks.push(
        i < piece.length - 1 ? rankAt(i, i, i + 2) : Number.POSITIVE_INFINITY,
      );
    }

    while (starts.length > 1) {
      let lowestRank = Number.POSITIVE_INFINITY;
      let lowestIndex = -1;
      for (let i = 0; i < ranks.length - 1; i++) {
        if (ranks[i]! < lowestRank) {
          lowestRank = ranks[i]!;
          lowestIndex = i;
        }
      }
      if (lowestRank === Number.POSITIVE_INFINITY || lowestIndex === -1) break;
      starts.splice(lowestIndex + 1, 1);
      ranks.splice(lowestIndex, 1);
      ranks[lowestIndex] = rankAt(lowestIndex);
      if (lowestIndex > 0) ranks[lowestIndex - 1] = rankAt(lowestIndex - 1);
    }

    // One token per surviving partition.
    return starts.length - 1;
  }
}
