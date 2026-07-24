/**
 * Claude vocab: the rank table (`./claude-ranks`) paired with its
 * pre-tokenisation regex and special tokens. The rank table is a
 * reverse-engineered approximation of Anthropic's (unpublished) tokenizer — see
 * `./claude-ranks` — so its counts are approximate and calibrated per-provider
 * by `PROVIDER_TOKEN_ESTIMATE_FACTOR`.
 *
 * The split pattern + special-token ranks come from ai-tokenizer's claude.json
 * (MIT, (c) 2025 Coder Technologies Inc.).
 */
import type { TBpeVocab, TRawBytePairRanks } from "../bpe-core";
import claudeRanks from "./claude-ranks";

const CLAUDE_TOKEN_SPLIT_REGEX =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export const claudeVocab: TBpeVocab = {
  // Unused/special ranks are `null` (JSON can't hold array holes); the core
  // skips them.
  bytePairRankDecoder: claudeRanks as TRawBytePairRanks,
  tokenSplitRegex: CLAUDE_TOKEN_SPLIT_REGEX,
};
