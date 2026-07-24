/**
 * o200k_base vocab: the rank table (`./o200k-ranks`) paired with its
 * pre-tokenisation regex and special tokens. This is OpenAI's o200k_base, and
 * the best local stand-in for the tiktoken-derived Qwen/Kimi/DeepSeek/GLM
 * vocabs (they land within a few %).
 *
 * The split pattern + special-token ranks are copied from gpt-tokenizer's
 * `encodingParams/o200k_base` (MIT, (c) 2023-2024 Bazyli Brzoska).
 */
import type { TBpeVocab, TRawBytePairRanks } from "../bpe-core";
import o200kRanks from "./o200k-ranks";

const O200K_TOKEN_SPLIT_REGEX =
  /[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'(?:[sS]|[dD]|[mM]|[tT]|[lL][lL]|[vV][eE]|[rR][eE]))?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'(?:[sS]|[dD]|[mM]|[tT]|[lL][lL]|[vV][eE]|[rR][eE]))?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

export const o200kVocab: TBpeVocab = {
  bytePairRankDecoder: o200kRanks as TRawBytePairRanks,
  tokenSplitRegex: O200K_TOKEN_SPLIT_REGEX,
};
