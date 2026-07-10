import type { TAnthropicRequest } from "@quantidexyz/openllmp";

/**
 * Anthropic `anthropic-beta` header policy.
 *
 * We forward whatever beta tokens the inbound client sent, with one
 * tiny exception: a denylist of tokens we know cause harm (auth path
 * mismatches, schema conflicts, …). Anything else passes through
 * verbatim.
 *
 * Why deny-list, not allow-list:
 *
 *   Anthropic ships new betas on every Claude Code release. The
 *   previous allow-list approach was mirrored from LiteLLM's
 *   `anthropic_beta_headers_config.json`, but LiteLLM lags Claude
 *   Code by weeks-to-months. Every time Anthropic shipped a new beta
 *   (`mid-conversation-system-2026-04-07` was the smoking gun for the
 *   Opus 4.8 "cuts off mid-task" symptom we hunted in the
 *   `claude-harness` diagnostic — opus 4.8 needs that beta to handle
 *   the mid-conversation system updates Claude Code v2.1.157+ sends),
 *   we silently dropped it for every passthrough request until a
 *   human noticed and PR'd it in. That's a regression treadmill:
 *   tracker-bug-fix-rinse-repeat. The cost of forwarding an unknown
 *   beta is at worst a 400 from Anthropic ("unknown beta"), which the
 *   fallback chain already treats as a transient error and walks past;
 *   the cost of dropping a real beta is silent quality degradation.
 *
 *   Denylist + passthrough is forward-compatible: any beta shipped by
 *   any future Claude Code build (or any other Anthropic-format
 *   client) flows through unchanged. The denylist only carries tokens
 *   we have a concrete reason to block — historical and concrete, not
 *   speculative.
 *
 * Today the denylist is empty. If a beta starts breaking the proxy,
 * add it here with a comment naming the failure mode.
 */
const ANTHROPIC_BLOCKED_BETAS: ReadonlySet<string> = new Set<string>();

/** Beta required for the subscription-OAuth (`sk-ant-oat01-…`) path. */
export const ANTHROPIC_OAUTH_BETA = "oauth-2025-04-20";

/**
 * Anthropic SUBSCRIPTION / OAuth credential matcher — the `claude setup-token`
 * and "Sign in with Claude" tokens. These ride `Authorization: Bearer` + the
 * oauth beta header, NOT `x-api-key` (which is for Console keys `sk-ant-api03-`).
 * Anthropic has shipped TWO prefixes over time: the original `sk-ant-oat01-` and
 * the current `sk-ant-at01-` (Claude CLI ≥ 2.1.x) — match BOTH, never a Console
 * key. Single source of truth for the daemon (setup-token capture + validation)
 * and the cloud (auth-header split, terms gate, count-tokens).
 */
export const ANTHROPIC_OAUTH_TOKEN_RE = /sk-ant-o?at01-[A-Za-z0-9_-]+/;
export const isAnthropicOAuthToken = (token: string): boolean =>
  /^sk-ant-o?at01-/.test(token);

const splitBetaHeader = (raw: string | null): string[] => {
  if (raw === null) return [];
  return raw
    .split(",")
    .map((t) => t.trim())
    .filter((t) => t.length > 0);
};

/** Beta required for Files-API (`file_id`) sources on image/document blocks. */
export const ANTHROPIC_FILES_API_BETA = "files-api-2025-04-14";

/**
 * True when any content block references an uploaded file (`source:
 * {type:"file"}` on an image/document, or a `container_upload` block) —
 * Anthropic only honours those under the Files API beta.
 */
const usesFilesApi = (req: TAnthropicRequest): boolean => {
  for (const m of req.messages) {
    if (typeof m.content === "string") continue;
    for (const block of m.content) {
      if (
        (block.type === "image" || block.type === "document") &&
        block.source.type === "file"
      ) {
        return true;
      }
      if (block.type === "container_upload") return true;
    }
  }
  return false;
};

/**
 * Derive the betas implied by the request body itself, independent of
 * what the client sent. Our `TAnthropicRequest` schema is intentionally
 * minimal, so the safe inferences today are the native server-side
 * `web_search`/`web_fetch` tools and Files-API (`file_id`) content
 * sources (Anthropic requires the matching beta for those to be
 * honoured). Kept as a single chokepoint so future schema fields
 * (context_management, output_format, speed) extend here rather than
 * at every call site — the LiteLLM equivalent is
 * `anthropic_beta_headers_manager.get_anthropic_beta_headers`.
 */
const derivedBetasFor = (req: TAnthropicRequest): string[] => {
  const out: string[] = [];
  const tools = req.tools ?? [];
  for (const tool of tools) {
    const t = tool.type ?? "";
    if (t.startsWith("web_search")) out.push("web-search-2025-03-05");
    if (t.startsWith("web_fetch")) out.push("web-fetch-2025-09-10");
  }
  if (usesFilesApi(req)) out.push(ANTHROPIC_FILES_API_BETA);
  return out;
};

export type TBetaHeaderInput = {
  /** Raw inbound `anthropic-beta` header value (comma-separated) or null. */
  readonly inboundBeta: string | null;
  /** The Anthropic request being forwarded (for body-derived betas). */
  readonly request: TAnthropicRequest;
  /** True when the credential is a subscription OAuth token. */
  readonly isOAuth: boolean;
  /**
   * Extra betas the caller derived from a NON-Anthropic-shaped inbound
   * body (cross-wire hops, where `request` is an empty stand-in) —
   * e.g. the Files-API beta when a canonical `file_id` part will be
   * encoded to an Anthropic file source.
   */
  readonly extraBetas?: ReadonlyArray<string>;
};

/**
 * Compute the final, de-duplicated `anthropic-beta` token list to send
 * upstream. Returns `undefined` when there is nothing to send so the
 * caller can omit the header entirely.
 *
 * Order of precedence (first occurrence wins for dedupe, order is
 * otherwise stable): inbound betas (deny-list filtered) → body-derived
 * betas → the OAuth beta when authing via a subscription token.
 *
 * Tokens the client sends that aren't in `ANTHROPIC_BLOCKED_BETAS`
 * forward verbatim — see the module header for the rationale.
 */
export const deriveAnthropicBetaHeader = (
  input: TBetaHeaderInput,
): string | undefined => {
  const seen = new Set<string>();
  const out: string[] = [];
  const push = (token: string): void => {
    if (token.length === 0 || seen.has(token)) return;
    if (ANTHROPIC_BLOCKED_BETAS.has(token)) return;
    seen.add(token);
    out.push(token);
  };

  for (const token of splitBetaHeader(input.inboundBeta)) push(token);
  for (const token of derivedBetasFor(input.request)) push(token);
  for (const token of input.extraBetas ?? []) push(token);
  if (input.isOAuth) push(ANTHROPIC_OAUTH_BETA);

  return out.length > 0 ? out.join(",") : undefined;
};
