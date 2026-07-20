import type {
  TAnthropicRequest,
  TChatCompletionRequest,
  TResponsesRequest,
} from "@openllmsh/protocol";
import { fromAnthropicMessagesRequest } from "../adapters/messages/request";
import { fromResponsesRequest } from "../adapters/responses";
import { normaliseAdaptiveThinking } from "./anthropic/adaptive-thinking";
import {
  ANTHROPIC_FILES_API_BETA,
  ANTHROPIC_OAUTH_BETA,
  deriveAnthropicBetaHeader,
} from "./anthropic/beta-headers";
import { toAnthropicRequest } from "./anthropic/request";
import { deriveChatGptSessionId, toChatGptRequest } from "./chatgpt/request";
import {
  ensureClaudeCodeSystemPreamble,
  injectGatewayPromptPrefix,
} from "./prompt-prefix";

// Re-exported for existing consumers — the definitions (and every other
// system-prompt injection) live in `./prompt-prefix`.
export {
  CLAUDE_CODE_SYSTEM_PREAMBLE,
  ensureClaudeCodeSystemPreamble,
} from "./prompt-prefix";

/**
 * The SINGLE recipe for preparing an upstream provider request from an inbound
 * one — body + wire-derived headers — for every `(clientWire × upstreamWire)`
 * pairing. The cloud runner (`@openllm/core`) and the coreless daemon walker
 * (`@openllmsh/daemon`) BOTH call this; neither re-derives the recipe.
 *
 * This exists because the pure transforms were single-sourced in `@openllmsh/wire`
 * but their COMPOSITION was open-coded in core's runner AND the daemon's
 * walker — which can't share (the daemon is core-free) — and drifted, dropping
 * the client's `anthropic-beta` and skipping `normaliseAdaptiveThinking`. See
 * `docs/proposals/unified-upstream-request-builder.md`.
 *
 * Response decode (upstream → canonical) + encode (canonical → client) are
 * already shared wire concerns; this module is the REQUEST side.
 */

export type TClientSurface = "messages" | "chat_completions" | "responses";
export type TUpstreamWire = "anthropic" | "chatgpt" | "openai";

const DEFAULT_ANTHROPIC_VERSION = "2023-06-01";

/** The client's upstream wire, derived from the surface it hit. `responses`
 *  rides the OpenAI family for the passthrough decision (it never passes
 *  through — its body is Responses-shaped — so it always transforms). */
export const clientWireOf = (surface: TClientSurface): TUpstreamWire =>
  surface === "messages" ? "anthropic" : "openai";

/** Inbound (client-shaped) body → canonical ChatCompletion, per surface. */
export const canonicalFromInbound = (
  surface: TClientSurface,
  rawBody: unknown,
): TChatCompletionRequest =>
  surface === "messages"
    ? fromAnthropicMessagesRequest(rawBody as TAnthropicRequest)
    : surface === "responses"
      ? fromResponsesRequest(rawBody as TResponsesRequest)
      : (rawBody as TChatCompletionRequest);

/**
 * Canonical request → an UPSTREAM body, per upstream wire. The cross-wire half
 * of the recipe: `toChatGptRequest` / `toAnthropicRequest` / OpenAI-identity.
 * Exported because the daemon's web_search agentic loop rebuilds the upstream
 * body each round from an ALREADY-canonical request (not the inbound body).
 */
export const canonicalToUpstreamBody = (
  upstreamWire: TUpstreamWire,
  canonical: TChatCompletionRequest,
  providerModelId: string,
  stream: boolean,
  // Whether the chatgpt (Responses) encode injects the Codex preamble. Undefined
  // → inject (Codex default); `false` suppresses it for non-Codex Responses-wire
  // providers (xAI Grok). No effect for the anthropic / openai wires.
  codexInstructions?: boolean,
): unknown => {
  if (upstreamWire === "chatgpt") {
    return toChatGptRequest(canonical, { providerModelId, codexInstructions });
  }
  const options = { providerModelId };
  if (upstreamWire === "anthropic") {
    return { ...toAnthropicRequest(canonical, options), stream };
  }
  // openai-identity passthrough: forward canonical verbatim, but DROP the
  // `responses_tools` carrier — it's a Codex/Responses-only field that
  // chatgpt re-emits; openai-compatible upstreams 400 on the unknown key.
  const { responses_tools: _responsesTools, ...openai } = canonical;
  const streamOptions =
    stream === true
      ? { ...openai.stream_options, include_usage: true }
      : openai.stream_options;
  return {
    ...openai,
    ...(streamOptions !== undefined ? { stream_options: streamOptions } : {}),
    model: providerModelId,
    stream,
  };
};

/** Inbound (client-shaped) body → an upstream body, per `(surface,
 *  upstreamWire)`. Passthrough (+ adaptive-thinking normalise for anthropic)
 *  or cross-wire via canonical. Exported so the daemon's single-shot serve and
 *  the cloud runner share the exact body recipe. */
export const buildUpstreamBody = (
  surface: TClientSurface,
  upstreamWire: TUpstreamWire,
  rawBody: unknown,
  providerModelId: string,
  // `undefined` → PRESERVE the body's own stream flag (the cloud passthrough
  // forwards verbatim); a boolean pins it (the daemon, off the 307's intent).
  stream: boolean | undefined,
  // Codex-preamble injection for the chatgpt wire (see canonicalToUpstreamBody).
  codexInstructions?: boolean,
  // Inject the Claude Code system preamble — the OAuth Anthropic upstream
  // (handrolled claude_code) requires it as the first system block; see
  // {@link ensureClaudeCodeSystemPreamble}. Set by buildUpstreamRequest from
  // `isOAuth && upstreamWire === "anthropic"`.
  oauthAnthropicPreamble?: boolean,
): unknown => {
  // Layered finish — ALL system-prompt injections live in
  // `./prompt-prefix`: (1) vendor-required Claude Code preamble (must
  // stay the FIRST system block), then (2) the gateway prompt prefix
  // (`GATEWAY_PROMPT_PREFIX` from `@openllmsh/protocol`), injected into
  // EVERY upstream chat body, slotted after the preamble when present.
  const withClaudePreamble = (upstreamBody: unknown): unknown =>
    injectGatewayPromptPrefix(
      upstreamWire,
      oauthAnthropicPreamble === true && upstreamWire === "anthropic"
        ? ensureClaudeCodeSystemPreamble(upstreamBody)
        : upstreamBody,
    );
  // Passthrough: same wire in + out (NEVER for `responses` — its body is
  // Responses-shaped). Only the model id + stream flag are pinned. The
  // Anthropic passthrough additionally normalises adaptive-thinking knobs
  // (`thinking:adaptive` / `output_config.effort` / top-level `effort`) for
  // the RESOLVED model — they 400 on haiku/claude-3.
  if (upstreamWire === clientWireOf(surface) && surface !== "responses") {
    // Pin the concrete model id + stream flag. The daemon always supplies a
    // resolved `providerModelId` (off the 307); the cloud passthrough (whose
    // body already carries the right model) passes the body's own model — and
    // an empty id means "preserve the body's model", so it stays a true
    // passthrough.
    const raw = rawBody as Record<string, unknown>;
    const effectiveStream = stream ?? (raw.stream === true);
    const pinned = {
      ...raw,
      ...(upstreamWire === "openai" && effectiveStream
        ? {
            stream_options: {
              ...(raw.stream_options as Record<string, unknown> | undefined),
              include_usage: true,
            },
          }
        : {}),
      ...(providerModelId.length > 0 ? { model: providerModelId } : {}),
      ...(stream !== undefined ? { stream } : {}),
    };
    return withClaudePreamble(
      upstreamWire === "anthropic" ? normaliseAdaptiveThinking(pinned) : pinned,
    );
  }
  // Cross-wire: route through canonical, then encode to the upstream's wire.
  return withClaudePreamble(
    canonicalToUpstreamBody(
      upstreamWire,
      canonicalFromInbound(surface, rawBody),
      providerModelId,
      stream ?? false,
      codexInstructions,
    ),
  );
};

/**
 * True when a cross-wire (canonical-shaped) body carries a `file` part
 * referencing an uploaded file — `toAnthropicRequest` encodes that to a
 * Files-API document source, which Anthropic only honours under the
 * Files API beta.
 */
const canonicalUsesFileIds = (canonical: TChatCompletionRequest): boolean =>
  canonical.messages.some(
    (m) =>
      typeof m.content !== "string" &&
      m.content != null &&
      m.content.some((p) => p.type === "file" && p.file.file_id !== undefined),
  );

/** Wire-derived headers (layered OVER the caller's auth/identity). Only the
 *  Anthropic upstream contributes here: version + the merged `anthropic-beta`
 *  (OAuth beta + the client's inbound betas + body-derived betas). */
const wireHeaders = (
  surface: TClientSurface,
  upstreamWire: TUpstreamWire,
  rawBody: unknown,
  inboundBeta: string | null,
  isOAuth: boolean,
  apiVersion: string | undefined,
): Record<string, string> => {
  if (upstreamWire !== "anthropic")
    return { "content-type": "application/json" };
  // Body-derived betas (web_search/web_fetch/files-api) only read the
  // Anthropic-shaped request; for a non-messages (cross-wire) request there's
  // no Anthropic-shaped request, so an empty stand-in yields no false derived
  // betas — the Files-API beta is instead derived from the canonical body's
  // `file_id` parts (which `toAnthropicRequest` encodes to file sources).
  const request =
    surface === "messages"
      ? (rawBody as TAnthropicRequest)
      : ({ model: "", messages: [], max_tokens: 0 } as TAnthropicRequest);
  const extraBetas =
    surface !== "messages" &&
    canonicalUsesFileIds(canonicalFromInbound(surface, rawBody))
      ? [ANTHROPIC_FILES_API_BETA]
      : [];
  const beta = deriveAnthropicBetaHeader({
    inboundBeta,
    request,
    isOAuth,
    extraBetas,
  });
  return {
    "anthropic-version": apiVersion ?? DEFAULT_ANTHROPIC_VERSION,
    "content-type": "application/json",
    ...(beta !== undefined
      ? { "anthropic-beta": beta }
      : isOAuth
        ? { "anthropic-beta": ANTHROPIC_OAUTH_BETA }
        : {}),
  };
};

export type TBuildUpstreamRequestInput = {
  readonly surface: TClientSurface;
  readonly upstreamWire: TUpstreamWire;
  /** The inbound body in the CLIENT's wire shape. */
  readonly rawBody: unknown;
  /** Concrete upstream model id to pin. */
  readonly providerModelId: string;
  /** The client's stream intent. */
  readonly stream: boolean;
  /**
   * Auth + identity the CALLER owns — BYOK `x-api-key` / OAuth `authorization`
   * (cloud), or the local CLI's `authorization` + vendor identity headers
   * (daemon). Wire-derived headers (anthropic-version / anthropic-beta /
   * content-type) are layered ON TOP so they win on collision.
   */
  readonly baseHeaders: Record<string, string>;
  /** The client's inbound `anthropic-beta` header (or null). */
  readonly inboundBeta?: string | null;
  /** True when the Anthropic auth is a subscription OAuth token. */
  readonly isOAuth?: boolean;
  /** Pinned `anthropic-version` (else the ground-floor default). */
  readonly apiVersion?: string;
  /**
   * chatgpt-wire only: inject the Codex preamble. Undefined → inject (Codex
   * default); `false` suppresses it for a non-Codex Responses-wire provider
   * (xAI Grok). See {@link canonicalToUpstreamBody}.
   */
  readonly codexInstructions?: boolean;
};

/**
 * The caller's auth/identity (`baseHeaders`) with the wire-derived headers
 * (anthropic-version / merged anthropic-beta / content-type) layered ON TOP so
 * they win on collision. Exported so the daemon's web_search loop (which
 * rebuilds the body per round but keeps one header set) can compute headers
 * once, independently of the body.
 */
export const buildUpstreamHeaders = (
  i: TBuildUpstreamRequestInput,
): Record<string, string> => {
  const headers: Record<string, string> = {
    ...i.baseHeaders,
    ...wireHeaders(
      i.surface,
      i.upstreamWire,
      i.rawBody,
      i.inboundBeta ?? null,
      i.isOAuth ?? false,
      i.apiVersion,
    ),
  };
  if (i.upstreamWire === "chatgpt") {
    ensureChatGptSessionAffinity(headers, i.surface, i.rawBody);
  }
  return headers;
};

/**
 * Guarantee a STABLE `session_id` on the ChatGPT Codex wire (chatgpt + grok).
 * The backend routes prompt-cache affinity by `session_id`; with none, every
 * request lands on a cold machine and caches nothing (`cached=0`), re-billing
 * the full conversation input every turn — a subscription drain for agentic
 * coding (audit 2026-07-14-codex-handrolled-quota-drain). We PRESERVE the
 * client's own session (a real Codex CLI already sends one, in either
 * `session_id` or codex-rs `session-id` form) and only SYNTHESIZE a stable
 * per-conversation one — derived from the immutable conversation prefix — when
 * the client sent none, so bare clients (chat-completions, custom agents) also
 * get cache affinity. `x-client-request-id` rides along for vendor-client
 * parity (caching keys on `session_id`, not this).
 */
const ensureChatGptSessionAffinity = (
  headers: Record<string, string>,
  surface: TClientSurface,
  rawBody: unknown,
): void => {
  const hasSession = Object.keys(headers).some((k) => {
    const lk = k.toLowerCase();
    return lk === "session_id" || lk === "session-id";
  });
  if (hasSession) return;
  const sessionId = deriveChatGptSessionId(
    canonicalFromInbound(surface, rawBody),
  );
  headers.session_id = sessionId;
  headers["x-client-request-id"] = sessionId;
};

/**
 * Prepare the `{ body, headers }` for ONE upstream call. The only place the
 * `(clientWire × upstreamWire)` request recipe lives.
 */
export const buildUpstreamRequest = (
  i: TBuildUpstreamRequestInput,
): { readonly body: unknown; readonly headers: Record<string, string> } => ({
  body: buildUpstreamBody(
    i.surface,
    i.upstreamWire,
    i.rawBody,
    i.providerModelId,
    i.stream,
    i.codexInstructions,
    (i.isOAuth ?? false) && i.upstreamWire === "anthropic",
  ),
  headers: buildUpstreamHeaders(i),
});
