# `packages/wire` Architecture

> `@openllm/wire` — the **pure wire-format transforms** shared by the
> cloud proxy pipeline (`packages/core`) and the local subscription
> daemon (`packages/daemon`). Depends only on `@openllmsh/protocol`: **no
> Effect-DI, no provider registry, no `fetch`, no HTTP framework, no
> DB.** Every export is a deterministic function (or a small stateful
> stream transform) over `@openllmsh/protocol` wire types.
>
> Extracted from `packages/core` per
> [`docs/proposals/coreless-daemon-passthrough.md`](../../docs/proposals/coreless-daemon-passthrough.md)
> §4 + §7.1. Referenced from the root
> [`ARCHITECTURE.md`](../../ARCHITECTURE.md) §3.

---

## Why this package exists

The daemon must adapt a request to each hop's wire, encode the response
back, and synthesize usage — **without** linking the whole `core`
pipeline (the runner + provider graph + Effect-DI layers). Those
transforms were already pure functions inside `core`; pulling them into
a `@openllmsh/protocol`-only package lets the daemon import them directly and
lets `core`'s churn (new providers, runner fixes) stop forcing a daemon
re-ship. `core` re-imports them — the move is **zero behaviour change**,
pinned by the existing `tests/{inbound,outbound}-*` + matrix suites.

## Modules

```
wire/
  index.ts                              barrel — re-exports every module below
  adapters/messages/
    index.ts                            `messages` namespace barrel
    request.ts                          fromAnthropicMessagesRequest (Anthropic → canonical)
    response.ts                         toAnthropicMessagesResponse (canonical → Anthropic)
    streaming.ts                        chunk→Anthropic SSE events + encoder + byte stream
    reasoning-signature.ts              reasoning-item base64/JSON round-trip codec
    reasoning-from-items.ts             plain-text from litellm-shaped reasoning items
  lib/canonical/
    message.ts                          extractMessageText, parseToolArguments
    token-estimate.ts                   estimateBodyTokens (generic JSON body estimator)
  lib/streaming/
    sse.ts                              SSE encode/decode, heartbeat + deadline wrappers, tee
    accumulate.ts                       drain a chunk stream into one non-streaming response
    peek.ts                             pre-commit first-event peek (shared overflow-walk gate)
    strip-tool-calls.ts                 compaction tool-call removal predicates
    upstream-error.ts                   UpstreamStreamError + upstreamErrorFrom
  features/
    context-skip.ts                     per-hop context-skip gate (shared cloud + daemon)
  features/compaction/
    compaction-text.ts                  visible-text compaction-safety helpers
```

## Layering rules

- Depends **only** on `@openllmsh/protocol`. No `@openllm/core`, no
  `@openllm/api`/`db`/`vault`, no `effect` runtime (Layer/Context/Effect),
  no Next/Vercel. (Schema types only; `effect`'s `Schema` decode lives in
  `core/lib/streaming/event-stream.ts`, which is deliberately **not** part
  of this package.)
- Every export is a pure function or a self-contained stream transform —
  no I/O, no DI, no global state. Runs identically in the browser, on the
  Vercel proxy, and in the compiled daemon binary.
- The exact cut is the boundary that keeps `event-stream.ts` in `core`
  (it binds `TChatProviderSpec` + `effect`'s `Schema`) while everything
  upstream/downstream of it that is provider-agnostic lives here.
