<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="./assets/openllm-light.svg">
    <img alt="OpenLLM" src="./assets/openllm.svg" width="300">
  </picture>
</p>

<p align="center"><b>openllmw</b> — the wire-format engine behind OpenLLM.</p>

<p align="center">
  <a href="./LICENSE"><img alt="License: BUSL-1.1" src="https://img.shields.io/badge/license-BUSL--1.1-blue.svg"></a>
  <img alt="source-available" src="https://img.shields.io/badge/source-available-informational.svg">
</p>

---

The pure transforms that let one request speak every dialect — the
**Anthropic ↔ OpenAI ↔ Responses** translation layer, with no DI, no
providers, and no `fetch`. Same code the hosted gateway and the local
daemon both run:

- request / response / streaming adapters across all three wire formats
- reasoning-signature codec · canonical-message + token-estimate helpers
- SSE primitives (accumulate · tool-strip · provider-decode · heartbeat)
- the single upstream-request builder (model/stream pinning, Anthropic
  adaptive-thinking + `anthropic-beta`)

Depends only on [`@quantidexyz/openllmp`](https://github.com/quantidexyz/openllmp)
+ `effect`.

## Install

```sh
bun install github:quantidexyz/openllmw#latest
```

```ts
import { toAnthropicMessagesResponse } from "@quantidexyz/openllmw/adapters/messages/response";
```

## License

**Source-available** under the [Business Source License 1.1](./LICENSE)
(© Quantide LLC) — use it freely except to run a competing hosted service;
converts to MIT on the Change Date. This is not OSI open-source.

---

> **Read-only mirror.** Regenerated from the OpenLLM monorepo each release.
> PRs welcome — ingested upstream with your authorship preserved. BUSL
> contributions require the CLA (the bot will prompt you).
