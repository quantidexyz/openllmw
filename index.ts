/**
 * `@openllmsh/wire` — pure wire-format transforms shared by the cloud
 * pipeline (`@openllm/core`) and the local daemon. Depends only on
 * `@openllmsh/protocol`: no Effect-DI, no provider registry, no `fetch`.
 *
 * These are the functions a coreless daemon needs to adapt a request
 * to a hop's wire, encode the response back, and synthesize usage —
 * carried verbatim from `core` so the matrix tests pin them unchanged.
 */
export * from "./adapters/messages/reasoning-from-items";
export * from "./adapters/messages/reasoning-signature";
export * from "./adapters/messages/request";
export * from "./adapters/messages/response";
export * from "./adapters/messages/streaming";
export * from "./features/compaction/compaction-text";
export * from "./features/context-skip";
export * from "./lib/canonical/message";
export * from "./lib/canonical/token-estimate";
export * from "./lib/encrypted-content";
export * from "./lib/refusal";
export * from "./lib/streaming/accumulate";
export * from "./lib/streaming/peek";
export * from "./lib/streaming/sse";
export * from "./lib/streaming/strip-tool-calls";
export * from "./lib/streaming/upstream-error";
