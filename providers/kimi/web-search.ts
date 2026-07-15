/**
 * Kimi (Moonshot) provider-native search — the kimi analogue of grok's
 * native search and codex's hosted search. The `kimi_code` subscription
 * upstream (OpenAI wire, `/coding/v1/chat/completions`) supports Moonshot's
 * builtin SERVER-executed `$web_search` tool (probed live 2026-07-15):
 *
 *   round 1 → the model emits a `tool_calls` entry of type
 *   `builtin_function` named `$web_search` whose `arguments` are an OPAQUE
 *   server reference (`{"search_result":{"search_id":…},"usage":…}`) — the
 *   search ALREADY RAN server-side; round 2 → the caller echoes those
 *   arguments back verbatim as the tool result and Moonshot injects the
 *   stored results into context and answers.
 *
 * The echo is Moonshot's documented builtin-function protocol, not gateway
 * agency: the walker extracts nothing, executes nothing, and synthesizes
 * nothing — it echoes opaque bytes and re-posts. Execution and continuation
 * stay fully provider-side. The builtin `tool_calls` never enter the
 * canonical world (their `type` isn't `"function"`), so this module works on
 * the RAW OpenAI-wire JSON.
 */

/** The builtin tool injected on a search-declared kimi hop. */
export const KIMI_BUILTIN_SEARCH_TOOL: Readonly<Record<string, unknown>> = {
  type: "builtin_function",
  function: { name: "$web_search" },
};

/** Echo-round ceiling — Moonshot may search several times before answering;
 *  a runaway guard, far above observed behaviour (1-2 rounds). */
export const KIMI_SEARCH_MAX_ROUNDS = 6;

/**
 * Swap the canonicalised `web_search` function tool for Moonshot's builtin
 * on a BUILT OpenAI-wire payload (one search owner per turn). Other tools
 * pass through unchanged.
 */
export const withKimiBuiltinSearch = (body: unknown): unknown => {
  if (body === null || typeof body !== "object") return body;
  const record = body as Record<string, unknown>;
  const tools = Array.isArray(record.tools) ? record.tools : [];
  const kept = tools.filter((t) => {
    if (t === null || typeof t !== "object") return true;
    const fn = (t as { readonly function?: { readonly name?: unknown } })
      .function;
    return !(
      (t as { readonly type?: unknown }).type === "function" &&
      fn?.name === "web_search"
    );
  });
  // Only the EXACT `$web_search` builtin counts — another builtin function
  // must not suppress the injection.
  const alreadyInjected = kept.some(
    (t) =>
      t !== null &&
      typeof t === "object" &&
      (t as { readonly type?: unknown }).type === "builtin_function" &&
      (t as { readonly function?: { readonly name?: unknown } }).function
        ?.name === "$web_search",
  );
  return {
    ...record,
    tools: alreadyInjected ? kept : [...kept, KIMI_BUILTIN_SEARCH_TOOL],
  };
};

export type TKimiBuiltinSearchCall = {
  readonly id: string;
  readonly argumentsJson: string;
};

/**
 * The round's `$web_search` builtin calls when Moonshot paused for the
 * protocol echo — raw-level detection on the upstream JSON. Returns null
 * unless the round's tool calls are EXCLUSIVELY builtin `$web_search`
 * (a round mixing client function calls is the client's to continue).
 */
export const kimiBuiltinSearchCalls = (
  rawResponse: unknown,
): ReadonlyArray<TKimiBuiltinSearchCall> | null => {
  if (rawResponse === null || typeof rawResponse !== "object") return null;
  const choice = (
    rawResponse as {
      readonly choices?: ReadonlyArray<{
        readonly message?: {
          readonly tool_calls?: ReadonlyArray<{
            readonly id?: unknown;
            readonly type?: unknown;
            readonly function?: {
              readonly name?: unknown;
              readonly arguments?: unknown;
            };
          }>;
        };
      }>;
    }
  ).choices?.[0];
  const calls = choice?.message?.tool_calls;
  if (!Array.isArray(calls) || calls.length === 0) return null;
  const builtin: TKimiBuiltinSearchCall[] = [];
  for (const call of calls) {
    if (
      call.type !== "builtin_function" ||
      call.function?.name !== "$web_search" ||
      typeof call.id !== "string"
    ) {
      return null;
    }
    builtin.push({
      id: call.id,
      argumentsJson:
        typeof call.function.arguments === "string"
          ? call.function.arguments
          : JSON.stringify(call.function.arguments ?? {}),
    });
  }
  return builtin;
};

/**
 * The OpenAI-wire messages appended for one echo round: the assistant turn
 * carrying its builtin tool calls VERBATIM, then one `tool` message per call
 * echoing the opaque arguments back (Moonshot's protocol contract).
 */
export const kimiSearchEchoMessages = (
  rawResponse: unknown,
  calls: ReadonlyArray<TKimiBuiltinSearchCall>,
): ReadonlyArray<Record<string, unknown>> => {
  const message =
    rawResponse !== null && typeof rawResponse === "object"
      ? ((
          rawResponse as {
            readonly choices?: ReadonlyArray<{ readonly message?: unknown }>;
          }
        ).choices?.[0]?.message as Record<string, unknown> | undefined)
      : undefined;
  return [
    message ?? { role: "assistant", content: "" },
    ...calls.map((call) => ({
      role: "tool",
      tool_call_id: call.id,
      name: "$web_search",
      content: call.argumentsJson,
    })),
  ];
};
