import type { TChatCompletionChunk } from "@openllmsh/protocol";

/**
 * Pre-commit first-event peek — shared by the daemon walker and the cloud
 * dispatch chain (like the context ladder, ONE implementation so the two
 * paths cannot drift). An in-stream rejection that precedes any output —
 * the overflow incident shape: HTTP 200, then (possibly after
 * housekeeping events) `error: Your input exceeds the context window` —
 * must surface while the hop is still an UNCOMMITTED candidate so the
 * chain can walk to the next hop, instead of failing inside an
 * already-committed 200 stream.
 */

/** Outcome of peeking a decoded upstream stream's first meaningful event
 *  before committing the client response:
 *  - `chunks`: the first meaningful event was real output — the
 *    (replayable) chunk stream is committed to the client;
 *  - `refusal`: the first meaningful event was a STRUCTURED refusal
 *    (canonical `finish_reason: "content_filter"`) that preceded any real
 *    output — the (replayable) chunk stream is still returned so a FINAL
 *    hop can surface the authentic provider refusal, while a non-final hop
 *    can discard it and walk;
 *  - `error`: the rejection that arrived before any output — including a
 *    stream that closed without producing a single event. */
export type TPeekedChunks<T> =
  | { readonly kind: "chunks"; readonly chunks: ReadableStream<T> }
  | { readonly kind: "refusal"; readonly chunks: ReadableStream<T> }
  | { readonly kind: "error"; readonly error: unknown };

/**
 * Does this decoded chunk carry actual model output (content, reasoning,
 * tool calls, a finish, usage)? Housekeeping chunks — the role-only
 * empty-content chunk decoded from `response.created` (chatgpt) or
 * `message_start` (anthropic) — do NOT commit the response: an in-stream
 * rejection can still follow them (`response.failed`), and committing on
 * housekeeping would forfeit the pre-commit walk for exactly the overflow
 * shape the peek exists to catch.
 */
export const isMeaningfulChunk = (chunk: TChatCompletionChunk): boolean => {
  if (chunk.usage !== undefined && chunk.usage !== null) return true;
  return chunk.choices.some((choice) => {
    if (choice.finish_reason !== undefined && choice.finish_reason !== null) {
      return true;
    }
    const d = choice.delta;
    const nonEmpty = (s: string | null | undefined): boolean =>
      typeof s === "string" && s.length > 0;
    const nonEmptyArr = (
      a: ReadonlyArray<unknown> | null | undefined,
    ): boolean => Array.isArray(a) && a.length > 0;
    return (
      nonEmpty(d.content) ||
      nonEmpty(d.reasoning_content) ||
      nonEmpty(d.refusal) ||
      nonEmptyArr(d.tool_calls) ||
      nonEmptyArr(d.reasoning_items) ||
      nonEmptyArr(d.server_search_calls)
    );
  });
};

/**
 * Wait for the first MEANINGFUL event of a decoded upstream stream (see
 * the module doc for why), then return a stream that replays what was
 * buffered ahead of the live tail.
 *
 * The wait is UNBOUNDED by design — bounded only by the request's abort
 * signal, which the upstream fetch carries (an aborted client errors the
 * read). The rejection we're waiting for arrives after the vendor
 * tokenizes the prompt, and a near-cap prompt (the exact overflow case)
 * prefills for many seconds; any fixed deadline commits the response
 * right before the rejection lands, stranding the client in a committed
 * stream that then dies (the pre-fix stall). The client loses nothing
 * from waiting: it would receive zero bytes before the first event
 * regardless. (Same rule as ref/CLIProxyAPI's stream bootstrap: no
 * timeouts after the upstream connection is established.)
 *
 * A stream that CLOSES before any event is an error by default (walkable
 * — the upstream produced nothing), not an empty success. EXCEPT when the
 * peeked stream is a lossy DECODED VIEW of a byte-verbatim passthrough
 * (`opts.emptyStreamIsError: false`): there, zero decoded chunks can mean
 * the decoder dropped frames the schema doesn't model — not an empty
 * upstream — and the client's bytes ride the other tee branch untouched,
 * so the peek must commit rather than 502 a healthy response. Buffered
 * housekeeping chunks are replayed ahead of the live tail: no event is
 * lost or duplicated, and a rejection that arrives after real output
 * began still propagates mid-stream as before (committed = never
 * re-dispatched).
 */
export const peekFirstChunk = async <T>(
  source: ReadableStream<T>,
  isMeaningful: (chunk: T) => boolean = () => true,
  opts?: {
    readonly emptyStreamIsError?: boolean;
    /**
     * Classifies the FIRST meaningful chunk as a structured refusal. When
     * it returns true the peek resolves `kind: "refusal"` (still with the
     * replayable stream) instead of `kind: "chunks"`. Evaluated only on
     * the first meaningful event, so a refusal that arrives AFTER real
     * output has already committed the response is never reported here —
     * exactly the pre-output boundary the fallback walk needs. Refusal
     * thus takes precedence over any usage the same terminal chunk
     * carries, because we classify the chunk before treating it as
     * committed output.
     */
    readonly isRefusal?: (chunk: T) => boolean;
  },
): Promise<TPeekedChunks<T>> => {
  const reader = source.getReader();
  const buffered: T[] = [];
  const replayed = (): ReadableStream<T> =>
    new ReadableStream<T>({
      pull: async (controller) => {
        const next = buffered.shift();
        if (next !== undefined) {
          controller.enqueue(next);
          return;
        }
        const r = await reader.read();
        if (r.done) {
          controller.close();
          return;
        }
        controller.enqueue(r.value);
      },
      cancel: (reason) => reader.cancel(reason),
    });
  try {
    for (;;) {
      const r = await reader.read();
      if (r.done) {
        if (buffered.length === 0 && opts?.emptyStreamIsError !== false) {
          return {
            kind: "error",
            error: new Error(
              "upstream closed the stream before producing any event",
            ),
          };
        }
        // Housekeeping-only stream that closed cleanly: commit and replay
        // what arrived rather than synthesizing a failure.
        return { kind: "chunks", chunks: replayed() };
      }
      buffered.push(r.value);
      if (isMeaningful(r.value)) {
        return opts?.isRefusal?.(r.value) === true
          ? { kind: "refusal", chunks: replayed() }
          : { kind: "chunks", chunks: replayed() };
      }
    }
  } catch (error) {
    return { kind: "error", error };
  }
};
