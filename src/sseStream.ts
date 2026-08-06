/**
 * Transport-level helpers for the OpenAI-compatible SSE stream.
 *
 * Kept free of `vscode` imports so it can be unit tested outside the extension
 * host — see src/test/streamDecode.test.ts.
 */

/** Refuse to buffer more than this without seeing a newline (runaway stream). */
export const MAX_SSE_LINE_CHARS = 8 * 1024 * 1024;

/**
 * Split a byte stream into complete lines.
 *
 * Decoding is stateful so a multi-byte UTF-8 character split across two TCP
 * chunks survives intact, and the trailing partial line is flushed at the end so
 * a final event that arrives without a newline is not silently dropped.
 */
export async function* decodeSseLines(
  chunks: AsyncIterable<Uint8Array | string>
): AsyncGenerator<string> {
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  for await (const chunk of chunks) {
    buffer += typeof chunk === "string" ? chunk : decoder.decode(chunk, { stream: true });
    if (buffer.length > MAX_SSE_LINE_CHARS) {
      throw new Error("llama-server stream produced a line over 8 MB — aborting.");
    }
    const parts = buffer.split("\n");
    buffer = parts.pop() || "";
    for (const line of parts) {
      yield line;
    }
  }
  buffer += decoder.decode();
  if (buffer.length) {
    yield buffer;
  }
}

/**
 * Pick the accumulator slot for a streamed tool-call delta.
 *
 * OpenAI identifies each parallel call by `index`, but not every server sends
 * one. Falling back to 0 merged separate calls into a single malformed one, so
 * fall back to the call `id`, and only then to the call already in flight.
 */
export function toolCallSlot(
  calls: Array<{ id: string; name: string; arguments: string }>,
  delta: { index?: number; id?: string }
): number {
  if (typeof delta.index === "number" && delta.index >= 0) {
    return delta.index;
  }
  if (delta.id) {
    const existing = calls.findIndex((c) => c && c.id === delta.id);
    return existing >= 0 ? existing : calls.length;
  }
  return calls.length ? calls.length - 1 : 0;
}
