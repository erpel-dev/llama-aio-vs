/**
 * Transport-level helpers for the OpenAI-compatible SSE stream.
 *
 * Kept free of `vscode` imports so it can be unit tested outside the extension
 * host — see src/test/streamDecode.test.ts.
 */

/** Refuse to buffer more than this without seeing a newline (runaway stream). */
export const MAX_SSE_LINE_CHARS = 8 * 1024 * 1024;

/** Mid-stream llama-server failure (`error:` event or a JSON `error` object). */
export class SseStreamError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SseStreamError";
  }
}

/**
 * Message from an SSE `error:` line, or undefined when the line is not an error event.
 * Plain text and `{"error":{"message":"..."}}` payloads are both accepted.
 */
export function messageFromSseErrorLine(line: string): string | undefined {
  const trimmed = line.trim();
  if (!/^error:/i.test(trimmed)) {
    return undefined;
  }
  const payload = trimmed.replace(/^error:\s*/i, "").trim();
  if (!payload) {
    return "llama-server stream error";
  }
  return errorMessageFromSseJson(tryParseJson(payload)) || payload.slice(0, 500);
}

/**
 * Message from a parsed SSE JSON object that carries `error`, or undefined for a normal chunk.
 * llama-server reports context overflow this way on an otherwise 200 stream.
 */
export function errorMessageFromSseJson(json: unknown): string | undefined {
  if (!json || typeof json !== "object") {
    return undefined;
  }
  const error = (json as { error?: unknown }).error;
  if (error == null) {
    return undefined;
  }
  if (typeof error === "string" && error.trim()) {
    return error.trim();
  }
  if (typeof error === "object") {
    const message = (error as { message?: unknown }).message;
    if (typeof message === "string" && message.trim()) {
      return message.trim();
    }
  }
  return "llama-server stream error";
}

function tryParseJson(payload: string): unknown {
  if (!payload.startsWith("{") && !payload.startsWith("[")) {
    return undefined;
  }
  try {
    return JSON.parse(payload);
  } catch {
    return undefined;
  }
}

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
