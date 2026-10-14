/**
 * Minimal OpenAI-compatible chat streaming client for llama-server.
 * Used by the TUI (and any non-VS Code frontend). Abort with AbortSignal.
 */
import type * as http from "http";
import { openChatStream, parseChatStream } from "./chatStream";
import { ConnectionError } from "./serverHttp";

export type ChatRole = "system" | "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export type ChatStreamEvent =
  | { kind: "text"; text: string }
  | {
      kind: "stats";
      promptTokens?: number;
      completionTokens?: number;
      promptTokPerSec?: number;
      genTokPerSec?: number;
      cachedPromptTokens?: number;
      processedPromptTokens?: number;
    }
  | { kind: "done" };

export interface ChatCompletionOptions {
  endpoint: string;
  messages: ChatMessage[];
  model?: string;
  temperature?: number;
  topP?: number;
  topK?: number;
  maxTokens?: number;
  /** Min-p sampling (0 = disabled). */
  minP?: number;
  /** Presence penalty (OpenAI-style, 0 = disabled). */
  presencePenalty?: number;
  /** Frequency penalty (OpenAI-style, 0 = disabled). */
  frequencyPenalty?: number;
  /** Repetition penalty (llama.cpp style, 1.0 = disabled). */
  repeatPenalty?: number;
  /** Token-id string → logit delta. Omitted when empty. */
  logitBias?: Record<string, number>;
  signal?: AbortSignal;
}

function formatHttpError(status: number, body: string): string {
  const snippet = body.replace(/\s+/g, " ").trim().slice(0, 240);
  return snippet ? `HTTP ${status}: ${snippet}` : `HTTP ${status}`;
}

/**
 * Stream a chat completion from `{endpoint}/v1/chat/completions`.
 * Yields text deltas and a final stats/done pair. Abort ends the stream quietly.
 */
export async function* streamChatCompletion(
  options: ChatCompletionOptions
): AsyncGenerator<ChatStreamEvent> {
  const body = {
    model: options.model || "llama-aio",
    messages: options.messages,
    temperature: options.temperature,
    top_p: options.topP,
    top_k: options.topK,
    max_tokens: options.maxTokens,
    // Sent explicitly (even at disabled values) so a request never silently
    // inherits llama-server's built-in defaults (min_p 0.05, repeat-last-n 64).
    min_p: options.minP ?? 0,
    presence_penalty: options.presencePenalty ?? 0,
    frequency_penalty: options.frequencyPenalty ?? 0,
    repeat_penalty: options.repeatPenalty ?? 1,
    ...(options.logitBias && Object.keys(options.logitBias).length > 0
      ? { logit_bias: options.logitBias }
      : {}),
    stream: true,
    stream_options: { include_usage: true },
  };
  const signal = options.signal;

  let stream: http.IncomingMessage;
  try {
    stream = await openChatStream(options.endpoint, body, signal);
  } catch (err) {
    if (err instanceof ConnectionError && err.status) {
      throw new Error(formatHttpError(err.status, err.body || ""));
    }
    throw err instanceof ConnectionError && err.kind === "cancelled" ? new Error("Cancelled") : err;
  }

  // Same parser as the VS Code chat provider; the TUI sends no tools, so
  // reasoning-only streams show live and nothing is held back.
  let last: ChatStreamEvent | undefined;
  try {
    for await (const part of parseChatStream(stream, { toolsEnabled: false, signal })) {
      if (part.kind === "text") {
        yield part;
      } else if (part.kind === "stats") {
        last = {
          kind: "stats",
          promptTokens: part.promptTokens,
          completionTokens: part.completionTokens,
          promptTokPerSec: part.promptTokPerSec,
          genTokPerSec: part.genTokPerSec,
          cachedPromptTokens: part.cachedPromptTokens,
          processedPromptTokens: part.processedPromptTokens,
        };
      }
    }
  } catch (err) {
    if (signal?.aborted) {
      return;
    }
    throw err;
  } finally {
    stream.destroy();
  }
  if (signal?.aborted) {
    return;
  }
  if (last) {
    yield last;
  }
  yield { kind: "done" };
}
