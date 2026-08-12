/**
 * Minimal OpenAI-compatible chat streaming client for llama-server.
 * Used by the TUI (and any non-VS Code frontend). Abort with AbortSignal.
 */
import * as http from "http";
import { decodeSseLines } from "./sseStream";

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
  signal?: AbortSignal;
}

interface LlamaTimings {
  cache_n?: number;
  prompt_n?: number;
  prompt_ms?: number;
  prompt_per_second?: number;
  predicted_n?: number;
  predicted_ms?: number;
  predicted_per_second?: number;
}

interface LlamaUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  prompt_tokens_details?: { cached_tokens?: number };
}

function formatHttpError(status: number, body: string): string {
  const snippet = body.replace(/\s+/g, " ").trim().slice(0, 240);
  return snippet ? `HTTP ${status}: ${snippet}` : `HTTP ${status}`;
}

/**
 * Stream a chat completion from `{endpoint}/v1/chat/completions`.
 * Yields text deltas and a final stats/done pair.
 */
export async function* streamChatCompletion(
  options: ChatCompletionOptions
): AsyncGenerator<ChatStreamEvent> {
  const endpoint = options.endpoint.replace(/\/$/, "");
  const u = new URL(`${endpoint}/v1/chat/completions`);
  const body = {
    model: options.model || "llama-aio",
    messages: options.messages,
    temperature: options.temperature,
    top_p: options.topP,
    top_k: options.topK,
    max_tokens: options.maxTokens,
    stream: true,
    stream_options: { include_usage: true },
  };
  const payload = JSON.stringify(body);
  const signal = options.signal;

  const stream = await new Promise<http.IncomingMessage>((resolve, reject) => {
    const req = http.request(
      {
        hostname: u.hostname,
        port: u.port,
        path: u.pathname,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(payload),
          Accept: "text/event-stream",
        },
        timeout: 600_000,
      },
      (res) => {
        if (!res.statusCode || res.statusCode >= 400) {
          const chunks: Buffer[] = [];
          res.on("data", (c) => chunks.push(c));
          res.on("end", () =>
            reject(new Error(formatHttpError(res.statusCode || 0, Buffer.concat(chunks).toString("utf8"))))
          );
          return;
        }
        resolve(res);
      }
    );
    req.on("error", reject);
    if (signal) {
      if (signal.aborted) {
        req.destroy();
        reject(new Error("Cancelled"));
        return;
      }
      signal.addEventListener(
        "abort",
        () => {
          req.destroy();
          reject(new Error("Cancelled"));
        },
        { once: true }
      );
    }
    req.write(payload);
    req.end();
  });

  let lastTimings: LlamaTimings | undefined;
  let lastUsage: LlamaUsage | undefined;
  let completionChars = 0;

  try {
    for await (const line of decodeSseLines(stream)) {
      if (signal?.aborted) {
        stream.destroy();
        return;
      }
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) {
        continue;
      }
      const data = trimmed.slice(5).trim();
      if (data === "[DONE]") {
        continue;
      }
      try {
        const json = JSON.parse(data) as {
          choices?: Array<{
            delta?: {
              content?: string | null;
              reasoning_content?: string | null;
            };
          }>;
          timings?: LlamaTimings;
          usage?: LlamaUsage;
        };
        if (json.timings) {
          lastTimings = json.timings;
        }
        if (json.usage) {
          lastUsage = json.usage;
        }

        const delta = json.choices?.[0]?.delta;
        const content = delta?.content;
        const reasoning = delta?.reasoning_content;
        if (typeof content === "string" && content.length) {
          completionChars += content.length;
          yield { kind: "text", text: content };
        } else if (typeof reasoning === "string" && reasoning.length && !content) {
          // deepseek format: stream reasoning when content is empty
          completionChars += reasoning.length;
          yield { kind: "text", text: reasoning };
        }
      } catch (err) {
        console.warn(
          "Llama AIO chat: could not parse SSE event:",
          data.slice(0, 200),
          err instanceof Error ? err.message : err
        );
      }
    }
  } finally {
    stream.destroy();
  }

  yield {
    kind: "stats",
    promptTokens: lastUsage?.prompt_tokens,
    completionTokens:
      lastUsage?.completion_tokens ??
      (completionChars ? Math.max(1, Math.ceil(completionChars / 4)) : undefined),
    promptTokPerSec: lastTimings?.prompt_per_second,
    genTokPerSec: lastTimings?.predicted_per_second,
    cachedPromptTokens:
      lastUsage?.prompt_tokens_details?.cached_tokens ??
      (typeof lastTimings?.cache_n === "number" && lastTimings.cache_n >= 0
        ? lastTimings.cache_n
        : undefined),
    processedPromptTokens: lastTimings?.prompt_n,
  };
  yield { kind: "done" };
}
