/**
 * The one parser for llama-server's OpenAI-compatible chat SSE stream, shared
 * by the VS Code chat provider and the TUI. It turns raw stream chunks into
 * neutral events; each front end maps them to its own UI types.
 */
import type * as http from "http";
import { ratesFromTimings, type LlamaTimings } from "./llamaTimings";
import { joinServerUrl, serverStream } from "./serverHttp";
import {
  decodeSseLines,
  errorMessageFromSseJson,
  messageFromSseErrorLine,
  SseStreamError,
  toolCallSlot,
} from "./sseStream";
import { LiveTextGate } from "./streamTextGate";
import { parseXmlToolCalls, stripXmlToolCalls } from "./xmlToolCalls";

export interface ChatStreamStats {
  kind: "stats";
  promptTokens?: number;
  completionTokens?: number;
  promptTokPerSec?: number;
  genTokPerSec?: number;
  draftTokens?: number;
  draftTokensAccepted?: number;
  /** Prompt tokens reused from the KV prefix cache (`timings.cache_n`). */
  cachedPromptTokens?: number;
  /** Prompt tokens actually evaluated this call (`timings.prompt_n`). */
  processedPromptTokens?: number;
}

export interface ChatStreamTrace {
  kind: "trace";
  finishReason?: string;
  assembledText: string;
  assembledReasoning: string;
  visibleText: string;
  structuredToolCalls: Array<{ id: string; name: string; arguments: string }>;
  xmlToolCalls: Array<{ name: string; input: object; raw: string }>;
  toolsEnabled: boolean;
}

export type ChatStreamPart =
  | { kind: "text"; text: string }
  | { kind: "reasoning"; text: string }
  | { kind: "tool_call"; callId: string; name: string; input: object }
  | ChatStreamStats
  | ChatStreamTrace;

interface LlamaUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  prompt_tokens_details?: { cached_tokens?: number };
}

function promptTokensFrom(usage?: LlamaUsage, timings?: LlamaTimings): number | undefined {
  if (typeof usage?.prompt_tokens === "number" && usage.prompt_tokens > 0) {
    return usage.prompt_tokens;
  }
  if (typeof timings?.cache_n === "number" && timings.cache_n >= 0) {
    const processed = typeof timings.prompt_n === "number" && timings.prompt_n > 0 ? timings.prompt_n : 0;
    const total = timings.cache_n + processed;
    return total > 0 ? total : undefined;
  }
  if (typeof timings?.prompt_n === "number" && timings.prompt_n > 0) {
    return timings.prompt_n;
  }
  return undefined;
}

function estimateCompletionTokens(chars: number, usage?: LlamaUsage, timings?: LlamaTimings): number | undefined {
  if (typeof usage?.completion_tokens === "number" && usage.completion_tokens > 0) {
    return usage.completion_tokens;
  }
  if (typeof timings?.predicted_n === "number" && timings.predicted_n > 0) {
    return timings.predicted_n;
  }
  if (chars > 0) {
    return Math.max(1, Math.ceil(chars / 4));
  }
  return undefined;
}

function statsFromTimings(timings?: LlamaTimings, usage?: LlamaUsage, chars = 0): ChatStreamStats {
  const rates = ratesFromTimings(timings);
  return {
    kind: "stats",
    promptTokens: promptTokensFrom(usage, timings),
    completionTokens: estimateCompletionTokens(chars, usage, timings),
    promptTokPerSec: rates.promptTokPerSec,
    genTokPerSec: rates.genTokPerSec,
    draftTokens: timings?.draft_n,
    draftTokensAccepted: timings?.draft_n_accepted,
    cachedPromptTokens:
      usage?.prompt_tokens_details?.cached_tokens ??
      (typeof timings?.cache_n === "number" && timings.cache_n >= 0 ? timings.cache_n : undefined),
    processedPromptTokens: timings?.prompt_n,
  };
}

/** Remove model "thinking" blocks from assistant text (Qwen / DeepSeek-style). */
export function stripThinkTags(text: string): string {
  return text
    .replace(/<think\b[^>]*>[\s\S]*?<\/think>/gi, "")
    .replace(/<thinking\b[^>]*>[\s\S]*?<\/thinking>/gi, "")
    .replace(/<\/?think\b[^>]*>/gi, "")
    .trim();
}

/**
 * POST a streaming chat completion and return the open SSE response. HTTP and
 * network failures reject with a `ConnectionError` for the caller to word.
 */
export function openChatStream(
  endpoint: string,
  body: Record<string, unknown>,
  signal?: AbortSignal
): Promise<http.IncomingMessage> {
  return serverStream(joinServerUrl(endpoint, "/v1/chat/completions"), {
    method: "POST",
    body: JSON.stringify({ ...body, stream: true }),
    headers: { Accept: "text/event-stream" },
    // llama-server can close an SSE socket just after the response ends. A
    // follow-up tool turn must not race Node's keep-alive reuse of that socket.
    // Open a fresh connection instead of retrying a potentially delivered POST.
    freshConnection: true,
    // Prompt processing of a long agent context can be silent for minutes.
    timeoutMs: 600_000,
    signal,
  });
}

type SseChunk = {
  choices?: Array<{
    finish_reason?: string | null;
    delta?: {
      content?: string | null;
      reasoning_content?: string | null;
      tool_calls?: Array<{ index?: number; id?: string; function?: { name?: string; arguments?: string } }>;
    };
    message?: {
      content?: string | null;
      reasoning_content?: string | null;
      tool_calls?: Array<{ id?: string; function?: { name?: string; arguments?: string } }>;
    };
  }>;
  timings?: LlamaTimings;
  usage?: LlamaUsage;
};

/**
 * Parse an OpenAI-compatible chat SSE stream into text, tool-call, stats and
 * a final trace event.
 *
 * With `toolsEnabled`, content goes through a {@link LiveTextGate} that holds
 * back only a short tail, so Qwen-style `<tool_call>` XML in plain content can
 * still become a tool call instead of showing as text — without freezing the
 * reply until the end. Without tools, reasoning-only streams (deepseek format)
 * are shown live.
 *
 * Server error events throw {@link SseStreamError}. When `signal` aborts, the
 * generator returns without trailing stats or trace.
 */
export async function* parseChatStream(
  chunks: AsyncIterable<Uint8Array | string>,
  options: { toolsEnabled: boolean; signal?: AbortSignal; separateReasoning?: boolean; onProgress?: (trace: ChatStreamTrace) => void }
): AsyncGenerator<ChatStreamPart> {
  const { toolsEnabled, signal } = options;
  let assembledText = "";
  let assembledReasoning = "";
  const liveGate = toolsEnabled || options.separateReasoning ? new LiveTextGate(!!options.separateReasoning) : undefined;
  let inlineReasoning = "";
  let emittedReasoning = "";
  let reasoningSource: "inline" | "structured" | undefined;
  function reasoningDelta(increment: string, source: "inline" | "structured"): string {
    if (source === "inline" && assembledReasoning) return "";
    const target = source === "structured" ? assembledReasoning : inlineReasoning;
    // Compare full prefixes only when switching channels. Checking the entire
    // accumulated thought on every token makes long reasoning quadratic.
    if (source !== reasoningSource && !target.startsWith(emittedReasoning)) return "";
    const delta = source === reasoningSource ? increment : target.slice(emittedReasoning.length);
    emittedReasoning = target;
    reasoningSource = source;
    return delta;
  }
  const toolCalls: Array<{ id: string; name: string; arguments: string }> = [];
  let lastTimings: LlamaTimings | undefined;
  let lastUsage: LlamaUsage | undefined;
  let completionChars = 0;
  let finishReason: string | undefined;
  let lastTraceAt = -Infinity;

  for await (const line of decodeSseLines(chunks)) {
    if (signal?.aborted) {
      return;
    }
    const trimmed = line.trim();
    const errorLine = messageFromSseErrorLine(trimmed);
    if (errorLine) {
      throw new SseStreamError(errorLine);
    }
    if (!trimmed.startsWith("data:")) {
      continue;
    }
    const data = trimmed.slice(5).trim();
    if (data === "[DONE]") {
      break;
    }
    let json: SseChunk;
    try {
      json = JSON.parse(data) as SseChunk;
    } catch (err) {
      // decodeSseLines only yields complete lines, so this is a real protocol problem.
      console.warn(
        "Llama AIO: could not parse SSE event:",
        data.slice(0, 300),
        err instanceof Error ? err.message : err
      );
      continue;
    }
    const streamError = errorMessageFromSseJson(json);
    if (streamError) {
      throw new SseStreamError(streamError);
    }
    if (json.timings) {
      lastTimings = json.timings;
    }
    if (json.usage) {
      lastUsage = json.usage;
    }

    const choice = json.choices?.[0];
    if (typeof choice?.finish_reason === "string") finishReason = choice.finish_reason;
    const delta = choice?.delta;
    let progressed = false;

    if (typeof delta?.reasoning_content === "string" && delta.reasoning_content.length) {
      assembledReasoning += delta.reasoning_content;
      completionChars += delta.reasoning_content.length;
      progressed = true;
      if (options.separateReasoning) {
        const text = reasoningDelta(delta.reasoning_content, "structured");
        if (text) yield { kind: "reasoning", text };
      } else if (!toolsEnabled && !delta.content) {
        // Live-stream reasoning when the server only emits reasoning_content
        // (deepseek format). deepseek-legacy also puts tags in content.
        yield { kind: "text", text: delta.reasoning_content };
      }
    }
    if (typeof delta?.content === "string" && delta.content.length) {
      assembledText += delta.content;
      completionChars += delta.content.length;
      progressed = true;
      // Tools enabled: stream everything except a short tail that could
      // still become a <tool_call> / <think> opener.
      const live = liveGate ? liveGate.push(delta.content) : delta.content;
      const inline = liveGate?.takeReasoning() || "";
      inlineReasoning += inline;
      if (options.separateReasoning) {
        const text = reasoningDelta(inline, "inline");
        if (text) yield { kind: "reasoning", text };
      }
      if (live) {
        yield { kind: "text", text: live };
      }
    }

    if (Array.isArray(delta?.tool_calls)) {
      for (const tc of delta.tool_calls) {
        const idx = toolCallSlot(toolCalls, tc);
        if (!toolCalls[idx]) {
          toolCalls[idx] = { id: "", name: "", arguments: "" };
        }
        if (tc.id) {
          toolCalls[idx].id = tc.id;
        }
        if (tc.function?.name) {
          toolCalls[idx].name = tc.function.name;
        }
        if (tc.function?.arguments) {
          toolCalls[idx].arguments += tc.function.arguments;
          completionChars += tc.function.arguments.length;
          progressed = true;
        }
      }
    }

    // Non-streaming servers answer with one `message` instead of deltas.
    if (choice?.message?.reasoning_content && !assembledReasoning) {
      assembledReasoning = choice.message.reasoning_content;
      completionChars += choice.message.reasoning_content.length;
      progressed = true;
      if (options.separateReasoning) {
        const text = reasoningDelta(choice.message.reasoning_content, "structured");
        if (text) yield { kind: "reasoning", text };
      } else if (!toolsEnabled && !choice.message.content && !assembledText) {
        yield { kind: "text", text: choice.message.reasoning_content };
      }
    }
    if (choice?.message?.content && !assembledText) {
      assembledText = choice.message.content;
      completionChars += choice.message.content.length;
      progressed = true;
      const live = liveGate ? liveGate.push(choice.message.content) : choice.message.content;
      const inline = liveGate?.takeReasoning() || "";
      inlineReasoning += inline;
      if (options.separateReasoning) {
        const text = reasoningDelta(inline, "inline");
        if (text) yield { kind: "reasoning", text };
      }
      if (live) yield { kind: "text", text: live };
    }
    if (Array.isArray(choice?.message?.tool_calls)) {
      for (const [idx, tc] of choice.message.tool_calls.entries()) {
        const hadArgs = !!toolCalls[idx]?.arguments;
        toolCalls[idx] = {
          id: tc.id || `call_${idx}`,
          name: tc.function?.name || "",
          arguments: tc.function?.arguments || "",
        };
        if (!hadArgs && tc.function?.arguments) {
          completionChars += tc.function.arguments.length;
          progressed = true;
        }
      }
    }

    // Keep partial diagnostics available if the stream fails or is cancelled.
    if (options.onProgress && progressed && Date.now() - lastTraceAt >= 1000) {
      lastTraceAt = Date.now();
      options.onProgress({ kind: "trace", assembledText, assembledReasoning: assembledReasoning || inlineReasoning,
        visibleText: liveGate ? liveGate.emittedText : assembledText,
        structuredToolCalls: toolCalls.filter(tc => !!tc?.name).map(tc => ({ ...tc })),
        xmlToolCalls: [], toolsEnabled, ...(finishReason ? { finishReason } : {}) });
    }
    // Live tok/s ticks even while tools buffer the text.
    if (progressed || json.timings || json.usage) {
      const live = statsFromTimings(json.timings, lastUsage, completionChars);
      if (live.completionTokens || live.genTokPerSec || live.promptTokPerSec || live.promptTokens) {
        yield live;
      }
    }
  }
  if (signal?.aborted) {
    return;
  }

  // Older reasoning formats put thoughts inside content; legacy servers may
  // also duplicate them in reasoning_content. Emit exactly one thinking part.
  const inlineThinking = /<think(?:ing)?\b[^>]*>([\s\S]*?)(?:<\/think(?:ing)?\s*>|$)/gi;
  if (options.separateReasoning && !assembledReasoning) {
    assembledReasoning = [...assembledText.matchAll(inlineThinking)].map(match => match[1]).join("").trim();
    const text = reasoningDelta(assembledReasoning, "structured");
    if (text) yield { kind: "reasoning", text };
  }

  if (lastTimings || lastUsage) {
    yield statsFromTimings(lastTimings, lastUsage, completionChars);
  } else if (assembledText || assembledReasoning || completionChars) {
    yield { kind: "stats", completionTokens: estimateCompletionTokens(completionChars) };
  }

  // OpenAI-structured tool calls first. One id per call, shared by the
  // emitted event and the trace.
  const stamp = Date.now();
  const structuredOut: Array<{ id: string; name: string; arguments: string }> = [];
  for (const [idx, tc] of toolCalls.entries()) {
    if (!tc?.name) {
      continue;
    }
    const id = tc.id || `call_${idx}_${stamp}`;
    structuredOut.push({ id, name: tc.name, arguments: tc.arguments || "" });
    let input: object = {};
    try {
      input = JSON.parse(tc.arguments || "{}") as object;
    } catch {
      input = { raw: tc.arguments || "" };
    }
    yield { kind: "tool_call", callId: id, name: tc.name, input };
  }
  const emittedStructured = structuredOut.length > 0;

  // Prefer content; fall back to reasoning_content when the server used deepseek format.
  const answer = options.separateReasoning ? assembledText.replace(inlineThinking, "") : assembledText;
  const textForParse = stripThinkTags(answer || (options.separateReasoning ? "" : assembledReasoning));
  const xmlCalls = emittedStructured ? [] : parseXmlToolCalls(textForParse);
  const visible = stripXmlToolCalls(textForParse);

  // Tools enabled: emit whatever the live gate still owes (held-back tail, or
  // text after a tool-call block) so the shown text equals `visible`.
  if (liveGate) {
    const remainder = liveGate.finish(visible);
    if (remainder) {
      yield { kind: "text", text: remainder };
    } else if (!options.separateReasoning && !liveGate.emittedText && !visible && !emittedStructured && !xmlCalls.length && assembledReasoning) {
      // Still thinking-only after the budget — surface it so the reply is not empty.
      const reasoningVisible = stripThinkTags(assembledReasoning);
      if (reasoningVisible) {
        yield { kind: "text", text: reasoningVisible };
      }
    }
  }

  for (const [idx, call] of xmlCalls.entries()) {
    yield { kind: "tool_call", callId: `call_xml_${idx}_${stamp}`, name: call.name, input: call.input };
  }

  yield {
    kind: "trace",
    ...(finishReason ? { finishReason } : {}),
    assembledText,
    assembledReasoning,
    visibleText: visible,
    structuredToolCalls: structuredOut,
    xmlToolCalls: xmlCalls,
    toolsEnabled,
  };
}

/**
 * Add fixed instructions to the leading system message (or a new one at the
 * front). Chat templates that only accept a system message at index 0, or
 * enforce strict user/assistant alternation (Gemma, Mistral), reject a system
 * message appended after the conversation with a Jinja exception. A constant
 * hint at the front also keeps the KV prefix stable across turns.
 */
export function withSystemHints<M extends { role: string; content?: unknown }>(messages: M[], hints: string[]): M[] {
  const extra = hints.map((h) => h.trim()).filter(Boolean);
  if (!extra.length) {
    return messages;
  }
  const first = messages[0];
  if (first?.role === "system" && (typeof first.content === "string" || first.content == null)) {
    const base = typeof first.content === "string" ? first.content : "";
    return [{ ...first, content: [base, ...extra].filter(Boolean).join("\n\n") }, ...messages.slice(1)];
  }
  return [{ role: "system", content: extra.join("\n\n") } as M, ...messages];
}
