import * as http from "http";
import * as path from "path";
import * as vscode from "vscode";
import { PerfStats } from "./perfStats";
import { ProcessManager } from "./processManager";
import {
  applyReplacementsToSystemMessages,
  buildReplacementStats,
  estimateRequestTokens,
  loadPromptReplacements,
  type PromptReplacement,
  type PromptReplacementStats,
} from "./promptReplacer";
import {
  applyModeToRequestBody,
  buildModeConfigurationSchema,
  resolveModeParams,
  resolveModelModes,
} from "./modelModes";
import { estimateContextBreakdown } from "./contextBreakdown";
import { decodeSseLines, toolCallSlot } from "./sseStream";
import { SettingsStore } from "./settings";

interface OpenAiModel {
  id: string;
}

interface OpenAiModelsResponse {
  data?: OpenAiModel[];
}

interface ServerProps {
  default_generation_settings?: { n_ctx?: number };
  total_slots?: number;
  model_path?: string;
  model_alias?: string;
}

type OpenAiChatMessage =
  | { role: "system" | "user" | "assistant"; content: string | null; tool_calls?: OpenAiToolCall[] }
  | { role: "tool"; content: string; tool_call_id: string };

type OpenAiToolCall = {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
};

type StreamEvent =
  | { kind: "text"; text: string }
  | { kind: "tool_call"; callId: string; name: string; input: object }
  | {
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
  | {
      kind: "trace";
      assembledText: string;
      assembledReasoning: string;
      visibleText: string;
      structuredToolCalls: Array<{ id: string; name: string; arguments: string }>;
      xmlToolCalls: Array<{ name: string; input: object; raw: string }>;
      toolsEnabled: boolean;
    };

interface LlamaTimings {
  /** Prompt tokens served from the KV prefix cache (-1 / absent on older builds). */
  cache_n?: number;
  prompt_n?: number;
  prompt_ms?: number;
  prompt_per_second?: number;
  predicted_n?: number;
  predicted_ms?: number;
  predicted_per_second?: number;
  /** Speculative / MTP draft tokens generated. */
  draft_n?: number;
  /** Speculative / MTP draft tokens accepted. */
  draft_n_accepted?: number;
}

interface LlamaUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  prompt_tokens_details?: { cached_tokens?: number };
}

function httpJson<T>(
  url: string,
  options: { method?: string; body?: unknown; timeoutMs?: number } = {}
): Promise<T> {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const bodyStr = options.body !== undefined ? JSON.stringify(options.body) : undefined;
    const req = http.request(
      {
        hostname: u.hostname,
        port: u.port,
        path: u.pathname + u.search,
        method: options.method || "GET",
        headers: {
          "Content-Type": "application/json",
          ...(bodyStr ? { "Content-Length": Buffer.byteLength(bodyStr) } : {}),
        },
        timeout: options.timeoutMs ?? 120_000,
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (c) => chunks.push(c));
        res.on("end", () => {
          const text = Buffer.concat(chunks).toString("utf8");
          if (!res.statusCode || res.statusCode >= 400) {
            reject(new Error(formatHttpError(res.statusCode || 0, text)));
            return;
          }
          try {
            resolve(JSON.parse(text) as T);
          } catch (e) {
            reject(e);
          }
        });
      }
    );
    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("Request timed out"));
    });
    if (bodyStr) {
      req.write(bodyStr);
    }
    req.end();
  });
}

function formatHttpError(status: number, text: string): string {
  try {
    const json = JSON.parse(text) as {
      error?: { message?: string; type?: string; n_prompt_tokens?: number; n_ctx?: number };
      message?: string;
    };
    const err = json.error || json;
    const msg = (err as { message?: string }).message || text;
    const nPrompt = (err as { n_prompt_tokens?: number }).n_prompt_tokens;
    const nCtx = (err as { n_ctx?: number }).n_ctx;
    if (
      /exceed.*context|exceed_context_size/i.test(msg) ||
      (typeof nPrompt === "number" && typeof nCtx === "number" && nPrompt > nCtx)
    ) {
      return (
        `Context too small for this Copilot Chat request` +
        (nPrompt && nCtx ? ` (${nPrompt} tokens > ${nCtx} slot context)` : "") +
        `.\n\nFix in Llama AIO sidebar:\n` +
        `1. Set Context Length ≥ 65536 (or higher)\n` +
        `2. Set Max Concurrent Predictions = 1 (parallel slots split the context)\n` +
        `3. Click "Reload to apply changes"\n\n` +
        `Server said: ${msg}`
      );
    }
    return `HTTP ${status}: ${msg}`;
  } catch {
    return `HTTP ${status}: ${text.slice(0, 400)}`;
  }
}

function extractTextFromPart(part: unknown): string {
  if (typeof part === "string") {
    return part;
  }
  if (part instanceof vscode.LanguageModelTextPart) {
    return part.value;
  }
  if (part && typeof part === "object") {
    if ("value" in part && typeof (part as { value: unknown }).value === "string") {
      return (part as { value: string }).value;
    }
    if ("text" in part && typeof (part as { text: unknown }).text === "string") {
      return (part as { text: string }).text;
    }
  }
  return "";
}

function stringifyToolResult(content: readonly unknown[]): string {
  return content
    .map((item) => extractTextFromPart(item) || JSON.stringify(item))
    .filter((item) => item && item !== "null" && item !== "undefined")
    .join("\n");
}

function toOpenAiMessages(
  messages: readonly vscode.LanguageModelChatRequestMessage[]
): OpenAiChatMessage[] {
  const out: OpenAiChatMessage[] = [];

  for (const msg of messages) {
    const textParts: string[] = [];
    const toolCalls: OpenAiToolCall[] = [];
    const toolResults: Array<{ callId: string; content: string }> = [];

    for (const part of msg.content) {
      if (part instanceof vscode.LanguageModelToolCallPart) {
        toolCalls.push({
          id: part.callId,
          type: "function",
          function: {
            name: part.name,
            arguments: JSON.stringify(part.input ?? {}),
          },
        });
        continue;
      }
      if (part instanceof vscode.LanguageModelToolResultPart) {
        toolResults.push({
          callId: part.callId,
          content: stringifyToolResult(part.content),
        });
        continue;
      }
      const text = extractTextFromPart(part);
      if (text) {
        textParts.push(text);
      }
    }

    const textContent = textParts.join("");
    const role =
      msg.role === vscode.LanguageModelChatMessageRole.User
        ? "user"
        : msg.role === vscode.LanguageModelChatMessageRole.Assistant
          ? "assistant"
          : "system";

    if (role === "system") {
      if (textContent) {
        out.push({ role: "system", content: textContent });
      }
      continue;
    }

    if (role === "assistant") {
      if (textContent || toolCalls.length > 0) {
        out.push({
          role: "assistant",
          content: textContent || null,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        });
      }
      continue;
    }

    // user
    if (textContent) {
      out.push({ role: "user", content: textContent });
    }
    for (const toolResult of toolResults) {
      out.push({
        role: "tool",
        tool_call_id: toolResult.callId,
        content: toolResult.content,
      });
    }
  }

  return out;
}

/** Parse Qwen/Hermes-style tool calls embedded in assistant text. */
function parseXmlToolCalls(text: string): Array<{ name: string; input: object; raw: string }> {
  const results: Array<{ name: string; input: object; raw: string }> = [];
  const re = /<tool_call>\s*<function=([^>\n]+)>\s*([\s\S]*?)\s*<\/function>\s*<\/tool_call>/gi;
  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) !== null) {
    const name = match[1].trim();
    const body = match[2];
    const input: Record<string, unknown> = {};
    const paramRe = /<parameter=([^>\n]+)>\s*([\s\S]*?)\s*<\/parameter>/gi;
    let p: RegExpExecArray | null;
    while ((p = paramRe.exec(body)) !== null) {
      const key = p[1].trim();
      let value: unknown = p[2];
      const trimmed = p[2].trim();
      try {
        value = JSON.parse(trimmed);
      } catch {
        value = trimmed;
      }
      input[key] = value;
    }
    results.push({ name, input, raw: match[0] });
  }
  return results;
}

function stripXmlToolCalls(text: string): string {
  return text.replace(/<tool_call>[\s\S]*?<\/tool_call>/gi, "").trim();
}

/** Remove model "thinking" blocks from assistant text (Qwen / DeepSeek-style). */
function stripThinkTags(text: string): string {
  return text
    .replace(/<think\b[^>]*>[\s\S]*?<\/think>/gi, "")
    .replace(/<thinking\b[^>]*>[\s\S]*?<\/thinking>/gi, "")
    .replace(/<\/?think\b[^>]*>/gi, "")
    .trim();
}

/**
 * Streams OpenAI-compatible SSE and yields text / tool-call events.
 * When tools are enabled, content is buffered until the end so Qwen-style
 * <tool_call> XML (content-only chat format) can be converted into tool calls
 * instead of being shown as plain text / silently dropped.
 */
async function* streamChatCompletions(
  endpoint: string,
  body: Record<string, unknown>,
  token: vscode.CancellationToken
): AsyncGenerator<StreamEvent> {
  const u = new URL(`${endpoint}/v1/chat/completions`);
  const payload = JSON.stringify({ ...body, stream: true });
  const toolsEnabled = Array.isArray(body.tools) && (body.tools as unknown[]).length > 0;

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
    token.onCancellationRequested(() => {
      req.destroy();
      reject(new Error("Cancelled"));
    });
    req.write(payload);
    req.end();
  });

  let assembledText = "";
  let assembledReasoning = "";
  const toolCalls: Array<{ id: string; name: string; arguments: string }> = [];
  let lastTimings: LlamaTimings | undefined;
  let lastUsage: LlamaUsage | undefined;
  let completionChars = 0;

  for await (const line of decodeSseLines(stream)) {
    if (token.isCancellationRequested) {
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
          finish_reason?: string | null;
          delta?: {
            content?: string | null;
            reasoning_content?: string | null;
            tool_calls?: Array<{
              index?: number;
              id?: string;
              function?: { name?: string; arguments?: string };
            }>;
          };
          message?: {
            content?: string | null;
            reasoning_content?: string | null;
            tool_calls?: Array<{
              id?: string;
              function?: { name?: string; arguments?: string };
            }>;
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

      const choice = json.choices?.[0];
      if (!choice) {
        continue;
      }

      const delta = choice.delta;
      if (typeof delta?.reasoning_content === "string" && delta.reasoning_content.length) {
        assembledReasoning += delta.reasoning_content;
        completionChars += delta.reasoning_content.length;
        if (!toolsEnabled) {
          // Live-stream reasoning when the server only emits reasoning_content
          // (deepseek format). deepseek-legacy also puts tags in content.
          if (!delta.content) {
            yield { kind: "text", text: delta.reasoning_content };
            yield {
              kind: "stats",
              completionTokens: Math.max(1, Math.ceil(completionChars / 4)),
            };
          }
        }
      }
      if (typeof delta?.content === "string" && delta.content.length) {
        assembledText += delta.content;
        completionChars += delta.content.length;
        // Live-stream only when tools are not involved (no XML tool-call risk).
        if (!toolsEnabled) {
          yield { kind: "text", text: delta.content };
          yield {
            kind: "stats",
            completionTokens: Math.max(1, Math.ceil(completionChars / 4)),
          };
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
          }
        }
      }

      if (choice.message?.reasoning_content && !assembledReasoning) {
        assembledReasoning = choice.message.reasoning_content;
      }
      if (choice.message?.content && !assembledText) {
        assembledText = choice.message.content;
      }
      if (Array.isArray(choice.message?.tool_calls)) {
        for (const [idx, tc] of choice.message.tool_calls.entries()) {
          toolCalls[idx] = {
            id: tc.id || `call_${idx}`,
            name: tc.function?.name || "",
            arguments: tc.function?.arguments || "",
          };
        }
      }
    } catch (err) {
      // A truncated line can no longer happen here (decodeSseLines only yields
      // complete lines), so this is a real protocol problem worth surfacing.
      console.warn(
        "Llama AIO: could not parse SSE event:",
        data.slice(0, 300),
        err instanceof Error ? err.message : err
      );
    }
  }

  if (lastTimings || lastUsage) {
    yield {
      kind: "stats",
      promptTokens: lastUsage?.prompt_tokens ?? lastTimings?.prompt_n,
      completionTokens: lastUsage?.completion_tokens ?? lastTimings?.predicted_n,
      promptTokPerSec: lastTimings?.prompt_per_second,
      genTokPerSec: lastTimings?.predicted_per_second,
      draftTokens: lastTimings?.draft_n,
      draftTokensAccepted: lastTimings?.draft_n_accepted,
      cachedPromptTokens:
        lastUsage?.prompt_tokens_details?.cached_tokens ??
        // cache_n is -1 on builds that don't report prefix reuse.
        (typeof lastTimings?.cache_n === "number" && lastTimings.cache_n >= 0
          ? lastTimings.cache_n
          : undefined),
      processedPromptTokens: lastTimings?.prompt_n,
    };
  } else if (assembledText || assembledReasoning) {
    yield {
      kind: "stats",
      completionTokens: Math.max(
        1,
        Math.ceil((assembledText.length + assembledReasoning.length) / 4)
      ),
    };
  }

  // Emit OpenAI-structured tool calls first.
  let emittedStructured = false;
  const structuredOut: Array<{ id: string; name: string; arguments: string }> = [];
  for (const [idx, tc] of toolCalls.entries()) {
    if (!tc?.name) {
      continue;
    }
    emittedStructured = true;
    structuredOut.push({
      id: tc.id || `call_${idx}`,
      name: tc.name,
      arguments: tc.arguments || "",
    });
    let input: object = {};
    try {
      input = JSON.parse(tc.arguments || "{}") as object;
    } catch {
      input = { raw: tc.arguments || "" };
    }
    yield {
      kind: "tool_call",
      callId: tc.id || `call_${idx}_${Date.now()}`,
      name: tc.name,
      input,
    };
  }

  // Prefer content; fall back to reasoning_content when the server used deepseek format.
  const textForParse = stripThinkTags(assembledText || assembledReasoning);
  const xmlCalls = emittedStructured ? [] : parseXmlToolCalls(textForParse);
  const visible = stripXmlToolCalls(textForParse);

  // When tools were enabled we buffered; emit clean visible text once.
  if (toolsEnabled && visible) {
    yield { kind: "text", text: visible };
  } else if (toolsEnabled && !visible && !emittedStructured && !xmlCalls.length && assembledReasoning) {
    // Still thinking-only after the budget — surface it so Chat is not empty.
    const reasoningVisible = stripThinkTags(assembledReasoning);
    if (reasoningVisible) {
      yield { kind: "text", text: reasoningVisible };
    }
  }

  for (const [idx, call] of xmlCalls.entries()) {
    yield {
      kind: "tool_call",
      callId: `call_xml_${idx}_${Date.now()}`,
      name: call.name,
      input: call.input,
    };
  }

  yield {
    kind: "trace",
    assembledText,
    assembledReasoning,
    visibleText: visible,
    structuredToolCalls: structuredOut,
    xmlToolCalls: xmlCalls,
    toolsEnabled,
  };
}

export class LlamaAioChatProvider implements vscode.LanguageModelChatProvider {
  private readonly _onDidChangeModelInformation = new vscode.EventEmitter<void>();
  readonly onDidChangeLanguageModelChatInformation = this._onDidChangeModelInformation.event;

  constructor(
    private readonly store: SettingsStore,
    private readonly processManager: ProcessManager,
    private readonly perf: PerfStats,
    private readonly extensionPath: string
  ) {}

  private cachedRules: PromptReplacement[] | undefined;
  private cachedRulesKey: string | undefined;

  private async getReplacementRules(): Promise<PromptReplacement[]> {
    const custom = this.store.getPromptReplacementsFile();
    const file =
      custom ||
      path.join(this.extensionPath, "prompt-replacements", "default-prompt-replacements.json");
    if (this.cachedRules && this.cachedRulesKey === file) {
      return this.cachedRules;
    }
    try {
      this.cachedRules = await loadPromptReplacements(file);
      this.cachedRulesKey = file;
      return this.cachedRules;
    } catch (e) {
      console.warn(
        `Llama AIO: failed to load prompt replacements from ${file}:`,
        e instanceof Error ? e.message : e
      );
      this.cachedRules = [];
      this.cachedRulesKey = file;
      return [];
    }
  }

  private async prepareMessagesWithReplacements(
    messages: OpenAiChatMessage[],
    tools: unknown[] | undefined
  ): Promise<{ messages: OpenAiChatMessage[]; stats: PromptReplacementStats }> {
    const tokensBefore = estimateRequestTokens(messages, tools);
    const enabled = this.store.isPromptReplacementsEnabled();
    if (!enabled) {
      return {
        messages,
        stats: buildReplacementStats({
          enabled: false,
          tokensBefore,
          tokensAfter: tokensBefore,
          matchedRuleNames: [],
        }),
      };
    }
    const rules = await this.getReplacementRules();
    const { messages: next, matchedRuleNames } = applyReplacementsToSystemMessages(messages, rules);
    const tokensAfter = estimateRequestTokens(next, tools);
    return {
      messages: next,
      stats: buildReplacementStats({
        enabled: true,
        tokensBefore,
        tokensAfter,
        matchedRuleNames,
      }),
    };
  }

  notifyChanged(): void {
    this._onDidChangeModelInformation.fire();
  }

  private async fetchServerProps(): Promise<ServerProps | undefined> {
    try {
      return await httpJson<ServerProps>(`${this.store.getEndpoint()}/props`, { timeoutMs: 3000 });
    } catch {
      return undefined;
    }
  }

  async provideLanguageModelChatInformation(
    options: { silent: boolean },
    _token: vscode.CancellationToken
  ): Promise<vscode.LanguageModelChatInformation[]> {
    const ready = await this.processManager.isHttpReady();
    if (!ready) {
      if (!options.silent) {
        // Soft prompt — don't block model picker forever.
      }
      return [];
    }

    let modelId = "local-model";
    try {
      const models = await httpJson<OpenAiModelsResponse>(`${this.store.getEndpoint()}/v1/models`, {
        timeoutMs: 3000,
      });
      modelId = models.data?.[0]?.id || modelId;
    } catch {
      const state = this.store.getState();
      if (state.selectedModelPath) {
        modelId = state.selectedModelPath.split(/[/\\]/).pop() || modelId;
      }
    }

    const state = this.store.getState();
    const props = await this.fetchServerProps();
    const serverSlotCtx =
      props?.default_generation_settings?.n_ctx || this.store.getSlotContextSize(state.loadSettings);
    const maxOutput = Math.min(state.requestSettings.maxTokens, Math.floor(serverSlotCtx / 4));
    const maxInput = Math.max(1024, serverSlotCtx - maxOutput);
    const shortName = modelId.split(/[/\\]/).pop() || modelId;
    const modeSet = resolveModelModes(state.modelCapabilities, modelId);
    const info: vscode.LanguageModelChatInformation & {
      configurationSchema?: { properties: Record<string, unknown> };
    } = {
      id: modelId,
      name: `Llama AIO: ${shortName}`,
      family: "llama-aio",
      version: "1.0.0",
      maxInputTokens: maxInput,
      maxOutputTokens: maxOutput,
      tooltip: modeSet
        ? `Local llama.cpp at ${this.store.getEndpoint()} · slot context ${serverSlotCtx} · modes: ${Object.keys(modeSet.modes).join(", ")}`
        : `Local llama.cpp at ${this.store.getEndpoint()} · slot context ${serverSlotCtx}`,
      detail: `${serverSlotCtx} ctx/slot · ${state.selectedModelPath || this.store.getEndpoint()}`,
      capabilities: {
        toolCalling: true,
        imageInput: false,
      },
    };
    if (modeSet) {
      info.configurationSchema = buildModeConfigurationSchema(modeSet);
    }
    return [info];
  }

  async provideLanguageModelChatResponse(
    model: vscode.LanguageModelChatInformation,
    messages: readonly vscode.LanguageModelChatRequestMessage[],
    options: vscode.ProvideLanguageModelChatResponseOptions,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    token: vscode.CancellationToken
  ): Promise<void> {
    if (!(await this.processManager.isHttpReady())) {
      try {
        await this.processManager.start();
      } catch (e) {
        throw new Error(
          `Llama AIO server is not running: ${e instanceof Error ? e.message : String(e)}`
        );
      }
    }

    const state = this.store.getState();
    const props = await this.fetchServerProps();
    const slotCtx =
      props?.default_generation_settings?.n_ctx || this.store.getSlotContextSize(state.loadSettings);
    if (slotCtx < 16384) {
      throw new Error(
        `Llama-server slot context is only ${slotCtx} tokens — too small for Copilot Chat/agent.\n` +
          `In Llama AIO: set Context Length ≥ 65536, Max Concurrent Predictions = 1, then Reload.`
      );
    }

    const convertedRaw = toOpenAiMessages(messages);
    const tools =
      options.tools?.map((t) => ({
        type: "function",
        function: {
          name: t.name,
          description: t.description,
          parameters: t.inputSchema,
        },
      })) ?? undefined;

    const { messages: converted, stats: replacementStats } =
      await this.prepareMessagesWithReplacements(convertedRaw, tools);

    const body: Record<string, unknown> = {
      model: model.id,
      messages: converted,
      temperature: state.requestSettings.temperature,
      top_p: state.requestSettings.topP,
      top_k: state.requestSettings.topK,
      max_tokens: Math.min(state.requestSettings.maxTokens, Math.floor(slotCtx / 4)),
      stream: true,
      stream_options: { include_usage: true },
    };
    if (tools?.length) {
      body.tools = tools;
      body.tool_choice = "auto";
    }

    const modeSet = resolveModelModes(state.modelCapabilities, model.id);
    const modelConfiguration = (options as { modelConfiguration?: Record<string, unknown> })
      .modelConfiguration;
    let appliedModeName: string | undefined;
    if (modeSet) {
      const { modeName, params } = resolveModeParams(modeSet, modelConfiguration);
      appliedModeName = modeName;
      applyModeToRequestBody(body, params);
    }

    const estimatedPromptTokens = estimateRequestTokens(converted, tools);
    const contextBreakdown = estimateContextBreakdown(
      converted,
      tools ?? [],
      slotCtx
    );
    this.perf.recordRequestContext({
      model: model.id,
      slotContext: slotCtx,
      estimatedPromptTokens,
      messages: converted,
      tools: tools ?? [],
      promptReplacements: replacementStats,
      modelMode: appliedModeName,
      contextBreakdown,
    });
    this.perf.begin({
      contextLimit: slotCtx,
      estimatedPromptTokens,
      promptReplacements: replacementStats,
      speculativeMode: state.loadSettings.speculativeMode || "off",
      contextBreakdown,
    });
    let sawFinalServerStats = false;
    let lastCompletionTokens = 0;
    let lastPromptTokens: number | undefined;
    let lastTickAt = 0;
    let emittedTextChars = 0;
    let emittedToolCallCount = 0;
    let lastTrace:
      | {
          assembledText: string;
          assembledReasoning: string;
          visibleText: string;
          structuredToolCalls: Array<{ id: string; name: string; arguments: string }>;
          xmlToolCalls: Array<{ name: string; input: object; raw: string }>;
          toolsEnabled: boolean;
        }
      | undefined;
    try {
      for await (const event of streamChatCompletions(this.store.getEndpoint(), body, token)) {
        if (event.kind === "text") {
          emittedTextChars += event.text.length;
          progress.report(new vscode.LanguageModelTextPart(event.text));
        } else if (event.kind === "tool_call") {
          emittedToolCallCount += 1;
          progress.report(
            new vscode.LanguageModelToolCallPart(event.callId, event.name, event.input)
          );
        } else if (event.kind === "trace") {
          lastTrace = event;
        } else if (event.kind === "stats") {
          if (typeof event.completionTokens === "number") {
            lastCompletionTokens = event.completionTokens;
          }
          if (typeof event.promptTokens === "number") {
            lastPromptTokens = event.promptTokens;
            this.perf.updateContext({
              promptTokens: event.promptTokens,
              contextLimit: slotCtx,
              estimated: false,
            });
          }
          const hasSpeed =
            typeof event.genTokPerSec === "number" ||
            typeof event.promptTokPerSec === "number";
          if (hasSpeed) {
            sawFinalServerStats = true;
            this.perf.complete({
              promptTokens: event.promptTokens ?? lastPromptTokens,
              completionTokens:
                (event.completionTokens ?? lastCompletionTokens) || undefined,
              promptTokPerSec: event.promptTokPerSec,
              genTokPerSec: event.genTokPerSec,
              draftTokens: event.draftTokens,
              draftTokensAccepted: event.draftTokensAccepted,
              cachedPromptTokens: event.cachedPromptTokens,
              processedPromptTokens: event.processedPromptTokens,
              contextLimit: slotCtx,
              source: "server",
            });
          } else if (typeof event.completionTokens === "number") {
            const now = Date.now();
            if (now - lastTickAt >= 250) {
              lastTickAt = now;
              this.perf.tick(event.completionTokens);
            }
          }
        }
      }
      if (!sawFinalServerStats) {
        this.perf.complete({
          promptTokens: lastPromptTokens ?? estimatedPromptTokens,
          completionTokens: lastCompletionTokens || undefined,
          contextLimit: slotCtx,
          source: lastPromptTokens !== undefined ? "server" : "estimate",
        });
      }

      const emptyToChat = emittedTextChars <= 0 && emittedToolCallCount <= 0;
      if (lastTrace) {
        this.perf.recordResponseTrace({
          model: model.id,
          toolsEnabled: lastTrace.toolsEnabled,
          assembledText: lastTrace.assembledText,
          assembledReasoning: lastTrace.assembledReasoning,
          visibleText: lastTrace.visibleText,
          structuredToolCalls: lastTrace.structuredToolCalls,
          xmlToolCalls: lastTrace.xmlToolCalls,
          emittedTextChars,
          emittedToolCallCount,
          emptyToChat,
          note: emptyToChat
            ? "No text/tool calls emitted to Chat."
            : undefined,
        });
      } else if (emptyToChat) {
        this.perf.recordResponseTrace({
          model: model.id,
          toolsEnabled: !!(tools && tools.length),
          assembledText: "",
          assembledReasoning: "",
          visibleText: "",
          structuredToolCalls: [],
          xmlToolCalls: [],
          emittedTextChars,
          emittedToolCallCount,
          emptyToChat: true,
          note: "Stream ended without a response trace (cancelled or failed before completion).",
        });
      }

      if (emptyToChat) {
        throw new Error(
          "Empty reply to Chat (no text or tool calls). Open Llama AIO → View last response."
        );
      }
    } catch (e) {
      this.perf.abort();
      throw e;
    }
  }

  async provideTokenCount(
    _model: vscode.LanguageModelChatInformation,
    text: string | vscode.LanguageModelChatRequestMessage,
    _token: vscode.CancellationToken
  ): Promise<number> {
    if (typeof text === "string") {
      return Math.ceil(text.length / 4);
    }
    const joined = text.content.map((p) => extractTextFromPart(p)).join("");
    return Math.ceil(joined.length / 4);
  }
}
