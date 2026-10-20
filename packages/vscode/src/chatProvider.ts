import type * as http from "http";
import * as path from "path";
import * as vscode from "vscode";
import {
  applyModeToRequestBody,
  applyReplacementsToSystemMessages,
  buildModeConfigurationSchema,
  buildReplacementStats,
  ConnectionError,
  connectionHint,
  duplicateToolCallHint,
  estimateContextBreakdown,
  estimateTokensFromChars,
  fingerprintsSinceLastUserMessage,
  formatExceedContextError,
  formatToolResultContent,
  joinServerUrl,
  loadPromptReplacements,
  messageContentChars,
  messagesHaveImageParts,
  openChatStream,
  overthinkingLogitBias,
  parseChatStream,
  PerfStats,
  positiveRate,
  ProcessManager,
  remoteSlotContext,
  remoteStateUsable,
  resolveModelModes,
  resolveModeParams,
  rewriteToolInput,
  serverJson,
  SettingsStore,
  SseStreamError,
  toolCallFingerprint,
  WIKIPEDIA_LOOKUP_SYSTEM_HINT,
  WIKIPEDIA_LOOKUP_TOOL_NAME,
  wikipediaLookupToolDefinition,
  withSystemHints,
  type ChatStreamPart,
  type ChatStreamStats,
  type PromptReplacement,
  type PromptReplacementStats,
  type RemoteServerInfo,
} from "@llama-aio/core";

const DUPLICATE_TOOL_CALL_SYSTEM_HINT =
  "Do not repeat a tool call with the same arguments in this turn. If file search returned no matches, broaden the glob (**/*llama* not **/llama) or grep for a symbol. Re-reading a file later is fine.";

interface OpenAiModel {
  id: string;
  owned_by?: string;
  meta?: { label?: string };
}

interface OpenAiModelsResponse {
  data?: OpenAiModel[];
}

interface ServerProps {
  default_generation_settings?: { n_ctx?: number };
  total_slots?: number;
  model_path?: string;
  model_alias?: string;
  modalities?: { vision?: boolean };
}

type OpenAiContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

type OpenAiChatMessage =
  | {
      role: "system" | "user" | "assistant";
      content: string | null | OpenAiContentPart[];
      tool_calls?: OpenAiToolCall[];
    }
  | { role: "tool"; content: string; tool_call_id: string };

type OpenAiToolCall = {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
};

type StatsEvent = ChatStreamStats;

/** Chat error text for a failed request: HTTP bodies are parsed, network errors get one hint. */
function describeRequestError(
  err: unknown,
  endpoint: string,
  extras?: { maxContext?: number; hasImages?: boolean; operatorLabel?: string }
): Error {
  if (!(err instanceof ConnectionError)) {
    return err instanceof Error ? err : new Error(String(err));
  }
  if (err.kind === "cancelled") {
    return new Error("Cancelled");
  }
  if (err.kind === "auth") {
    return new Error(
      `${err.message} ${connectionHint("auth", endpoint)} Use “Set API key…” in the Llama AIO sidebar.`
    );
  }
  if (err.kind === "loading") {
    return new Error(`${err.message} ${connectionHint("loading", endpoint)} Send the chat again when the sidebar shows Connected.`);
  }
  if (err.status) {
    return new Error(formatHttpError(err.status, err.body || "", extras));
  }
  return new Error(`${err.message} ${connectionHint(err.kind, endpoint)}`);
}

function formatHttpError(
  status: number,
  text: string,
  extras?: { maxContext?: number; hasImages?: boolean; operatorLabel?: string }
): string {
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
      return formatExceedContextError({
        nPrompt: typeof nPrompt === "number" ? nPrompt : undefined,
        nCtx: typeof nCtx === "number" ? nCtx : undefined,
        maxContext: extras?.maxContext,
        hasImages: extras?.hasImages,
        operatorLabel: extras?.operatorLabel,
        serverMessage: msg,
      });
    }
    return `HTTP ${status}: ${msg}`;
  } catch {
    return `HTTP ${status}: ${text.slice(0, 400)}`;
  }
}

function asImagePart(part: unknown): { mimeType: string; data: Uint8Array } | undefined {
  if (!part || typeof part !== "object") {
    return undefined;
  }
  const rec = part as { mimeType?: unknown; data?: unknown; value?: unknown };
  const mime = typeof rec.mimeType === "string" ? rec.mimeType : "";
  if (!/^image\//i.test(mime)) {
    return undefined;
  }
  const raw = rec.data ?? rec.value;
  if (raw instanceof Uint8Array) {
    return { mimeType: mime, data: raw };
  }
  if (typeof Buffer !== "undefined" && Buffer.isBuffer(raw)) {
    return { mimeType: mime, data: raw };
  }
  return undefined;
}

function imageToDataUrl(img: { mimeType: string; data: Uint8Array }): string {
  return `data:${img.mimeType};base64,${Buffer.from(img.data).toString("base64")}`;
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
  return formatToolResultContent(content);
}

function toOpenAiMessages(
  messages: readonly vscode.LanguageModelChatRequestMessage[]
): OpenAiChatMessage[] {
  const out: OpenAiChatMessage[] = [];

  for (const msg of messages) {
    const textParts: string[] = [];
    const imageParts: Array<{ mimeType: string; data: Uint8Array }> = [];
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
      const image = asImagePart(part);
      if (image) {
        imageParts.push(image);
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

    // Tool results must follow the assistant tool_calls immediately. Copilot
    // puts the follow-up user text in the same turn, so emit tools first.
    for (const toolResult of toolResults) {
      out.push({
        role: "tool",
        tool_call_id: toolResult.callId,
        content: toolResult.content,
      });
    }
    if (imageParts.length > 0) {
      const content: OpenAiContentPart[] = [];
      if (textContent) {
        content.push({ type: "text", text: textContent });
      }
      for (const img of imageParts) {
        content.push({ type: "image_url", image_url: { url: imageToDataUrl(img) } });
      }
      out.push({ role: "user", content });
    } else if (textContent) {
      out.push({ role: "user", content: textContent });
    }
  }

  return out;
}

/**
 * Open the chat stream and run it through core's shared SSE parser. Request
 * and stream failures become chat-facing errors; a cancelled request ends
 * quietly instead of surfacing a socket error.
 */
async function* streamChatCompletions(
  endpoint: string,
  body: Record<string, unknown>,
  token: vscode.CancellationToken,
  extras?: { maxContext?: number; hasImages?: boolean; operatorLabel?: string }
): AsyncGenerator<ChatStreamPart> {
  const toolsEnabled = Array.isArray(body.tools) && (body.tools as unknown[]).length > 0;
  const abort = new AbortController();
  const cancelSub = token.onCancellationRequested(() => abort.abort());
  try {
    let stream: http.IncomingMessage;
    try {
      stream = await openChatStream(endpoint, body, abort.signal);
    } catch (err) {
      if (abort.signal.aborted) {
        return;
      }
      throw describeRequestError(err, endpoint, extras);
    }
    try {
      yield* parseChatStream(stream, { toolsEnabled, signal: abort.signal });
    } catch (err) {
      if (abort.signal.aborted) {
        return;
      }
      if (err instanceof SseStreamError) {
        throw new vscode.LanguageModelError(err.message);
      }
      throw describeRequestError(err, endpoint, extras);
    } finally {
      stream.destroy();
    }
  } finally {
    cancelSub.dispose();
  }
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
  /** One toast per session when the slot is under 16k but the prompt still fits. */
  private warnedSmallSlot = false;
  /** Successful /health, /props, and /v1/models probes shared across one chat turn. */
  private static readonly PROBE_TTL_MS = 2_000;
  private readyProbe: { endpoint: string; at: number } | undefined;
  private propsProbe: { endpoint: string; at: number; props: ServerProps } | undefined;
  private modelProbe: { endpoint: string; at: number; modelId: string; modelName?: string } | undefined;

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
    messages: OpenAiChatMessage[]
  ): Promise<{
    messages: OpenAiChatMessage[];
    enabled: boolean;
    matchedRuleNames: string[];
    /** Characters removed from system text. Negative when a rule expands it. */
    charsSaved: number;
  }> {
    const enabled = this.store.isPromptReplacementsEnabled();
    if (!enabled) {
      return { messages, enabled: false, matchedRuleNames: [], charsSaved: 0 };
    }
    const rules = await this.getReplacementRules();
    const { messages: next, matchedRuleNames } = applyReplacementsToSystemMessages(messages, rules);
    let charsSaved = 0;
    for (let i = 0; i < messages.length; i++) {
      const before = messages[i]?.content;
      const after = next[i]?.content;
      if (typeof before === "string" && typeof after === "string") {
        charsSaved += before.length - after.length;
      }
    }
    return { messages: next, enabled: true, matchedRuleNames, charsSaved };
  }

  notifyChanged(): void {
    this._onDidChangeModelInformation.fire();
  }

  private async fetchServerProps(): Promise<ServerProps | undefined> {
    const endpoint = this.store.getEndpoint();
    const hit = this.propsProbe;
    if (hit && hit.endpoint === endpoint && Date.now() - hit.at < LlamaAioChatProvider.PROBE_TTL_MS) {
      return hit.props;
    }
    if (this.store.isRemote() && this.processManager.remote.current()?.isLlamaCpp === false) {
      return undefined;
    }
    try {
      const props = await serverJson<ServerProps>(joinServerUrl(endpoint, "/props"), {
        timeoutMs: this.store.isRemote() ? 5000 : 3000,
      });
      if (props) {
        this.propsProbe = { endpoint, at: Date.now(), props };
      }
      return props;
    } catch {
      return undefined;
    }
  }

  /** Last probe of the remote server, or undefined in local mode. */
  private remoteInfo(): RemoteServerInfo | undefined {
    return this.store.isRemote() ? this.processManager.remote.current() : undefined;
  }

  /** One successful /health probe is reused for a short window (P-12). */
  private async serverReady(): Promise<boolean> {
    const endpoint = this.store.getEndpoint();
    const hit = this.readyProbe;
    if (hit && hit.endpoint === endpoint && Date.now() - hit.at < LlamaAioChatProvider.PROBE_TTL_MS) {
      return true;
    }
    let ready: boolean;
    if (this.store.isRemote()) {
      const info = await this.processManager.remote.refresh();
      ready = remoteStateUsable(info?.state);
    } else {
      ready = await this.processManager.isHttpReady();
    }
    if (ready) {
      this.readyProbe = { endpoint, at: Date.now() };
    }
    return ready;
  }

  /** Why a remote chat cannot be sent right now, with one targeted hint. */
  private remoteNotReadyMessage(): string {
    const target = this.store.getServerTarget();
    const info = this.processManager.remote.current();
    switch (info?.state) {
      case "loading":
        return `${target.label} is still loading the model. Send the chat again when the Llama AIO sidebar shows Connected.`;
      case "auth":
        return `${target.label} rejected the request (401): it needs an API key. Use “Set API key…” in the Llama AIO sidebar.`;
      case "unreachable":
        return `Llama AIO cannot reach ${target.label} (${info.error?.message || "no answer"}). ${connectionHint(info.error?.kind, target.baseUrl)}`;
      default:
        return `Llama AIO is still connecting to ${target.label}. Send the chat again in a moment.`;
    }
  }

  /**
   * Per-request context for the current server. Remote: what the server says,
   * never this machine's load settings.
   */
  private slotContext(props: ServerProps | undefined): number {
    const fromServer = props?.default_generation_settings?.n_ctx;
    if (fromServer) {
      return fromServer;
    }
    if (this.store.isRemote()) {
      return remoteSlotContext(this.remoteInfo());
    }
    return this.store.getSlotContextSize(this.store.getState().loadSettings);
  }

  /**
   * Model-mode lookup. Remote servers match their canonical alias, filename,
   * then served id; the local GGUF selection describes a different model.
   */
  private modeSetFor(modelId: string, props: ServerProps | undefined) {
    if (!this.store.isRemote()) {
      return resolveModelModes(this.store.getState().modelCapabilities, modelId);
    }
    const file = props?.model_path || this.remoteInfo()?.modelPath;
    const canonical = props?.model_alias || this.remoteInfo()?.modelAlias;
    // The remote server may run on a different OS. Match only its filename,
    // never a parent folder or this machine's selected GGUF.
    const filename = file?.split(/[/\\]/).pop();
    const cached = this.modelProbe;
    const remote = this.remoteInfo();
    const cachedName = cached?.endpoint === this.store.getEndpoint() && cached.modelId === modelId
      ? cached.modelName : undefined;
    const name = cachedName ?? (remote?.modelId === modelId ? remote.modelName : undefined);
    return resolveModelModes(undefined, canonical)
      ?? resolveModelModes(undefined, filename)
      ?? resolveModelModes(undefined, name)
      ?? resolveModelModes(undefined, modelId);
  }

  private async cachedModelId(fallback: string): Promise<string> {
    const endpoint = this.store.getEndpoint();
    const hit = this.modelProbe;
    if (hit && hit.endpoint === endpoint && Date.now() - hit.at < LlamaAioChatProvider.PROBE_TTL_MS) {
      return hit.modelId;
    }
    try {
      const models = await serverJson<OpenAiModelsResponse>(joinServerUrl(endpoint, "/v1/models"), {
        timeoutMs: this.store.isRemote() ? 5000 : 3000,
      });
      const first = models.data?.[0];
      const modelId = first?.id;
      if (modelId) {
        const modelName = first.owned_by === "strata" && typeof first.meta?.label === "string"
          ? first.meta.label : undefined;
        this.modelProbe = { endpoint, at: Date.now(), modelId, modelName };
        return modelId;
      }
    } catch {
      // fall through to the selected file name
    }
    return fallback;
  }

  async provideLanguageModelChatInformation(
    _options: { silent: boolean },
    _token: vscode.CancellationToken
  ): Promise<vscode.LanguageModelChatInformation[]> {
    const target = this.store.getServerTarget();
    const remote = target.mode === "remote";
    const ready = await this.serverReady();
    const state = this.store.getState();
    const remoteInfo = this.remoteInfo();
    const selected = state.selectedModelPath;
    const fallback = remote
      ? remoteInfo?.modelId || this.store.getServerPreferences().lastRemoteModelId || "remote-model"
      : selected
        ? selected.split(/[/\\]/).pop() || "local-model"
        : "local-model";
    const modelId = ready ? await this.cachedModelId(fallback) : fallback;
    if (remote && ready && modelId !== "remote-model") {
      void this.store.setLastRemoteModelId(modelId).catch(() => undefined);
    }

    const props = ready ? await this.fetchServerProps() : undefined;
    const serverSlotCtx = this.slotContext(props);
    const maxOutput = Math.min(state.requestSettings.maxTokens, Math.floor(serverSlotCtx / 4));
    const maxInput = Math.max(1024, serverSlotCtx - maxOutput);
    const shortName = modelId.split(/[/\\]/).pop() || modelId;
    const vision = remote
      ? !!(props?.modalities?.vision ?? remoteInfo?.vision)
      : !!(state.loadSettings.mmprojPath || "").trim();
    const modeSet = this.modeSetFor(modelId, props);
    const where = remote
      ? `${remoteInfo?.isLlamaCpp === false ? "OpenAI-compatible server" : "llama.cpp"} on ${target.label} · ${target.baseUrl}`
      : `Local llama.cpp at ${target.baseUrl}`;
    const info: vscode.LanguageModelChatInformation & {
      configurationSchema?: { properties: Record<string, unknown> };
      statusIcon?: vscode.ThemeIcon;
    } = {
      id: modelId,
      name: `Llama AIO: ${shortName}`,
      family: "llama-aio",
      version: "1.0.0",
      maxInputTokens: maxInput,
      maxOutputTokens: maxOutput,
      statusIcon: new vscode.ThemeIcon("llama-aio-model"),
      tooltip: modeSet
        ? `${where} · slot context ${serverSlotCtx} · modes: ${Object.keys(modeSet.modes).join(", ")}${vision ? " · vision" : ""}`
        : `${where} · slot context ${serverSlotCtx}${vision ? " · vision" : ""}`,
      detail: remote
        ? `${serverSlotCtx} ctx/slot · ${target.label}`
        : `${serverSlotCtx} ctx/slot · ${state.selectedModelPath || target.baseUrl}`,
      capabilities: {
        toolCalling: true,
        imageInput: vision,
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
    const remote = this.store.isRemote();
    if (!(await this.serverReady())) {
      if (remote) {
        throw new vscode.LanguageModelError(this.remoteNotReadyMessage());
      }
      const starting = this.processManager.getStatus().starting;
      throw new vscode.LanguageModelError(
        starting
          ? "Llama AIO server is still starting. Wait until it is ready, then send the chat again."
          : "Llama AIO server is not running. Start it from the Llama AIO sidebar, then send the chat again."
      );
    }
    if (!remote && this.processManager.getStatus().configDirty) {
      throw new vscode.LanguageModelError(
        "Llama AIO load settings changed. Reload the server from the sidebar before chatting. A chat request will not restart it."
      );
    }

    const state = this.store.getState();
    const remoteInfo = this.remoteInfo();
    const serverLabel = this.store.getServerTarget().label;
    const props = await this.fetchServerProps();
    const slotCtx = this.slotContext(props);

    const convertedRaw = toOpenAiMessages(messages);
    let tools =
      options.tools?.map((t) => ({
        type: "function",
        function: {
          name: t.name,
          description: t.description,
          parameters: t.inputSchema,
        },
      })) ?? undefined;

    if (this.store.isWikipediaLookupEnabled()) {
      const wiki = wikipediaLookupToolDefinition();
      const already = tools?.some((t) => t.function.name === WIKIPEDIA_LOOKUP_TOOL_NAME);
      // Ask mode has no tools; don't emit a call Copilot cannot run.
      if (!already && tools?.length) {
        tools = [...tools, wiki];
      }
    }

    const prepared = await this.prepareMessagesWithReplacements(convertedRaw);
    const guardDuplicates = this.store.isDuplicateToolCallGuardEnabled();
    const hints: string[] = [];
    if (guardDuplicates && tools?.length) {
      hints.push(DUPLICATE_TOOL_CALL_SYSTEM_HINT);
    }
    if (
      this.store.isWikipediaLookupEnabled() &&
      tools?.some((t) => t.function.name === WIKIPEDIA_LOOKUP_TOOL_NAME)
    ) {
      hints.push(WIKIPEDIA_LOOKUP_SYSTEM_HINT);
    }
    // Front, not end: strict templates reject a trailing system message.
    const converted = withSystemHints(prepared.messages, hints);

    const body: Record<string, unknown> = {
      model: model.id,
      messages: converted,
      temperature: state.requestSettings.temperature,
      top_p: state.requestSettings.topP,
      top_k: state.requestSettings.topK,
      max_tokens: Math.min(state.requestSettings.maxTokens, Math.floor(slotCtx / 4)),
      // Explicit so requests never inherit llama-server's built-in sampler
      // defaults (min_p 0.05 …). Curated model modes still override below.
      min_p: state.requestSettings.minP,
      presence_penalty: state.requestSettings.presencePenalty,
      frequency_penalty: state.requestSettings.frequencyPenalty,
      repeat_penalty: state.requestSettings.repeatPenalty,
      stream: true,
      stream_options: { include_usage: true },
    };
    if (tools?.length) {
      body.tools = tools;
      body.tool_choice = options.toolMode === vscode.LanguageModelChatToolMode.Required ? "required" : "auto";
    }

    const modeSet = this.modeSetFor(model.id, props);
    const modelConfiguration = (options as { modelConfiguration?: Record<string, unknown> })
      .modelConfiguration;
    let appliedModeName: string | undefined;
    if (modeSet) {
      const { modeName, params } = resolveModeParams(modeSet, modelConfiguration);
      appliedModeName = modeName;
      applyModeToRequestBody(body, params);
    }

    let toolChars = 0;
    if (tools?.length) {
      try {
        toolChars = JSON.stringify(tools).length;
      } catch {
        toolChars = 0;
      }
    }
    const requestChars = messageContentChars(converted) + toolChars;
    const tokensAfter = estimateTokensFromChars(requestChars);
    const tokensBefore = estimateTokensFromChars(requestChars + prepared.charsSaved);
    const replacementStats: PromptReplacementStats = buildReplacementStats({
      enabled: prepared.enabled,
      tokensBefore,
      tokensAfter,
      matchedRuleNames: prepared.matchedRuleNames,
    });
    const estimatedPromptTokens = tokensAfter;
    if (estimatedPromptTokens >= slotCtx) {
      throw new vscode.LanguageModelError(
        remote
          ? `This prompt is about ${estimatedPromptTokens} tokens and ${serverLabel}'s context window is ${slotCtx} per slot. ` +
              `The server must be restarted with a larger --ctx-size or fewer -np slots.`
          : `This prompt is about ${estimatedPromptTokens} tokens and the context window is ${slotCtx} per slot. ` +
              `It does not fit. Raise the Context window or set Max Concurrent Predictions to 1, then Reload.`
      );
    }
    if (slotCtx < 16384 && !this.warnedSmallSlot) {
      this.warnedSmallSlot = true;
      void vscode.window.showWarningMessage(
        remote
          ? `${serverLabel}'s context window is ${slotCtx} tokens per slot (under 16k). This prompt fits, but Copilot agent turns often need more. ` +
              `The server must be restarted with a larger --ctx-size and fewer parallel slots.`
          : `Llama AIO's context window is ${slotCtx} tokens per slot (under 16k). This prompt fits, but Copilot agent turns often need more. ` +
              `Set the Context window ≥ 65536 and Max Concurrent Predictions = 1, then Reload.`
      );
    }
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
      speculativeMode: remote ? undefined : state.loadSettings.speculativeMode || "off",
      contextBreakdown,
    });
    let lastServerStats: StatsEvent | undefined;
    let lastCompletionTokens = 0;
    let lastPromptTokens: number | undefined;
    let lastTickAt = 0;
    let emittedTextChars = 0;
    let emittedToolCallCount = 0;
    const priorToolCalls = guardDuplicates
      ? fingerprintsSinceLastUserMessage(convertedRaw)
      : new Set<string>();
    const seenThisTurn = new Set<string>();
    let skippedDuplicateHint = false;
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
    // After mode sampling, so a Copilot mode that rewrites temperature does not
    // drop the penalty. Ids are resolved once per model from /tokenize.
    const skipBias = remote && remoteInfo?.isLlamaCpp === false;
    const markerBias = skipBias
      ? undefined
      : await overthinkingLogitBias(
          this.store.getEndpoint(),
          remote ? model.id || remoteInfo?.modelId || "" : state.selectedModelPath || "",
          state.requestSettings.overthinkingPenalty,
          state.requestSettings.overthinkingPenaltyStrength
        );
    if (markerBias) {
      body.logit_bias = markerBias;
    }
    try {
      for await (const event of streamChatCompletions(this.store.getEndpoint(), body, token, {
        maxContext: remote ? remoteInfo?.trainedContext : state.modelCapabilities?.maxContextLength,
        hasImages: messagesHaveImageParts(converted),
        operatorLabel: remote ? serverLabel : undefined,
      })) {
        if (event.kind === "text") {
          emittedTextChars += event.text.length;
          progress.report(new vscode.LanguageModelTextPart(event.text));
        } else if (event.kind === "tool_call") {
          const input = rewriteToolInput(event.name, event.input);
          const fp = toolCallFingerprint(event.name, input);
          if (guardDuplicates && (priorToolCalls.has(fp) || seenThisTurn.has(fp))) {
            if (!skippedDuplicateHint) {
              skippedDuplicateHint = true;
              const hint = duplicateToolCallHint(event.name, input);
              emittedTextChars += hint.length;
              progress.report(new vscode.LanguageModelTextPart(hint));
            }
            continue;
          }
          seenThisTurn.add(fp);
          emittedToolCallCount += 1;
          progress.report(
            new vscode.LanguageModelToolCallPart(
              event.callId,
              event.name,
              input && typeof input === "object" ? (input as object) : {}
            )
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
          const gen = positiveRate(event.genTokPerSec);
          const prompt = positiveRate(event.promptTokPerSec);
          const hasServerExtras =
            gen !== undefined ||
            prompt !== undefined ||
            event.draftTokens !== undefined ||
            event.cachedPromptTokens !== undefined ||
            event.processedPromptTokens !== undefined;
          if (hasServerExtras) {
            lastServerStats = event;
          }
          // Tick during the stream — do not complete() until the SSE ends, or
          // a mid-stream timings object (prompt tok/s) would hide "Generating…".
          if (typeof event.completionTokens === "number" || gen || prompt) {
            const now = Date.now();
            if (now - lastTickAt >= 250 || gen || prompt) {
              lastTickAt = now;
              this.perf.tick(event.completionTokens ?? lastCompletionTokens, {
                genTokPerSec: gen,
                promptTokPerSec: prompt,
              });
            }
          }
        }
      }
      if (token.isCancellationRequested) {
        // User stopped the request: no stats, no "empty reply" error.
        this.perf.abort();
        return;
      }
      if (lastServerStats) {
        this.perf.complete({
          promptTokens: lastServerStats.promptTokens ?? lastPromptTokens,
          completionTokens:
            (lastServerStats.completionTokens ?? lastCompletionTokens) || undefined,
          promptTokPerSec: lastServerStats.promptTokPerSec,
          genTokPerSec: lastServerStats.genTokPerSec,
          draftTokens: lastServerStats.draftTokens,
          draftTokensAccepted: lastServerStats.draftTokensAccepted,
          cachedPromptTokens: lastServerStats.cachedPromptTokens,
          processedPromptTokens: lastServerStats.processedPromptTokens,
          contextLimit: slotCtx,
          source: "server",
        });
      } else {
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
      if (token.isCancellationRequested) {
        return;
      }
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
    // Same serialization the request uses, so tool calls and tool results count.
    return Math.ceil(messageContentChars(toOpenAiMessages([text])) / 4);
  }
}
