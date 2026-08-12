import { Emitter } from "./events";
import type { ContextBreakdown } from "./contextBreakdown";
import { scaleBreakdownToServerPrompt } from "./contextBreakdown";
import type { PromptReplacementStats } from "./promptReplacer";

export type ContextLevel = "ok" | "warn" | "critical";

/** Last (or in-flight) generation performance from llama-server. */
export interface GenerationPerf {
  /** True while a Copilot chat response is streaming. */
  generating: boolean;
  /** Wall-clock start of the current/last generation. */
  startedAt?: number;
  /** Wall-clock end of the last completed generation. */
  finishedAt?: number;
  promptTokens?: number;
  completionTokens?: number;
  /** Slot context size (n_ctx) used for the meter. */
  contextLimit?: number;
  /** promptTokens / contextLimit · 100 */
  contextPct?: number;
  contextLevel?: ContextLevel;
  /** True when prompt token count is a client-side estimate. */
  contextEstimated?: boolean;
  /** Prompt processing speed (tok/s), from llama timings when available. */
  promptTokPerSec?: number;
  /** Generation speed (tok/s). Prefer server timings; else live estimate. */
  genTokPerSec?: number;
  /** True when genTokPerSec came from a live estimate, not server timings. */
  estimated?: boolean;
  source?: "server" | "estimate";
  /** Speculative / MTP draft tokens generated (`timings.draft_n`). */
  draftTokens?: number;
  /** Speculative / MTP draft tokens accepted (`timings.draft_n_accepted`). */
  draftTokensAccepted?: number;
  /** 100 * draftTokensAccepted / draftTokens when draftTokens > 0. */
  draftAcceptancePct?: number;
  /** Prompt tokens reused from the KV prefix cache (`timings.cache_n`). */
  cachedPromptTokens?: number;
  /** Prompt tokens evaluated this call (`timings.prompt_n`). */
  processedPromptTokens?: number;
  /** 100 * cached / (cached + processed) — prompt cache hit rate. */
  cacheHitPct?: number;
  /** Current load-settings speculative mode (`off` | `mtp` | `dflash`). */
  speculativeMode?: "off" | "mtp" | "dflash";
  /** System-prompt find/replace savings for the last request. */
  promptReplacements?: PromptReplacementStats;
  /** Segmented context-bar breakdown (Tools / System / History / …). */
  contextBreakdown?: ContextBreakdown;
}

/** Snapshot of the last Copilot → llama.cpp chat request (for debug "View context"). */
export interface LastChatRequestContext {
  capturedAt: string;
  model: string;
  slotContext: number;
  estimatedPromptTokens: number;
  messageCount: number;
  toolCount: number;
  /** OpenAI-style messages forwarded to llama.cpp. */
  messages: unknown[];
  /** OpenAI-style tool definitions (may be large). */
  tools: unknown[];
  promptReplacements?: PromptReplacementStats;
  /** Selected Copilot model mode (e.g. Think / No Think), if any. */
  modelMode?: string;
  contextBreakdown?: ContextBreakdown;
}

/** Snapshot of the last streamed assistant reply (for empty-response debugging). */
export interface LastChatResponseTrace {
  capturedAt: string;
  model: string;
  toolsEnabled: boolean;
  /** Raw concatenated delta/message content from the stream. */
  assembledText: string;
  /** Raw reasoning_content from the stream (think models). */
  assembledReasoning: string;
  /** Text after stripping think tags + recognized <tool_call> XML blocks. */
  visibleText: string;
  /** OpenAI-style tool_calls assembled from the stream. */
  structuredToolCalls: Array<{ id: string; name: string; arguments: string }>;
  /** Qwen/Hermes XML tool calls parsed from assembledText. */
  xmlToolCalls: Array<{ name: string; input: object; raw: string }>;
  /** Parts actually reported to Copilot Chat. */
  emittedTextChars: number;
  emittedToolCallCount: number;
  emptyToChat: boolean;
  note?: string;
}

function formatRate(n: number | undefined): string | undefined {
  if (typeof n !== "number" || !Number.isFinite(n) || n <= 0) {
    return undefined;
  }
  return n >= 100 ? n.toFixed(0) : n.toFixed(1);
}

function contextLevel(pct: number | undefined): ContextLevel | undefined {
  if (typeof pct !== "number" || !Number.isFinite(pct)) {
    return undefined;
  }
  if (pct >= 90) {
    return "critical";
  }
  if (pct >= 80) {
    return "warn";
  }
  return "ok";
}

function withContextFields(
  base: GenerationPerf,
  promptTokens: number | undefined,
  contextLimit: number | undefined,
  contextEstimated?: boolean
): Pick<
  GenerationPerf,
  "promptTokens" | "contextLimit" | "contextPct" | "contextLevel" | "contextEstimated"
> {
  const limit = contextLimit ?? base.contextLimit;
  const prompt = promptTokens ?? base.promptTokens;
  let contextPct: number | undefined;
  if (typeof prompt === "number" && typeof limit === "number" && limit > 0) {
    contextPct = Math.min(999, Math.round((prompt / limit) * 1000) / 10);
  }
  return {
    promptTokens: prompt,
    contextLimit: limit,
    contextPct,
    contextLevel: contextLevel(contextPct),
    contextEstimated:
      contextEstimated !== undefined ? contextEstimated : base.contextEstimated,
  };
}

export class PerfStats {
  private readonly _onDidChange = new Emitter<GenerationPerf>();
  readonly onDidChange = this._onDidChange.event;

  private current: GenerationPerf = { generating: false };
  /** Highest alert band we've already toasted for this conversation fill. */
  private alertBand: 0 | 80 | 90 = 0;
  /** Last Copilot → llama.cpp chat request (for debug "View context"). */
  private lastRequest: LastChatRequestContext | undefined;
  /** Last streamed assistant payload (for debug "View last response"). */
  private lastResponse: LastChatResponseTrace | undefined;

  get(): GenerationPerf {
    return { ...this.current };
  }

  hasLastRequestContext(): boolean {
    return !!this.lastRequest;
  }

  hasLastResponseTrace(): boolean {
    return !!this.lastResponse;
  }

  getLastRequestContext(): LastChatRequestContext | undefined {
    return this.lastRequest ? structuredClone(this.lastRequest) : undefined;
  }

  getLastResponseTrace(): LastChatResponseTrace | undefined {
    return this.lastResponse ? structuredClone(this.lastResponse) : undefined;
  }

  /** Store the outbound chat body for later inspection in an editor. */
  recordRequestContext(
    ctx: Omit<LastChatRequestContext, "capturedAt" | "messageCount" | "toolCount"> & {
      capturedAt?: string;
    }
  ): void {
    this.lastRequest = {
      capturedAt: ctx.capturedAt || new Date().toISOString(),
      model: ctx.model,
      slotContext: ctx.slotContext,
      estimatedPromptTokens: ctx.estimatedPromptTokens,
      messageCount: Array.isArray(ctx.messages) ? ctx.messages.length : 0,
      toolCount: Array.isArray(ctx.tools) ? ctx.tools.length : 0,
      messages: ctx.messages,
      tools: ctx.tools,
      promptReplacements: ctx.promptReplacements,
      modelMode: ctx.modelMode,
      contextBreakdown: ctx.contextBreakdown,
    };
  }

  /** Store the last streamed assistant payload (including empty-to-Chat cases). */
  recordResponseTrace(
    trace: Omit<LastChatResponseTrace, "capturedAt" | "emptyToChat"> & {
      capturedAt?: string;
      emptyToChat?: boolean;
    }
  ): void {
    const emptyToChat =
      trace.emptyToChat ??
      (trace.emittedTextChars <= 0 && trace.emittedToolCallCount <= 0);
    this.lastResponse = {
      capturedAt: trace.capturedAt || new Date().toISOString(),
      model: trace.model,
      toolsEnabled: trace.toolsEnabled,
      assembledText: trace.assembledText,
      assembledReasoning: trace.assembledReasoning || "",
      visibleText: trace.visibleText,
      structuredToolCalls: trace.structuredToolCalls,
      xmlToolCalls: trace.xmlToolCalls,
      emittedTextChars: trace.emittedTextChars,
      emittedToolCallCount: trace.emittedToolCallCount,
      emptyToChat,
      note: trace.note,
    };
    this._onDidChange.fire(this.get());
  }

  /** Human-readable dump of the last request (markdown + JSON). */
  formatLastRequestContext(): string | undefined {
    const ctx = this.lastRequest;
    if (!ctx) {
      return undefined;
    }
    const lines: string[] = [
      "# Llama AIO — last Copilot → llama.cpp call",
      "",
      `- Captured: ${ctx.capturedAt}`,
      `- Model: ${ctx.model}`,
      `- Slot context: ${ctx.slotContext.toLocaleString()} tokens`,
      `- Estimated prompt tokens: ${ctx.estimatedPromptTokens.toLocaleString()}`,
      `- Messages: ${ctx.messageCount}`,
      `- Tools: ${ctx.toolCount}`,
    ];
    if (ctx.modelMode) {
      lines.push(`- Model mode: ${ctx.modelMode}`);
    }
    if (ctx.contextBreakdown) {
      const parts = ctx.contextBreakdown.segments
        .filter((s) => s.tokens > 0)
        .map((s) => `${s.label} ≈${s.tokens.toLocaleString()}`);
      if (parts.length) {
        lines.push(`- Context breakdown: ${parts.join(" · ")}`);
      }
    }
    if (ctx.promptReplacements) {
      const pr = ctx.promptReplacements;
      lines.push(
        `- Prompt replacements: ${pr.enabled ? "on" : "off"}` +
          (pr.enabled
            ? ` · saved ≈${pr.tokensSaved.toLocaleString()} tokens (${pr.pctSaved}% of request)`
            : "")
      );
      if (pr.matchedRuleNames.length) {
        lines.push(`- Matched rules: ${pr.matchedRuleNames.join("; ")}`);
      }
    }
    lines.push("", "");

    if (Array.isArray(ctx.tools) && ctx.tools.length) {
      lines.push("## Tool names");
      lines.push("");
      for (const t of ctx.tools) {
        const fn = (t as { function?: { name?: string; description?: string } }).function;
        const name = fn?.name || "(unnamed)";
        const desc = (fn?.description || "").replace(/\s+/g, " ").trim();
        lines.push(`- \`${name}\`${desc ? ` — ${desc.slice(0, 160)}${desc.length > 160 ? "…" : ""}` : ""}`);
      }
      lines.push("");
    }

    if (Array.isArray(ctx.messages)) {
      lines.push("## Messages");
      lines.push("");
      ctx.messages.forEach((m, i) => {
        const msg = m as {
          role?: string;
          content?: string | null;
          tool_call_id?: string;
          tool_calls?: unknown;
        };
        const role = msg.role || "?";
        lines.push(`### [${i}] ${role}${msg.tool_call_id ? ` (tool_call_id=${msg.tool_call_id})` : ""}`);
        lines.push("");
        if (msg.content != null && msg.content !== "") {
          lines.push("```");
          lines.push(String(msg.content));
          lines.push("```");
          lines.push("");
        }
        if (msg.tool_calls) {
          lines.push("```json");
          lines.push(JSON.stringify(msg.tool_calls, null, 2));
          lines.push("```");
          lines.push("");
        }
        if ((msg.content == null || msg.content === "") && !msg.tool_calls) {
          lines.push("_(empty)_");
          lines.push("");
        }
      });
    }

    lines.push("## Raw JSON (messages + tools)");
    lines.push("");
    lines.push("```json");
    lines.push(
      JSON.stringify(
        {
          model: ctx.model,
          slotContext: ctx.slotContext,
          estimatedPromptTokens: ctx.estimatedPromptTokens,
          messages: ctx.messages,
          tools: ctx.tools,
        },
        null,
        2
      )
    );
    lines.push("```");
    lines.push("");
    return lines.join("\n");
  }

  /** Human-readable dump of the last assistant stream (markdown). */
  formatLastResponseTrace(): string | undefined {
    const ctx = this.lastResponse;
    if (!ctx) {
      return undefined;
    }
    const lines: string[] = [
      "# Llama AIO — last llama.cpp → Copilot response",
      "",
      `- Captured: ${ctx.capturedAt}`,
      `- Model: ${ctx.model}`,
      `- Tools enabled: ${ctx.toolsEnabled ? "yes" : "no"}`,
      `- Emitted to Chat: ${ctx.emittedTextChars.toLocaleString()} text chars, ${ctx.emittedToolCallCount} tool call(s)`,
      `- Empty to Chat: ${ctx.emptyToChat ? "yes" : "no"}`,
    ];
    if (ctx.note) {
      lines.push(`- Note: ${ctx.note}`);
    }
    lines.push("");

    lines.push("## Assembled stream text (raw content)");
    lines.push("");
    if (ctx.assembledText) {
      lines.push("```");
      lines.push(ctx.assembledText);
      lines.push("```");
    } else {
      lines.push("_(empty — no delta/message content in the SSE stream)_");
    }
    lines.push("");

    lines.push("## Assembled reasoning_content");
    lines.push("");
    if (ctx.assembledReasoning) {
      lines.push("```");
      lines.push(ctx.assembledReasoning);
      lines.push("```");
    } else {
      lines.push("_(empty)_");
    }
    lines.push("");

    lines.push("## Visible text (after stripping think tags + tool XML)");
    lines.push("");
    if (ctx.visibleText) {
      lines.push("```");
      lines.push(ctx.visibleText);
      lines.push("```");
    } else {
      lines.push("_(empty)_");
    }
    lines.push("");

    lines.push("## Structured tool_calls (OpenAI stream)");
    lines.push("");
    if (ctx.structuredToolCalls.length) {
      lines.push("```json");
      lines.push(JSON.stringify(ctx.structuredToolCalls, null, 2));
      lines.push("```");
    } else {
      lines.push("_(none)_");
    }
    lines.push("");

    lines.push("## Parsed XML tool_calls");
    lines.push("");
    if (ctx.xmlToolCalls.length) {
      for (const [i, call] of ctx.xmlToolCalls.entries()) {
        lines.push(`### [${i}] \`${call.name}\``);
        lines.push("");
        lines.push("```json");
        lines.push(JSON.stringify(call.input, null, 2));
        lines.push("```");
        lines.push("");
        lines.push("<details><summary>Raw XML</summary>");
        lines.push("");
        lines.push("```");
        lines.push(call.raw);
        lines.push("```");
        lines.push("");
        lines.push("</details>");
        lines.push("");
      }
    } else {
      lines.push("_(none matched `<tool_call><function=…>` / `<parameter=…>`)_");
      lines.push("");
    }

    lines.push("## Tips");
    lines.push("");
    lines.push(
      "- If **Empty to Chat** is yes but Assembled text/reasoning is not empty, the model likely used a tool format we do not parse, or tools were enabled and everything was stripped without a valid parse."
    );
    lines.push(
      "- If both content and reasoning are empty while the terminal showed tokens, the stream may use a field we still do not read."
    );
    lines.push("");
    return lines.join("\n");
  }

  /** Keep speculative label in sync with Load settings (clears stale drafts when off). */
  setSpeculativeMode(mode: "off" | "mtp" | "dflash"): void {
    const clearDrafts = mode === "off";
    const hadDrafts =
      this.current.draftTokens !== undefined ||
      this.current.draftTokensAccepted !== undefined ||
      this.current.draftAcceptancePct !== undefined;
    if (this.current.speculativeMode === mode && !(clearDrafts && hadDrafts)) {
      return;
    }
    this.current = {
      ...this.current,
      speculativeMode: mode,
      ...(clearDrafts
        ? {
            draftTokens: undefined,
            draftTokensAccepted: undefined,
            draftAcceptancePct: undefined,
          }
        : {}),
    };
    this._onDidChange.fire(this.get());
  }

  begin(opts?: {
    contextLimit?: number;
    estimatedPromptTokens?: number;
    promptReplacements?: PromptReplacementStats;
    speculativeMode?: "off" | "mtp" | "dflash";
    contextBreakdown?: ContextBreakdown;
  }): void {
    const prev = this.current;
    const ctx = withContextFields(
      prev,
      opts?.estimatedPromptTokens,
      opts?.contextLimit ?? prev.contextLimit,
      opts?.estimatedPromptTokens !== undefined ? true : prev.contextEstimated
    );
    this.current = {
      generating: true,
      startedAt: Date.now(),
      estimated: true,
      source: "estimate",
      completionTokens: undefined,
      promptTokPerSec: undefined,
      genTokPerSec: undefined,
      // Clear draft stats so a disabled / no-draft run never shows a prior %.
      draftTokens: undefined,
      draftTokensAccepted: undefined,
      draftAcceptancePct: undefined,
      speculativeMode: opts?.speculativeMode ?? prev.speculativeMode,
      promptReplacements: opts?.promptReplacements ?? prev.promptReplacements,
      contextBreakdown: opts?.contextBreakdown ?? prev.contextBreakdown,
      ...ctx,
    };
    this.maybeResetAlertBand(ctx.contextPct);
    this._onDidChange.fire(this.get());
  }

  /** Live update while tokens stream (estimate from elapsed time). */
  tick(completionTokens: number): void {
    if (!this.current.generating || !this.current.startedAt) {
      return;
    }
    const elapsedSec = Math.max(0.05, (Date.now() - this.current.startedAt) / 1000);
    this.current = {
      ...this.current,
      completionTokens,
      genTokPerSec: completionTokens / elapsedSec,
      estimated: true,
      source: "estimate",
    };
    this._onDidChange.fire(this.get());
  }

  /** Update context meter without ending the generation (e.g. early usage). */
  updateContext(opts: {
    promptTokens?: number;
    contextLimit?: number;
    estimated?: boolean;
  }): void {
    const ctx = withContextFields(
      this.current,
      opts.promptTokens,
      opts.contextLimit ?? this.current.contextLimit,
      opts.estimated
    );
    let contextBreakdown = this.current.contextBreakdown;
    if (
      contextBreakdown &&
      typeof opts.promptTokens === "number" &&
      opts.estimated === false
    ) {
      contextBreakdown = scaleBreakdownToServerPrompt(
        contextBreakdown,
        opts.promptTokens,
        opts.contextLimit ?? this.current.contextLimit
      );
    }
    this.current = { ...this.current, ...ctx, contextBreakdown };
    this.maybeResetAlertBand(ctx.contextPct);
    this._onDidChange.fire(this.get());
  }

  /** Final numbers from llama-server `timings` / `usage`. */
  complete(partial: Partial<GenerationPerf>): void {
    const finishedAt = Date.now();
    const startedAt = this.current.startedAt;
    let genTokPerSec = partial.genTokPerSec ?? this.current.genTokPerSec;
    const completionTokens = partial.completionTokens ?? this.current.completionTokens;
    const source = partial.source || (typeof partial.genTokPerSec === "number" ? "server" : "estimate");
    if (
      (genTokPerSec === undefined || source === "estimate") &&
      typeof completionTokens === "number" &&
      startedAt &&
      typeof partial.genTokPerSec !== "number"
    ) {
      const elapsedSec = Math.max(0.05, (finishedAt - startedAt) / 1000);
      genTokPerSec = completionTokens / elapsedSec;
    }

    const ctx = withContextFields(
      this.current,
      partial.promptTokens,
      partial.contextLimit ?? this.current.contextLimit,
      partial.promptTokens !== undefined ? source !== "server" : this.current.contextEstimated
    );
    // Server usage is authoritative.
    if (partial.promptTokens !== undefined && source === "server") {
      ctx.contextEstimated = false;
    }

    let draftTokens: number | undefined;
    let draftTokensAccepted: number | undefined;
    let draftAcceptancePct: number | undefined;
    if ("draftTokens" in partial) {
      if (typeof partial.draftTokens === "number" && partial.draftTokens > 0) {
        draftTokens = partial.draftTokens;
        draftTokensAccepted =
          typeof partial.draftTokensAccepted === "number" ? partial.draftTokensAccepted : 0;
        draftAcceptancePct = Math.round((draftTokensAccepted / draftTokens) * 1000) / 10;
      }
      // else: timings omitted draft_n / draft_n===0 → leave undefined (MTP: —)
    } else {
      draftTokens = this.current.draftTokens;
      draftTokensAccepted = this.current.draftTokensAccepted;
      draftAcceptancePct = this.current.draftAcceptancePct;
    }

    let cachedPromptTokens: number | undefined;
    let processedPromptTokens: number | undefined;
    let cacheHitPct: number | undefined;
    if ("cachedPromptTokens" in partial || "processedPromptTokens" in partial) {
      cachedPromptTokens = partial.cachedPromptTokens;
      processedPromptTokens = partial.processedPromptTokens;
      const total = (cachedPromptTokens ?? 0) + (processedPromptTokens ?? 0);
      if (typeof cachedPromptTokens === "number" && total > 0) {
        cacheHitPct = Math.round((cachedPromptTokens / total) * 1000) / 10;
      }
    } else {
      cachedPromptTokens = this.current.cachedPromptTokens;
      processedPromptTokens = this.current.processedPromptTokens;
      cacheHitPct = this.current.cacheHitPct;
    }

    const speculativeMode = partial.speculativeMode ?? this.current.speculativeMode;
    if (speculativeMode === "off") {
      draftTokens = undefined;
      draftTokensAccepted = undefined;
      draftAcceptancePct = undefined;
    }

    let contextBreakdown = partial.contextBreakdown ?? this.current.contextBreakdown;
    if (
      contextBreakdown &&
      typeof partial.promptTokens === "number" &&
      source === "server"
    ) {
      contextBreakdown = scaleBreakdownToServerPrompt(
        contextBreakdown,
        partial.promptTokens,
        partial.contextLimit ?? this.current.contextLimit
      );
    }

    this.current = {
      ...this.current,
      ...partial,
      completionTokens,
      genTokPerSec,
      draftTokens,
      draftTokensAccepted,
      draftAcceptancePct,
      cachedPromptTokens,
      processedPromptTokens,
      cacheHitPct,
      speculativeMode,
      contextBreakdown,
      generating: false,
      finishedAt,
      estimated: source === "estimate",
      source,
      ...ctx,
    };
    this.maybeResetAlertBand(ctx.contextPct);
    this._onDidChange.fire(this.get());
  }

  abort(): void {
    if (!this.current.generating) {
      return;
    }
    this.current = { ...this.current, generating: false, finishedAt: Date.now() };
    this._onDidChange.fire(this.get());
  }

  /**
   * If context crossed 80% / 90%, return an alert payload once per band
   * (resets when usage drops below 70%, e.g. new chat).
   */
  consumeContextAlert():
    | { level: "warn" | "critical"; pct: number; used: number; limit: number }
    | undefined {
    const p = this.current;
    if (
      typeof p.contextPct !== "number" ||
      typeof p.promptTokens !== "number" ||
      typeof p.contextLimit !== "number"
    ) {
      return undefined;
    }
    if (p.contextPct >= 90 && this.alertBand < 90) {
      this.alertBand = 90;
      return {
        level: "critical",
        pct: p.contextPct,
        used: p.promptTokens,
        limit: p.contextLimit,
      };
    }
    if (p.contextPct >= 80 && this.alertBand < 80) {
      this.alertBand = 80;
      return {
        level: "warn",
        pct: p.contextPct,
        used: p.promptTokens,
        limit: p.contextLimit,
      };
    }
    return undefined;
  }

  private maybeResetAlertBand(pct: number | undefined): void {
    if (typeof pct === "number" && pct < 70) {
      this.alertBand = 0;
    }
  }

  /** Compact label for the status bar. */
  statusBarText(serverReady: boolean): string {
    if (!serverReady) {
      return "$(circle-slash) Llama AIO";
    }
    const p = this.current;
    const rate = formatRate(p.genTokPerSec);
    const ctx =
      typeof p.contextPct === "number"
        ? `${p.contextPct >= 10 ? Math.round(p.contextPct) : p.contextPct}% ctx`
        : undefined;
    if (p.generating) {
      const bits = ["$(sync~spin) Llama AIO", rate ? `${rate} t/s` : undefined, ctx].filter(
        Boolean
      );
      return bits.join(" ");
    }
    const icon =
      p.contextLevel === "critical"
        ? "$(warning)"
        : p.contextLevel === "warn"
          ? "$(info)"
          : "$(rocket)";
    const bits = [icon + " Llama AIO", rate ? `${rate} t/s` : undefined, ctx].filter(Boolean);
    return bits.join(" ") || "$(rocket) Llama AIO";
  }

  private mtpDetailLine(p: GenerationPerf): string | undefined {
    if (p.speculativeMode === undefined) {
      return undefined;
    }
    const label = p.speculativeMode === "dflash" ? "DFlash" : "MTP";
    if (p.speculativeMode === "off") {
      return "Speculative: off";
    }
    if (typeof p.draftTokens === "number" && p.draftTokens > 0) {
      const accepted = p.draftTokensAccepted ?? 0;
      const pct =
        typeof p.draftAcceptancePct === "number"
          ? p.draftAcceptancePct
          : Math.round((accepted / p.draftTokens) * 1000) / 10;
      return `${label}: ${pct.toFixed(1)}% accepted (${accepted.toLocaleString()}/${p.draftTokens.toLocaleString()})`;
    }
    return `${label}: —`;
  }

  /** Multi-line tooltip / sidebar summary. */
  detailLines(): string[] {
    const p = this.current;
    if (!p.startedAt && p.genTokPerSec === undefined && p.contextPct === undefined) {
      const mtp = this.mtpDetailLine(p);
      return mtp ? ["No generation yet", mtp] : ["No generation yet"];
    }
    const lines: string[] = [];
    if (p.generating) {
      lines.push("Generating…");
    }
    if (
      typeof p.promptTokens === "number" &&
      typeof p.contextLimit === "number" &&
      typeof p.contextPct === "number"
    ) {
      const approx = p.contextEstimated ? " ≈" : "";
      lines.push(
        `Context:${approx} ${p.promptTokens.toLocaleString()} / ${p.contextLimit.toLocaleString()} (${p.contextPct}%)${
          p.contextLevel === "critical"
            ? " · nearly full"
            : p.contextLevel === "warn"
              ? " · running low"
              : ""
        }`
      );
    }
    const gen = formatRate(p.genTokPerSec);
    const prompt = formatRate(p.promptTokPerSec);
    if (gen) {
      lines.push(`Generation: ${gen} tok/s` + (p.estimated ? " (est.)" : ""));
    }
    if (prompt) {
      lines.push(`Prompt: ${prompt} tok/s`);
    }
    if (typeof p.promptTokens === "number" || typeof p.completionTokens === "number") {
      lines.push(
        `Tokens: ${p.promptTokens?.toLocaleString() ?? "—"} prompt · ${
          p.completionTokens?.toLocaleString() ?? "—"
        } completion`
      );
    }
    if (typeof p.cacheHitPct === "number") {
      lines.push(
        `Prompt reuse: ${p.cacheHitPct}% (${(p.cachedPromptTokens ?? 0).toLocaleString()} cached · ${(
          p.processedPromptTokens ?? 0
        ).toLocaleString()} evaluated)`
      );
    }
    const mtp = this.mtpDetailLine(p);
    if (mtp) {
      lines.push(mtp);
    }
    if (p.finishedAt && p.startedAt && !p.generating) {
      const sec = ((p.finishedAt - p.startedAt) / 1000).toFixed(1);
      lines.push(`Duration: ${sec}s`);
    }
    return lines.length ? lines : ["No generation yet"];
  }

  dispose(): void {
    this._onDidChange.dispose();
  }
}
