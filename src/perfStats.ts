import * as vscode from "vscode";

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
  private readonly _onDidChange = new vscode.EventEmitter<GenerationPerf>();
  readonly onDidChange = this._onDidChange.event;

  private current: GenerationPerf = { generating: false };
  /** Highest alert band we've already toasted for this conversation fill. */
  private alertBand: 0 | 80 | 90 = 0;

  get(): GenerationPerf {
    return { ...this.current };
  }

  begin(opts?: { contextLimit?: number; estimatedPromptTokens?: number }): void {
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
    this.current = { ...this.current, ...ctx };
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

    this.current = {
      ...this.current,
      ...partial,
      completionTokens,
      genTokPerSec,
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

  /** Multi-line tooltip / sidebar summary. */
  detailLines(): string[] {
    const p = this.current;
    if (!p.startedAt && p.genTokPerSec === undefined && p.contextPct === undefined) {
      return ["No generation yet"];
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
        `Tokens: ${p.promptTokens ?? "—"} prompt · ${p.completionTokens ?? "—"} completion`
      );
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
