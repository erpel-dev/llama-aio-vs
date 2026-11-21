import { setTimeout as delay } from "node:timers/promises";
import {
  ConnectionError,
  joinServerUrl,
  normalizeServerUrl,
  serverJson,
} from "@llama-aio/core";
import * as vscode from "vscode";

const CONFIG = "llamaAio.inlineAutocomplete";
const PREFIX_CHARS = 4000;
const SUFFIX_CHARS = 1000;
/** The prefix window starts on this grid, so its start stays put while the user types. */
const PREFIX_GRID = 1024;
const REQUEST_TIMEOUT_MS = 2500;
/** A cold prompt cache (first request, or a newly opened large file) takes longer. */
const COLD_REQUEST_TIMEOUT_MS = 10000;
const SNOOZE_KEY = "llamaAio.inlineAutocomplete.snoozedUntil";

/** A setting the user has to change: reported once instead of failing silently on every keystroke. */
class SetupError extends Error {}

/**
 * Plain HTTP is allowed only to an address that cannot leave this machine:
 * localhost, ::1, or an IPv4 literal in 127.0.0.0/8. A DNS name that merely
 * starts with "127." (127.example.com) or ends in ".localhost" is resolved by
 * DNS here and could reach another host.
 */
export function isLoopbackAddress(hostname: string): boolean {
  const h = hostname.trim().toLowerCase().replace(/^\[|\]$/g, "");
  if (h === "localhost" || h === "::1") return true;
  const parts = h.split(".");
  return parts.length === 4 && parts[0] === "127" && parts.every(p => /^\d{1,3}$/.test(p) && Number(p) <= 255);
}

function settings() {
  const config = vscode.workspace.getConfiguration(CONFIG);
  const int = (name: string, fallback: number, min: number, max: number) => {
    const value = Number(config.get<number>(name));
    return Number.isFinite(value) ? Math.min(max, Math.max(min, Math.round(value))) : fallback;
  };
  return {
    enabled: config.get<boolean>("enabled") === true,
    endpoint: config.get<string>("endpoint")?.trim() || "",
    disabledLanguages: (config.get<string[]>("disabledLanguages") || []).filter(l => typeof l === "string"),
    maxTokens: int("maxTokens", 32, 8, 256),
    delayMs: int("delayMs", 120, 0, 2000),
  };
}

function completionEndpoint(configured = settings().endpoint): string {
  if (!configured) return "";
  let endpoint: string;
  try {
    endpoint = normalizeServerUrl(configured);
  } catch (error) {
    throw new SetupError(`The autocomplete endpoint "${configured}" is not a valid URL: ` +
      (error instanceof Error ? error.message : String(error)));
  }
  if (new URL(endpoint).protocol !== "https:" && !isLoopbackAddress(new URL(endpoint).hostname)) {
    throw new SetupError("Use HTTPS for an autocomplete server outside this machine.");
  }
  return endpoint;
}

function secretName(endpoint: string): string {
  return "llama-aio.inlineAutocomplete.apiKey:" + new URL(endpoint).origin;
}

const SET_KEY = "Set API Key";
const OPEN_SETTINGS = "Open Settings";

/**
 * What the user must fix, for errors a retry on the next keystroke cannot fix.
 * Slow, refused or cancelled requests are shown in the status bar only.
 */
export function autocompleteProblem(error: unknown): { message: string; action: string } | undefined {
  if (error instanceof SetupError) return { message: error.message, action: OPEN_SETTINGS };
  if (error instanceof ConnectionError) {
    if (error.kind === "auth") {
      return {
        message: `The autocomplete server rejected the request (HTTP ${error.status ?? 401}). ` +
          "Set its API key with \"Llama AIO: Set Autocomplete API Key…\".",
        action: SET_KEY,
      };
    }
    if (error.status === 404 || error.status === 501) {
      return {
        message: `The autocomplete server answered HTTP ${error.status} for /infill. ` +
          "Point the endpoint at a llama.cpp server running a model with fill-in-the-middle support.",
        action: OPEN_SETTINGS,
      };
    }
  }
  return undefined;
}

const isLowSurrogate = (c: string) => c >= "\uDC00" && c <= "\uDFFF";
const isHighSurrogate = (c: string) => c >= "\uD800" && c <= "\uDBFF";

/** Drop half an emoji (or other surrogate pair) where a window was cut at a raw UTF-16 offset. */
export function wholeCharacters(prefix: string, suffix: string): { prefix: string; suffix: string } {
  return {
    prefix: prefix && isLowSurrogate(prefix[0]) ? prefix.slice(1) : prefix,
    suffix: suffix && isHighSurrogate(suffix[suffix.length - 1]) ? suffix.slice(0, -1) : suffix,
  };
}

/**
 * Offset where the prefix window starts. A window of "the last N characters"
 * moves with every keystroke, and llama.cpp reuses its prompt cache only for a
 * shared start, so each keystroke reprocessed the whole window. The start is
 * rounded down to a grid and then to a line start, so it stays put while the
 * user types nearby; the window holds between N and N + grid characters.
 */
export function prefixWindowStart(
  offset: number,
  textBetween: (from: number, to: number) => string,
  maxChars = PREFIX_CHARS,
  grid = PREFIX_GRID
): number {
  if (offset <= maxChars) return 0;
  const gridStart = Math.floor((offset - maxChars) / grid) * grid;
  // Back up to the start of that line. A very long line keeps the grid offset.
  const from = Math.max(0, gridStart - 512);
  const newline = textBetween(from, gridStart).lastIndexOf("\n");
  return newline >= 0 ? from + newline + 1 : from === 0 ? 0 : gridStart;
}

/** Text after the cursor that an editor auto-inserts: closing brackets, quotes, `;` and `,`. */
export function isClosingOnly(text: string): boolean {
  return /^[\s)\]}>"'`;,]*$/.test(text);
}

/**
 * Where generation stops. In the middle of a line, finish that line. At the end
 * of a line (only auto-closed characters after the cursor count as the end), or
 * on a blank one, the model's first token is usually a newline, so allow a short
 * block and stop before the next paragraph.
 */
export function stopSequences(prefix: string, suffix: string): string[] {
  const linePrefix = prefix.slice(prefix.lastIndexOf("\n") + 1);
  const lineSuffix = suffix.split("\n", 1)[0];
  return linePrefix.trim() && !isClosingOnly(lineSuffix) ? ["\n"] : ["\n\n"];
}

const indentOf = (line: string) => line.length - line.trimStart().length;

/** True when `needle`'s non-space characters appear in `haystack` in the same order. */
function inOrder(needle: string, haystack: string): boolean {
  let i = 0;
  for (const c of needle.replace(/\s+/g, "")) {
    i = haystack.indexOf(c, i);
    if (i < 0) return false;
    i++;
  }
  return true;
}

/**
 * Fit a reply to the text around the cursor.
 *
 * - A block stops before a line that repeats the next line after the cursor
 *   (same indentation), and before a line indented less than the cursor's line:
 *   that is code outside the block being written.
 * - When only auto-closed characters follow the cursor and the reply contains
 *   them, the suggestion replaces them (`replaceChars`), so `a, b);` at `foo(|)`
 *   gives `foo(a, b);` and not `foo(a, b);)`. A multi-line reply without them
 *   gets them on a closing line of its own.
 * - Otherwise a reply that ends with the start of the rest of the line loses
 *   that overlap.
 */
export function shapeSuggestion(content: string, prefix: string, suffix: string): { text: string; replaceChars: number } {
  const linePrefix = prefix.slice(prefix.lastIndexOf("\n") + 1);
  const lineSuffix = suffix.split("\n", 1)[0];
  let text = content;
  if (text.includes("\n")) {
    const lines = text.split("\n");
    const nextLine = suffix.split("\n").slice(1).find(line => line.trim())?.trimEnd();
    const base = indentOf(linePrefix);
    const at = lines.findIndex((line, i) => i > 0 && line.trim() !== "" &&
      ((nextLine !== undefined && line.trimEnd() === nextLine) || indentOf(line) < base));
    if (at > 0) text = lines.slice(0, at).join("\n").replace(/\s+$/, "");
  }
  if (lineSuffix.trim() && isClosingOnly(lineSuffix)) {
    // The replaced characters end the line, so trailing whitespace after them would add a blank line.
    if (inOrder(lineSuffix, text)) return { text: text.replace(/\s+$/, ""), replaceChars: lineSuffix.length };
    // VS Code shows a multi-line suggestion before text on the same line only
    // when the suggestion contains that text. Close the block with it, on its
    // own line at the cursor line's indentation: `{|}` -> `{\n  body\n}`.
    if (text.includes("\n")) {
      const body = text.replace(/\s+$/, "");
      if (!body.trim()) return { text: "", replaceChars: 0 };
      return { text: body + "\n" + linePrefix.slice(0, indentOf(linePrefix)) + lineSuffix.trimStart(), replaceChars: lineSuffix.length };
    }
  }
  for (let k = Math.min(text.length, lineSuffix.length); k > 0; k--) {
    if (text.endsWith(lineSuffix.slice(0, k))) return { text: text.slice(0, -k), replaceChars: 0 };
  }
  return { text, replaceChars: 0 };
}

type Status =
  | { kind: "off" }
  | { kind: "snoozed"; until: number }
  | { kind: "idle" }
  | { kind: "working" }
  | { kind: "ready"; ms: number }
  | { kind: "slow" }
  | { kind: "unreachable"; message: string }
  | { kind: "problem"; message: string };

/** Supply native VS Code ghost text from a FIM model without changing the chat target. */
export function registerInlineAutocomplete(context: vscode.ExtensionContext): void {
  let pending: AbortController | undefined;
  /** Problems already shown, per endpoint, until the settings or the key change. */
  const reported = new Set<string>();
  /** Endpoints that answered since the settings changed: their prompt cache is warm. */
  const warm = new Set<string>();
  let status: Status = { kind: "idle" };
  /** The status before the current request, restored when that request is cancelled. */
  let settled: Status = status;
  let snoozeTimer: ReturnType<typeof setTimeout> | undefined;

  const statusItem = vscode.window.createStatusBarItem("llamaAio.inlineAutocomplete", vscode.StatusBarAlignment.Right, 99);
  statusItem.name = "Llama AIO Autocomplete";
  statusItem.command = "llamaAio.inlineAutocomplete.menu";

  const snoozedUntil = () => {
    const until = Number(context.globalState?.get<number>(SNOOZE_KEY)) || 0;
    return until > Date.now() ? until : 0;
  };

  const render = () => {
    const s = settings();
    const until = snoozedUntil();
    const shown: Status = !s.enabled ? { kind: "off" } : until ? { kind: "snoozed", until } : status;
    // Only people who set autocomplete up see the item; others find it through the command palette.
    if (!s.enabled && !s.endpoint) {
      statusItem.hide();
      return;
    }
    const where = s.endpoint || "no endpoint set";
    const icon = "$(llama-aio-model)";
    statusItem.backgroundColor = undefined;
    switch (shown.kind) {
      case "off":
        statusItem.text = icon + " Off";
        statusItem.tooltip = `Llama AIO autocomplete is off.\nEndpoint: ${where}\nClick for options.`;
        break;
      case "snoozed": {
        const minutes = Math.max(1, Math.ceil((shown.until - Date.now()) / 60000));
        statusItem.text = `${icon} Snoozed ${minutes} min`;
        statusItem.tooltip = `Llama AIO autocomplete is snoozed for ${minutes} more min.\nClick to resume or for other options.`;
        break;
      }
      case "working":
        statusItem.text = "$(loading~spin) Autocomplete";
        statusItem.tooltip = `Asking ${where} for a suggestion…`;
        break;
      case "ready":
        statusItem.text = icon + " Autocomplete";
        statusItem.tooltip = `Llama AIO autocomplete is on.\nLast suggestion took ${shown.ms} ms from ${where}.\nClick for options.`;
        break;
      case "slow":
        statusItem.text = icon + " Autocomplete";
        statusItem.tooltip = `The last request took too long and was dropped. A large file can take several seconds the first time; ` +
          `the server keeps working, so the next suggestions are usually fast.\nEndpoint: ${where}`;
        break;
      case "unreachable":
        statusItem.text = "$(warning) Autocomplete";
        statusItem.tooltip = `${shown.message}\nStart the server or change the endpoint. Click for options.`;
        statusItem.backgroundColor = new vscode.ThemeColor("statusBarItem.warningBackground");
        break;
      case "problem":
        statusItem.text = "$(warning) Autocomplete";
        statusItem.tooltip = shown.message + "\nClick for options.";
        statusItem.backgroundColor = new vscode.ThemeColor("statusBarItem.warningBackground");
        break;
      default:
        statusItem.text = icon + " Autocomplete";
        statusItem.tooltip = `Llama AIO autocomplete is on and waits for you to type.\nEndpoint: ${where}\nClick for options.`;
    }
    statusItem.show();
  };
  const setStatus = (next: Status) => {
    status = next;
    if (next.kind !== "working") settled = next;
    render();
  };

  const report = (endpoint: string, problem: { message: string; action: string }) => {
    setStatus({ kind: "problem", message: problem.message });
    const key = endpoint + "\u0000" + problem.message;
    if (reported.has(key)) return;
    reported.add(key);
    void vscode.window.showWarningMessage("Llama AIO autocomplete: " + problem.message, problem.action).then(choice => {
      if (choice === SET_KEY) void vscode.commands.executeCommand("llamaAio.setInlineAutocompleteApiKey");
      else if (choice === OPEN_SETTINGS) void vscode.commands.executeCommand("workbench.action.openSettings", CONFIG);
    });
  };

  const scheduleSnoozeEnd = () => {
    if (snoozeTimer) clearTimeout(snoozeTimer);
    const until = snoozedUntil();
    snoozeTimer = undefined;
    if (!until) return;
    // Refresh the minutes shown, and switch back on when the snooze ends.
    snoozeTimer = setTimeout(() => {
      render();
      scheduleSnoozeEnd();
    }, Math.min(60000, until - Date.now() + 50));
  };

  const update = async (key: string, value: unknown) =>
    vscode.workspace.getConfiguration(CONFIG).update(key, value, vscode.ConfigurationTarget.Global);

  const snooze = async (minutes: number) => {
    pending?.abort();
    await context.globalState?.update(SNOOZE_KEY, minutes > 0 ? Date.now() + minutes * 60000 : 0);
    scheduleSnoozeEnd();
    render();
  };

  const editEndpoint = async () => {
    const value = await vscode.window.showInputBox({
      title: "Autocomplete endpoint",
      prompt: "Base URL of a llama.cpp server running a fill-in-the-middle (FIM) model. Use loopback HTTP or HTTPS.",
      placeHolder: "http://127.0.0.1:8012",
      value: settings().endpoint,
      ignoreFocusOut: true,
      validateInput: text => {
        if (!text.trim()) return undefined;
        try {
          completionEndpoint(text.trim());
          return undefined;
        } catch (error) {
          return error instanceof Error ? error.message : String(error);
        }
      },
    });
    if (value === undefined) return;
    await update("endpoint", value.trim());
    if (value.trim() && !settings().enabled) await update("enabled", true);
  };

  context.subscriptions.push(
    statusItem,
    { dispose: () => { pending?.abort(); if (snoozeTimer) clearTimeout(snoozeTimer); } },
    vscode.workspace.onDidChangeConfiguration(event => {
      if (event.affectsConfiguration(CONFIG)) {
        pending?.abort();
        reported.clear();
        warm.clear();
        status = { kind: "idle" };
        render();
      }
    }),
    vscode.commands.registerCommand("llamaAio.setInlineAutocompleteApiKey", async () => {
      try {
        const endpoint = completionEndpoint();
        if (!endpoint) {
          void vscode.window.showInformationMessage("Set Llama AIO’s inline autocomplete endpoint first.");
          await editEndpoint();
          return;
        }
        const key = await vscode.window.showInputBox({
          title: "Autocomplete API key · " + new URL(endpoint).origin,
          prompt: "Stored in VS Code Secret Storage. Submit an empty value to remove it.",
          password: true,
          ignoreFocusOut: true,
        });
        if (key === undefined) return;
        if (key.trim()) await context.secrets.store(secretName(endpoint), key.trim());
        else await context.secrets.delete(secretName(endpoint));
        pending?.abort();
        reported.clear();
        setStatus({ kind: "idle" });
      } catch (error) {
        void vscode.window.showErrorMessage(error instanceof Error ? error.message : String(error));
      }
    }),
    vscode.commands.registerCommand("llamaAio.inlineAutocomplete.toggle", async () => {
      const s = settings();
      if (!s.enabled && !s.endpoint) {
        await editEndpoint();
        return;
      }
      if (!s.enabled && snoozedUntil()) await snooze(0);
      await update("enabled", !s.enabled);
    }),
    vscode.commands.registerCommand("llamaAio.inlineAutocomplete.snooze", async () => {
      const choice = await vscode.window.showQuickPick(
        [
          { label: "30 minutes", minutes: 30 },
          { label: "1 hour", minutes: 60 },
          { label: "Until VS Code restarts", minutes: 24 * 60 },
        ],
        { title: "Snooze Llama AIO autocomplete" }
      );
      if (choice) await snooze(choice.minutes);
    }),
    vscode.commands.registerCommand("llamaAio.inlineAutocomplete.menu", async () => {
      const s = settings();
      const language = vscode.window.activeTextEditor?.document.languageId;
      const languageOff = !!language && s.disabledLanguages.includes(language);
      type Item = vscode.QuickPickItem & { run: () => Thenable<unknown> | Promise<unknown> };
      const items: Item[] = [
        s.enabled
          ? { label: "$(circle-slash) Turn off", run: () => update("enabled", false) }
          : { label: "$(check) Turn on", run: () => vscode.commands.executeCommand("llamaAio.inlineAutocomplete.toggle") },
        ...(s.enabled ? [snoozedUntil()
          ? { label: "$(play) Resume now", run: () => snooze(0) }
          : { label: "$(clock) Snooze…", run: () => vscode.commands.executeCommand("llamaAio.inlineAutocomplete.snooze") }] : []),
        ...(language ? [{
          label: languageOff ? `$(check) Turn on for ${language}` : `$(circle-slash) Turn off for ${language}`,
          run: () => update("disabledLanguages", languageOff
            ? s.disabledLanguages.filter(l => l !== language) : [...s.disabledLanguages, language]),
        }] : []),
        { label: "$(link) Change endpoint…", description: s.endpoint || "not set", run: editEndpoint },
        { label: "$(key) Set API key…", run: () => vscode.commands.executeCommand("llamaAio.setInlineAutocompleteApiKey") },
        { label: "$(gear) Open settings", run: () => vscode.commands.executeCommand("workbench.action.openSettings", CONFIG) },
      ];
      const picked = await vscode.window.showQuickPick(items, { title: "Llama AIO autocomplete" });
      if (picked) await picked.run();
    }),
    vscode.languages.registerInlineCompletionItemProvider(
      [{ scheme: "file" }, { scheme: "vscode-remote" }, { scheme: "untitled" }, { scheme: "vscode-notebook-cell" }],
      {
        async provideInlineCompletionItems(document, position, completionContext, token) {
          const s = settings();
          if (!vscode.workspace.isTrusted || token.isCancellationRequested ||
              completionContext.selectedCompletionInfo || !s.enabled || snoozedUntil() ||
              s.disabledLanguages.includes(document.languageId)) return [];

          pending?.abort();
          const request = new AbortController();
          pending = request;
          const cancellation = token.onCancellationRequested(() => request.abort());
          const version = document.version;
          let endpoint = "";
          let timeout: AbortSignal | undefined;
          try {
            endpoint = completionEndpoint(s.endpoint);
            if (!endpoint) return [];
            if (completionContext.triggerKind === vscode.InlineCompletionTriggerKind.Automatic && s.delayMs) {
              await delay(s.delayMs, undefined, { signal: request.signal });
            }
            if (request.signal.aborted || document.version !== version) return [];

            const offset = document.offsetAt(position);
            const start = prefixWindowStart(offset,
              (from, to) => document.getText(new vscode.Range(document.positionAt(from), document.positionAt(to))));
            const { prefix, suffix } = wholeCharacters(
              document.getText(new vscode.Range(document.positionAt(start), position)),
              document.getText(new vscode.Range(position, document.positionAt(offset + SUFFIX_CHARS))));
            if (!prefix.trim()) return [];
            const apiKey = await context.secrets.get(secretName(endpoint));
            if (request.signal.aborted || document.version !== version) return [];

            setStatus({ kind: "working" });
            const started = Date.now();
            timeout = AbortSignal.timeout(warm.has(endpoint) ? REQUEST_TIMEOUT_MS : COLD_REQUEST_TIMEOUT_MS);
            const reply = await serverJson<{ content?: string }>(joinServerUrl(endpoint, "/infill"), {
              // No stored autocomplete key: fall back to the key registered for this
              // origin, or this machine's own key for a loopback server.
              apiKey: apiKey || undefined,
              signal: AbortSignal.any([request.signal, timeout]),
              body: {
                input_prefix: prefix,
                input_suffix: suffix,
                n_predict: s.maxTokens,
                stop: stopSequences(prefix, suffix),
                temperature: 0.2,
                cache_prompt: true,
                stream: false,
              },
            });
            warm.add(endpoint);
            setStatus({ kind: "ready", ms: Date.now() - started });
            if (request.signal.aborted || document.version !== version || typeof reply?.content !== "string") return [];
            const { text, replaceChars } = shapeSuggestion(reply.content, prefix, suffix);
            if (!text.trim()) return [];

            const end = replaceChars ? document.positionAt(offset + replaceChars) : position;
            return [new vscode.InlineCompletionItem(text, new vscode.Range(position, end))];
          } catch (error) {
            if (request.signal.aborted) {
              // Superseded by a newer keystroke or cancelled by the editor. A newer
              // request shows its own progress; otherwise go back to the last state.
              if (pending === request && status.kind === "working") setStatus(settled);
              return [];
            }
            const problem = autocompleteProblem(error);
            if (problem) {
              report(endpoint || "(endpoint)", problem);
            } else if (timeout?.aborted) {
              setStatus({ kind: "slow" });
            } else if (error instanceof ConnectionError && ["refused", "dns", "tls", "network", "timeout"].includes(error.kind)) {
              setStatus({ kind: "unreachable", message: `The autocomplete server at ${endpoint} is not reachable. ${error.message}` });
            } else if (status.kind === "working") {
              setStatus(settled);
            }
            return [];
          } finally {
            cancellation.dispose();
            if (pending === request) pending = undefined;
          }
        },
      }
    )
  );
  scheduleSnoozeEnd();
  render();
}
