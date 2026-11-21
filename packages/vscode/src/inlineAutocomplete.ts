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
const DEBOUNCE_MS = 120;
const REQUEST_TIMEOUT_MS = 2500;

/** A setting the user has to change: reported once instead of failing silently on every keystroke. */
class SetupError extends Error {}

function completionEndpoint(): string {
  const configured = vscode.workspace.getConfiguration(CONFIG).get<string>("endpoint")?.trim();
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

const SET_KEY = "Set API Key";
const OPEN_SETTINGS = "Open Settings";

/**
 * What the user must fix, for errors a retry on the next keystroke cannot fix.
 * Slow, refused or cancelled requests stay silent: autocomplete is opportunistic.
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
 * Where generation stops. In the middle of a line, finish that line. At the end
 * of a line, or on a blank one, the model's first token is usually a newline, so
 * allow a short block and stop before the next paragraph.
 */
export function stopSequences(prefix: string, suffix: string): string[] {
  const linePrefix = prefix.slice(prefix.lastIndexOf("\n") + 1);
  const lineSuffix = suffix.split("\n", 1)[0];
  return linePrefix.trim() && lineSuffix.trim() ? ["\n"] : ["\n\n"];
}

/**
 * Remove the end of a suggestion that repeats the text after the cursor, so
 * accepting `a, b)` at `foo(|)` gives `foo(a, b)` and not `foo(a, b))`.
 */
export function withoutSuffixOverlap(content: string, suffix: string): string {
  // A block suggestion tends to run on into the code that already follows the
  // cursor (the closing brace and beyond). Cut it before the first line that
  // repeats the next non-blank line after the cursor, indentation included, so
  // an inner brace at another depth is kept.
  const nextLine = suffix.split("\n").slice(1).find(line => line.trim())?.trimEnd();
  if (nextLine && content.includes("\n")) {
    const lines = content.split("\n");
    const at = lines.findIndex((line, i) => i > 0 && line.trimEnd() === nextLine);
    if (at > 0) content = lines.slice(0, at).join("\n").replace(/\s+$/, "");
  }
  const lineSuffix = suffix.split("\n", 1)[0];
  for (let k = Math.min(content.length, lineSuffix.length); k > 0; k--) {
    if (content.endsWith(lineSuffix.slice(0, k))) return content.slice(0, -k);
  }
  return content;
}

function secretName(endpoint: string): string {
  return "llama-aio.inlineAutocomplete.apiKey:" + new URL(endpoint).origin;
}

/** Supply native VS Code ghost text from a FIM model without changing the chat target. */
export function registerInlineAutocomplete(context: vscode.ExtensionContext): void {
  let pending: AbortController | undefined;
  /** Problems already shown, per endpoint, until the settings or the key change. */
  const reported = new Set<string>();
  const report = (endpoint: string, problem: { message: string; action: string }) => {
    const key = endpoint + "\u0000" + problem.message;
    if (reported.has(key)) return;
    reported.add(key);
    void vscode.window.showWarningMessage("Llama AIO autocomplete: " + problem.message, problem.action).then(choice => {
      if (choice === SET_KEY) void vscode.commands.executeCommand("llamaAio.setInlineAutocompleteApiKey");
      else if (choice === OPEN_SETTINGS) void vscode.commands.executeCommand("workbench.action.openSettings", CONFIG);
    });
  };

  context.subscriptions.push(
    { dispose: () => pending?.abort() },
    vscode.workspace.onDidChangeConfiguration(event => {
      if (event.affectsConfiguration(CONFIG)) {
        pending?.abort();
        reported.clear();
      }
    }),
    vscode.commands.registerCommand("llamaAio.setInlineAutocompleteApiKey", async () => {
      try {
        const endpoint = completionEndpoint();
        if (!endpoint) {
          void vscode.window.showInformationMessage("Set Llama AIO’s inline autocomplete endpoint first.");
          await vscode.commands.executeCommand("workbench.action.openSettings", CONFIG + ".endpoint");
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
      } catch (error) {
        void vscode.window.showErrorMessage(error instanceof Error ? error.message : String(error));
      }
    }),
    vscode.languages.registerInlineCompletionItemProvider([{ scheme: "file" }, { scheme: "vscode-remote" }], {
      async provideInlineCompletionItems(document, position, completionContext, token) {
        if (!vscode.workspace.isTrusted || token.isCancellationRequested ||
            completionContext.selectedCompletionInfo ||
            !vscode.workspace.getConfiguration(CONFIG).get<boolean>("enabled")) return [];

        pending?.abort();
        const request = new AbortController();
        pending = request;
        const cancellation = token.onCancellationRequested(() => request.abort());
        const version = document.version;
        let endpoint = "";
        try {
          endpoint = completionEndpoint();
          if (!endpoint) return [];
          if (completionContext.triggerKind === vscode.InlineCompletionTriggerKind.Automatic) {
            await delay(DEBOUNCE_MS, undefined, { signal: request.signal });
          }
          if (request.signal.aborted || document.version !== version) return [];

          const offset = document.offsetAt(position);
          const { prefix, suffix } = wholeCharacters(
            document.getText(new vscode.Range(document.positionAt(Math.max(0, offset - PREFIX_CHARS)), position)),
            document.getText(new vscode.Range(position, document.positionAt(offset + SUFFIX_CHARS))));
          if (!prefix.trim()) return [];
          const apiKey = await context.secrets.get(secretName(endpoint));
          if (request.signal.aborted || document.version !== version) return [];

          const reply = await serverJson<{ content?: string }>(joinServerUrl(endpoint, "/infill"), {
            // No stored autocomplete key: fall back to the key registered for this
            // origin, or this machine's own key for a loopback server.
            apiKey: apiKey || undefined,
            signal: AbortSignal.any([request.signal, AbortSignal.timeout(REQUEST_TIMEOUT_MS)]),
            body: {
              input_prefix: prefix,
              input_suffix: suffix,
              n_predict: 32,
              stop: stopSequences(prefix, suffix),
              temperature: 0.2,
              cache_prompt: true,
              stream: false,
            },
          });
          if (request.signal.aborted || document.version !== version || typeof reply?.content !== "string") return [];
          const text = withoutSuffixOverlap(reply.content, suffix);
          if (!text.trim()) return [];

          return [new vscode.InlineCompletionItem(text, new vscode.Range(position, position))];
        } catch (error) {
          // A slow, refused or cancelled request stays silent. A setting or key
          // the user must fix is shown once, not on every keystroke.
          const problem = request.signal.aborted ? undefined : autocompleteProblem(error);
          if (problem) report(endpoint || "(endpoint)", problem);
          return [];
        } finally {
          cancellation.dispose();
          if (pending === request) pending = undefined;
        }
      },
    })
  );
}
