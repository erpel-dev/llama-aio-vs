import { setTimeout as delay } from "node:timers/promises";
import {
  isLoopbackOrigin,
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

function completionEndpoint(): string {
  const configured = vscode.workspace.getConfiguration(CONFIG).get<string>("endpoint")?.trim();
  if (!configured) return "";
  const endpoint = normalizeServerUrl(configured);
  const url = new URL(endpoint);
  if (url.protocol !== "https:" && !isLoopbackOrigin(url.origin)) {
    throw new Error("Use HTTPS for an autocomplete server outside this machine.");
  }
  return endpoint;
}

function secretName(endpoint: string): string {
  return "llama-aio.inlineAutocomplete.apiKey:" + new URL(endpoint).origin;
}

/** Supply native VS Code ghost text from a FIM model without changing the chat target. */
export function registerInlineAutocomplete(context: vscode.ExtensionContext): void {
  let pending: AbortController | undefined;

  context.subscriptions.push(
    { dispose: () => pending?.abort() },
    vscode.workspace.onDidChangeConfiguration(event => {
      if (event.affectsConfiguration(CONFIG)) pending?.abort();
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
        try {
          const endpoint = completionEndpoint();
          if (!endpoint) return [];
          if (completionContext.triggerKind === vscode.InlineCompletionTriggerKind.Automatic) {
            await delay(DEBOUNCE_MS, undefined, { signal: request.signal });
          }
          if (request.signal.aborted || document.version !== version) return [];

          const offset = document.offsetAt(position);
          const prefix = document.getText(new vscode.Range(document.positionAt(Math.max(0, offset - PREFIX_CHARS)), position));
          if (!prefix.trim()) return [];
          const suffix = document.getText(new vscode.Range(position, document.positionAt(offset + SUFFIX_CHARS)));
          const apiKey = await context.secrets.get(secretName(endpoint));
          if (request.signal.aborted || document.version !== version) return [];

          const reply = await serverJson<{ content?: string }>(joinServerUrl(endpoint, "/infill"), {
            apiKey: apiKey ?? "",
            signal: AbortSignal.any([request.signal, AbortSignal.timeout(REQUEST_TIMEOUT_MS)]),
            body: {
              input_prefix: prefix,
              input_suffix: suffix,
              n_predict: 32,
              // Finish the current line quickly. On an indented blank line,
              // allow a short block but stop before the next paragraph.
              stop: [prefix.slice(prefix.lastIndexOf("\n") + 1).trim() ? "\n" : "\n\n"],
              temperature: 0.2,
              cache_prompt: true,
              stream: false,
            },
          });
          if (request.signal.aborted || document.version !== version ||
              typeof reply?.content !== "string" || !reply.content.trim()) return [];

          return [new vscode.InlineCompletionItem(reply.content, new vscode.Range(position, position))];
        } catch {
          // Autocomplete is opportunistic. A slow or unavailable server must
          // not interrupt typing or prompt for credentials on each keystroke.
          return [];
        } finally {
          cancellation.dispose();
          if (pending === request) pending = undefined;
        }
      },
    })
  );
}
