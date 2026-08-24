import * as vscode from "vscode";
import {
  lookupWikipedia,
  wikipediaQueryFromInput,
  WIKIPEDIA_LOOKUP_TOOL_NAME,
  type SettingsStore,
} from "@llama-aio/core";

export const WIKIPEDIA_LOOKUP_CONTEXT_KEY = "llamaAio.wikipediaLookup";

export function syncWikipediaLookupContext(store: SettingsStore): void {
  void vscode.commands.executeCommand(
    "setContext",
    WIKIPEDIA_LOOKUP_CONTEXT_KEY,
    store.isWikipediaLookupEnabled()
  );
}

function abortSignalFrom(token: vscode.CancellationToken): AbortSignal {
  const ac = new AbortController();
  if (token.isCancellationRequested) {
    ac.abort();
  } else {
    token.onCancellationRequested(() => ac.abort());
  }
  return ac.signal;
}

export class WikipediaLookupTool implements vscode.LanguageModelTool<{ query?: string }> {
  constructor(private readonly store: SettingsStore) {}

  prepareInvocation(
    options: vscode.LanguageModelToolInvocationPrepareOptions<{ query?: string }>,
    _token: vscode.CancellationToken
  ): vscode.PreparedToolInvocation {
    const q = wikipediaQueryFromInput(options.input) || "Wikipedia";
    return {
      invocationMessage: `Looking up Wikipedia: ${q}`,
    };
  }

  async invoke(
    options: vscode.LanguageModelToolInvocationOptions<{ query?: string }>,
    token: vscode.CancellationToken
  ): Promise<vscode.LanguageModelToolResult> {
    if (!this.store.isWikipediaLookupEnabled()) {
      return new vscode.LanguageModelToolResult([
        new vscode.LanguageModelTextPart(
          "Wikipedia lookup is off. Enable it under Prompt replacements in Llama AIO settings."
        ),
      ]);
    }
    const query = wikipediaQueryFromInput(options.input);
    const text = await lookupWikipedia(query, { signal: abortSignalFrom(token) });
    return new vscode.LanguageModelToolResult([new vscode.LanguageModelTextPart(text)]);
  }
}

export { WIKIPEDIA_LOOKUP_TOOL_NAME };
