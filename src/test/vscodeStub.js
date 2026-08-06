/**
 * Minimal stand-in for the `vscode` module.
 *
 * Extension-host modules import `vscode` at load time, which would make them
 * unrequirable from `node --test`. Only the surface touched during module
 * evaluation needs to be real; anything called inside a function can stay a
 * stub, and a test that reaches for missing API will fail loudly rather than
 * silently pass.
 */

class EventEmitter {
  constructor() {
    this.listeners = new Set();
    this.event = (listener) => {
      this.listeners.add(listener);
      return { dispose: () => this.listeners.delete(listener) };
    };
  }
  fire(value) {
    for (const listener of [...this.listeners]) {
      listener(value);
    }
  }
  dispose() {
    this.listeners.clear();
  }
}

class CancellationError extends Error {
  constructor() {
    super("Canceled");
    this.name = "Canceled";
  }
}

const notImplemented = (name) => () => {
  throw new Error(`vscode.${name} is not available in unit tests`);
};

module.exports = {
  EventEmitter,
  CancellationError,
  Disposable: { from: (...items) => ({ dispose: () => items.forEach((i) => i.dispose?.()) }) },
  Uri: {
    file: (p) => ({ scheme: "file", fsPath: p, path: p, toString: () => `file://${p}` }),
    parse: (s) => ({ scheme: "https", fsPath: s, path: s, toString: () => s }),
    joinPath: (base, ...parts) => ({ fsPath: [base.fsPath, ...parts].join("/") }),
  },
  ConfigurationTarget: { Global: 1, Workspace: 2, WorkspaceFolder: 3 },
  ProgressLocation: { SourceControl: 1, Window: 10, Notification: 15 },
  StatusBarAlignment: { Left: 1, Right: 2 },
  ViewColumn: { One: 1, Two: 2, Beside: -2 },
  ExtensionMode: { Production: 1, Development: 2, Test: 3 },
  LanguageModelChatMessageRole: { User: 1, Assistant: 2 },
  LanguageModelChatMessage: {
    User: (content) => ({ role: 1, content }),
    Assistant: (content) => ({ role: 2, content }),
  },
  LanguageModelTextPart: class LanguageModelTextPart {
    constructor(value) {
      this.value = value;
    }
  },
  LanguageModelToolCallPart: class LanguageModelToolCallPart {
    constructor(callId, name, input) {
      Object.assign(this, { callId, name, input });
    }
  },
  LanguageModelToolResultPart: class LanguageModelToolResultPart {
    constructor(callId, content) {
      Object.assign(this, { callId, content });
    }
  },
  ThemeIcon: class ThemeIcon {
    constructor(id) {
      this.id = id;
    }
  },
  window: {
    showInformationMessage: notImplemented("window.showInformationMessage"),
    showWarningMessage: notImplemented("window.showWarningMessage"),
    showErrorMessage: notImplemented("window.showErrorMessage"),
    showQuickPick: notImplemented("window.showQuickPick"),
    createOutputChannel: () => ({
      appendLine() {},
      append() {},
      show() {},
      dispose() {},
    }),
    createStatusBarItem: () => ({ show() {}, hide() {}, dispose() {} }),
    withProgress: (_opts, task) => task({ report() {} }, { isCancellationRequested: false }),
  },
  workspace: {
    getConfiguration: () => ({ get: (_k, fallback) => fallback, update: async () => undefined }),
    onDidChangeConfiguration: () => ({ dispose() {} }),
    fs: {},
  },
  commands: {
    registerCommand: () => ({ dispose() {} }),
    executeCommand: notImplemented("commands.executeCommand"),
  },
  lm: {
    registerLanguageModelChatProvider: () => ({ dispose() {} }),
  },
  extensions: { getExtension: () => undefined },
  env: { openExternal: notImplemented("env.openExternal"), clipboard: { writeText: async () => {} } },
};
