const assert = require("node:assert/strict");
const http = require("node:http");
const Module = require("node:module");
const { it } = require("node:test");

let config, provider, selector, changeConfiguration, statusItem, quickPick, inputBox, activeEditor;
const warnings = [];
const commands = new Map();
const vscode = {
  window: {
    showWarningMessage: async (message, ...actions) => { warnings.push({ message, actions }); return undefined; },
    showInformationMessage: async () => undefined,
    showErrorMessage: async () => undefined,
    showQuickPick: async (items, options) => (quickPick ? quickPick(items, options) : undefined),
    showInputBox: async options => (inputBox ? inputBox(options) : undefined),
    createStatusBarItem: () => {
      statusItem = { text: "", tooltip: "", visible: false, show() { this.visible = true; }, hide() { this.visible = false; }, dispose() {} };
      return statusItem;
    },
    get activeTextEditor() { return activeEditor; },
  },
  workspace: {
    isTrusted: true,
    getConfiguration: () => ({
      get: name => config[name],
      update: async (name, value) => {
        config[name] = value;
        changeConfiguration?.({ affectsConfiguration: () => true });
      },
    }),
    onDidChangeConfiguration: listener => { changeConfiguration = listener; },
  },
  commands: {
    registerCommand: (name, run) => { commands.set(name, run); },
    executeCommand: async (name, ...args) => commands.get(name)?.(...args),
  },
  languages: { registerInlineCompletionItemProvider: (value, item) => { selector = value; provider = item; } },
  StatusBarAlignment: { Left: 1, Right: 2 },
  ConfigurationTarget: { Global: 1 },
  ThemeColor: class { constructor(id) { this.id = id; } },
  InlineCompletionTriggerKind: { Invoke: 0, Automatic: 1 },
  InlineCompletionItem: class { constructor(insertText, range) { Object.assign(this, { insertText, range }); } },
  Range: class { constructor(start, end) { Object.assign(this, { start, end }); } },
};

const resolveFilename = Module._resolveFilename;
Module._resolveFilename = function (name, ...args) {
  return name === "vscode" ? "vscode" : resolveFilename.call(this, name, ...args);
};
require.cache.vscode = { id: "vscode", filename: "vscode", loaded: true, exports: vscode };
const {
  registerInlineAutocomplete, autocompleteProblem, isLoopbackAddress, isClosingOnly, prefixWindowStart, shapeSuggestion,
  stopSequences, wholeCharacters,
} = require("../out-harness/inlineAutocomplete.js");
const core = require("@llama-aio/core");

function document(text, languageId = "javascript") {
  return {
    version: 1,
    languageId,
    offsetAt: position => position,
    positionAt: offset => Math.min(offset, text.length),
    getText: range => text.slice(range.start, range.end),
  };
}

async function setup(t, reply, storedKey = "test-key") {
  warnings.length = 0;
  const requests = [];
  const server = http.createServer(async (req, res) => {
    let json = "";
    for await (const chunk of req) json += chunk;
    requests.push({ url: req.url, key: req.headers.authorization, body: JSON.parse(json) });
    reply(res);
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  t.after(() => new Promise(resolve => server.close(resolve)));
  const origin = `http://127.0.0.1:${server.address().port}`;
  const lookups = [];
  config = { enabled: true, endpoint: origin + "/coder", delayMs: 0 };
  quickPick = inputBox = activeEditor = undefined;
  const memento = new Map();
  registerInlineAutocomplete({
    subscriptions: [],
    globalState: { get: key => memento.get(key), update: async (key, value) => { memento.set(key, value); } },
    secrets: { get: async name => {
      lookups.push(name);
      return name.endsWith(origin) ? storedKey : undefined;
    } },
  });

  return {
    requests, lookups, origin,
    invoke: (doc, cursor, triggerKind = 0) => provider.provideInlineCompletionItems(doc, cursor, { triggerKind }, {
      isCancellationRequested: false,
      onCancellationRequested: () => ({ dispose() {} }),
    }),
  };
}

it("sends a bounded prefix and suffix to the authenticated infill endpoint and inserts at the cursor", async t => {
  const fixture = await setup(t, res => res.end(JSON.stringify({ content: "return a + b;" })));
  const left = "x".repeat(5000);
  const right = "y".repeat(2000);
  const result = await fixture.invoke(document(left + right), left.length);
  assert.equal(result[0].insertText, "return a + b;");
  assert.equal(result[0].range.start, left.length);
  assert.equal(result[0].range.end, left.length);
  const [request] = fixture.requests;
  assert.equal(request.url, "/coder/infill");
  assert.equal(request.key, "Bearer test-key");
  // One 5,000-character line: the window starts on the 1,024-character grid (0 here), not 4,000 back.
  assert.equal(request.body.input_prefix, left);
  assert.equal(request.body.input_suffix, right.slice(0, 1000));
  assert.equal(request.body.n_predict, 32);
  assert.deepEqual(request.body.stop, ["\n"]);
  assert.deepEqual(fixture.lookups, ["llama-aio.inlineAutocomplete.apiKey:" + fixture.origin]);
});

it("skips requests while disabled and rejects external plain HTTP before reading a key", async t => {
  const fixture = await setup(t, () => assert.fail("unexpected completion request"));
  config.enabled = false;
  assert.deepEqual(await fixture.invoke(document("code"), 4), []);
  config.enabled = true;
  config.endpoint = "http://127.example.invalid";
  assert.deepEqual(await fixture.invoke(document("code"), 4), []);
  assert.deepEqual(fixture.requests, []);
  assert.deepEqual(fixture.lookups, []);
});

it("reports a setup problem once per endpoint, and again after the settings change", async t => {
  const fixture = await setup(t, () => assert.fail("unexpected completion request"));
  config.endpoint = "http://gpu-box.lan:8012";
  await fixture.invoke(document("code"), 4);
  await fixture.invoke(document("code"), 4);
  assert.equal(warnings.length, 1);
  assert.match(warnings[0].message, /Use HTTPS for an autocomplete server outside this machine/);
  assert.deepEqual(warnings[0].actions, ["Open Settings"]);
  changeConfiguration({ affectsConfiguration: () => true });
  await fixture.invoke(document("code"), 4);
  assert.equal(warnings.length, 2);
  config.endpoint = "http://[bad";
  await fixture.invoke(document("code"), 4);
  assert.match(warnings[2].message, /is not a valid URL/);
});

it("treats all loopback addresses as this machine, with no popup when nothing listens", async t => {
  const fixture = await setup(t, () => assert.fail("unexpected completion request"));
  config.endpoint = "http://127.0.1.1:9";
  assert.deepEqual(await fixture.invoke(document("code"), 4), []);
  assert.deepEqual(warnings, [], "a refused connection is not a setup problem");
});

it("reports a rejected key and a server without /infill once, with the matching action", async t => {
  let status = 401;
  const fixture = await setup(t, res => { res.statusCode = status; res.end("{}"); });
  await fixture.invoke(document("code"), 4);
  await fixture.invoke(document("code"), 4);
  assert.equal(warnings.length, 1);
  assert.match(warnings[0].message, /rejected the request \(HTTP 401\)/);
  assert.deepEqual(warnings[0].actions, ["Set API Key"]);
  status = 404;
  await fixture.invoke(document("code"), 4);
  assert.equal(warnings.length, 2);
  assert.match(warnings[1].message, /HTTP 404 for \/infill/);
  status = 500;
  await fixture.invoke(document("code"), 4);
  assert.equal(warnings.length, 2, "other server errors stay silent");
});

it("falls back to the key registered for the origin when no autocomplete key is stored", async t => {
  const fixture = await setup(t, res => res.end(JSON.stringify({ content: "x" })), null);
  core.setServerApiKeyProvider(origin => (origin === fixture.origin ? "registered-key" : undefined));
  t.after(() => core.setServerApiKeyProvider(undefined));
  await fixture.invoke(document("code"), 4);
  assert.equal(fixture.requests[0].key, "Bearer registered-key");
});

it("lets a suggestion start a new line at the end of a line", async t => {
  // The reply real llama-server gave for this prompt (Qwen3.5-2B): it runs on past the existing brace.
  const fixture = await setup(t, res => res.end(JSON.stringify({ content: "\n  return a + b;\n}\nconsole.log(add(2, 3));\n" })));
  const text = "function add(a, b) {\n}";
  const result = await fixture.invoke(document(text), "function add(a, b) {".length);
  assert.deepEqual(fixture.requests[0].body.stop, ["\n\n"]);
  assert.equal(result[0].insertText, "\n  return a + b;");
});

it("replaces auto-closed characters the suggestion already contains", async t => {
  // What real llama-server (Qwen2.5-Coder 0.5B) returned at `add(|)` in VS Code.
  const fixture = await setup(t, res => res.end(JSON.stringify({ content: "2, 3);" })));
  const result = await fixture.invoke(document("const total = add()"), 18);
  assert.equal(result[0].insertText, "2, 3);");
  assert.deepEqual([result[0].range.start, result[0].range.end], [18, 19], "the existing ) is replaced");
});

it("keeps auto-closed characters the suggestion does not repeat", async t => {
  const fixture = await setup(t, res => res.end(JSON.stringify({ content: "2, 3" })));
  const result = await fixture.invoke(document("const total = add()"), 18);
  assert.equal(result[0].insertText, "2, 3");
  assert.deepEqual([result[0].range.start, result[0].range.end], [18, 18]);
});

it("suggests a body after an auto-closed brace, with or without the brace in the reply", async t => {
  // What real llama-server returned at `function sub(a, b) {|}` in VS Code: no closing brace.
  const bare = await setup(t, res => res.end(JSON.stringify({ content: "\n  return a - b;\n" })));
  const shown = await bare.invoke(document("function sub(a, b) {}\n"), 20);
  assert.equal(shown[0].insertText, "\n  return a - b;\n}", "the existing } closes the block on its own line");
  assert.deepEqual([shown[0].range.start, shown[0].range.end], [20, 21]);
  const fixture = await setup(t, res => res.end(JSON.stringify({ content: "\n  return a - b;\n}" })));
  const text = "function sub(a, b) {}\n";
  const result = await fixture.invoke(document(text), 20);
  assert.deepEqual(fixture.requests[0].body.stop, ["\n\n"], "only a } after the cursor counts as the end of the line");
  assert.equal(result[0].insertText, "\n  return a - b;\n}");
  assert.deepEqual([result[0].range.start, result[0].range.end], [20, 21]);
});

it("stops a block suggestion at code outside the block", async t => {
  // What real llama-server returned on the blank line inside `def area(r):`.
  const fixture = await setup(t, res => res.end(JSON.stringify({ content: "return 3.14*r*r\nr=int(input(\"enter the radius\"))\nprint(area(r))" })));
  const result = await fixture.invoke(document("def area(r):\n    \n", "python"), 17);
  assert.equal(result[0].insertText, "return 3.14*r*r");
});

it("offers suggestions in untitled files and notebook cells", () => {
  assert.deepEqual(selector.map(s => s.scheme), ["file", "vscode-remote", "untitled", "vscode-notebook-cell"]);
});

it("skips languages turned off, and everything while snoozed", async t => {
  const fixture = await setup(t, res => res.end(JSON.stringify({ content: "x" })));
  config.disabledLanguages = ["markdown"];
  assert.deepEqual(await fixture.invoke(document("# Title", "markdown"), 7), []);
  assert.equal(fixture.requests.length, 0);
  quickPick = items => items.find(i => i.minutes === 30);
  await vscode.commands.executeCommand("llamaAio.inlineAutocomplete.snooze");
  assert.match(statusItem.text, /Snoozed 30 min/);
  assert.deepEqual(await fixture.invoke(document("code"), 4), []);
  assert.equal(fixture.requests.length, 0);
  quickPick = items => items.find(i => /Resume now/.test(i.label));
  await vscode.commands.executeCommand("llamaAio.inlineAutocomplete.menu");
  await fixture.invoke(document("code"), 4);
  assert.equal(fixture.requests.length, 1, "resumed");
});

it("shows the state in the status bar: ready, not reachable, off", async t => {
  const fixture = await setup(t, res => res.end(JSON.stringify({ content: "x" })));
  assert.equal(statusItem.visible, true);
  await fixture.invoke(document("code"), 4);
  assert.match(statusItem.text, /Autocomplete/);
  assert.match(statusItem.tooltip, /Last suggestion took \d+ ms/);
  config.endpoint = "http://127.0.0.1:9";
  changeConfiguration({ affectsConfiguration: () => true });
  await fixture.invoke(document("code"), 4);
  assert.match(statusItem.text, /\$\(warning\)/);
  assert.match(statusItem.tooltip, /not reachable/);
  assert.deepEqual(warnings, [], "a stopped server is shown in the status bar, not as a popup");
  await vscode.commands.executeCommand("llamaAio.inlineAutocomplete.toggle");
  assert.equal(config.enabled, false);
  assert.match(statusItem.text, /Off/);
  config.endpoint = "";
  changeConfiguration({ affectsConfiguration: () => true });
  assert.equal(statusItem.visible, false, "hidden for people who never set autocomplete up");
});

it("turning on without an endpoint asks for one and validates it", async t => {
  await setup(t, () => assert.fail("unexpected request"));
  config = { enabled: false, endpoint: "" };
  let validate;
  inputBox = options => { validate = options.validateInput; return "http://127.0.0.1:8012"; };
  await vscode.commands.executeCommand("llamaAio.inlineAutocomplete.toggle");
  assert.equal(config.endpoint, "http://127.0.0.1:8012");
  assert.equal(config.enabled, true);
  assert.match(validate("http://gpu-box.lan:8012"), /Use HTTPS/);
  assert.equal(validate("https://gpu-box.lan"), undefined);
});

it("the menu turns the current language off and on", async t => {
  await setup(t, () => undefined);
  activeEditor = { document: { languageId: "markdown" } };
  quickPick = items => items.find(i => /Turn off for markdown/.test(i.label));
  await vscode.commands.executeCommand("llamaAio.inlineAutocomplete.menu");
  assert.deepEqual(config.disabledLanguages, ["markdown"]);
  quickPick = items => items.find(i => /Turn on for markdown/.test(i.label));
  await vscode.commands.executeCommand("llamaAio.inlineAutocomplete.menu");
  assert.deepEqual(config.disabledLanguages, []);
});

it("never sends half of a surrogate pair at a window edge", async t => {
  const fixture = await setup(t, res => res.end(JSON.stringify({ content: "z" })));
  // A long line keeps the grid offset (1,024), which here falls inside the emoji at 1,023–1,024.
  const left = "a".repeat(1023) + "\u{1F600}" + "x".repeat(5000);
  const right = "y".repeat(999) + "\u{1F600}";
  await fixture.invoke(document(left + right), left.length);
  assert.equal(fixture.requests[0].body.input_prefix, "x".repeat(5000));
  assert.equal(fixture.requests[0].body.input_suffix, "y".repeat(999));
});

it("helpers: loopback addresses, window start, stops, shaping, surrogates and problem classes", () => {
  for (const host of ["localhost", "127.0.0.1", "127.0.1.1", "127.255.255.255", "[::1]", "::1"]) {
    assert.equal(isLoopbackAddress(host), true, host);
  }
  for (const host of ["127.example.invalid", "127.0.0.1.nip.io", "box.localhost", "128.0.0.1", "127.0.0.256", "10.0.0.1"]) {
    assert.equal(isLoopbackAddress(host), false, host);
  }
  // The window start stays put while typing, and lands on a line start.
  const code = Array.from({ length: 400 }, (_, i) => `const value${i} = ${i};`).join("\n");
  const between = (from, to) => code.slice(from, to);
  const starts = new Set();
  for (let typed = 0; typed < 40; typed++) starts.add(prefixWindowStart(code.length - 200 + typed, between));
  assert.equal(starts.size, 1, "the same start for 40 keystrokes");
  const [start] = starts;
  assert.equal(code[start - 1], "\n");
  assert.ok(code.length - 200 - start >= 4000 && code.length - 200 - start < 4000 + 1024 + 512);
  assert.equal(prefixWindowStart(3000, between), 0);
  assert.equal(isClosingOnly(");"), true);
  assert.equal(isClosingOnly(" } "), true);
  assert.equal(isClosingOnly("x)"), false);
  assert.deepEqual(stopSequences("  const x = ", "1;"), ["\n"]);
  assert.deepEqual(stopSequences("  const x = add(", ");"), ["\n\n"]);
  assert.deepEqual(stopSequences("  const x = 1;", "\nnext"), ["\n\n"]);
  assert.deepEqual(stopSequences("a\n    ", "b"), ["\n\n"]);
  assert.deepEqual(shapeSuggestion("bar()", "foo(", ")"), { text: "bar()", replaceChars: 1 });
  assert.deepEqual(shapeSuggestion("x", "foo(", ");"), { text: "x", replaceChars: 0 });
  assert.deepEqual(shapeSuggestion("a, b)", "foo(", ") + c"), { text: "a, b", replaceChars: 0 }, "not closing-only: trim the overlap");
  assert.deepEqual(shapeSuggestion("\n    return a + b;\n}\nconsole.log(1);\n", "function add(a, b) {", "\n}\n"),
    { text: "\n    return a + b;", replaceChars: 0 });
  assert.deepEqual(shapeSuggestion("\n    if (x) {\n      y();\n    }", "  function f() {", "\n  }\n"),
    { text: "\n    if (x) {\n      y();\n    }", replaceChars: 0 }, "a closing brace at another depth is kept");
  assert.deepEqual(shapeSuggestion("\nreturn 1;", "x", ""), { text: "\nreturn 1;", replaceChars: 0 });
  assert.deepEqual(shapeSuggestion("\n    y();\n", "  if (x) {", "}"), { text: "\n    y();\n  }", replaceChars: 1 },
    "the closing line uses the cursor line's indentation");
  // What real llama-server returned for `  norm() {|}` inside a class.
  assert.deepEqual(shapeSuggestion("\n    return Math.sqrt(x);\n  }\n", "  norm() {", "}"),
    { text: "\n    return Math.sqrt(x);\n  }", replaceChars: 1 }, "no blank line after the replaced brace");
  assert.deepEqual(shapeSuggestion("\n\n", "  if (x) {", "}"), { text: "", replaceChars: 0 });
  assert.deepEqual(wholeCharacters("\uDE00ab", "cd\uD83D"), { prefix: "ab", suffix: "cd" });
  assert.deepEqual(wholeCharacters("\u{1F600}", "\u{1F600}"), { prefix: "\u{1F600}", suffix: "\u{1F600}" });
  assert.equal(autocompleteProblem(new core.ConnectionError("timeout", "slow")), undefined);
  assert.equal(autocompleteProblem(new core.ConnectionError("refused", "no")), undefined);
  assert.equal(autocompleteProblem(new Error("x")), undefined);
  assert.equal(autocompleteProblem(new core.ConnectionError("http", "x", 501)).action, "Open Settings");
});
