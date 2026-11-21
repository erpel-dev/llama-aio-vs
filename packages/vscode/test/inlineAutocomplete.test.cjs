const assert = require("node:assert/strict");
const http = require("node:http");
const Module = require("node:module");
const { it } = require("node:test");

let config, provider, changeConfiguration;
const warnings = [];
const vscode = {
  window: { showWarningMessage: async (message, ...actions) => { warnings.push({ message, actions }); return undefined; } },
  workspace: {
    isTrusted: true,
    getConfiguration: () => ({ get: name => config[name] }),
    onDidChangeConfiguration: listener => { changeConfiguration = listener; },
  },
  commands: { registerCommand() {}, executeCommand: async () => undefined },
  languages: { registerInlineCompletionItemProvider: (_selector, value) => { provider = value; } },
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
  registerInlineAutocomplete, autocompleteProblem, isLoopbackAddress, stopSequences, wholeCharacters, withoutSuffixOverlap,
} = require("../out-harness/inlineAutocomplete.js");
const core = require("@llama-aio/core");

function document(text) {
  return {
    version: 1,
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
  config = { enabled: true, endpoint: origin + "/coder" };
  registerInlineAutocomplete({
    subscriptions: [],
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
  assert.equal(request.body.input_prefix, left.slice(-4000));
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

it("treats all loopback addresses as this machine, and stays silent when nothing listens", async t => {
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

it("does not repeat the text after the cursor", async t => {
  const fixture = await setup(t, res => res.end(JSON.stringify({ content: "a, b)" })));
  const result = await fixture.invoke(document("foo();"), 4);
  assert.equal(result[0].insertText, "a, b");
});

it("never sends half of a surrogate pair at a window edge", async t => {
  const fixture = await setup(t, res => res.end(JSON.stringify({ content: "z" })));
  const left = "\u{1F600}" + "x".repeat(3999);
  const right = "y".repeat(999) + "\u{1F600}";
  await fixture.invoke(document(left + right), left.length);
  assert.equal(fixture.requests[0].body.input_prefix, "x".repeat(3999));
  assert.equal(fixture.requests[0].body.input_suffix, "y".repeat(999));
});

it("helpers: loopback addresses, stops, overlap, surrogates and problem classes", () => {
  for (const host of ["localhost", "127.0.0.1", "127.0.1.1", "127.255.255.255", "[::1]", "::1"]) {
    assert.equal(isLoopbackAddress(host), true, host);
  }
  for (const host of ["127.example.invalid", "127.0.0.1.nip.io", "box.localhost", "128.0.0.1", "127.0.0.256", "10.0.0.1"]) {
    assert.equal(isLoopbackAddress(host), false, host);
  }
  assert.deepEqual(stopSequences("  const x = ", "1;"), ["\n"]);
  assert.deepEqual(stopSequences("  const x = 1;", "\nnext"), ["\n\n"]);
  assert.deepEqual(stopSequences("a\n    ", "b"), ["\n\n"]);
  assert.equal(withoutSuffixOverlap("bar()", ")"), "bar(", "the existing ) completes the call");
  assert.equal(withoutSuffixOverlap("x", ");"), "x");
  assert.equal(withoutSuffixOverlap(")", ")"), "");
  assert.equal(withoutSuffixOverlap("\n    return a + b;\n}\nconsole.log(1);\n", "\n}\n"), "\n    return a + b;");
  assert.equal(withoutSuffixOverlap("\n    if (x) {\n      y();\n    }", "\n}\n"), "\n    if (x) {\n      y();\n    }",
    "a closing brace at another depth is kept");
  assert.equal(withoutSuffixOverlap("\nreturn 1;", ""), "\nreturn 1;");
  assert.deepEqual(wholeCharacters("\uDE00ab", "cd\uD83D"), { prefix: "ab", suffix: "cd" });
  assert.deepEqual(wholeCharacters("\u{1F600}", "\u{1F600}"), { prefix: "\u{1F600}", suffix: "\u{1F600}" });
  assert.equal(autocompleteProblem(new core.ConnectionError("timeout", "slow")), undefined);
  assert.equal(autocompleteProblem(new core.ConnectionError("refused", "no")), undefined);
  assert.equal(autocompleteProblem(new Error("x")), undefined);
  assert.equal(autocompleteProblem(new core.ConnectionError("http", "x", 501)).action, "Open Settings");
});

it("supersedes automatic requests and stops in-flight requests when settings change", async t => {
  let onRequest;
  const received = new Promise(resolve => { onRequest = resolve; });
  const fixture = await setup(t, () => onRequest());
  const first = fixture.invoke(document("old"), 3, 1);
  const latest = fixture.invoke(document("new"), 3, 1);
  assert.deepEqual(await first, []);
  await received;
  changeConfiguration({ affectsConfiguration: () => true });
  assert.deepEqual(await latest, []);
  assert.equal(fixture.requests.length, 1);
  assert.equal(fixture.requests[0].body.input_prefix, "new");
});

it("drops a completion when the document changes in flight", async t => {
  const doc = document("code");
  const fixture = await setup(t, res => {
    doc.version++;
    res.end(JSON.stringify({ content: " stale" }));
  });
  assert.deepEqual(await fixture.invoke(doc, 4), []);
});
