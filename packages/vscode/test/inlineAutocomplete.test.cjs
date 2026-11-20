const assert = require("node:assert/strict");
const http = require("node:http");
const Module = require("node:module");
const { it } = require("node:test");

let config, provider, changeConfiguration;
const vscode = {
  workspace: {
    isTrusted: true,
    getConfiguration: () => ({ get: name => config[name] }),
    onDidChangeConfiguration: listener => { changeConfiguration = listener; },
  },
  commands: { registerCommand() {} },
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
const { registerInlineAutocomplete } = require("../out-harness/inlineAutocomplete.js");

function document(text) {
  return {
    version: 1,
    offsetAt: position => position,
    positionAt: offset => Math.min(offset, text.length),
    getText: range => text.slice(range.start, range.end),
  };
}

async function setup(t, reply) {
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
      return name.endsWith(origin) ? "test-key" : undefined;
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
