const assert = require('node:assert/strict');
const { it } = require('node:test');
const path = require('node:path');
const vm = require('node:vm');
const { buildSync } = require('esbuild');

const source = buildSync({ entryPoints: [path.join(__dirname, '../src/webview/format.ts')], bundle: true,
  platform: 'node', format: 'cjs', write: false }).outputFiles[0].text;
const context = vm.createContext({ module: { exports: {} } });
vm.runInContext(source, context);
const { friendlyNetworkError } = context.module.exports;

it('rate limits never echo the response body or the IP address', () => {
  const raw = 'HF HTTP 429: {"error":"We had to rate limit your IP (160.79.106.23). To continue..."}';
  const msg = friendlyNetworkError(raw, 'huggingface');
  assert.match(msg, /rate-limiting/);
  assert.doesNotMatch(msg, /160\.79|\{|error/);
});

it('a blocked GitHub download points at the archive and PATH alternatives', () => {
  const msg = friendlyNetworkError('HTTP 403 resolving latest llama.cpp nightly from https://github.com/ggml-org/llama.cpp/releases', 'github');
  assert.match(msg, /Install from archive/);
  assert.match(msg, /PATH/);
});

it('DNS and generic errors become one short sentence', () => {
  assert.match(friendlyNetworkError('getaddrinfo ENOTFOUND huggingface.co', 'huggingface'), /no network or DNS/);
  assert.ok(friendlyNetworkError('x'.repeat(400), 'github').length <= 160);
});
