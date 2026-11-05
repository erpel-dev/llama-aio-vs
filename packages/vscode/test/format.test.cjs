const assert = require('node:assert/strict');
const { it } = require('node:test');
const path = require('node:path');
const vm = require('node:vm');
const { buildSync } = require('esbuild');

const source = buildSync({ entryPoints: [path.join(__dirname, '../src/webview/format.ts')], bundle: true,
  platform: 'node', format: 'cjs', write: false }).outputFiles[0].text;
const context = vm.createContext({ module: { exports: {} } });
vm.runInContext(source, context);
const { friendlyNetworkError, promptEtaSeconds } = context.module.exports;

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

it('prompt ETA waits for real evidence (PR #34 review: 1,604 s after one token)', () => {
  assert.equal(promptEtaSeconds({ total: 45826, cache: 0, processed: 1, timeMs: 35 }), 0);
  assert.equal(promptEtaSeconds({ total: 45826, cache: 0, processed: 200, timeMs: 9000 }), 0, 'too few tokens');
  assert.equal(promptEtaSeconds({ total: 45826, cache: 0, processed: 2048, timeMs: 900 }), 0, 'too little time');
  assert.equal(promptEtaSeconds({ total: 8192, cache: 4096, processed: 4100, timeMs: 5000 }), 0, 'cache hits are not evidence');
  // 2,048 fresh tokens in 45 s → ~45.5 tok/s → 5,466 left ≈ 121 s
  assert.equal(promptEtaSeconds({ total: 7514, cache: 0, processed: 2048, timeMs: 45000 }), 121);
  assert.equal(promptEtaSeconds({ total: 100, cache: 0, processed: 100, timeMs: 9000 }), 0, 'done');
});
