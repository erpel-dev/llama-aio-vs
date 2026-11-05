const assert = require('node:assert/strict');
const { it } = require('node:test');
const path = require('node:path');
const Module = require('node:module');

// perfTarget.ts imports only types from vscode/core at runtime; load the harness build.
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === '@llama-aio/core') return require.resolve(path.join(__dirname, '../../core/out/index.js'));
  return origResolve.call(this, request, ...rest);
};
const core = require(path.join(__dirname, '../../core/out/index.js'));
const { syncPerfTarget } = require(path.join(__dirname, '../out-harness/perfTarget.js'));

function env(remote, launched, remoteModel) {
  return {
    store: { isRemote: () => remote.value, getEndpoint: () => (remote.value ? 'http://box:8766' : 'http://127.0.0.1:8742') },
    pm: {
      getLaunchedConfig: () => (launched.value ? { modelPath: launched.value } : undefined),
      remote: { current: () => (remote.value && remoteModel ? { modelId: remoteModel } : undefined) },
    },
  };
}

it('remote numbers never show after switching back to a stopped local server (#35)', () => {
  const perf = new core.PerfStats();
  const remote = { value: true }; const launched = { value: '' };
  const { store, pm } = env(remote, launched, 'unsloth/Qwen3.5-2B-GGUF');
  syncPerfTarget(perf, store, pm);
  const id = perf.begin();
  perf.complete({ completionTokens: 5, genTokPerSec: 10, source: 'server' }, id);
  assert.equal(perf.get().genTokPerSec, 10);
  remote.value = false;
  syncPerfTarget(perf, store, pm);
  assert.equal(perf.get().genTokPerSec, undefined);
  assert.deepEqual(perf.get().history, []);
});

it('stopping the same local server keeps its last numbers', () => {
  const perf = new core.PerfStats();
  const remote = { value: false }; const launched = { value: '/m/a.gguf' };
  const { store, pm } = env(remote, launched);
  syncPerfTarget(perf, store, pm);
  const id = perf.begin();
  perf.complete({ completionTokens: 5, genTokPerSec: 9, source: 'server' }, id);
  launched.value = '';
  syncPerfTarget(perf, store, pm);
  assert.equal(perf.get().genTokPerSec, 9);
});
