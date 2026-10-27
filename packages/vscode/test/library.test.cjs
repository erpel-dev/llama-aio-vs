const assert = require('node:assert/strict');
const { it } = require('node:test');
const path = require('node:path');
const vm = require('node:vm');
const { buildSync } = require('esbuild');

// Run the actual library controller/rendering module with a minimal DOM sink.
// Browser layout and accessibility are checked separately in Chromium.
const source = buildSync({ entryPoints: [path.join(__dirname, '../src/webview/library.ts')], bundle: true,
  platform: 'node', format: 'cjs', write: false }).outputFiles[0].text;
function setup() {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, { value: '', innerHTML: '', textContent: '', open: true,
      classList: { add() {}, remove() {} } });
    return elements.get(id);
  };
  const messages = [];
  const context = vm.createContext({ module: { exports: {} },
    acquireVsCodeApi: () => ({ postMessage: m => messages.push(m) }),
    document: { getElementById: element, querySelector: () => null } });
  vm.runInContext(source, context);
  const api = context.module.exports;
  const click = act => {
    const button = { getAttribute: key => key === 'data-act' ? act : '', closest: () => null };
    api.onLibraryClick({ target: { closest: selector => selector === '[data-act]' ? button : null } });
  };
  return { ...api, element, messages, click };
}

it('ignores late repo files and errors after returning to a local filter', () => {
  const t = setup();
  t.resetLibraryNavigation('files');
  t.lib.hfFiles = { modelId: 'Qwen/repo' };
  const requestId = t.lib.hfRequest;
  t.element('libraryFilter').value = 'smollm2';
  t.resetLibraryNavigation();
  const original = t.element('libraryList').innerHTML = 'current local result';
  assert.equal(t.applyLibraryHfMessage({ type: 'libraryHfFiles', requestId, modelId: 'Qwen/repo', files: [] }), false);
  assert.equal(t.applyLibraryHfMessage({ type: 'libraryHfError', requestId, query: '', message: 'late failure' }), false);
  assert.equal(t.element('libraryList').innerHTML, original);
  assert.equal(t.lib.hfMode, 'local');
});

it('accepts current repo files but rejects another repo and a repeated-query generation', () => {
  const t = setup();
  t.resetLibraryNavigation('files'); t.lib.hfFiles = { modelId: 'current/repo' };
  assert.equal(t.applyLibraryHfMessage({ type: 'libraryHfFiles', requestId: t.lib.hfRequest, modelId: 'old/repo', files: [] }), false);
  assert.equal(t.applyLibraryHfMessage({ type: 'libraryHfFiles', requestId: t.lib.hfRequest, modelId: 'current/repo', files: [{ label: 'right.gguf', path: 'right.gguf' }] }), true);
  assert.match(t.element('libraryList').innerHTML, /right.gguf/);
  t.element('libraryFilter').value = 'same query';
  t.startLibrarySearch(); const previous = t.lib.hfRequest;
  t.startLibrarySearch();
  assert.equal(t.applyLibraryHfMessage({ type: 'libraryHfResults', requestId: previous, query: 'same query', repos: [] }), false);
});

it('pages a 600-model library without losing access to its last model', () => {
  const t = setup();
  t.lib.payload.groups = Array.from({ length: 600 }, (_, i) => ({ title: `Model ${i}`, detail: '', source: 'Llama AIO',
    quants: [{ path: `/models/${i}.gguf`, label: `Model ${i}`, sizeBytes: 1 }], companions: [] }));
  t.renderLibraryList();
  const rendered = () => t.element('libraryList').innerHTML;
  assert.equal((rendered().match(/class="lib-quant/g) || []).length, 50);
  assert.match(rendered(), /1–50 of 600/);
  for (let i = 0; i < 11; i++) t.click('lib-next');
  assert.match(rendered(), /551–600 of 600/);
  assert.match(rendered(), /data-path="\/models\/599.gguf"/);
  t.element('libraryFilter').value = 'Model 599';
  t.resetLibraryNavigation(); t.renderLibraryList();
  assert.equal((rendered().match(/class="lib-quant/g) || []).length, 1);
  assert.match(rendered(), /data-path="\/models\/599.gguf"/);
});

it('finds a model by filename even when GGUF metadata supplies its display name', () => {
  const t = setup();
  t.lib.payload.groups = [{ title: 'LFM2.5', detail: '', source: 'Local',
    quants: [{ path: 'C:\\models\\audit-0599-LFM2.5-Q8_0.gguf', label: 'Q8_0', sizeBytes: 1 }], companions: [] }];
  t.element('libraryFilter').value = 'AUDIT-0599';
  t.renderLibraryList();
  assert.match(t.element('libraryList').innerHTML, /class="lib-quant/);
  t.element('libraryFilter').value = 'nonexistent';
  t.renderLibraryList();
  assert.match(t.element('libraryList').innerHTML, /No local model matches/);
});
