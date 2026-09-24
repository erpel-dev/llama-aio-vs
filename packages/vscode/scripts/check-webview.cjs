#!/usr/bin/env node
/*
 * Static check of the sidebar webview produced by SettingsViewProvider.getHtml():
 * the inline script parses, element ids are unique, and every $('id') the
 * script uses exists in the markup.
 *
 *   npx tsc -p packages/vscode/tsconfig.json --outDir packages/vscode/out-harness
 *   node packages/vscode/scripts/check-webview.cjs
 */
const Module = require("module");
const path = require("path");

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  return request === "vscode" ? "vscode" : origResolve.call(this, request, ...rest);
};
require.cache.vscode = {
  id: "vscode",
  filename: "vscode",
  loaded: true,
  exports: { window: {}, workspace: {}, commands: {}, env: {}, Uri: {}, ProgressLocation: {}, ViewColumn: {}, EventEmitter: class {} },
};

const outDir = path.resolve(__dirname, "..", process.argv[2] || "out-harness");
const { SettingsViewProvider } = require(path.join(outDir, "settingsView.js"));
const provider = new SettingsViewProvider({}, {}, {}, {}, {}, async () => undefined, {}, () => undefined);
const html = provider.getHtml({ cspSource: "" });

let failed = false;
const fail = (msg) => {
  failed = true;
  console.error("FAIL:", msg);
};

const script = /<script nonce="[^"]+">([\s\S]*)<\/script>/.exec(html);
if (!script) {
  fail("no inline script");
} else {
  try {
    new Function(script[1]);
    console.log(`ok: webview script parses (${script[1].split("\n").length} lines)`);
  } catch (e) {
    fail(`webview script does not parse: ${e.message}`);
  }
}

const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]);
const dup = [...new Set(ids.filter((x, i) => ids.indexOf(x) !== i))];
if (dup.length) fail(`duplicate ids: ${dup.join(", ")}`);
else console.log(`ok: ${ids.length} unique element ids`);

if (script) {
  const refs = [...new Set([...script[1].matchAll(/\$\('([A-Za-z0-9_]+)'\)/g)].map((m) => m[1]))];
  const missing = refs.filter((r) => !ids.includes(r));
  if (missing.length) fail(`$() references without an element: ${missing.join(", ")}`);
  else console.log(`ok: all ${refs.length} $() references resolve`);
}

process.exit(failed ? 1 : 0);
