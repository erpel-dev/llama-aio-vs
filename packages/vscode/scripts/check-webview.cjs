#!/usr/bin/env node
/*
 * Static check of the sidebar webview: the page SettingsViewProvider.getHtml()
 * renders from src/webview/sidebar.html has every placeholder filled, unique
 * element ids, a nonce'd external script, and every $('id') and data-act value
 * that the src/webview/*.ts modules use resolves. (Syntax and the message protocol
 * are checked by `tsc -p tsconfig.webview.json`.)
 *
 *   npx tsc -p packages/vscode/tsconfig.json --outDir packages/vscode/out-harness
 *   node packages/vscode/esbuild.mjs --webview
 *   node packages/vscode/scripts/check-webview.cjs
 */
const Module = require("module");
const fs = require("fs");
const path = require("path");

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  return request === "vscode" ? "vscode" : origResolve.call(this, request, ...rest);
};
require.cache.vscode = {
  id: "vscode",
  filename: "vscode",
  loaded: true,
  exports: {
    window: {},
    workspace: {},
    commands: {},
    env: {},
    Uri: { joinPath: (base, ...parts) => ({ fsPath: path.join(base.fsPath, ...parts) }) },
    ProgressLocation: {},
    ViewColumn: {},
    EventEmitter: class {},
  },
};

const pkgDir = path.resolve(__dirname, "..");
const outDir = path.resolve(pkgDir, process.argv[2] || "out-harness");
const { SettingsViewProvider } = require(path.join(outDir, "settingsView.js"));
const provider = new SettingsViewProvider({ fsPath: pkgDir }, {}, {}, {}, {}, async () => undefined, {}, () => undefined);
const html = provider.getHtml({
  cspSource: "vscode-resource:",
  asWebviewUri: (u) => ({ toString: () => `vscode-resource:${u.fsPath}` }),
});
// Every webview module: the main script plus library.ts, format.ts, … share one page.
const webviewSrc = path.join(pkgDir, "src", "webview");
const script = fs
  .readdirSync(webviewSrc)
  .filter((f) => f.endsWith(".ts"))
  .map((f) => fs.readFileSync(path.join(webviewSrc, f), "utf8"))
  .join("\n");

let failed = false;
const fail = (msg) => {
  failed = true;
  console.error("FAIL:", msg);
};

const left = [...new Set([...html.matchAll(/\{\{(\w+)\}\}/g)].map((m) => m[1]))];
if (left.length) fail(`unfilled template placeholders: ${left.join(", ")}`);
else console.log("ok: all template placeholders filled");

const tag = /<script nonce="([^"]+)" src="([^"]+)"><\/script>/.exec(html);
const csp = /script-src 'nonce-([^']+)'/.exec(html);
if (!tag || !csp || tag[1] !== csp[1]) fail("sidebar script tag is missing or its nonce does not match the CSP");
else if (!fs.existsSync(tag[2].replace(/^vscode-resource:/, ""))) fail(`bundled script not found: ${tag[2]} (run esbuild.mjs --webview)`);
else console.log("ok: nonce'd sidebar.js is built");
if (/<script(?![^>]*\bsrc=)[^>]*>/.test(html)) fail("inline <script> in the page; the CSP only allows the nonce'd bundle");

const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]);
const dup = [...new Set(ids.filter((x, i) => ids.indexOf(x) !== i))];
if (dup.length) fail(`duplicate ids: ${dup.join(", ")}`);
else console.log(`ok: ${ids.length} unique element ids`);

const refs = [...new Set([...script.matchAll(/\$\('([A-Za-z0-9_]+)'\)/g)].map((m) => m[1]))];
const missing = refs.filter((r) => !ids.includes(r));
if (missing.length) fail(`$() references without an element: ${missing.join(", ")}`);
else console.log(`ok: all ${refs.length} $() references resolve`);

// A rendered data-act button needs a click handler that compares against it.
const acts = [...new Set([...`${html}\n${script}`.matchAll(/data-act="([a-z0-9-]+)"/g)].map((m) => m[1]))];
const unhandled = acts.filter((a) => !new RegExp(`[!=]==\\s*'${a}'`).test(script));
if (unhandled.length) fail(`data-act values without a handler: ${unhandled.join(", ")}`);
else console.log(`ok: all ${acts.length} data-act values are handled`);

// `lib` is the library card's shared state object (library.ts). A local
// variable or parameter with that name silently writes somewhere else.
const shadows = [...script.matchAll(/(?:\b(?:const|let|var)\s+lib\b|\bfunction\s*\w*\s*\([^)]*\blib\b[^)]*\))/g)];
const legit = (script.match(/export const lib = \{/g) || []).length;
if (shadows.length > legit) fail(`a local "lib" shadows the library state (${shadows.length - legit}×)`);
else console.log("ok: no local shadows the shared lib state");

process.exit(failed ? 1 : 0);
