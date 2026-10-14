import { build, context } from "esbuild";
import { cpSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const coreRoot = join(here, "..", "core");
const watch = process.argv.includes("--watch");

/**
 * The core is a workspace package, and `vsce package --no-dependencies` ships
 * no node_modules — so the extension is bundled into a single file instead.
 */
const options = {
  entryPoints: [join(here, "src", "extension.ts")],
  bundle: true,
  outfile: join(here, "out", "extension.js"),
  platform: "node",
  target: "node20",
  format: "cjs",
  external: ["vscode"],
  sourcemap: !watch ? "linked" : "inline",
  minify: !watch,
  logLevel: "info",
};

/** Sidebar webview script: browser bundle loaded by SettingsViewProvider.getHtml. */
const webviewOptions = {
  entryPoints: [join(here, "src", "webview", "sidebar.ts")],
  bundle: true,
  outfile: join(here, "out", "webview", "sidebar.js"),
  platform: "browser",
  target: "es2022",
  format: "iife",
  sourcemap: !watch ? "linked" : "inline",
  minify: !watch,
  logLevel: "info",
};

/** Runtime assets the bundle reads relative to the extension root. */
function copyAssets() {
  const dest = join(here, "prompt-replacements");
  rmSync(dest, { recursive: true, force: true });
  mkdirSync(dirname(dest), { recursive: true });
  cpSync(join(coreRoot, "prompt-replacements"), dest, { recursive: true });
  copyWebviewStatic();
}

/** sidebar.html (template) and sidebar.css ship next to the bundled script. */
function copyWebviewStatic() {
  const out = join(here, "out", "webview");
  mkdirSync(out, { recursive: true });
  for (const file of ["sidebar.html", "sidebar.css"]) {
    cpSync(join(here, "src", "webview", file), join(out, file));
  }
}

/** `--webview`: only the sidebar (tests and the webview check need it, not the host bundle). */
const webviewOnly = process.argv.includes("--webview");
copyAssets();

if (watch) {
  const contexts = await Promise.all([context(options), context(webviewOptions)]);
  await Promise.all(contexts.map((ctx) => ctx.watch()));
  console.log("watching… (re-run to pick up sidebar.html / sidebar.css edits)");
} else if (webviewOnly) {
  await build(webviewOptions);
} else {
  await Promise.all([build(options), build(webviewOptions)]);
}
