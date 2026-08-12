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

/** Runtime assets the bundle reads relative to the extension root. */
function copyAssets() {
  const dest = join(here, "prompt-replacements");
  rmSync(dest, { recursive: true, force: true });
  mkdirSync(dirname(dest), { recursive: true });
  cpSync(join(coreRoot, "prompt-replacements"), dest, { recursive: true });
}

copyAssets();

if (watch) {
  const ctx = await context(options);
  await ctx.watch();
  console.log("watching…");
} else {
  await build(options);
}
