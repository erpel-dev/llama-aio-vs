import { build } from "esbuild";
import {
  chmodSync,
  cpSync,
  mkdirSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { execFileSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Pack a relocatable TUI for the current platform:
 *   dist/llama-aio            — executable (Node shebang, ESM)
 *   dist/llama-aio.assets/    — OpenTUI native/wasm assets (OTUI_ASSET_ROOT)
 *   dist/prompt-replacements/ — defaults used by @llama-aio/core
 *
 * Must be ESM: @opentui/core uses top-level await. Core is bundled from its
 * TypeScript sources (not the CJS out/) so Node builtins stay real imports.
 */
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const tuiSrc = join(root, "packages", "tui", "src", "main.ts");
const coreSrc = join(root, "packages", "core", "src", "index.ts");
const corePrompt = join(root, "packages", "core", "prompt-replacements");
const outDir = join(root, "dist");
const assetsDir = join(outDir, "llama-aio.assets");
const binPath = join(outDir, "llama-aio");
const stagingJs = join(outDir, "llama-aio.mjs");

function detectLinuxLibc() {
  if (process.platform !== "linux") {
    return undefined;
  }
  try {
    const out = execFileSync("ldd", ["--version"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    });
    if (String(out).toLowerCase().includes("musl")) {
      return "musl";
    }
  } catch (err) {
    const blob = `${err?.stderr || err?.stdout || err?.message || ""}`.toLowerCase();
    if (blob.includes("musl")) {
      return "musl";
    }
  }
  return "glibc";
}

rmSync(binPath, { force: true });
rmSync(stagingJs, { force: true });
rmSync(`${stagingJs}.map`, { force: true });
rmSync(`${binPath}.map`, { force: true });
rmSync(join(outDir, "llama-aio.cjs"), { force: true });
rmSync(join(outDir, "llama-aio.cjs.map"), { force: true });
rmSync(assetsDir, { recursive: true, force: true });
rmSync(join(outDir, "prompt-replacements"), { recursive: true, force: true });
rmSync(join(outDir, "README.txt"), { force: true });
mkdirSync(outDir, { recursive: true });
mkdirSync(assetsDir, { recursive: true });

// OpenTUI's Zig renderer needs Node's experimental node:ffi (Node ≥ 26.4).
const banner = `#!/usr/bin/env -S node --experimental-ffi
import { dirname as __otuiDirname, join as __otuiJoin } from "node:path";
import { fileURLToPath as __otuiFileURLToPath } from "node:url";
const __filename = __otuiFileURLToPath(import.meta.url);
const __dirname = __otuiDirname(__filename);
if (!process.env.OTUI_ASSET_ROOT) {
  process.env.OTUI_ASSET_ROOT = __otuiJoin(__dirname, "llama-aio.assets");
}
`;

await build({
  entryPoints: [tuiSrc],
  bundle: true,
  outfile: stagingJs,
  platform: "node",
  target: "node26",
  format: "esm",
  sourcemap: true,
  minify: false,
  logLevel: "info",
  banner: { js: banner },
  alias: {
    "@llama-aio/core": coreSrc,
  },
  packages: "bundle",
});

const { getNodeAssets } = await import("@opentui/core/node-assets");
const target = {
  platform: process.platform,
  arch: process.arch,
};
const libc = detectLinuxLibc();
if (libc === "musl") {
  target.libc = "musl";
}

const assets = getNodeAssets(target);
for (const asset of assets) {
  const dest = join(assetsDir, asset.key);
  mkdirSync(dirname(dest), { recursive: true });
  cpSync(asset.source, dest);
}

cpSync(corePrompt, join(outDir, "prompt-replacements"), { recursive: true });

renameSync(stagingJs, binPath);
chmodSync(binPath, 0o755);

writeFileSync(
  join(outDir, "README.txt"),
  [
    "Llama AIO TUI",
    "",
    "Run:",
    "  ./dist/llama-aio",
    "",
    "Requires Node.js 26.4+ on PATH (OpenTUI uses --experimental-ffi).",
    "OpenTUI native assets live in dist/llama-aio.assets/",
    "(set automatically via OTUI_ASSET_ROOT).",
    "",
    "Keys: Tab/F1-F5 panes · click or ←→ sections · ↑↓ · Enter · s/x/r · q · Esc cancel chat",
    "Mouse: click tabs/lists/actions · wheel scrolls lists",
    "",
    `Built for ${process.platform}-${process.arch}` +
      (libc ? ` (${libc})` : "") +
      ".",
    "",
  ].join("\n"),
  "utf8"
);

console.log(`TUI executable: ${binPath}`);
console.log(`OpenTUI assets: ${assetsDir} (${assets.length} files)`);
