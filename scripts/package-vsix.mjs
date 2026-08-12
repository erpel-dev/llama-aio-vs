import { execFileSync } from "node:child_process";
import { cpSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * The README and screenshots live at the repo root (they are the GitHub
 * landing page). vsce can only pack files inside the package directory, so
 * stage copies, build the .vsix into dist/, then clean up.
 */
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const pkg = join(root, "packages", "vscode");
const dist = join(root, "dist");
const staged = [join(pkg, "README.md"), join(pkg, "media")];

function cleanStaged() {
  for (const p of staged) {
    rmSync(p, { recursive: true, force: true });
  }
}

cleanStaged();
mkdirSync(dist, { recursive: true });
cpSync(join(root, "README.md"), join(pkg, "README.md"));
cpSync(join(root, "media"), join(pkg, "media"), { recursive: true });

try {
  execFileSync(
    "npx",
    ["--yes", "@vscode/vsce", "package", "--no-dependencies", "--out", dist],
    {
      cwd: pkg,
      stdio: "inherit",
    }
  );
} finally {
  cleanStaged();
}
