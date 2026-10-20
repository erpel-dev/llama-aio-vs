/**
 * Per-user API key for the locally launched llama-server.
 *
 * The server used to accept unauthenticated requests from anything that could
 * reach loopback (other local users, web pages via DNS rebinding, LAN clients
 * when `host` was wildcarded). The key file is 0600 in the runtime directory,
 * shared by the extension and the TUI so both authenticate transparently.
 */
import { randomBytes } from "crypto";
import * as fs from "fs";
import * as path from "path";
import { ensureDirs, getLockDir } from "./paths";

const KEY_FILE_NAME = "local-api.key";

let cached: { key: string; mtimeMs: number } | undefined;

export function getLocalApiKeyPath(): string {
  return path.join(getLockDir(), KEY_FILE_NAME);
}

/** `localhost`, `127.0.0.1` and `::1` are the only origins we treat as local. */
export function isLoopbackOrigin(origin: string): boolean {
  try {
    const host = new URL(origin).hostname.toLowerCase().replace(/^\[|\]$/g, "");
    return host === "localhost" || host === "127.0.0.1" || host === "::1";
  } catch {
    return false;
  }
}

/** Current key, or undefined when no local server has ever been launched. */
export function readLocalApiKey(): string | undefined {
  try {
    const file = getLocalApiKeyPath();
    const stat = fs.statSync(file);
    if (cached && cached.mtimeMs === stat.mtimeMs) {
      return cached.key;
    }
    const key = fs.readFileSync(file, "utf8").trim();
    if (!key) {
      return undefined;
    }
    cached = { key, mtimeMs: stat.mtimeMs };
    return key;
  } catch {
    return undefined;
  }
}

/**
 * Read the key or create one. `wx` makes creation atomic across the extension
 * and the TUI; the loser of a race re-reads the winner's key.
 */
export function ensureLocalApiKey(): string {
  const existing = readLocalApiKey();
  if (existing) {
    return existing;
  }
  ensureDirs(getLockDir());
  const key = randomBytes(32).toString("hex");
  const file = getLocalApiKeyPath();
  try {
    fs.writeFileSync(file, `${key}\n`, { encoding: "utf8", flag: "wx", mode: 0o600 });
    cached = { key, mtimeMs: fs.statSync(file).mtimeMs };
    return key;
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "EEXIST") {
      return readLocalApiKey() ?? key;
    }
    throw err;
  }
}

/** Test hook: forget the mtime cache after rewriting the key file. */
export function resetLocalApiKeyCache(): void {
  cached = undefined;
}
