import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { after, before, describe, it } from "node:test";

const home = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-key-"));
process.env.HOME = home;
process.env.USERPROFILE = home;

import {
  ensureLocalApiKey,
  getLocalApiKeyPath,
  isLoopbackOrigin,
  readLocalApiKey,
  resetLocalApiKeyCache,
} from "../src/localApiKey";

describe("local API key", () => {
  before(() => {
    resetLocalApiKeyCache();
  });
  after(() => {
    fs.rmSync(home, { recursive: true, force: true });
  });

  it("creates a 0600 key file and reuses it", () => {
    const first = ensureLocalApiKey();
    assert.match(first, /^[a-f0-9]{64}$/);
    assert.equal(fs.readFileSync(getLocalApiKeyPath(), "utf8").trim(), first);
    if (process.platform !== "win32") {
      assert.equal(fs.statSync(getLocalApiKeyPath()).mode & 0o777, 0o600);
    }
    assert.equal(ensureLocalApiKey(), first);
    assert.equal(readLocalApiKey(), first);
  });

  it("picks up a key written by another frontend", () => {
    const replacement = "a".repeat(64);
    fs.writeFileSync(getLocalApiKeyPath(), `${replacement}\n`, { mode: 0o600 });
    resetLocalApiKeyCache();
    assert.equal(readLocalApiKey(), replacement);
  });

  it("treats only loopback origins as local", () => {
    assert.equal(isLoopbackOrigin("http://127.0.0.1:8742"), true);
    assert.equal(isLoopbackOrigin("http://localhost:8742"), true);
    assert.equal(isLoopbackOrigin("http://[::1]:8742"), true);
    assert.equal(isLoopbackOrigin("http://192.168.0.5:8742"), false);
    assert.equal(isLoopbackOrigin("https://example.com"), false);
    assert.equal(isLoopbackOrigin("not a url"), false);
  });
});
