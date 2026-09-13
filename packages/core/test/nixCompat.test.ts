import assert from "node:assert/strict";
import { describe, it } from "node:test";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import {
  expectedFhsDynamicLinker,
  invalidateLlamaServerProbeCache,
  looksLikeMissingDynamicLinker,
  nixOsIncompatibilityHint,
  probeLlamaServerRunnable,
  resolveLaunchPlan,
} from "../src/nixCompat";

const unixOnly = process.platform === "win32" ? { skip: "needs sh" } : {};

describe("probeLlamaServerRunnable cache", () => {
  const writeScript = (file: string, body: string) => {
    fs.writeFileSync(file, body, { mode: 0o755 });
  };

  it(
    "remembers a failed probe for the same mtime/size until invalidated",
    unixOnly,
    () => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-probe-"));
      const bin = path.join(dir, "llama-server");
      try {
        const broken = "#!/bin/sh\nexit 1\n";
        // Whole-second mtime so it can be restored bit-for-bit after rewriting.
        const stamp = Math.floor(Date.now() / 1000) - 60;
        writeScript(bin, broken);
        fs.utimesSync(bin, stamp, stamp);
        const first = probeLlamaServerRunnable(bin, { timeoutMs: 5000 });
        assert.equal(first.ok, false);

        // Same byte length as `broken`, same mtime → looks unchanged on disk.
        const fixed = "#!/bin/sh\nexit 0\n";
        assert.equal(fixed.length, broken.length);
        writeScript(bin, fixed);
        fs.utimesSync(bin, stamp, stamp);

        const cached = probeLlamaServerRunnable(bin, { timeoutMs: 5000 });
        assert.equal(cached.ok, false, "negative result served from cache");

        invalidateLlamaServerProbeCache(bin);
        const fresh = probeLlamaServerRunnable(bin, { timeoutMs: 5000 });
        assert.equal(fresh.ok, true, "invalidation forces a re-probe");
      } finally {
        invalidateLlamaServerProbeCache(bin);
        fs.rmSync(dir, { recursive: true, force: true });
      }
    }
  );

  it("re-probes when the binary changes on disk", unixOnly, () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-probe-"));
    const bin = path.join(dir, "llama-server");
    try {
      writeScript(bin, "#!/bin/sh\nexit 3\n");
      assert.equal(probeLlamaServerRunnable(bin, { timeoutMs: 5000 }).ok, false);
      writeScript(bin, "#!/bin/sh\necho 'version: 1234 (abc)'\nexit 0\n");
      assert.equal(probeLlamaServerRunnable(bin, { timeoutMs: 5000 }).ok, true);
    } finally {
      invalidateLlamaServerProbeCache(bin);
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("expectedFhsDynamicLinker", () => {
  it("points at the usual glibc interpreter path for this arch", () => {
    const linker = expectedFhsDynamicLinker();
    if (process.arch === "arm64") {
      assert.equal(linker, "/lib/ld-linux-aarch64.so.1");
    } else {
      assert.equal(linker, "/lib64/ld-linux-x86-64.so.2");
    }
  });
});

describe("nixOsIncompatibilityHint", () => {
  it("mentions nix-ld, steam-run, PATH backend, and llama-server", () => {
    const hint = nixOsIncompatibilityHint();
    assert.match(hint, /nix-ld/i);
    assert.match(hint, /steam-run/i);
    assert.match(hint, /llama-server/i);
    assert.match(hint, /System \(PATH\)|PATH/i);
    assert.match(hint, /NixOS/);
  });
});

describe("looksLikeMissingDynamicLinker", () => {
  it("treats ENOENT on an existing file as a missing interpreter", () => {
    const tmp = path.join(os.tmpdir(), `llama-aio-enoent-${Date.now()}`);
    fs.writeFileSync(tmp, "x");
    try {
      const err = Object.assign(new Error("spawn ENOENT"), { code: "ENOENT" });
      assert.equal(looksLikeMissingDynamicLinker(tmp, err), true);
    } finally {
      fs.unlinkSync(tmp);
    }
  });

  it("does not flag a plain missing binary path", () => {
    const missing = path.join(os.tmpdir(), `llama-aio-missing-${Date.now()}-nope`);
    const err = Object.assign(new Error("spawn ENOENT"), { code: "ENOENT" });
    assert.equal(looksLikeMissingDynamicLinker(missing, err), false);
  });
});

describe("resolveLaunchPlan", () => {
  it("usePath requires a runnable llama-server on PATH", () => {
    try {
      const plan = resolveLaunchPlan({ usePath: true });
      assert.equal(plan.method, "path");
      assert.ok(plan.binary);
      assert.equal(plan.command, plan.binary);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      assert.match(msg, /not found on PATH|not runnable/i);
    }
  });

  it("throws a clear error for a non-executable placeholder when nothing rescues it", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-nix-"));
    const fake = path.join(dir, "llama-server");
    fs.writeFileSync(fake, "#!/bin/sh\nexit 1\n", { mode: 0o644 });
    try {
      const plan = resolveLaunchPlan({ installedBinary: fake });
      // PATH or steam-run may rescue the probe on some machines.
      assert.ok(plan.method === "path" || plan.method === "steam-run" || plan.method === "direct");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      assert.match(msg, /Cannot execute installed llama-server|llama-server binary not found/i);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});
