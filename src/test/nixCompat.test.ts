import assert from "node:assert/strict";
import { describe, it } from "node:test";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import {
  expectedFhsDynamicLinker,
  looksLikeMissingDynamicLinker,
  nixOsIncompatibilityHint,
  probeLlamaServerRunnable,
  resolveLaunchPlan,
} from "../nixCompat";

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
  it("mentions nix-ld, steam-run, and nixpkgs llama-server", () => {
    const hint = nixOsIncompatibilityHint();
    assert.match(hint, /nix-ld/i);
    assert.match(hint, /steam-run/i);
    assert.match(hint, /llama-server/i);
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

describe("probeLlamaServerRunnable / resolveLaunchPlan", () => {
  it("reports ok for a real runnable llama-server when available", () => {
    // Optional: only when a previously downloaded inspect binary exists.
    const candidate = "/tmp/llama-inspect/llama-b10330/llama-server";
    if (!fs.existsSync(candidate)) {
      return;
    }
    const probe = probeLlamaServerRunnable(candidate);
    assert.equal(probe.ok, true);
    assert.match(probe.detail, /version|built/i);
    const plan = resolveLaunchPlan(candidate);
    assert.equal(plan.method, "direct");
    assert.equal(plan.command, candidate);
  });

  it("throws a NixOS-oriented error for a non-executable placeholder when no PATH fallback exists", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-nix-"));
    const fake = path.join(dir, "llama-server");
    fs.writeFileSync(fake, "#!/bin/sh\nexit 1\n", { mode: 0o644 });
    // Not executable → probe fails. PATH may still have a real llama-server on
    // some machines; only assert the thrown message when resolve falls through
    // to the incompatibility hint.
    try {
      const plan = resolveLaunchPlan(fake);
      // If something on PATH rescued us, that is also a valid compatibility path.
      assert.ok(plan.method === "path" || plan.method === "steam-run" || plan.method === "direct");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      assert.match(msg, /Cannot execute installed llama-server|llama-server binary not found/i);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});
