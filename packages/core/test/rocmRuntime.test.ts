import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { describe, it } from "node:test";
import {
  discoverRocmLibraryDirs,
  rocmRuntimePresent,
  rocmSupport,
  withRocmRuntimeEnv,
} from "../src/rocmRuntime";

describe("discoverRocmLibraryDirs", () => {
  it("finds libhipblas in lib/ and in a TheRock core-* directory", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "rocm-"));
    const lib = path.join(root, "lib");
    const core = path.join(root, "core-7.14", "lib");
    fs.mkdirSync(lib, { recursive: true });
    fs.mkdirSync(core, { recursive: true });
    fs.writeFileSync(path.join(core, "libhipblas.so.3"), "");
    const dirs = discoverRocmLibraryDirs(root);
    assert.ok(dirs.includes(lib));
    assert.ok(dirs.includes(core));
    assert.equal(rocmRuntimePresent({ ROCM_PATH: root }), true);
  });

  it("returns nothing for a missing root", () => {
    assert.deepEqual(discoverRocmLibraryDirs(path.join(os.tmpdir(), "no-such-rocm-root")), []);
  });
});

describe("withRocmRuntimeEnv", () => {
  it("prepends the hipBLAS directory and sets ROCM_PATH on Linux", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "rocm-env-"));
    const lib = path.join(root, "lib");
    fs.mkdirSync(lib, { recursive: true });
    fs.writeFileSync(path.join(lib, "libhipblas.so"), "");
    const env = withRocmRuntimeEnv({
      LD_LIBRARY_PATH: "/opt/already",
      PATH: "",
      ROCM_PATH: root,
    });
    assert.equal(env.ROCM_PATH, root);
    assert.equal(env.HSA_OVERRIDE_GFX_VERSION, undefined);
    if (process.platform !== "win32") {
      const escaped = lib.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      assert.match(env.LD_LIBRARY_PATH || "", new RegExp(`^${escaped}`));
      assert.match(env.LD_LIBRARY_PATH || "", /\/opt\/already$/);
    }
  });
});

describe("rocmSupport", () => {
  it("is limited to Windows and Linux x64, and Linux needs libhipblas", () => {
    const missing = path.join(os.tmpdir(), "missing-rocm");
    const support = rocmSupport({ ROCM_PATH: missing, PATH: "" });
    if (process.arch !== "x64" || (process.platform !== "win32" && process.platform !== "linux")) {
      assert.equal(support.available, false);
      assert.match(support.reason || "", /No ROCm release packages/);
    } else if (process.platform === "linux") {
      const onlyFake = rocmRuntimePresent({ ROCM_PATH: missing });
      if (!onlyFake && !rocmRuntimePresent({ ROCM_PATH: "" })) {
        assert.equal(support.available, false);
        assert.match(support.reason || "", /libhipblas/);
      }
    } else {
      assert.equal(support.available, true);
    }
  });
});
