import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { describe, it } from "node:test";
import {
  discoverOneApiLibraryDirs,
  oneApiRuntimePresent,
  syclSupport,
  withSyclRuntimeEnv,
} from "../src/syclRuntime";

describe("discoverOneApiLibraryDirs", () => {
  it("finds libsycl and the sibling component lib dirs", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "oneapi-"));
    const syclLib = path.join(root, "compiler", "latest", "lib");
    const mklLib = path.join(root, "mkl", "latest", "lib");
    fs.mkdirSync(syclLib, { recursive: true });
    fs.mkdirSync(mklLib, { recursive: true });
    fs.writeFileSync(path.join(syclLib, "libsycl.so"), "");
    const dirs = discoverOneApiLibraryDirs(root);
    assert.ok(dirs.includes(syclLib));
    assert.ok(dirs.includes(mklLib));
    assert.equal(oneApiRuntimePresent({ ONEAPI_ROOT: root, PATH: "" }), true);
  });

  it("returns nothing for a missing root", () => {
    assert.deepEqual(discoverOneApiLibraryDirs(path.join(os.tmpdir(), "no-such-oneapi-root")), []);
  });
});

describe("withSyclRuntimeEnv", () => {
  it("sets ZES_ENABLE_SYSMAN and prepends oneAPI libs on Linux", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "oneapi-env-"));
    const syclLib = path.join(root, "compiler", "2026.1", "lib");
    fs.mkdirSync(syclLib, { recursive: true });
    fs.writeFileSync(path.join(syclLib, "libsycl.so"), "");
    const env = withSyclRuntimeEnv({
      ONEAPI_ROOT: root,
      LD_LIBRARY_PATH: "/opt/already",
      PATH: "",
    });
    assert.equal(env.ZES_ENABLE_SYSMAN, "1");
    if (process.platform !== "win32") {
      assert.match(env.LD_LIBRARY_PATH || "", new RegExp(`^${syclLib.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`));
      assert.match(env.LD_LIBRARY_PATH || "", /\/opt\/already$/);
    }
  });

  it("does not override an explicit ZES_ENABLE_SYSMAN", () => {
    const env = withSyclRuntimeEnv({ ZES_ENABLE_SYSMAN: "0", ONEAPI_ROOT: "" });
    assert.equal(env.ZES_ENABLE_SYSMAN, "0");
  });
});

describe("syclSupport", () => {
  it("is limited to Windows and Linux x64, and Linux needs libsycl", () => {
    const support = syclSupport({ ONEAPI_ROOT: path.join(os.tmpdir(), "missing-oneapi"), PATH: "" });
    if (process.arch !== "x64" || (process.platform !== "win32" && process.platform !== "linux")) {
      assert.equal(support.available, false);
      assert.match(support.reason || "", /No SYCL release packages/);
    } else if (process.platform === "linux") {
      // A fake root must not count. A real /opt/intel/oneapi on this machine still can.
      const onlyFake = oneApiRuntimePresent({ ONEAPI_ROOT: path.join(os.tmpdir(), "missing-oneapi") });
      if (!onlyFake && !oneApiRuntimePresent({ ONEAPI_ROOT: "" })) {
        assert.equal(support.available, false);
        assert.match(support.reason || "", /oneAPI/);
      }
    } else {
      assert.equal(support.available, true);
    }
  });
});
