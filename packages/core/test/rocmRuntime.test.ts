import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { describe, it } from "node:test";
import {
  amdGpuMissingReason,
  detectAmdGpu,
  discoverRocmLibraryDirs,
  lspciTextHasAmdGpu,
  pciIdsLookLikeAmdGpu,
  pnpTextHasAmdGpu,
  resetAmdGpuDetectionCache,
  rocmRuntimePresent,
  rocmSmiTextHasGpu,
  rocminfoTextHasGpu,
  rocmSupport,
  sysfsPciHasAmdGpu,
  withRocmRuntimeEnv,
} from "../src/rocmRuntime";

function writePci(root: string, slot: string, vendor: string, deviceClass: string): void {
  const dir = path.join(root, slot);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "vendor"), `${vendor}\n`);
  fs.writeFileSync(path.join(dir, "class"), `${deviceClass}\n`);
}

const NVIDIA_ONLY_LSPCI = [
  "01:00.0 VGA compatible controller [0300]: NVIDIA Corporation GA102 [10de:2206]",
  "01:00.1 Audio device [0403]: NVIDIA Corporation GA102 Audio [10de:1aef]",
].join("\n");

const AMD_LSPCI = [
  NVIDIA_ONLY_LSPCI,
  "28:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 48 [1002:7550]",
  "28:00.1 Audio device [0403]: Advanced Micro Devices, Inc. [AMD/ATI] HDMI Audio [1002:ab40]",
  "c5:00.0 Processing accelerators [1200]: Advanced Micro Devices, Inc. [AMD/ATI] Instinct MI210 [1002:740f]",
].join("\n");

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

function hostSupportsRocmPackages(): boolean {
  return process.arch === "x64" && (process.platform === "win32" || process.platform === "linux");
}

function writeHipblasRoot(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "rocm-hipblas-"));
  const lib = path.join(root, "lib");
  fs.mkdirSync(lib, { recursive: true });
  fs.writeFileSync(path.join(lib, "libhipblas.so"), "");
  return root;
}

describe("AMD GPU detection", () => {
  it("accepts an AMD display or Instinct device and rejects HDMI audio", () => {
    assert.equal(pciIdsLookLikeAmdGpu("0x1002\n", "0x030000\n"), true);
    assert.equal(pciIdsLookLikeAmdGpu("0x1002", "0x120000"), true);
    assert.equal(pciIdsLookLikeAmdGpu("0x1002", "0x040300"), false);
    assert.equal(pciIdsLookLikeAmdGpu("0x10de", "0x030000"), false);

    const root = fs.mkdtempSync(path.join(os.tmpdir(), "pci-amd-"));
    writePci(root, "0000:28:00.0", "0x1002", "0x030000");
    writePci(root, "0000:28:00.1", "0x1002", "0x040300");
    writePci(root, "0000:01:00.0", "0x10de", "0x030000");
    assert.equal(sysfsPciHasAmdGpu(root), true);

    const nvidiaOnly = fs.mkdtempSync(path.join(os.tmpdir(), "pci-nv-"));
    writePci(nvidiaOnly, "0000:01:00.0", "0x10de", "0x030000");
    writePci(nvidiaOnly, "0000:01:00.1", "0x1002", "0x040300");
    assert.equal(sysfsPciHasAmdGpu(nvidiaOnly), false);
    assert.equal(sysfsPciHasAmdGpu(path.join(os.tmpdir(), "no-such-pci-root")), false);
  });

  it("reads lspci, rocminfo, rocm-smi, and Windows PNP ids", () => {
    assert.equal(lspciTextHasAmdGpu(AMD_LSPCI), true);
    assert.equal(lspciTextHasAmdGpu(NVIDIA_ONLY_LSPCI), false);
    assert.equal(lspciTextHasAmdGpu("28:00.1 Audio device [0403]: AMD [1002:ab40]"), false);
    assert.equal(rocminfoTextHasGpu("  Device Type:             CPU\n"), false);
    assert.equal(rocminfoTextHasGpu("  Device Type:             GPU\n"), true);
    assert.equal(rocmSmiTextHasGpu("ERROR: No AMD GPUs specified\nGPU[0] leftover"), false);
    assert.equal(rocmSmiTextHasGpu("GPU[0] : Card series: AMD Radeon RX 9070 XT"), true);
    assert.equal(pnpTextHasAmdGpu("PCI\\VEN_10DE&DEV_2206\nPCI\\VEN_1002&DEV_7550"), true);
    assert.equal(pnpTextHasAmdGpu("PCI\\VEN_10DE&DEV_2206"), false);
  });

  it("uses an injected sysfs tree and does not overwrite the live cache", () => {
    resetAmdGpuDetectionCache();
    const live = detectAmdGpu();
    assert.equal(detectAmdGpu(), live);

    const empty = fs.mkdtempSync(path.join(os.tmpdir(), "pci-empty-"));
    assert.equal(detectAmdGpu({ pciRoot: empty }), false);
    assert.equal(detectAmdGpu(), live, "injected probe must not replace the cached live verdict");

    const amd = fs.mkdtempSync(path.join(os.tmpdir(), "pci-inject-"));
    writePci(amd, "0000:03:00.0", "0x1002", "0x030000");
    assert.equal(detectAmdGpu({ pciRoot: amd, lspciText: NVIDIA_ONLY_LSPCI }), true);
    assert.equal(
      detectAmdGpu({ pciRoot: empty, lspciText: "c5:00.0 Processing accelerators [1200]: AMD [1002:740f]" }),
      true
    );
    assert.equal(
      detectAmdGpu({
        pciRoot: empty,
        lspciText: NVIDIA_ONLY_LSPCI,
        rocminfoText: "Device Type: GPU",
      }),
      true
    );
    assert.equal(detectAmdGpu(), live);
    resetAmdGpuDetectionCache();
  });
});

describe("rocmSupport", () => {
  it("rejects unsupported OS/arch before the GPU check", () => {
    const support = rocmSupport({ PATH: "" }, { amdGpu: true });
    if (!hostSupportsRocmPackages()) {
      assert.equal(support.available, false);
      assert.match(support.reason || "", /No ROCm release packages/);
    } else {
      assert.notEqual(support.reason, amdGpuMissingReason());
    }
  });

  it("stays unavailable when no AMD GPU is present, even if hipBLAS is installed", () => {
    const support = rocmSupport({ ROCM_PATH: writeHipblasRoot(), PATH: "" }, { amdGpu: false });
    if (!hostSupportsRocmPackages()) {
      assert.match(support.reason || "", /No ROCm release packages/);
      return;
    }
    assert.equal(support.available, false);
    assert.equal(support.reason, amdGpuMissingReason());
  });

  it("reports the missing ROCm runtime when an AMD GPU is present", () => {
    if (process.platform !== "linux" || process.arch !== "x64") {
      return;
    }
    const missing = path.join(os.tmpdir(), "missing-rocm-no-such");
    if (rocmRuntimePresent({ ROCM_PATH: missing })) {
      return;
    }
    const support = rocmSupport({ ROCM_PATH: missing, PATH: "" }, { amdGpu: true });
    assert.equal(support.available, false);
    assert.match(support.reason || "", /libhipblas/);
  });

  it("is available when an AMD GPU and, on Linux, libhipblas are present", () => {
    if (!hostSupportsRocmPackages()) {
      return;
    }
    const support = rocmSupport({ ROCM_PATH: writeHipblasRoot(), PATH: "" }, { amdGpu: true });
    assert.equal(support.available, true);
    assert.equal(support.reason, undefined);
  });
});
