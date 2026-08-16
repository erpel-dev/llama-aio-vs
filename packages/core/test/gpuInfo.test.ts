import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  gpuModelTokens,
  orderGpusLikeLlama,
  parseLlamaListDevices,
  type GpuMemoryInfo,
} from "../src/gpuInfo";

const LIST_DEVICES = `
WARNING: radv is not a conformant Vulkan implementation, testing use only.
Available devices:
  Vulkan0: AMD Radeon RX 9070 XT (RADV GFX1201) (16304 MiB, 4810 MiB free)
  Vulkan1: AMD Radeon RX 9060 XT (RADV GFX1200) (16304 MiB, 1809 MiB free)
`;

describe("parseLlamaListDevices", () => {
  it("reads Vulkan ids in llama.cpp order, not PCI order", () => {
    const d = parseLlamaListDevices(LIST_DEVICES);
    assert.equal(d.length, 2);
    assert.equal(d[0]?.id, "Vulkan0");
    assert.equal(d[0]?.index, 0);
    assert.match(d[0]?.name || "", /9070/);
    assert.equal(d[1]?.id, "Vulkan1");
    assert.match(d[1]?.name || "", /9060/);
  });
});

describe("gpuModelTokens", () => {
  it("extracts RX 9070 from both llama and lspci-style names", () => {
    assert.deepEqual(gpuModelTokens("AMD Radeon RX 9070 XT"), ["rx9070"]);
    assert.ok(gpuModelTokens("Radeon RX 9070/9070 XT/9070 GRE").includes("rx9070"));
    assert.ok(gpuModelTokens("Radeon RX 9060 XT").includes("rx9060"));
  });
});

describe("orderGpusLikeLlama", () => {
  it("reverses PCI-sorted cards to match Vulkan0 = 9070", () => {
    const pci: GpuMemoryInfo[] = [
      {
        totalBytes: 16e9,
        name: "Radeon RX 9060 XT",
        source: "sysfs:card1",
        pciSlot: "0000:27:00.0",
        index: 0,
      },
      {
        totalBytes: 16e9,
        name: "Radeon RX 9070/9070 XT/9070 GRE",
        source: "sysfs:card2",
        pciSlot: "0000:2a:00.0",
        index: 1,
      },
    ];
    const ordered = orderGpusLikeLlama(pci, parseLlamaListDevices(LIST_DEVICES));
    assert.equal(ordered[0]?.llamaDeviceId, "Vulkan0");
    assert.match(ordered[0]?.name || "", /9070/);
    assert.equal(ordered[0]?.index, 0);
    assert.equal(ordered[1]?.llamaDeviceId, "Vulkan1");
    assert.match(ordered[1]?.name || "", /9060/);
    assert.equal(ordered[1]?.index, 1);
  });
});
