import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  adapterRamLooksCapped,
  decodeRegistryQueryOutput,
  gpuModelTokens,
  orderGpusLikeLlama,
  parseLlamaListDevices,
  parseQwMemorySizeField,
  parseWindowsGpuRegistry,
  parseWindowsWmiVideoControllers,
  preferStableGpuList,
  resolveWindowsGpuList,
  windowsVramNeedsRegistry,
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

  it("keeps sysfs VRAM when --list-devices reports 0 MiB", () => {
    const pci: GpuMemoryInfo[] = [
      { totalBytes: 16e9, name: "Radeon RX 9070 XT", source: "sysfs", pciSlot: "0000:2a:00.0" },
      { totalBytes: 16e9, name: "Radeon RX 9060 XT", source: "sysfs", pciSlot: "0000:27:00.0" },
    ];
    const listed = parseLlamaListDevices(`
Available devices:
  Vulkan0: AMD Radeon RX 9070 XT (RADV GFX1201) (0 MiB, 0 MiB free)
  Vulkan1: AMD Radeon RX 9060 XT (RADV GFX1200) (0 MiB, 0 MiB free)
`);
    const ordered = orderGpusLikeLlama(pci, listed);
    assert.equal(ordered[0]?.totalBytes, 16e9);
    assert.equal(ordered[1]?.totalBytes, 16e9);
  });
});

describe("preferStableGpuList", () => {
  const two: GpuMemoryInfo[] = [
    { totalBytes: 16e9, name: "RX 9070 XT", source: "sysfs", llamaDeviceId: "Vulkan0" },
    { totalBytes: 16e9, name: "RX 9060 XT", source: "sysfs", llamaDeviceId: "Vulkan1" },
  ];

  it("keeps the previous dual-GPU list when a re-probe drops a card", () => {
    const next = [{ totalBytes: 16e9, name: "RX 9070 XT", source: "sysfs", llamaDeviceId: "Vulkan0" }];
    const kept = preferStableGpuList(next, two);
    assert.equal(kept.length, 2);
    assert.equal(kept[1]?.llamaDeviceId, "Vulkan1");
  });

  it("keeps llama.cpp ids when the next probe has none", () => {
    const next: GpuMemoryInfo[] = [
      { totalBytes: 16e9, name: "RX 9070 XT", source: "sysfs" },
      { totalBytes: 16e9, name: "RX 9060 XT", source: "sysfs" },
    ];
    const kept = preferStableGpuList(next, two);
    assert.equal(kept[0]?.llamaDeviceId, "Vulkan0");
    assert.equal(kept[1]?.llamaDeviceId, "Vulkan1");
  });
});

const WMI_CAP = 0xfff0_0000;
const GiB = 1024 ** 3;

const REGISTRY_DUMP = `
HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0000
    DriverDesc    REG_SZ    AMD Radeon RX 7900 XTX
    HardwareInformation.MemorySize    REG_DWORD    0xfff00000
    HardwareInformation.qwMemorySize    REG_QWORD    0x600000000

HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0001
    DriverDesc    REG_SZ    Microsoft Basic Display Adapter
    HardwareInformation.qwMemorySize    REG_QWORD    0x20000000

HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0002
    DriverDesc    REG_SZ    NVIDIA GeForce RTX 3060
    HardwareInformation.qwMemorySize    REG_BINARY    00 00 00 00 03 00 00 00

HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\Properties
    DeviceDesc    REG_SZ    not a gpu instance
`;

describe("adapterRamLooksCapped", () => {
  it("treats the uint32 ceiling as capped and a real 2 GiB card as not", () => {
    assert.equal(adapterRamLooksCapped(WMI_CAP), true);
    assert.equal(adapterRamLooksCapped(0xffff_ffff), true);
    assert.equal(adapterRamLooksCapped(4 * GiB), false);
    assert.equal(adapterRamLooksCapped(2 * GiB), false);
    assert.equal(adapterRamLooksCapped(24 * GiB), false);
  });
});

describe("Windows registry qwMemorySize", () => {
  it("reads qword bytes, little-endian binary, and skips the uint32 MemorySize and basic display", () => {
    assert.equal(parseQwMemorySizeField("REG_QWORD", "0x600000000"), 24 * GiB);
    assert.equal(parseQwMemorySizeField("REG_BINARY", "00 00 00 00 03 00 00 00"), 12 * GiB);
    const gpus = parseWindowsGpuRegistry(REGISTRY_DUMP);
    assert.equal(gpus.length, 2);
    assert.equal(gpus[0]?.name, "AMD Radeon RX 7900 XTX");
    assert.equal(gpus[0]?.totalBytes, 24 * GiB);
    assert.equal(gpus[1]?.name, "NVIDIA GeForce RTX 3060");
    assert.equal(gpus[1]?.totalBytes, 12 * GiB);
  });

  it("decodes a UTF-16LE reg query buffer", () => {
    const text = "DriverDesc    REG_SZ    AMD Radeon RX 7900 XTX\n";
    const utf16 = Buffer.from(text, "utf16le");
    assert.equal(decodeRegistryQueryOutput(utf16), text);
    const bom = Buffer.concat([Buffer.from([0xff, 0xfe]), utf16]);
    assert.equal(decodeRegistryQueryOutput(bom), text);
    assert.equal(decodeRegistryQueryOutput(Buffer.from(text, "utf8")), text);
  });
});

describe("parseWindowsWmiVideoControllers", () => {
  it("keeps a capped AdapterRAM string and drops Basic Display and negative sizes", () => {
    const gpus = parseWindowsWmiVideoControllers(
      JSON.stringify([
        { Name: "AMD Radeon RX 7900 XTX", AdapterRAM: String(WMI_CAP) },
        { Name: "Microsoft Basic Display Adapter", AdapterRAM: WMI_CAP },
        { Name: "Broken signed", AdapterRAM: -1 },
      ])
    );
    assert.equal(gpus.length, 1);
    assert.equal(gpus[0]?.totalBytes, WMI_CAP);
    assert.equal(gpus[0]?.source, "wmi");
  });
});

describe("resolveWindowsGpuList", () => {
  const capped7900: GpuMemoryInfo = {
    totalBytes: WMI_CAP,
    name: "AMD Radeon RX 7900 XTX",
    source: "wmi",
  };

  it("replaces one capped WMI GPU with --list-devices", () => {
    const listed = parseLlamaListDevices(
      "  Vulkan0: AMD Radeon RX 7900 XTX (RADV GFX1100) (24564 MiB, 20000 MiB free)\n"
    );
    const gpus = resolveWindowsGpuList([capped7900], listed, [
      { name: "AMD Radeon RX 7900 XTX", totalBytes: 24 * GiB },
    ]);
    assert.equal(gpus.length, 1);
    assert.equal(gpus[0]?.totalBytes, 24564 * 1024 * 1024);
    assert.equal(gpus[0]?.source, "wmi+list-devices");
    assert.equal(gpus[0]?.llamaDeviceId, "Vulkan0");
    assert.equal(windowsVramNeedsRegistry([capped7900], listed), false);
  });

  it("uses the registry qword when --list-devices reports 0 MiB", () => {
    const listed = parseLlamaListDevices(
      "  Vulkan0: AMD Radeon RX 7900 XTX (RADV GFX1100) (0 MiB, 0 MiB free)\n"
    );
    const gpus = resolveWindowsGpuList([capped7900], listed, [
      { name: "AMD Radeon RX 7900 XTX", totalBytes: 24 * GiB },
    ]);
    assert.equal(gpus[0]?.totalBytes, 24 * GiB);
    assert.equal(gpus[0]?.source, "registry");
    assert.equal(gpus[0]?.llamaDeviceId, "Vulkan0");
    assert.equal(windowsVramNeedsRegistry([capped7900], listed), true);
  });

  it("does not replace a real 2 GiB AdapterRAM with a larger registry value", () => {
    const small: GpuMemoryInfo = { totalBytes: 2 * GiB, name: "Radeon RX 6400", source: "wmi" };
    const gpus = resolveWindowsGpuList([small], undefined, [{ name: "Radeon RX 6400", totalBytes: 16 * GiB }]);
    assert.equal(gpus[0]?.totalBytes, 2 * GiB);
    assert.equal(gpus[0]?.source, "wmi");
  });

  it("does not give an RX 7900 the RTX 4090 registry size", () => {
    const gpus = resolveWindowsGpuList([capped7900], undefined, [
      { name: "NVIDIA GeForce RTX 4090", totalBytes: 24 * GiB },
    ]);
    assert.equal(gpus[0]?.totalBytes, WMI_CAP);
    assert.equal(gpus[0]?.source, "wmi");
  });

  it("does not assign a named registry card to a generic adapter", () => {
    const generic: GpuMemoryInfo = {
      totalBytes: WMI_CAP,
      name: "AMD Radeon Graphics",
      source: "wmi",
    };
    const gpus = resolveWindowsGpuList([generic], undefined, [
      { name: "NVIDIA GeForce RTX 4090", totalBytes: 24 * GiB },
    ]);
    assert.equal(gpus[0]?.totalBytes, WMI_CAP);
  });

  it("builds the list from --list-devices when WMI returns nothing", () => {
    const listed = parseLlamaListDevices(`
Available devices:
  CUDA0: NVIDIA GeForce RTX 4090 (24576 MiB, 1000 MiB free)
  CUDA1: NVIDIA GeForce RTX 3060 (0 MiB, 0 MiB free)
`);
    const gpus = resolveWindowsGpuList([], listed, undefined);
    assert.equal(gpus.length, 1);
    assert.equal(gpus[0]?.llamaDeviceId, "CUDA0");
    assert.equal(gpus[0]?.totalBytes, 24576 * 1024 * 1024);
    assert.equal(gpus[0]?.source, "llama-list-devices");
  });

  it("keeps each card's own registry size", () => {
    const wmi: GpuMemoryInfo[] = [
      { totalBytes: WMI_CAP, name: "NVIDIA GeForce RTX 4090", source: "wmi" },
      { totalBytes: WMI_CAP, name: "NVIDIA GeForce RTX 3060", source: "wmi" },
    ];
    const gpus = resolveWindowsGpuList(wmi, undefined, [
      { name: "NVIDIA GeForce RTX 3060", totalBytes: 12 * GiB },
      { name: "NVIDIA GeForce RTX 4090", totalBytes: 24 * GiB },
    ]);
    assert.equal(gpus[0]?.totalBytes, 24 * GiB);
    assert.equal(gpus[1]?.totalBytes, 12 * GiB);
  });

  it("leaves a sysfs total in the 4 GiB band alone", () => {
    const pci: GpuMemoryInfo[] = [
      { totalBytes: WMI_CAP, name: "Radeon RX 7600", source: "sysfs" },
    ];
    const listed = parseLlamaListDevices(
      "  Vulkan0: AMD Radeon RX 7600 (8192 MiB, 100 MiB free)\n"
    );
    const ordered = orderGpusLikeLlama(pci, listed);
    assert.equal(ordered[0]?.totalBytes, WMI_CAP);
    assert.equal(ordered[0]?.source, "sysfs");
  });
});

describe("preferStableGpuList capped fallback", () => {
  it("keeps a previous >4 GiB reading when the next probe is the WMI cap", () => {
    const previous: GpuMemoryInfo[] = [
      { totalBytes: 24 * GiB, name: "AMD Radeon RX 7900 XTX", source: "registry" },
    ];
    const next: GpuMemoryInfo[] = [
      { totalBytes: WMI_CAP, name: "AMD Radeon RX 7900 XTX", source: "wmi" },
    ];
    const kept = preferStableGpuList(next, previous);
    assert.equal(kept[0]?.totalBytes, 24 * GiB);
  });
});
