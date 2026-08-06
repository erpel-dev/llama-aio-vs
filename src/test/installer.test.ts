import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { candidateAssetNames, describeMissingAsset, pickAsset } from "../llamaInstaller";

const asset = (name: string) => ({
  name,
  browser_download_url: `https://example/${name}`,
  size: 1,
});

describe("candidateAssetNames", () => {
  it("includes the tag in every candidate", () => {
    for (const backend of ["cpu", "cuda", "vulkan"] as const) {
      for (const name of candidateAssetNames("b10297", backend)) {
        assert.ok(name.includes("b10297"), `${name} is missing the tag`);
      }
    }
  });

  it("offers at least one candidate per backend", () => {
    for (const backend of ["cpu", "cuda", "vulkan"] as const) {
      assert.ok(candidateAssetNames("b10297", backend).length > 0, backend);
    }
  });
});

describe("pickAsset", () => {
  it("returns nothing for an empty release", () => {
    assert.equal(pickAsset([], "cpu"), undefined);
  });

  it("ignores assets for other platforms", () => {
    // Only assets for a platform we are not on — must not be selected.
    const foreign =
      process.platform === "linux"
        ? ["llama-b1-bin-win-cuda-12.4-x64.zip", "llama-b1-bin-macos-arm64.zip"]
        : ["llama-b1-bin-ubuntu-x64.tar.gz"];
    assert.equal(pickAsset(foreign.map(asset), "cpu"), undefined);
  });
});

describe("describeMissingAsset", () => {
  const probes404 = [
    { name: "llama-b10297-bin-ubuntu-x64.tar.gz", probe: { status: 404 } },
    { name: "llama-b10297-bin-ubuntu-x64.zip", probe: { status: 404 } },
  ];

  it("says the release published nothing when the asset list is empty", () => {
    const msg = describeMissingAsset("b10297", "cpu", probes404, []);
    assert.match(msg, /no downloadable assets/i);
    assert.match(msg, /earlier tag/i);
  });

  it("lists what the release does contain so the mismatch is visible", () => {
    const listed = [asset("llama-b10297-bin-win-cuda-12.4-x64.zip"), asset("sources.zip")];
    const msg = describeMissingAsset("b10297", "cpu", probes404, listed);
    assert.match(msg, /llama-b10297-bin-win-cuda-12\.4-x64\.zip/);
    assert.match(msg, /None of them match/i);
  });

  it("truncates a long asset list", () => {
    const listed = Array.from({ length: 30 }, (_, i) => asset(`asset-${i}.zip`));
    const msg = describeMissingAsset("b10297", "cpu", probes404, listed);
    assert.match(msg, /and 18 more/);
  });

  it("blames the network, not the release, when GitHub was unreachable", () => {
    const offline = [{ name: "x.tar.gz", probe: { error: "getaddrinfo ENOTFOUND" } }];
    const msg = describeMissingAsset("b10297", "cpu", offline, undefined);
    assert.match(msg, /Could not reach GitHub/i);
    assert.match(msg, /ENOTFOUND/);
    assert.doesNotMatch(msg, /No cpu archive/i);
  });

  it("reports per-candidate status when the API is unavailable but GitHub answered", () => {
    const msg = describeMissingAsset("b10297", "cpu", probes404, undefined);
    assert.match(msg, /404 \(not uploaded\)/);
  });

  it("always points at the manual escape hatch", () => {
    const msg = describeMissingAsset("b10297", "cpu", probes404, []);
    assert.match(msg, /Install from archive/);
  });
});
