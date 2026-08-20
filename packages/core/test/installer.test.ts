import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  buildArchiveExtractCommand,
  candidateAssetNames,
  compareReleaseTags,
  createClearableTimeoutSignal,
  describeMissingAsset,
  pickAsset,
  resolveLatestReleaseTag,
} from "../src/llamaInstaller";

const asset = (name: string) => ({
  name,
  browser_download_url: `https://example/${name}`,
  size: 1,
});

describe("createClearableTimeoutSignal", () => {
  it("aborts after the timeout when not cleared", async () => {
    const { signal } = createClearableTimeoutSignal(20);
    await new Promise((r) => setTimeout(r, 50));
    assert.equal(signal.aborted, true);
  });

  it("does not abort after clear — so a long body is not killed", async () => {
    const { signal, clear } = createClearableTimeoutSignal(20);
    clear();
    await new Promise((r) => setTimeout(r, 50));
    assert.equal(signal.aborted, false);
  });
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

describe("compareReleaseTags", () => {
  it("treats a higher b-number as newer", () => {
    assert.ok(compareReleaseTags("b10375", "b10344") > 0);
    assert.ok(compareReleaseTags("b10344", "b10375") < 0);
    assert.equal(compareReleaseTags("b10344", "b10344"), 0);
  });
});

describe("buildArchiveExtractCommand", () => {
  it("never invokes PowerShell", () => {
    const zip = buildArchiveExtractCommand("llama.zip", "out");
    const tgz = buildArchiveExtractCommand("llama.tar.gz", "out");
    for (const plan of [zip, tgz]) {
      assert.notEqual(plan.command.toLowerCase(), "powershell.exe");
      assert.equal(plan.argv.join(" ").toLowerCase().includes("expand-archive"), false);
    }
  });

  it("unpacks zip with tar.exe on Windows and unzip elsewhere", () => {
    const plan = buildArchiveExtractCommand("llama.zip", "dest");
    if (process.platform === "win32") {
      assert.equal(plan.command, "tar.exe");
      assert.deepEqual(plan.argv, ["-xf", "llama.zip", "-C", "dest"]);
    } else {
      assert.equal(plan.command, "unzip");
      assert.deepEqual(plan.argv, ["-o", "llama.zip", "-d", "dest"]);
    }
  });

  it("unpacks tar.gz with tar on every platform", () => {
    const plan = buildArchiveExtractCommand("llama.tar.gz", "dest");
    assert.match(plan.command, /^tar(\.exe)?$/);
    assert.deepEqual(plan.argv, ["-xzf", "llama.tar.gz", "-C", "dest"]);
  });
});

const GH_HEADERS = {
  "User-Agent": "llama-aio-vs",
  Accept: "application/vnd.github+json",
};

describe("GitHub llama.cpp fetch", () => {
  async function latestTagOrSkip(t: { skip: (msg?: string) => void }): Promise<string | undefined> {
    try {
      return await resolveLatestReleaseTag();
    } catch (err) {
      t.skip(`GitHub unreachable: ${err instanceof Error ? err.message : err}`);
      return undefined;
    }
  }

  it("resolves the latest release tag", { timeout: 20_000 }, async (t) => {
    const tag = await latestTagOrSkip(t);
    if (!tag) {
      return;
    }
    assert.match(tag, /^b\d+$/);
  });

  it("HEADs a real archive for this platform", { timeout: 30_000 }, async (t) => {
    const tag = await latestTagOrSkip(t);
    if (!tag) {
      return;
    }
    const res = await fetch(
      `https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/${encodeURIComponent(tag)}`,
      { headers: GH_HEADERS, signal: AbortSignal.timeout(15_000) }
    );
    assert.equal(res.ok, true, `API ${res.status}`);
    const release = (await res.json()) as {
      assets: Array<{ name: string; browser_download_url: string; url: string; size: number }>;
    };
    const backend = process.platform === "linux" ? "vulkan" : "cpu";
    const picked = pickAsset(release.assets, backend);
    assert.ok(picked, `no ${backend} asset in ${tag}`);
    const assetUrl = picked.url || picked.browser_download_url;
    const head = await fetch(assetUrl, {
      method: "HEAD",
      headers: { "User-Agent": "llama-aio-vs", Accept: "application/octet-stream" },
      redirect: "follow",
      signal: AbortSignal.timeout(20_000),
    });
    assert.ok(head.ok, `HEAD ${picked.name} → HTTP ${head.status}`);
    const len = Number(head.headers.get("content-length") || 0);
    assert.ok(len > 1_000_000, `${picked.name} is only ${len} bytes`);
  });

  it("downloads 64KiB of the archive with gzip/zip magic", { timeout: 30_000 }, async (t) => {
    const tag = await latestTagOrSkip(t);
    if (!tag) {
      return;
    }
    const res = await fetch(
      `https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/${encodeURIComponent(tag)}`,
      { headers: GH_HEADERS, signal: AbortSignal.timeout(15_000) }
    );
    assert.equal(res.ok, true, `API ${res.status}`);
    const release = (await res.json()) as {
      assets: Array<{ name: string; browser_download_url: string; url: string; size: number }>;
    };
    const backend = process.platform === "linux" ? "vulkan" : "cpu";
    const picked = pickAsset(release.assets, backend);
    assert.ok(picked?.url, `no ${backend} API asset URL in ${tag}`);
    const part = await fetch(picked.url, {
      headers: {
        "User-Agent": "llama-aio-vs",
        Accept: "application/octet-stream",
        Range: "bytes=0-65535",
      },
      redirect: "follow",
      signal: AbortSignal.timeout(20_000),
    });
    assert.ok(part.ok || part.status === 206, `partial GET HTTP ${part.status}`);
    const buf = Buffer.from(await part.arrayBuffer());
    const gzip = buf[0] === 0x1f && buf[1] === 0x8b;
    const zip = buf[0] === 0x50 && buf[1] === 0x4b;
    assert.ok(gzip || zip, `unexpected magic ${buf.slice(0, 4).toString("hex")}`);
  });
});
