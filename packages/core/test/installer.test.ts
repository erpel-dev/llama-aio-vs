import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  buildArchiveExtractCommand,
  candidateAssetNames,
  compareReleaseTags,
  createClearableTimeoutSignal,
  describeGithubHttpError,
  describeMissingAsset,
  detectCudaHardware,
  parseNightlyTagFile,
  parseStableReleaseTag,
  pickAsset,
  pickNewestBuildTag,
  resetHardwareDetectionCache,
  resolveLatestReleaseTag,
} from "../src/llamaInstaller";

const asset = (name: string) => ({
  name,
  browser_download_url: `https://example/${name}`,
  size: 1,
});

describe("detectCudaHardware cache", () => {
  it("serves the cached verdict until reset", () => {
    const saved = { CUDA_PATH: process.env.CUDA_PATH, CUDA_HOME: process.env.CUDA_HOME };
    try {
      process.env.CUDA_PATH = "/opt/cuda-fake";
      delete process.env.CUDA_HOME;
      resetHardwareDetectionCache();
      assert.equal(detectCudaHardware(), true);

      delete process.env.CUDA_PATH;
      assert.equal(detectCudaHardware(), true, "cached — no re-probe on the next tick");

      resetHardwareDetectionCache();
      assert.equal(typeof detectCudaHardware(), "boolean");
    } finally {
      if (saved.CUDA_PATH === undefined) {
        delete process.env.CUDA_PATH;
      } else {
        process.env.CUDA_PATH = saved.CUDA_PATH;
      }
      if (saved.CUDA_HOME === undefined) {
        delete process.env.CUDA_HOME;
      } else {
        process.env.CUDA_HOME = saved.CUDA_HOME;
      }
      resetHardwareDetectionCache();
    }
  });
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
    assert.ok((compareReleaseTags("b10375", "b10344") ?? 0) > 0);
    assert.ok((compareReleaseTags("b10344", "b10375") ?? 0) < 0);
    assert.equal(compareReleaseTags("b10344", "b10344"), 0);
  });

  it("returns undefined for tags that are not llama.cpp build tags", () => {
    assert.equal(compareReleaseTags("b10375", "local"), undefined);
    assert.equal(compareReleaseTags("local", "b10375"), undefined);
    assert.equal(compareReleaseTags("PATH", "b1"), undefined);
    assert.equal(compareReleaseTags("", "b10375"), undefined);
  });
});

describe("pickNewestBuildTag", () => {
  it("skips stable vX.Y.Z releases that only ship nightly-tag.txt", () => {
    const tag = pickNewestBuildTag([
      {
        tag_name: "v0.2.0",
        assets: [{ name: "nightly-tag.txt" }],
      },
      {
        tag_name: "b10566",
        assets: [{ name: "llama-b10566-bin-ubuntu-vulkan-x64.tar.gz" }],
      },
      {
        tag_name: "b10587",
        assets: [{ name: "llama-b10587-bin-win-vulkan-x64.zip" }],
      },
    ]);
    assert.equal(tag, "b10587");
  });

  it("ignores a newer b-tag that has not uploaded binaries yet", () => {
    const tag = pickNewestBuildTag([
      { tag_name: "b10588", assets: [] },
      { tag_name: "b10587", assets: [{ name: "llama-b10587-bin-macos-arm64.tar.gz" }] },
    ]);
    assert.equal(tag, "b10587");
  });
});

describe("parseStableReleaseTag", () => {
  it("accepts v-prefixed tags and release URLs", () => {
    assert.equal(parseStableReleaseTag("v0.2.0"), "v0.2.0");
    assert.equal(parseStableReleaseTag("0.2.0"), "v0.2.0");
    assert.equal(
      parseStableReleaseTag("https://github.com/ggml-org/llama.cpp/releases/tag/v0.2.0"),
      "v0.2.0"
    );
  });

  it("does not treat nightly tags as stable", () => {
    assert.equal(parseStableReleaseTag("b10587"), undefined);
    assert.equal(
      parseStableReleaseTag("https://github.com/ggml-org/llama.cpp/releases/tag/b10587"),
      undefined
    );
  });
});

describe("parseNightlyTagFile", () => {
  it("reads the b-tag from nightly-tag.txt", () => {
    assert.equal(parseNightlyTagFile("b10566\n"), "b10566");
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

const GH_HEADERS: Record<string, string> = {
  "User-Agent": "llama-aio-vs",
  Accept: "application/vnd.github+json",
  // Unauthenticated api.github.com allows 60 requests/hour/IP; a dev machine
  // (or CI without a token) can burn that in one session and then every run
  // gets HTTP 403. A token lifts it to 5000/hour.
  ...(process.env.GITHUB_TOKEN ? { Authorization: `Bearer ${process.env.GITHUB_TOKEN}` } : {}),
};

interface GhAsset {
  name: string;
  browser_download_url: string;
  url?: string;
  size: number;
}

/**
 * GitHub answers 403/429 once the unauthenticated rate limit is hit, and
 * 5xx for its own outages. Those say nothing about our code, so a network
 * integration test must skip rather than fail the build (`make` runs this
 * suite). Any other unexpected status — a 404 for a release that vanished —
 * is a real signal and still fails.
 */
function isUnavailable(status: number): boolean {
  return status === 403 || status === 429 || status >= 500;
}

function skipMessage(what: string, status: number): string {
  const hint =
    status === 403 || status === 429
      ? "GitHub API rate limit (set GITHUB_TOKEN to raise it from 60/h)"
      : "GitHub unavailable";
  return `${hint}: ${what} → HTTP ${status}`;
}

/** Release JSON for `tag`, or undefined when GitHub is unavailable (test skips). */
async function releaseOrSkip(
  t: { skip: (msg?: string) => void },
  tag: string
): Promise<{ assets: GhAsset[] } | undefined> {
  const url = `https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/${encodeURIComponent(tag)}`;
  let res: Response;
  try {
    res = await fetch(url, { headers: GH_HEADERS, signal: AbortSignal.timeout(15_000) });
  } catch (err) {
    t.skip(`GitHub unreachable: ${err instanceof Error ? err.message : err}`);
    return undefined;
  }
  if (isUnavailable(res.status)) {
    t.skip(skipMessage("release lookup", res.status));
    return undefined;
  }
  assert.equal(res.ok, true, `API ${res.status}`);
  return (await res.json()) as { assets: GhAsset[] };
}

describe("describeGithubHttpError", () => {
  const res = (headers: Record<string, string> = {}) =>
    new Response("", { status: 403, headers });

  it("explains a rate limit instead of dumping the JSON body", () => {
    const msg = describeGithubHttpError(
      403,
      res({ "x-ratelimit-remaining": "0" }),
      '{"message":"API rate limit exceeded for 1.2.3.4."}'
    );
    assert.match(msg, /GitHub API rate limit reached \(HTTP 403\)/);
    assert.match(msg, /60\/hour/);
    assert.match(msg, /Install from archive/);
    assert.doesNotMatch(msg, /message/, "raw JSON must not leak into the message");
  });

  it("includes the reset wait when GitHub reports one", () => {
    const reset = Math.floor((Date.now() + 5 * 60_000) / 1000);
    const msg = describeGithubHttpError(
      429,
      res({ "x-ratelimit-reset": String(reset) }),
      "rate limit"
    );
    assert.match(msg, /Try again in ~[45] min/);
  });

  it("keeps the plain message for a 403 that is not a rate limit", () => {
    // A genuinely forbidden request (e.g. blocked) must not claim a limit.
    const msg = describeGithubHttpError(403, res(), '{"message":"Forbidden"}');
    assert.match(msg, /^HTTP 403: /);
    assert.doesNotMatch(msg, /rate limit reached/);
  });

  it("keeps the plain message for non-403 statuses", () => {
    const notFound = describeGithubHttpError(404, res(), '{"message":"Not Found"}');
    assert.match(notFound, /^HTTP 404: /);
    const server = describeGithubHttpError(500, res(), "boom");
    assert.match(server, /^HTTP 500: /);
  });
});

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
    const release = await releaseOrSkip(t, tag);
    if (!release) {
      return;
    }
    const backend = process.platform === "linux" ? "vulkan" : "cpu";
    const picked = pickAsset(release.assets, backend);
    assert.ok(picked, `no ${backend} asset in ${tag}`);
    // Prefer the browser_download_url: it is always present, and the
    // api.github.com asset endpoint redirects through another rate-limited hop.
    const assetUrl = picked.browser_download_url;
    const head = await fetch(assetUrl, {
      method: "HEAD",
      headers: { "User-Agent": "llama-aio-vs", Accept: "application/octet-stream" },
      redirect: "follow",
      signal: AbortSignal.timeout(20_000),
    });
    if (isUnavailable(head.status)) {
      t.skip(skipMessage(`HEAD ${picked.name}`, head.status));
      return;
    }
    assert.ok(head.ok, `HEAD ${picked.name} → HTTP ${head.status}`);
    const len = Number(head.headers.get("content-length") || 0);
    assert.ok(len > 1_000_000, `${picked.name} is only ${len} bytes`);
  });

  it("downloads 64KiB of the archive with gzip/zip magic", { timeout: 30_000 }, async (t) => {
    const tag = await latestTagOrSkip(t);
    if (!tag) {
      return;
    }
    const release = await releaseOrSkip(t, tag);
    if (!release) {
      return;
    }
    const backend = process.platform === "linux" ? "vulkan" : "cpu";
    const picked = pickAsset(release.assets, backend);
    assert.ok(picked, `no ${backend} asset in ${tag}`);
    const url = picked.browser_download_url;
    assert.ok(url, `no ${backend} asset URL in ${tag}`);
    const part = await fetch(url, {
      headers: {
        "User-Agent": "llama-aio-vs",
        Accept: "application/octet-stream",
        Range: "bytes=0-65535",
      },
      redirect: "follow",
      signal: AbortSignal.timeout(20_000),
    });
    if (isUnavailable(part.status)) {
      t.skip(skipMessage(`partial GET ${picked.name}`, part.status));
      return;
    }
    assert.ok(part.ok || part.status === 206, `partial GET HTTP ${part.status}`);
    const buf = Buffer.from(await part.arrayBuffer());
    const gzip = buf[0] === 0x1f && buf[1] === 0x8b;
    const zip = buf[0] === 0x50 && buf[1] === 0x4b;
    assert.ok(gzip || zip, `unexpected magic ${buf.slice(0, 4).toString("hex")}`);
  });
});
