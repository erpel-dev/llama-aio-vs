import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { after, describe, it } from "node:test";
import type { ConfigAccessor } from "../src/config";
import {
  buildLibraryGroups,
  buildLibraryGroupsAsync,
  findSiblingMmproj,
  findSiblingMtpDraft,
  invalidateModelLibraryCache,
  isOwnedModelPath,
  listLocalModelEntries,
  listMtpDraftEntries,
  listPartialDownloads,
  listShardPaths,
  removalPathsForQuant,
  withoutHiddenModels,
} from "../src/modelLibrary";

const LANGUAGE_MIN = 32 * 1024 * 1024;

function configFor(modelsDir: string): ConfigAccessor {
  return {
    get<T>(key: string, fallback?: T): T {
      if (key === "modelsDir") {
        return modelsDir as T;
      }
      if (key === "extraModelDirs") {
        return [] as T;
      }
      return fallback as T;
    },
    update: async () => undefined,
  };
}

function writeSparseGguf(file: string, size = LANGUAGE_MIN): void {
  const fd = fs.openSync(file, "w");
  try {
    fs.ftruncateSync(fd, size);
  } finally {
    fs.closeSync(fd);
  }
}

describe("listLocalModelEntries cache", () => {
  const dirs: string[] = [];
  after(() => {
    invalidateModelLibraryCache();
    for (const dir of dirs) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it("holds the scan until invalidate, then sees a newly added GGUF", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-lib-"));
    dirs.push(dir);
    writeSparseGguf(path.join(dir, "alpha.gguf"));
    const config = configFor(dir);
    invalidateModelLibraryCache();

    const first = listLocalModelEntries(config).filter((e) => e.path.startsWith(dir));
    assert.equal(first.length, 1);
    assert.equal(path.basename(first[0]!.path), "alpha.gguf");
    assert.equal(first[0]!.source, "Llama AIO");

    writeSparseGguf(path.join(dir, "bravo.gguf"));
    const cached = listLocalModelEntries(config).filter((e) => e.path.startsWith(dir));
    assert.equal(cached.length, 1, "second scan served from the 10 s cache");

    invalidateModelLibraryCache();
    const fresh = listLocalModelEntries(config).filter((e) => e.path.startsWith(dir));
    assert.equal(fresh.length, 2);
    assert.deepEqual(
      fresh.map((e) => path.basename(e.path)).sort(),
      ["alpha.gguf", "bravo.gguf"]
    );
  });

  it("returns copies so callers cannot poison the memo", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-lib-"));
    dirs.push(dir);
    writeSparseGguf(path.join(dir, "alpha.gguf"));
    const config = configFor(dir);
    invalidateModelLibraryCache();
    const first = listLocalModelEntries(config).find((e) => e.path.startsWith(dir));
    assert.ok(first);
    first.source = "mutated";
    const second = listLocalModelEntries(config).find((e) => e.path.startsWith(dir));
    assert.equal(second?.source, "Llama AIO");
  });
});

describe("library groups and removal", () => {
  const dirs: string[] = [];
  after(() => {
    for (const dir of dirs) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it("keeps MTP directory sidecars out of the language picker", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-mtp-context-"));
    dirs.push(dir);
    fs.mkdirSync(path.join(dir, "MTP"));
    const draft = path.join(dir, "MTP", "Model-Q4_0.gguf");
    writeSparseGguf(draft);
    writeSparseGguf(path.join(dir, "ggml-vocab-test.gguf"));
    const main = path.join(dir, "Model-Q4_0.gguf");
    writeSparseGguf(main);
    invalidateModelLibraryCache();
    const config = configFor(dir);
    assert.deepEqual(listLocalModelEntries(config).filter((e) => e.path.startsWith(dir)).map((e) => e.path), [main]);
    assert.deepEqual(listMtpDraftEntries(config).filter((e) => e.path.startsWith(dir)).map((e) => e.path), [draft]);
  });

  it("finds shared repo companions for preserved quant subdirectories", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-hf-companions-"));
    dirs.push(dir);
    const repo = path.join(dir, "org__repo");
    const quantDir = path.join(repo, "Q4");
    fs.mkdirSync(quantDir, { recursive: true });
    fs.mkdirSync(path.join(repo, "MTP"));
    const main = path.join(quantDir, "Model-Q4_0.gguf");
    const projector = path.join(repo, "mmproj-F16.gguf");
    const draft = path.join(repo, "MTP", "Model-Q4_0.gguf");
    writeSparseGguf(main);
    writeSparseGguf(projector, 1024 * 1024);
    writeSparseGguf(draft, 1024 * 1024);
    assert.equal(findSiblingMmproj(main), projector);
    assert.equal(findSiblingMtpDraft(main), draft);
    const entries = [{ path: main, source: "Llama AIO", sizeBytes: LANGUAGE_MIN }];
    const groups = buildLibraryGroups(entries, dir);
    assert.deepEqual(groups[0]?.companions.map((c) => c.path), [projector, draft]);
    assert.deepEqual(await buildLibraryGroupsAsync(entries, dir), groups);
    const nearby = path.join(quantDir, "mmproj-Q8_0.gguf");
    writeSparseGguf(nearby, 1024 * 1024);
    assert.equal(findSiblingMmproj(main), nearby, "prefer a nearby quant-specific projector");
  });

  for (const count of [100, 300, 600]) {
    it(`uses linear filesystem work for ${count} flat model families`, (t) => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-flat-"));
      dirs.push(dir);
      const entries = Array.from({ length: count }, (_, i) => {
        const file = path.join(dir, `Model${i}-Q4_K_M.gguf`);
        writeSparseGguf(file);
        return { path: file, source: "Custom", sizeBytes: LANGUAGE_MIN };
      });
      const stat = t.mock.method(require("fs"), "statSync");
      const read = t.mock.method(require("fs"), "readdirSync");
      const realpath = t.mock.method(require("fs"), "realpathSync");
      const groups = buildLibraryGroups(entries, dir);
      assert.equal(groups.length, count);
      assert.ok(stat.mock.callCount() <= count * 2, `${stat.mock.callCount()} stat calls`);
      assert.ok(read.mock.callCount() <= 4, `${read.mock.callCount()} directory reads`);
      assert.ok(realpath.mock.callCount() <= count * 4, `${realpath.mock.callCount()} realpath calls`);
    });
  }

  it("groups asynchronously, shares cached work, and invalidates companion changes", async (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-async-groups-"));
    dirs.push(dir);
    const file = path.join(dir, "Model-Q4_K_M-00001-of-00002.gguf");
    const otherShard = path.join(dir, "Model-Q4_K_M-00002-of-00002.gguf");
    writeSparseGguf(file);
    writeSparseGguf(otherShard);
    const mtpDir = path.join(dir, "MTP");
    fs.mkdirSync(mtpDir);
    const draft = path.join(mtpDir, "Model-Q4_K_M.gguf");
    writeSparseGguf(draft, 1024 * 1024);
    const entries = [{ path: file, source: "Custom", sizeBytes: LANGUAGE_MIN * 2, shardCount: 2 }];
    const expected = buildLibraryGroups(entries, dir);
    invalidateModelLibraryCache();
    const read = t.mock.method(fs.promises, "readdir");
    const noSync = () => { throw new Error("synchronous filesystem access during async grouping"); };
    const statSync = t.mock.method(require("fs"), "statSync", noSync);
    const readSync = t.mock.method(require("fs"), "readdirSync", noSync);
    const realSync = t.mock.method(require("fs"), "realpathSync", noSync);
    let yielded = false;
    setImmediate(() => { yielded = true; });
    const [first, second] = await Promise.all([
      buildLibraryGroupsAsync(entries, dir), buildLibraryGroupsAsync(entries, dir),
    ]);
    assert.ok(yielded, "cold grouping yields to the host event loop");
    assert.deepEqual(first, expected);
    assert.deepEqual(second, expected);
    assert.equal(read.mock.callCount(), 2, "one inventory for simultaneous identical payloads");
    assert.equal(first[0]?.companions[0]?.path, draft);
    first[0]!.companions.length = 0;
    assert.deepEqual(await buildLibraryGroupsAsync(entries, dir), expected, "cached results cannot be mutated by callers");
    statSync.mock.restore(); readSync.mock.restore(); realSync.mock.restore();
    const projector = path.join(dir, "mmproj-F16.gguf");
    writeSparseGguf(projector, 1024 * 1024);
    assert.deepEqual(await buildLibraryGroupsAsync(entries, dir), expected, "held until invalidate or TTL expiry");
    invalidateModelLibraryCache();
    const fresh = await buildLibraryGroupsAsync(entries, dir);
    assert.deepEqual(fresh, buildLibraryGroups(entries, dir));
    assert.equal(fresh[0]?.companions[0]?.path, projector);
  });

  it("keeps symlink ownership and baked-in MTP classification in indexed groups", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-indexed-"));
    const outside = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-indexed-outside-"));
    dirs.push(dir, outside);
    const file = path.join(dir, "Model-Q4_K_M.gguf");
    writeSparseGguf(file);
    writeSparseGguf(path.join(dir, "Model-Q4_K_M-mtp.gguf"));
    const target = path.join(outside, "external.gguf");
    writeSparseGguf(target);
    const link = path.join(dir, "Other-Q4_K_M.gguf");
    fs.symlinkSync(target, link);
    const entries = [file, link].map((p) => ({ path: p, source: "Custom", sizeBytes: LANGUAGE_MIN }));
    const groups = await buildLibraryGroupsAsync(entries, dir);
    assert.deepEqual(groups, buildLibraryGroups(entries, dir));
    assert.ok(groups.every((g) => g.companions.length === 0), "near-copy MTP is not a sidecar");
    assert.equal(groups.find((g) => g.quants[0]?.path === link)?.owned, false);
  });

  it("groups quants in one download folder and refuses paths outside it", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-own-"));
    dirs.push(root);
    const repo = path.join(root, "unsloth__Qwen3.5-9B-GGUF");
    fs.mkdirSync(repo);
    const q4 = path.join(repo, "Qwen3.5-9B-Q4_K_M.gguf");
    const q8 = path.join(repo, "Qwen3.5-9B-Q8_0.gguf");
    const proj = path.join(repo, "mmproj-F16.gguf");
    const projFd = fs.openSync(proj, "w");
    fs.ftruncateSync(projFd, 1024 * 1024);
    fs.closeSync(projFd);
    fs.writeFileSync(q4, "q4");
    fs.writeFileSync(q8, "q8");
    fs.writeFileSync(path.join(repo, "Qwen3.5-9B-Q4_K_M.gguf.partial"), "partial");

    const outside = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-foreign-"));
    dirs.push(outside);
    const foreign = path.join(outside, "other-Q4_K_M.gguf");
    fs.writeFileSync(foreign, "nope");

    const groups = buildLibraryGroups(
      [
        { path: q4, source: "Llama AIO", sizeBytes: 2 },
        { path: q8, source: "Llama AIO", sizeBytes: 4 },
        { path: foreign, source: "LM Studio", sizeBytes: 3 },
      ],
      root
    );
    const owned = groups.find((g) => g.owned);
    assert.ok(owned);
    assert.equal(owned!.quants.length, 2);
    assert.equal(owned!.companions[0]?.role, "mmproj");
    assert.equal(owned!.companions[0]?.shared, true);
    assert.equal(isOwnedModelPath(q4, root), true);
    assert.equal(isOwnedModelPath(foreign, root), false);

    const keep = removalPathsForQuant(q4, root, []);
    assert.deepEqual(keep.rejected, []);
    assert.equal(keep.files.length, 1);

    const withProj = removalPathsForQuant(q4, root, [owned!.companions[0]!.path]);
    assert.equal(withProj.files.length, 2);

    const blocked = removalPathsForQuant(foreign, root, []);
    assert.equal(blocked.files.length, 0);
    assert.ok(blocked.rejected.length > 0);

    const partials = listPartialDownloads(root);
    assert.equal(partials.length, 1);
    assert.ok(partials[0]!.path.endsWith(".partial"));
  });

  it("associates each flat-folder model only with its named projector", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-projector-families-"));
    dirs.push(root);
    const entries = ["VL", "Audio", "Omni"].map(kind => {
      const model = path.join(root, `LFM3-${kind}-2B-Q4_K_M.gguf`);
      writeSparseGguf(model);
      writeSparseGguf(path.join(root, `mmproj-LFM3-${kind}-2B-Q8_0.gguf`), 1024 * 1024);
      return { path: model, source: "Llama AIO" as const, sizeBytes: LANGUAGE_MIN };
    });
    const groups = buildLibraryGroups(entries, root);
    for (const kind of ["VL", "Audio", "Omni"]) {
      const group = groups.find(g => g.quants.some(q => q.path.includes(`LFM3-${kind}-2B`)));
      assert.equal(path.basename(group!.companions[0]!.path), `mmproj-LFM3-${kind}-2B-Q8_0.gguf`);
    }
  });

  it("lists every shard and drops hidden paths from a visible list", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-shard-"));
    dirs.push(root);
    const a = path.join(root, "Model-00001-of-00002.gguf");
    const b = path.join(root, "Model-00002-of-00002.gguf");
    fs.writeFileSync(a, "a");
    fs.writeFileSync(b, "b");
    const shards = listShardPaths(a).map((p) => path.basename(p));
    assert.deepEqual(shards, ["Model-00001-of-00002.gguf", "Model-00002-of-00002.gguf"]);
    const plan = removalPathsForQuant(a, root, []);
    assert.equal(plan.rejected.length, 0);
    assert.equal(plan.files.length, 2);

    const entries = [
      { path: a, source: "Llama AIO", sizeBytes: 1 },
      { path: b, source: "Llama AIO", sizeBytes: 1 },
    ];
    const visible = withoutHiddenModels(entries, [a]);
    assert.equal(visible.length, 1);
    assert.equal(visible[0]!.path, b);
  });

  it("refuses a symlink that points outside the models folder", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-link-"));
    dirs.push(root);
    const outside = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-blob-"));
    dirs.push(outside);
    const target = path.join(outside, "secret.gguf");
    fs.writeFileSync(target, "secret");
    const link = path.join(root, "alias.gguf");
    try {
      fs.symlinkSync(target, link);
    } catch {
      return;
    }
    assert.equal(isOwnedModelPath(link, root), false);
    const plan = removalPathsForQuant(link, root, []);
    assert.equal(plan.files.length, 0);
  });
});
