import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { after, describe, it } from "node:test";
import type { ConfigAccessor } from "../src/config";
import {
  buildLibraryGroups,
  invalidateModelLibraryCache,
  isOwnedModelPath,
  listLocalModelEntries,
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
