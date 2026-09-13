import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { after, describe, it } from "node:test";
import type { ConfigAccessor } from "../src/config";
import {
  invalidateModelLibraryCache,
  listLocalModelEntries,
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
