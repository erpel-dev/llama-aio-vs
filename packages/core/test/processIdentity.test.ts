import assert from "node:assert/strict";
import * as path from "node:path";
import { describe, it } from "node:test";
import {
  isLlamaServerProcess,
  looksLikeLlamaServer,
  sameModelFile,
  uniquePids,
} from "../src/processIdentity";

describe("looksLikeLlamaServer", () => {
  it("accepts a llama-server command line", () => {
    assert.ok(looksLikeLlamaServer("/home/u/.llama-aio/bin/llama-server -m model.gguf --port 8742"));
    assert.ok(looksLikeLlamaServer("C:\\Llama\\LLAMA-SERVER.EXE -m x.gguf"));
  });

  it("rejects unrelated processes that merely hold a port", () => {
    for (const cmd of ["node /app/server.js --port 8742", "python -m http.server", "nginx: worker"]) {
      assert.equal(looksLikeLlamaServer(cmd), false, cmd);
    }
  });

  it("treats an unreadable command line as foreign", () => {
    // Unknown must mean "do not touch", never "safe to kill".
    assert.equal(looksLikeLlamaServer(undefined), false);
    assert.equal(looksLikeLlamaServer(""), false);
  });
});

describe("isLlamaServerProcess", () => {
  it("does not claim the current test runner", () => {
    assert.equal(isLlamaServerProcess(process.pid), false);
  });

  it("returns false for a pid that cannot exist", () => {
    assert.equal(isLlamaServerProcess(0), false);
    assert.equal(isLlamaServerProcess(2 ** 30), false);
  });
});

describe("sameModelFile", () => {
  it("matches a path against itself", () => {
    assert.ok(sameModelFile("/models/a.gguf", "/models/a.gguf"));
  });

  it("ignores redundant path segments", () => {
    assert.ok(sameModelFile("/models/./a.gguf", "/models/sub/../a.gguf"));
  });

  it("does not confuse same-named models in different folders", () => {
    // The old basename comparison reused the wrong server for this pair.
    assert.equal(
      sameModelFile("/models/qwen/model-Q4_K_M.gguf", "/models/llama/model-Q4_K_M.gguf"),
      false
    );
  });

  it("is false when either side is missing", () => {
    assert.equal(sameModelFile(undefined, "/models/a.gguf"), false);
    assert.equal(sameModelFile("/models/a.gguf", ""), false);
  });

  it("resolves relative paths against the cwd", () => {
    assert.ok(sameModelFile("a.gguf", path.join(process.cwd(), "a.gguf")));
  });
});

describe("uniquePids", () => {
  it("parses and dedupes", () => {
    assert.deepEqual(uniquePids(["12", "12", "34"]), [12, 34]);
  });

  it("drops junk, zero and negative values", () => {
    assert.deepEqual(uniquePids(["", undefined, "abc", "0", "-1", "7"]), [7]);
  });
});
