import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  collapseSplitGgufFiles,
  describeLanguageGgufFile,
  ggufHitsFromTree,
  languageGgufFiles,
  parseNextLink,
  remoteShardPaths,
  splitGgufGroupKey,
} from "../src/huggingFace";

describe("parseNextLink", () => {
  it("reads the RFC 8288 next URL", () => {
    assert.equal(
      parseNextLink('</api/models/org/repo/tree/main?recursive=true&cursor=abc>; rel="next"'),
      "/api/models/org/repo/tree/main?recursive=true&cursor=abc"
    );
  });

  it("ignores other rels and missing headers", () => {
    assert.equal(parseNextLink('</x>; rel="prev"'), undefined);
    assert.equal(parseNextLink(undefined), undefined);
  });
});

describe("ggufHitsFromTree", () => {
  it("keeps nested .gguf files and skips directories", () => {
    const hits = ggufHitsFromTree("unsloth/Qwen3.8-Flash-Next-GGUF", [
      { type: "directory", path: "UD-Q4_K_XL", size: 0 },
      { type: "file", path: "README.md", size: 100 },
      { type: "file", path: "mmproj-F16.gguf", size: 9e8 },
      {
        type: "file",
        path: "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00002.gguf",
        size: 1000,
      },
    ]);
    assert.deepEqual(
      hits.map((f) => f.path).sort(),
      [
        "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00002.gguf",
        "mmproj-F16.gguf",
      ].sort()
    );
  });
});

describe("split GGUF picker collapse", () => {
  const shards = [
    {
      path: "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
      size: 10,
      url: "",
    },
    {
      path: "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf",
      size: 20,
      url: "",
    },
    {
      path: "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf",
      size: 30,
      url: "",
    },
    {
      path: "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf",
      size: 40,
      url: "",
    },
    { path: "mmproj-F16.gguf", size: 9e8, url: "" },
    { path: "MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf", size: 2e9, url: "" },
    { path: "UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00002.gguf", size: 5, url: "" },
    { path: "UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00002-of-00002.gguf", size: 7, url: "" },
  ];

  it("groups shards that share a stem and folder", () => {
    assert.equal(
      splitGgufGroupKey("UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf"),
      "UD-Q4_K_XL/qwen3.8-flash-next-ud-q4_k_xl"
    );
    assert.equal(splitGgufGroupKey("plain-Q4_K_M.gguf"), undefined);
  });

  it("lists one language row per quant with summed size", () => {
    const rows = languageGgufFiles(shards);
    assert.deepEqual(
      rows.map((f) => f.path),
      [
        "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
        "UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00002.gguf",
      ]
    );
    assert.equal(rows[0]?.size, 100);
    assert.equal(rows[0]?.shardCount, 4);
    assert.equal(rows[1]?.size, 12);
    assert.equal(rows[1]?.shardCount, 2);
  });

  it("labels the picker with shard count", () => {
    const [row] = collapseSplitGgufFiles(shards.slice(0, 4));
    assert.match(describeLanguageGgufFile(row!, shards), /4 shards/);
  });

  it("expands a picked shard into every remote part", () => {
    assert.deepEqual(remoteShardPaths("UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"), [
      "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
      "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf",
      "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf",
      "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf",
    ]);
    assert.deepEqual(remoteShardPaths("mmproj-F16.gguf"), ["mmproj-F16.gguf"]);
  });
});
