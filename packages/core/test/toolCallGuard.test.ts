import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  duplicateToolCallHint,
  expandLiteralFileGlob,
  fingerprintsSinceLastUserMessage,
  rewriteToolInput,
  toolCallFingerprint,
} from "../src/toolCallGuard";

describe("expandLiteralFileGlob", () => {
  it("expands **/llama to a contains-glob", () => {
    assert.equal(expandLiteralFileGlob("**/llama"), "**/*llama*");
    assert.equal(expandLiteralFileGlob("llama"), "**/*llama*");
  });

  it("leaves real globs and filenames alone", () => {
    assert.equal(expandLiteralFileGlob("**/*llama*"), undefined);
    assert.equal(expandLiteralFileGlob("**/*.ts"), undefined);
    assert.equal(expandLiteralFileGlob("**/llama.cpp"), undefined);
  });
});

describe("rewriteToolInput", () => {
  it("rewrites file_search globs", () => {
    assert.deepEqual(rewriteToolInput("file_search", { globPattern: "**/llama" }), {
      globPattern: "**/*llama*",
    });
    assert.deepEqual(rewriteToolInput("grep_search", { query: "llama" }), { query: "llama" });
  });
});

describe("fingerprintsSinceLastUserMessage", () => {
  it("records the original call, not the expanded glob", () => {
    const seen = fingerprintsSinceLastUserMessage([
      { role: "user" },
      {
        role: "assistant",
        tool_calls: [
          {
            function: {
              name: "file_search",
              arguments: JSON.stringify({ globPattern: "**/llama" }),
            },
          },
        ],
      },
    ]);
    assert.ok(seen.has(toolCallFingerprint("file_search", { globPattern: "**/llama" })));
    const expanded = rewriteToolInput("file_search", { globPattern: "**/llama" });
    assert.equal(seen.has(toolCallFingerprint("file_search", expanded)), false);
  });

  it("ignores tool calls from before the last user message", () => {
    const oldRead = {
      role: "assistant" as const,
      tool_calls: [
        {
          function: {
            name: "read_file",
            arguments: JSON.stringify({ path: "packages/core/src/toolCallGuard.ts" }),
          },
        },
      ],
    };
    const seen = fingerprintsSinceLastUserMessage([
      oldRead,
      { role: "user" },
    ]);
    assert.equal(seen.size, 0);
  });

  it("still records retries after the last user message", () => {
    const fp = toolCallFingerprint("read_file", { path: "foo.ts" });
    const seen = fingerprintsSinceLastUserMessage([
      {
        role: "assistant",
        tool_calls: [{ function: { name: "read_file", arguments: JSON.stringify({ path: "foo.ts" }) } }],
      },
      { role: "user" },
      {
        role: "assistant",
        tool_calls: [{ function: { name: "grep_search", arguments: JSON.stringify({ query: "bar" }) } }],
      },
    ]);
    assert.equal(seen.has(fp), false);
    assert.ok(seen.has(toolCallFingerprint("grep_search", { query: "bar" })));
  });
});

describe("duplicateToolCallHint", () => {
  it("mentions the tool", () => {
    assert.match(duplicateToolCallHint("file_search", { globPattern: "**/llama" }), /file_search/);
  });
});
