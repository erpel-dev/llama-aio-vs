import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { formatToolResultContent, OMITTED_BINARY_TOOL_RESULT } from "../src/toolResultText";

describe("formatToolResultContent", () => {
  it("keeps text parts", () => {
    assert.equal(
      formatToolResultContent([{ value: "file contents" }, { text: "more" }]),
      "file contents\nmore"
    );
  });

  it("keeps structured non-binary results", () => {
    assert.equal(formatToolResultContent([{ stdout: "ok", code: 0 }]), '{"stdout":"ok","code":0}');
  });

  it("replaces image byte maps and buffers", () => {
    const byteMap: Record<string, number> = {};
    for (let i = 0; i < 16; i++) {
      byteMap[String(i)] = i;
    }
    assert.equal(
      formatToolResultContent([{ mimeType: "image/png", data: byteMap }]),
      OMITTED_BINARY_TOOL_RESULT
    );
    assert.equal(
      formatToolResultContent([{ mimeType: "image/png", data: Buffer.from([137, 80, 78, 71]) }]),
      OMITTED_BINARY_TOOL_RESULT
    );
  });
});