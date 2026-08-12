import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { decodeSseLines, toolCallSlot } from "../src/sseStream";

async function* feed(chunks: Array<Uint8Array | string>): AsyncGenerator<Uint8Array | string> {
  for (const c of chunks) {
    yield c;
  }
}

async function collect(chunks: Array<Uint8Array | string>): Promise<string[]> {
  const out: string[] = [];
  for await (const line of decodeSseLines(feed(chunks))) {
    out.push(line);
  }
  return out;
}

describe("decodeSseLines", () => {
  it("splits complete lines", async () => {
    assert.deepEqual(await collect(["a\nb\nc\n"]), ["a", "b", "c"]);
  });

  it("joins a line split across chunks", async () => {
    assert.deepEqual(await collect(["data: {\"a\":", "1}\n"]), ['data: {"a":1}']);
  });

  it("keeps a multi-byte character split across chunks intact", async () => {
    // "ü" is 0xC3 0xBC — split the two bytes into separate chunks.
    const bytes = Buffer.from("data: über\n", "utf8");
    const cut = bytes.indexOf(0xc3) + 1;
    const lines = await collect([bytes.subarray(0, cut), bytes.subarray(cut)]);
    assert.deepEqual(lines, ["data: über"]);
  });

  it("survives an emoji split across three chunks", async () => {
    const bytes = Buffer.from("x 🚀 y\n", "utf8");
    const i = bytes.indexOf(0xf0);
    const lines = await collect([
      bytes.subarray(0, i + 1),
      bytes.subarray(i + 1, i + 3),
      bytes.subarray(i + 3),
    ]);
    assert.deepEqual(lines, ["x 🚀 y"]);
  });

  it("flushes a trailing line that never got a newline", async () => {
    assert.deepEqual(await collect(["a\ndata: last"]), ["a", "data: last"]);
  });

  it("emits nothing for an empty stream", async () => {
    assert.deepEqual(await collect([]), []);
  });

  it("preserves blank lines between SSE events", async () => {
    assert.deepEqual(await collect(["data: 1\n\ndata: 2\n\n"]), ["data: 1", "", "data: 2", ""]);
  });

  it("aborts instead of buffering forever without a newline", async () => {
    const huge = "x".repeat(1024 * 1024);
    await assert.rejects(
      () => collect(Array.from({ length: 10 }, () => huge)),
      /over 8 MB/
    );
  });
});

describe("toolCallSlot", () => {
  const empty = () => [] as Array<{ id: string; name: string; arguments: string }>;

  it("uses the index when the server sends one", () => {
    assert.equal(toolCallSlot(empty(), { index: 2 }), 2);
    assert.equal(toolCallSlot(empty(), { index: 0 }), 0);
  });

  it("keeps two id-only calls apart instead of merging them", () => {
    const calls = empty();
    const a = toolCallSlot(calls, { id: "call_a" });
    calls[a] = { id: "call_a", name: "read", arguments: "" };
    const b = toolCallSlot(calls, { id: "call_b" });
    assert.notEqual(a, b);
  });

  it("routes a follow-up delta back to its own call by id", () => {
    const calls = [
      { id: "call_a", name: "read", arguments: "{" },
      { id: "call_b", name: "write", arguments: "{" },
    ];
    assert.equal(toolCallSlot(calls, { id: "call_a" }), 0);
    assert.equal(toolCallSlot(calls, { id: "call_b" }), 1);
  });

  it("appends argument fragments to the call in flight when nothing identifies them", () => {
    const calls = [{ id: "call_a", name: "read", arguments: "{" }];
    assert.equal(toolCallSlot(calls, {}), 0);
    calls.push({ id: "call_b", name: "write", arguments: "{" });
    assert.equal(toolCallSlot(calls, {}), 1);
  });
});
