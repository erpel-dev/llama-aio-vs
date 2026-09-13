import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { LiveTextGate, partialTagTailLength } from "../src/streamTextGate";
import { stripXmlToolCalls } from "../src/xmlToolCalls";

function stripThink(text: string): string {
  return text
    .replace(/<think\b[^>]*>[\s\S]*?<\/think>/gi, "")
    .replace(/<\/?think\b[^>]*>/gi, "")
    .trim();
}

/** Feed `chunks` through a gate and reconcile like the chat provider does. */
function run(chunks: string[]): { live: string[]; shown: string } {
  const gate = new LiveTextGate();
  const live: string[] = [];
  let raw = "";
  for (const c of chunks) {
    raw += c;
    const out = gate.push(c);
    if (out) {
      live.push(out);
    }
  }
  const finalVisible = stripXmlToolCalls(stripThink(raw));
  const rest = gate.finish(finalVisible);
  return { live, shown: live.join("") + rest };
}

describe("partialTagTailLength", () => {
  it("holds fragments that could become <tool_call or <think", () => {
    assert.equal(partialTagTailLength("hello <to"), 3);
    assert.equal(partialTagTailLength("hello <tool_cal"), 9);
    assert.equal(partialTagTailLength("x <thi"), 4);
    assert.equal(partialTagTailLength("x </thinki"), 8);
    assert.equal(partialTagTailLength("<think foo="), 11);
  });

  it("does not hold ordinary text or completed tags", () => {
    assert.equal(partialTagTailLength("a < b"), 0);
    assert.equal(partialTagTailLength("x <div>"), 0);
    assert.equal(partialTagTailLength("x <span"), 0);
    assert.equal(partialTagTailLength("no tags"), 0);
  });
});

describe("LiveTextGate", () => {
  it("streams plain text immediately, in order", () => {
    const { live, shown } = run(["Hello", ", ", "world", "!"]);
    assert.deepEqual(live, ["Hello", ", ", "world", "!"]);
    assert.equal(shown, "Hello, world!");
  });

  it("withholds only the ambiguous tail and releases it when it is not a tag", () => {
    const gate = new LiveTextGate();
    assert.equal(gate.push("use a <to"), "use a ");
    assert.equal(gate.push("ken> here"), "<token> here");
    assert.equal(gate.isHalted, false);
  });

  it("halts on a tool call split across chunks and never shows the XML", () => {
    const { live, shown } = run([
      "Let me read that file.\n",
      "<tool_",
      "call>\nread_file\n<arg_key>path</arg_key><arg_value>/a.ts</arg_value>\n",
      "</tool_call>",
    ]);
    assert.deepEqual(live, ["Let me read that file.\n"]);
    // Whitespace already streamed before the block stays; no XML leaks.
    assert.equal(shown.trim(), "Let me read that file.");
    assert.ok(!/tool_call|arg_key/.test(shown));
  });

  it("shows text that follows a tool-call block once the stream has ended", () => {
    const { live, shown } = run([
      "Before. ",
      "<tool_call>\ngrep\n<arg_key>q</arg_key><arg_value>x</arg_value>\n</tool_call>",
      "\nAfter.",
    ]);
    assert.deepEqual(live, ["Before. "]);
    assert.equal(shown, "Before. \nAfter.");
  });

  it("removes think blocks from the live stream like the final pass", () => {
    const { live, shown } = run(["<think>plan", "ning…</think>", "Answer ", "here"]);
    assert.deepEqual(live, ["Answer ", "here"]);
    assert.equal(shown, "Answer here");
  });

  it("drops a stray closing think tag", () => {
    const { shown } = run(["oops</think>", " fine"]);
    assert.equal(shown, "oops fine");
  });

  it("trims leading whitespace of the first emission only", () => {
    const { live } = run(["\n\n", "Hi", " there"]);
    assert.deepEqual(live, ["Hi", " there"]);
  });

  it("does not duplicate already-shown text at finish", () => {
    const gate = new LiveTextGate();
    assert.equal(gate.push("Done."), "Done.");
    assert.equal(gate.finish("Done."), "");
  });

  it("shows the final text when nothing was streamed (reasoning-only reply)", () => {
    const gate = new LiveTextGate();
    assert.equal(gate.finish("from reasoning"), "from reasoning");
  });

  it("does not leak a partial opener when the stream ends mid-tag", () => {
    const gate = new LiveTextGate();
    assert.equal(gate.push("text <tool_"), "text ");
    // Final pass strips the unclosed <tool_ fragment? It does not (not a tag) —
    // so the remainder is returned to keep live + rest == final.
    assert.equal(gate.finish("text <tool_"), "<tool_");
  });

  it("tool call inside a longer chunk keeps preceding prose", () => {
    const { live, shown } = run(["Sure.\n\n<tool_call>\nls\n</tool_call>"]);
    assert.deepEqual(live, ["Sure.\n\n"]);
    assert.equal(shown.trim(), "Sure.");
    assert.ok(!shown.includes("<tool_call"));
  });
});
