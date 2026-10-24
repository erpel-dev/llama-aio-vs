import assert from "node:assert/strict";
import { messageContentChars } from "../src/promptReplacer";
import { describe, it } from "node:test";
import {
  estimateContextBreakdown,
  formatExceedContextError,
  messagesHaveImageParts,
} from "../src/contextBreakdown";

describe("estimateContextBreakdown images", () => {
  it("includes retained assistant reasoning in prompt and history budgets", () => {
    const plain = [{ role: "assistant", content: "answer" }, { role: "user", content: "continue" }];
    const retained = [{ ...plain[0], reasoning_content: "x".repeat(4000) }, plain[1]];
    assert.equal(messageContentChars(retained) - messageContentChars(plain), 4000);
    const before = estimateContextBreakdown(plain, [], 32768);
    const after = estimateContextBreakdown(retained, [], 32768);
    assert.equal(after.usedTokens - before.usedTokens, 1000);
    assert.equal(after.segments.find(s => s.key === "history")!.tokens - before.segments.find(s => s.key === "history")!.tokens, 1000);
  });
  it("does not treat image base64 as hundreds of thousands of tokens", () => {
    const huge = "A".repeat(2_000_000);
    const bd = estimateContextBreakdown(
      [
        {
          role: "user",
          content: [
            { type: "text", text: "what is this?" },
            { type: "image_url", image_url: { url: `data:image/png;base64,${huge}` } },
          ],
        },
      ],
      [],
      128000
    );
    assert.ok(bd.usedTokens < 2000, `expected a CLIP stub, got ${bd.usedTokens}`);
  });
});

describe("messagesHaveImageParts", () => {
  it("detects image_url parts", () => {
    assert.equal(
      messagesHaveImageParts([{ content: [{ type: "image_url", image_url: { url: "x" } }] }]),
      true
    );
    assert.equal(messagesHaveImageParts([{ content: "hello" }]), false);
  });
});

describe("formatExceedContextError", () => {
  it("does not recommend 65536 when the prompt cannot fit the model max", () => {
    const msg = formatExceedContextError({
      nPrompt: 569953,
      nCtx: 128000,
      maxContext: 262144,
      hasImages: true,
    });
    assert.match(msg, /569953 tokens > 128000 slot context/);
    assert.match(msg, /will not fit/i);
    assert.match(msg, /Pictures/);
    assert.doesNotMatch(msg, /≥ 65536/);
    assert.doesNotMatch(msg, /Reload to apply/);
  });

  it("tells the user the slot to raise when a larger context would fit", () => {
    const msg = formatExceedContextError({
      nPrompt: 90000,
      nCtx: 32768,
      maxContext: 262144,
    });
    assert.match(msg, /at least 90000/);
    assert.match(msg, /Reload to apply/);
    assert.doesNotMatch(msg, /will not fit/i);
  });

  it("tells the server operator to restart when this client cannot reload", () => {
    const msg = formatExceedContextError({
      nPrompt: 41000,
      nCtx: 32768,
      maxContext: 262144,
      operatorLabel: "gpu-box",
    });
    assert.match(msg, /gpu-box's slot is 32768/);
    assert.match(msg, /--ctx-size/);
    assert.doesNotMatch(msg, /Reload to apply/);
  });
});
