import assert from "node:assert/strict";
import { describe, it, beforeEach } from "node:test";
import {
  OVERTHINKING_MARKERS,
  cachedOverthinkingTokenIds,
  clearOverthinkingTokenCache,
  logitBiasMap,
  markerTexts,
  mergeTokenIds,
  overthinkingLogitBias,
  singleTokenId,
} from "../src/overthinkingBias";

describe("overthinking logit bias", () => {
  beforeEach(() => {
    clearOverthinkingTokenCache();
  });

  it("keeps a single token id and drops splits", () => {
    assert.equal(singleTokenId([{ id: 13428 }]), 13428);
    assert.equal(singleTokenId([{ id: 265 }, { id: 23934 }]), undefined);
    assert.equal(singleTokenId([]), undefined);
    assert.equal(singleTokenId([{ id: -1 }]), undefined);
  });

  it("dedupes ids", () => {
    assert.deepEqual(mergeTokenIds([1, undefined, 1, 2]), [1, 2]);
  });

  it("builds a negative logit_bias map", () => {
    assert.deepEqual(logitBiasMap([8106, 13428], 2), { "8106": -2, "13428": -2 });
    assert.deepEqual(logitBiasMap([1], -3), { "1": -3 });
  });

  it("tokenizes each marker with and without a leading space", () => {
    assert.equal(OVERTHINKING_MARKERS.length, 49);
    const texts = markerTexts();
    assert.equal(texts.length, 98);
    assert.equal(texts[0], " perhaps");
    assert.equal(texts[1], "perhaps");
  });

  it("caches single-token ids and skips a split bare word", async () => {
    let calls = 0;
    const assigned = new Map<string, number>();
    let next = 1;
    const ids = await cachedOverthinkingTokenIds("http://127.0.0.1:9", "model-a", async (text) => {
      calls += 1;
      if (text === "hmm") {
        return [{ id: 9001 }, { id: 9002 }];
      }
      let id = assigned.get(text);
      if (id === undefined) {
        id = next++;
        assigned.set(text, id);
      }
      return [{ id }];
    });
    assert.equal(calls, 98);
    assert.equal(ids.length, 97);
    assert.ok(!ids.includes(9001));
    assert.ok(!ids.includes(9002));
    assert.ok(ids.includes(assigned.get(" hmm")!));
    assert.ok(ids.includes(assigned.get(" Wait")!));
    assert.ok(ids.includes(assigned.get("Wait")!));
    assert.notEqual(assigned.get(" Wait"), assigned.get("Wait"));

    const again = await cachedOverthinkingTokenIds("http://127.0.0.1:9", "model-a", async () => {
      calls += 1;
      return [{ id: 1 }];
    });
    assert.equal(calls, 98);
    assert.deepEqual(again, ids);
  });

  it("returns undefined when the penalty is off or tokenize fails", async () => {
    assert.equal(await overthinkingLogitBias("http://127.0.0.1:9", "m", false, 2), undefined);
    assert.equal(
      await overthinkingLogitBias("http://127.0.0.1:1", "missing", true, 2),
      undefined
    );
  });
});
