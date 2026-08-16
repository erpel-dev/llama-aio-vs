import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { positiveRate, ratesFromTimings } from "../src/llamaTimings";

describe("positiveRate", () => {
  it("keeps a finite tok/s value", () => {
    assert.equal(positiveRate(37.18), 37.18);
  });

  it("drops zero, NaN, and non-numbers", () => {
    assert.equal(positiveRate(0), undefined);
    assert.equal(positiveRate(-1), undefined);
    assert.equal(positiveRate(Number.NaN), undefined);
    assert.equal(positiveRate("37"), undefined);
  });
});

describe("ratesFromTimings", () => {
  it("prefers predicted_per_second when it is positive", () => {
    assert.deepEqual(
      ratesFromTimings({ predicted_per_second: 32.5, predicted_n: 100, predicted_ms: 10_000 }),
      { genTokPerSec: 32.5 }
    );
  });

  it("derives tok/s from n/ms when per_second is missing or zero", () => {
    assert.deepEqual(ratesFromTimings({ predicted_n: 1695, predicted_ms: 45_589 }), {
      genTokPerSec: 1695 / (45_589 / 1000),
    });
    assert.deepEqual(
      ratesFromTimings({ predicted_per_second: 0, predicted_n: 80, predicted_ms: 2000 }),
      { genTokPerSec: 40 }
    );
  });

  it("derives prompt tok/s the same way", () => {
    assert.deepEqual(ratesFromTimings({ prompt_n: 2582, prompt_ms: 5089.51 }), {
      promptTokPerSec: 2582 / (5089.51 / 1000),
    });
  });

  it("returns nothing when timings are empty", () => {
    assert.deepEqual(ratesFromTimings(undefined), {});
    assert.deepEqual(ratesFromTimings({}), {});
  });
});
