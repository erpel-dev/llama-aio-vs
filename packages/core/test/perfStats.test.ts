import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { PerfStats } from "../src/perfStats";

describe("PerfStats live generation rate", () => {
  it("estimates tok/s from completion tokens after generation starts", () => {
    const perf = new PerfStats();
    perf.begin({ contextLimit: 4096, estimatedPromptTokens: 100 });
    assert.equal(perf.get().generating, true);
    assert.equal(perf.get().genTokPerSec, undefined);

    perf.tick(40);
    const live = perf.get();
    assert.equal(live.generating, true);
    assert.equal(live.completionTokens, 40);
    assert.ok(typeof live.genTokPerSec === "number" && live.genTokPerSec > 0);
    assert.equal(live.estimated, true);
  });

  it("keeps a server tok/s across later ticks that have no new rate", () => {
    const perf = new PerfStats();
    perf.begin({ contextLimit: 4096 });
    perf.tick(100, { genTokPerSec: 37.18 });
    perf.tick(200);
    const live = perf.get();
    assert.equal(live.genTokPerSec, 37.18);
    assert.equal(live.completionTokens, 200);
    assert.equal(live.estimated, false);
    assert.equal(live.generating, true);
  });

  it("prefers a positive server rate over the wall-clock estimate", () => {
    const perf = new PerfStats();
    perf.begin({ contextLimit: 4096 });
    perf.tick(100, { genTokPerSec: 37.18, promptTokPerSec: 508.9 });
    const live = perf.get();
    assert.equal(live.genTokPerSec, 37.18);
    assert.equal(live.promptTokPerSec, 508.9);
    assert.equal(live.estimated, false);
    assert.equal(live.source, "server");
    assert.equal(live.generating, true);
  });

  it("ignores a zero server gen rate and keeps the live estimate", () => {
    const perf = new PerfStats();
    perf.begin({ contextLimit: 4096 });
    perf.tick(80);
    const estimated = perf.get().genTokPerSec;
    assert.ok(typeof estimated === "number" && estimated > 0);

    perf.complete({
      genTokPerSec: 0,
      promptTokPerSec: 0,
      completionTokens: 80,
      promptTokens: 200,
      source: "server",
    });
    const done = perf.get();
    assert.equal(done.generating, false);
    assert.ok(typeof done.genTokPerSec === "number" && done.genTokPerSec > 0);
    assert.equal(done.estimated, true);
  });

  it("does not tick after complete()", () => {
    const perf = new PerfStats();
    perf.begin({ contextLimit: 4096 });
    perf.complete({ completionTokens: 10, genTokPerSec: 30, source: "server" });
    perf.tick(999);
    assert.equal(perf.get().completionTokens, 10);
    assert.equal(perf.get().generating, false);
  });
});
