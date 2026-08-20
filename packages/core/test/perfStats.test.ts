import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { PerfStats } from "../src/perfStats";

describe("PerfStats live generation rate", () => {
  it("does not publish a tok/s spike in the first few hundred milliseconds", () => {
    const perf = new PerfStats();
    perf.begin({ contextLimit: 4096, estimatedPromptTokens: 100 });
    assert.equal(perf.get().generating, true);
    perf.tick(40, { now: Date.now() });
    assert.equal(perf.get().genTokPerSec, undefined);
    assert.equal(perf.get().completionTokens, 40);
  });

  it("estimates tok/s after enough tokens and elapsed time", () => {
    const perf = new PerfStats();
    const t0 = 1_000_000;
    perf.begin({ contextLimit: 4096, estimatedPromptTokens: 100 });
    perf.tick(8, { now: t0 });
    assert.equal(perf.get().genTokPerSec, undefined);
    perf.tick(40, { now: t0 + 800 });
    const live = perf.get();
    assert.equal(live.generating, true);
    assert.equal(live.completionTokens, 40);
    assert.ok(typeof live.genTokPerSec === "number" && live.genTokPerSec > 0);
    assert.ok(live.genTokPerSec < 200, `expected a settled estimate, got ${live.genTokPerSec}`);
    assert.equal(live.estimated, true);
  });

  it("keeps the previous call's prompt / MTP tiles until this request completes", () => {
    const perf = new PerfStats();
    perf.begin({ contextLimit: 4096 });
    perf.complete({
      genTokPerSec: 42,
      promptTokPerSec: 900,
      completionTokens: 100,
      promptTokens: 2000,
      draftTokens: 80,
      draftTokensAccepted: 72,
      cachedPromptTokens: 1800,
      processedPromptTokens: 200,
      speculativeMode: "mtp",
      source: "server",
    });
    perf.begin({ contextLimit: 4096, speculativeMode: "mtp" });
    const live = perf.get();
    assert.equal(live.generating, true);
    assert.equal(live.showingPreviousCall, true);
    assert.equal(live.promptTokPerSec, 900);
    assert.equal(live.draftAcceptancePct, 90);
    assert.equal(live.cacheHitPct, 90);
    assert.equal(live.history?.length, 1);
  });

  it("records a short history of completed calls", () => {
    const perf = new PerfStats();
    for (let i = 0; i < 3; i++) {
      perf.begin({ contextLimit: 4096 });
      perf.complete({
        genTokPerSec: 30 + i,
        completionTokens: 10 + i,
        source: "server",
      });
    }
    const hist = perf.get().history || [];
    assert.equal(hist.length, 3);
    assert.equal(hist[0]?.genTokPerSec, 32);
    assert.equal(hist[2]?.genTokPerSec, 30);
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
    const t0 = 1_000_000;
    perf.begin({ contextLimit: 4096 });
    perf.tick(80, { now: t0 });
    perf.tick(80, { now: t0 + 800 });
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
