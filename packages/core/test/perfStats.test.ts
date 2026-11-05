import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { PerfStats } from "../src/perfStats";

it("keeps thinking counts and finish reasons in response diagnostics", () => {
  const perf = new PerfStats();
  perf.recordResponseTrace({ model: "Flint", toolsEnabled: true, assembledText: "", assembledReasoning: "Still thinking",
    visibleText: "", structuredToolCalls: [], xmlToolCalls: [], emittedTextChars: 0, emittedReasoningChars: 14,
    emittedToolCallCount: 0, finishReason: "length" });
  assert.equal(perf.getLastResponseTrace()?.finishReason, "length");
  assert.match(perf.formatLastResponseTrace() || "", /Thinking emitted: 14 chars/);
  assert.match(perf.formatLastResponseTrace() || "", /Finish reason: length/);
});

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

describe("PerfStats request and target identity (#33)", () => {
  const done = (perf: PerfStats, tokens: number, gen: number, id?: number) =>
    perf.complete({ completionTokens: tokens, genTokPerSec: gen, promptTokPerSec: 100, source: "server" }, id);

  it("a target change clears the live numbers before the new model's first request", () => {
    const perf = new PerfStats();
    perf.setTarget("local|a|LFM3");
    done(perf, 40, 10.1, perf.begin({ contextLimit: 8192, estimatedPromptTokens: 500 }));
    assert.equal(perf.get().genTokPerSec, 10.1);
    perf.setTarget("local|a|Spark");
    const p = perf.get();
    assert.equal(p.genTokPerSec, undefined);
    assert.equal(p.contextPct, undefined);
    assert.equal(p.completionTokens, undefined);
    assert.deepEqual(p.history, []);
  });

  it("local → remote → local shows each target's own history, not the other's rate", () => {
    const perf = new PerfStats();
    perf.setTarget("local|Spark");
    done(perf, 2, 9, perf.begin());
    perf.setTarget("remote|LFM2.5");
    done(perf, 30, 77.7, perf.begin());
    perf.setTarget("local|Spark");
    const p = perf.get();
    assert.equal(p.genTokPerSec, undefined, "live tile must not show the remote 77.7 tok/s");
    assert.deepEqual(p.history?.map((h) => h.genTokPerSec), [9]);
  });

  it("a request cancelled during prefill reports no tokens instead of the previous answer's", () => {
    const perf = new PerfStats();
    done(perf, 2, 20, perf.begin());
    const id = perf.begin({ estimatedPromptTokens: 2600, contextLimit: 8192 });
    perf.abort("cancelled", id);
    const p = perf.get();
    assert.equal(p.outcome, "cancelled");
    assert.equal(p.completionTokens, 0);
    assert.equal(p.history?.[0].outcome, "cancelled");
    assert.equal(p.history?.[0].completionTokens, 0);
    assert.equal(p.history?.[1].completionTokens, 2);
  });

  it("cancellation after partial output keeps only that request's tokens", () => {
    const perf = new PerfStats();
    const id = perf.begin();
    perf.tick(7, { requestId: id });
    perf.abort("cancelled", id);
    assert.equal(perf.get().completionTokens, 7);
    assert.equal(perf.get().outcome, "cancelled");
  });

  it("failed requests are distinguished from cancelled ones", () => {
    const perf = new PerfStats();
    perf.abort("failed", perf.begin());
    assert.equal(perf.get().outcome, "failed");
  });

  it("ignores late events from a request superseded by a target switch", () => {
    const perf = new PerfStats();
    perf.setTarget("local|old");
    const old = perf.begin();
    perf.setTarget("local|new");
    perf.tick(50, { genTokPerSec: 99, requestId: old });
    done(perf, 50, 99, old);
    perf.abort("cancelled", old);
    const p = perf.get();
    assert.equal(p.genTokPerSec, undefined);
    assert.equal(p.outcome, undefined);
    assert.deepEqual(p.history, []);
  });

  it("same target keeps last-call rates visible (flagged) during the next request", () => {
    const perf = new PerfStats();
    perf.setTarget("t");
    done(perf, 10, 30, perf.begin());
    perf.begin();
    const p = perf.get();
    assert.equal(p.genTokPerSec, 30);
    assert.equal(p.showingPreviousCall, true);
    assert.equal(p.outcome, undefined);
  });
});

describe("PerfStats request stages and prompt progress (#35)", () => {
  const pp = (processed: number, total = 8000, cache = 0) => ({ total, cache, processed, timeMs: processed });

  it("waiting → prompt (server-measured) → generating → cleared on completion", () => {
    const perf = new PerfStats();
    const id = perf.begin({ progressRequested: true, replyReserve: 4096 });
    assert.equal(perf.get().stage, "waiting");
    assert.equal(perf.get().progressRequested, true);
    assert.equal(perf.get().replyReserve, 4096);
    perf.promptProgress(pp(2048), id);
    assert.equal(perf.get().stage, "prompt");
    assert.deepEqual(perf.get().promptProgress, pp(2048));
    perf.tick(3, { requestId: id });
    assert.equal(perf.get().stage, "generating");
    perf.promptProgress(pp(8000), id); // late prompt event must not move the stage back
    assert.equal(perf.get().stage, "generating");
    perf.complete({ completionTokens: 3, source: "server" }, id);
    assert.equal(perf.get().stage, undefined);
  });

  it("cancel during prompt processing clears the stage and keeps the cancelled outcome", () => {
    const perf = new PerfStats();
    const id = perf.begin({ progressRequested: true });
    perf.promptProgress(pp(1024), id);
    perf.abort("cancelled", id);
    assert.equal(perf.get().stage, undefined);
    assert.equal(perf.get().outcome, "cancelled");
    assert.equal(perf.get().completionTokens, 0);
  });

  it("progress from an older request or another target is ignored", () => {
    const perf = new PerfStats();
    perf.setTarget("a");
    const old = perf.begin({ progressRequested: true });
    const now = perf.begin({ progressRequested: true });
    perf.promptProgress(pp(4000), old);
    assert.equal(perf.get().promptProgress, undefined);
    perf.setTarget("b");
    perf.promptProgress(pp(4000), now);
    assert.equal(perf.get().promptProgress, undefined);
    assert.equal(perf.get().stage, undefined);
  });

  it("a new request starts without the previous request's progress", () => {
    const perf = new PerfStats();
    const a = perf.begin({ progressRequested: true });
    perf.promptProgress(pp(8000), a);
    perf.complete({ completionTokens: 2, source: "server" }, a);
    perf.begin({ progressRequested: false });
    assert.equal(perf.get().promptProgress, undefined);
    assert.equal(perf.get().stage, "waiting");
    assert.equal(perf.get().progressRequested, false);
  });
});
