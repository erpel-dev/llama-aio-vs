import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import { describe, it, afterEach } from "node:test";
import {
  formatRemoteDiagnostics,
  parseLlamaMetrics,
  parseSlots,
  parseStrataMetrics,
  parseVllmMetrics,
  applyVllmThroughput,
  probeRemoteServer,
  type RemoteServerInfo,
  remoteLoadBusy,
  remoteSlotContext,
  remoteStateLabel,
  remoteStateUsable,
} from "../src/remoteServer";
import { legacyRemoteHostUrl } from "../src/settings";

describe("legacyRemoteHostUrl", () => {
  it("migrates a non-local host and leaves this machine alone", () => {
    assert.equal(legacyRemoteHostUrl("10.1.2.3", 8080, ["192.168.0.4"]), "http://10.1.2.3:8080");
    assert.equal(legacyRemoteHostUrl("127.0.0.1", 8080, []), undefined);
    assert.equal(legacyRemoteHostUrl("0.0.0.0", 8080, []), undefined);
    assert.equal(legacyRemoteHostUrl("192.168.0.4", 8080, ["192.168.0.4"]), undefined);
  });
});

describe("probeRemoteServer", () => {
  let server: Server;
  let base = "";
  let mode = "ready";

  afterEach(async () => {
    if (server) {
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });

  async function listen() {
    server = createServer((req, res) => {
      const path = req.url || "";
      if (mode === "loading" && path === "/health") {
        res.writeHead(503);
        res.end('{"error":{"message":"Loading model"}}');
        return;
      }
      if (mode === "no-health" && path === "/health") {
        res.writeHead(404);
        res.end("nope");
        return;
      }
      if (path === "/health") {
        res.writeHead(200);
        res.end('{"status":"ok"}');
        return;
      }
      if (path === "/v1/models") {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(
          JSON.stringify({
            data: [
              {
                id: "qwythos-9b.gguf",
                meta: { size: 1000, n_params: 9e9, n_ctx_train: 1048576 },
                max_model_len: 32768,
              },
            ],
          })
        );
        return;
      }
      if (path === "/slots") {
        if (mode === "slots-off" || mode === "ready" || mode === "metrics-only" || mode === "vllm") {
          res.writeHead(mode === "slots-off" ? 501 : 404);
          res.end('{"error":{"message":"This server does not support metrics endpoint."}}');
          return;
        }
        const busy = mode === "busy" || mode === "queued" ? (mode === "queued" ? 1 : 4) : 0;
        const total = mode === "busy" ? 4 : 2;
        const slots = Array.from({ length: total }, (_, id) => ({
          id,
          is_processing: id < busy,
          prompt: "secret prompt text",
          n_ctx: 32768,
        }));
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(slots));
        return;
      }
      if (path === "/metrics") {
        if (mode === "vllm") {
          res.writeHead(200, { "content-type": "text/plain" });
          res.end(
            [
              "# HELP vllm:num_requests_running Number of requests in model execution batches.",
              "# TYPE vllm:num_requests_running gauge",
              'vllm:num_requests_running{model_name="org/model-a"} 5',
              'vllm:num_requests_running{model_name="org/model-b"} 3',
              'vllm:num_requests_waiting{model_name="org/model-a"} 1',
              'vllm:generation_tokens_total{model_name="org/model-a"} 1000',
              'vllm:prompt_tokens_total{model_name="org/model-a"} 4000',
              "vllm:e2e_request_latency_seconds_bucket{le=\"1.0\"} 4",
            ].join("\n")
          );
          return;
        }
        if (mode !== "busy" && mode !== "queued" && mode !== "metrics-only") {
          res.writeHead(501);
          res.end('{"error":{"message":"This server does not support metrics endpoint."}}');
          return;
        }
        const deferred = mode === "busy" ? 3 : mode === "queued" ? 2 : 0;
        const processing = mode === "busy" ? 4 : mode === "queued" ? 1 : 0;
        res.writeHead(200, { "content-type": "text/plain" });
        res.end(
          [
            "# HELP llamacpp:predicted_tokens_seconds Average generation throughput in tokens/s.",
            "# TYPE llamacpp:predicted_tokens_seconds gauge",
            `llamacpp:predicted_tokens_seconds ${mode === "busy" ? 61 : 0}`,
            `llamacpp:prompt_tokens_seconds ${mode === "busy" ? 1900 : 0}`,
            `llamacpp:requests_processing{job="llama"} ${processing}`,
            `llamacpp:requests_deferred ${deferred}`,
            "llamacpp:kv_cache_usage_ratio 0.42",
            "llamacpp:prompt_tokens_total 100",
          ].join("\n")
        );
        return;
      }
      if (path === "/props") {
        if (mode === "auth" || mode === "limited" || mode === "vllm") {
          res.writeHead(mode === "auth" ? 401 : 404);
          res.end(mode === "auth" ? "unauthorized" : "missing");
          return;
        }
        res.writeHead(200, { "content-type": "application/json" });
        res.end(
          JSON.stringify({
            default_generation_settings: { n_ctx: 65536 },
            total_slots: 2,
            model_path: "/srv/models/qwythos-9b.gguf",
            build_info: "b10517-abc",
            modalities: { vision: true },
          })
        );
        return;
      }
      res.writeHead(404);
      res.end();
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
    const addr = server.address();
    if (!addr || typeof addr === "string") {
      throw new Error("no port");
    }
    base = `http://127.0.0.1:${addr.port}`;
  }

  it("reads a llama.cpp server as ready", async () => {
    mode = "ready";
    await listen();
    const info = await probeRemoteServer(base);
    assert.equal(info.state, "ready");
    assert.equal(info.isLlamaCpp, true);
    assert.equal(info.modelId, "qwythos-9b.gguf");
    assert.equal(info.slotContext, 65536);
    assert.equal(info.vision, true);
    assert.equal(info.servedContext, 32768);
    assert.equal(remoteStateUsable(info.state), true);
    assert.equal(remoteSlotContext(info), 65536);
    assert.equal(info.slotsBusy, undefined);
    const slots = info.probes.find((p) => p.path === "/slots");
    const metrics = info.probes.find((p) => p.path === "/metrics");
    assert.equal(slots?.ok, null);
    assert.match(slots?.detail || "", /--slots/);
    assert.equal(metrics?.ok, null);
    assert.match(metrics?.detail || "", /--metrics/);
  });

  it("marks every slot busy and keeps the queue from /metrics", async () => {
    mode = "busy";
    await listen();
    const info = await probeRemoteServer(base);
    assert.equal(info.state, "busy");
    assert.equal(remoteStateUsable(info.state), true);
    assert.equal(remoteStateLabel(info.state), "Connected · busy");
    assert.equal(info.slotsBusy, 4);
    assert.equal(info.slotsTotal, 4);
    assert.equal(info.requestsDeferred, 3);
    assert.equal(info.serverGenTokPerSec, 61);
    assert.equal(info.serverPromptTokPerSec, 1900);
    assert.equal(JSON.stringify(info).includes("secret prompt"), false);
  });

  it("is busy when requests are queued even if a slot is free", async () => {
    mode = "queued";
    await listen();
    const info = await probeRemoteServer(base);
    assert.equal(info.state, "busy");
    assert.equal(info.slotsBusy, 1);
    assert.equal(info.slotsTotal, 2);
    assert.equal(info.requestsDeferred, 2);
  });

  it("stays ready when /slots and /metrics are turned off", async () => {
    mode = "slots-off";
    await listen();
    const info = await probeRemoteServer(base);
    assert.equal(info.state, "ready");
    assert.equal(info.slotsBusy, undefined);
    assert.equal(info.probes.find((p) => p.path === "/slots")?.status, 501);
  });

  it("uses /metrics processing when /slots is off", async () => {
    mode = "metrics-only";
    await listen();
    const info = await probeRemoteServer(base);
    assert.equal(info.state, "ready");
    assert.equal(info.slotsBusy, 0);
    assert.equal(info.slotsTotal, 2);
    assert.equal(info.serverGenTokPerSec, 0);
  });

  it("turns a 401 on /props into auth while keeping the public model name", async () => {
    mode = "auth";
    await listen();
    const info = await probeRemoteServer(base);
    assert.equal(info.state, "auth");
    assert.equal(info.modelId, "qwythos-9b.gguf");
    assert.equal(remoteStateUsable(info.state), false);
  });

  it("reads vLLM Prometheus metrics on a server without /props", async () => {
    mode = "vllm";
    await listen();
    const info = await probeRemoteServer(base);
    assert.equal(info.state, "limited");
    assert.equal(info.isLlamaCpp, false);
    assert.equal(info.requestsProcessing, 8);
    assert.equal(info.requestsDeferred, 1);
    assert.equal(info.vllmGenerationTokensTotal, 1000);
    assert.equal(info.vllmPromptTokensTotal, 4000);
    assert.equal(info.serverGenTokPerSec, undefined);
    assert.equal(info.probes.find((p) => p.path === "/metrics")?.ok, true);
    assert.equal(info.probes.find((p) => p.path === "/slots")?.ok, null);
  });

  it("treats a server without /props as limited and uses max_model_len", async () => {
    mode = "limited";
    await listen();
    const info = await probeRemoteServer(base);
    assert.equal(info.state, "limited");
    assert.equal(info.isLlamaCpp, false);
    assert.equal(remoteSlotContext(info), 32768);
    assert.equal(remoteStateUsable(info.state), true);
  });

  it("maps /health 503 to loading", async () => {
    mode = "loading";
    await listen();
    const info = await probeRemoteServer(base);
    assert.equal(info.state, "loading");
  });

  it("redacts the API key in diagnostics", () => {
    const text = formatRemoteDiagnostics(
      { url: "http://box:8080", state: "ready", checkedAt: 1, isLlamaCpp: true, probes: [], modelId: "m" },
      { url: "http://box:8080", label: "box", hasApiKey: true }
    );
    assert.match(text, /redacted/);
    assert.doesNotMatch(text, /sk-|secret|Bearer/);
  });

  it("includes slot load in diagnostics and not slot prompt text", () => {
    const text = formatRemoteDiagnostics(
      {
        url: "http://box:8080",
        state: "busy",
        checkedAt: 1,
        isLlamaCpp: true,
        probes: [],
        slotsBusy: 4,
        slotsTotal: 4,
        requestsDeferred: 3,
        serverGenTokPerSec: 61,
      },
      { url: "http://box:8080", hasApiKey: false }
    );
    assert.match(text, /Slots busy: 4\/4/);
    assert.match(text, /Queued: 3/);
    assert.match(text, /Server gen: 61 tok\/s/);
    assert.doesNotMatch(text, /prompt text/i);
  });
});

describe("slot and metrics parsers", () => {
  it("counts is_processing and drops prompt text", () => {
    const parsed = parseSlots(
      JSON.stringify([
        { id: 0, is_processing: true, prompt: "do not keep" },
        { id: 1, is_processing: false, prompt: "also secret" },
      ])
    );
    assert.deepEqual(parsed, { busy: 1, total: 2 });
    assert.equal(parseSlots('{"error":{"message":"off"}}'), undefined);
    assert.equal(parseSlots("not json"), undefined);
  });

  it("reads the four gauges and ignores the rest", () => {
    const parsed = parseLlamaMetrics(
      [
        "# HELP llamacpp:requests_deferred queued",
        "llamacpp:requests_deferred 3",
        "llamacpp:requests_processing 4",
        "llamacpp:predicted_tokens_seconds 61.2",
        "llamacpp:prompt_tokens_seconds 1900",
        "llamacpp:kv_cache_tokens 100",
        "llamacpp:predicted_tokens_seconds NaN",
      ].join("\n")
    );
    assert.equal(parsed?.requestsDeferred, 3);
    assert.equal(parsed?.requestsProcessing, 4);
    assert.equal(parsed?.serverGenTokPerSec, 61.2);
    assert.equal(parsed?.serverPromptTokPerSec, 1900);
    assert.equal("kv" in (parsed || {}), false);
  });

  it("reads Strata's JSON metrics and ignores request history", () => {
    const idle = parseStrataMetrics(
      JSON.stringify({
        live: { state: "idle", queued: 0, tok_s: null, tok_s_mean: null, prefill_tok_s_mean: null },
        totals: { prompt_tokens: 1000, reused: 400, output_tokens: 200, prompt_ms: 2000, decode_ms: 4000 },
        requests: [{ input: "do not keep", output: "secret" }],
      })
    );
    assert.equal(idle?.requestsDeferred, 0);
    assert.equal(idle?.requestsProcessing, 0);
    assert.equal(idle?.serverGenTokPerSec, 50);
    assert.equal(idle?.serverPromptTokPerSec, 300);
    assert.equal(JSON.stringify(idle).includes("secret"), false);

    const live = parseStrataMetrics(
      JSON.stringify({
        live: { state: "generating", queued: 2, tok_s: 61, prefill_tok_s_mean: null, running: 1 },
        totals: { prompt_tokens: 1000, reused: 0, output_tokens: 10, prompt_ms: 1000, decode_ms: 1000 },
      })
    );
    assert.equal(live?.requestsDeferred, 2);
    assert.equal(live?.requestsProcessing, 1);
    assert.equal(live?.serverGenTokPerSec, 61);
    assert.equal(live?.serverPromptTokPerSec, 1000);
    assert.equal(parseStrataMetrics("not json"), undefined);
    assert.equal(parseStrataMetrics('{"error":{"message":"off"}}'), undefined);
  });

  it("sums vLLM series and keeps token totals out of the tok/s fields", () => {
    const parsed = parseVllmMetrics(
      [
        'vllm:num_requests_running{model_name="a"} 5',
        'vllm:num_requests_running{model_name="b"} 3',
        'vllm:num_requests_waiting{model_name="a"} 2',
        "vllm:generation_tokens_total 27453",
        "vllm:prompt_tokens_total 9000",
        'vllm:e2e_request_latency_seconds_bucket{le="1.0"} 4',
        "vllm:avg_generation_throughput_toks_per_s 40",
      ].join("\n")
    );
    assert.equal(parsed?.requestsProcessing, 8);
    assert.equal(parsed?.requestsDeferred, 2);
    assert.equal(parsed?.generationTokensTotal, 27453);
    assert.equal(parsed?.promptTokensTotal, 9000);
    assert.equal(parsed?.serverGenTokPerSec, 40);
    assert.equal(parseVllmMetrics("# just a comment\n"), undefined);
  });

  it("turns two vLLM counter scrapes into tok/s", () => {
    const prev: RemoteServerInfo = {
      url: "http://vllm:8000",
      state: "limited",
      checkedAt: 0,
      isLlamaCpp: false,
      probes: [],
      vllmGenerationTokensTotal: 1000,
      vllmPromptTokensTotal: 5000,
      vllmMetricsAt: 0,
    };
    const next: RemoteServerInfo = {
      ...prev,
      checkedAt: 15_000,
      vllmGenerationTokensTotal: 1180,
      vllmPromptTokensTotal: 6500,
      vllmMetricsAt: 15_000,
    };
    applyVllmThroughput(prev, next);
    assert.equal(next.serverGenTokPerSec, 12);
    assert.equal(next.serverPromptTokPerSec, 100);
    const reset = { ...next, vllmGenerationTokensTotal: 10, serverGenTokPerSec: undefined, serverPromptTokPerSec: undefined };
    applyVllmThroughput(next, reset);
    assert.equal(reset.serverGenTokPerSec, undefined);
    const kept = { ...next, serverGenTokPerSec: 40, vllmGenerationTokensTotal: 2000 };
    applyVllmThroughput(next, kept);
    assert.equal(kept.serverGenTokPerSec, 40);
  });

  it("treats a full slot set or a queue as busy", () => {
    assert.equal(remoteLoadBusy({ slotsBusy: 4, slotsTotal: 4 }), true);
    assert.equal(remoteLoadBusy({ slotsBusy: 1, slotsTotal: 4, requestsDeferred: 2 }), true);
    assert.equal(remoteLoadBusy({ slotsBusy: 0, slotsTotal: 4, requestsDeferred: 0 }), false);
    assert.equal(remoteLoadBusy({}), false);
  });
});
