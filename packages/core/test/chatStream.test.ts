import assert from "node:assert/strict";
import * as http from "node:http";
import { after, before, describe, it } from "node:test";
import { streamChatCompletion, type ChatStreamEvent } from "../src/chatClient";
import { openChatStream, parseChatStream, withSystemHints, type ChatStreamPart } from "../src/chatStream";
import { SseStreamError } from "../src/sseStream";

/** Recorded-style llama-server SSE: one `data:` line per object, split into odd-sized chunks. */
function sse(events: Array<object | string>, chunkSize = 7): string[] {
  const text = events.map((e) => `data: ${typeof e === "string" ? e : JSON.stringify(e)}\n\n`).join("");
  const out: string[] = [];
  for (let i = 0; i < text.length; i += chunkSize) {
    out.push(text.slice(i, i + chunkSize));
  }
  return out;
}

async function* from(chunks: string[]): AsyncGenerator<string> {
  for (const c of chunks) {
    yield c;
  }
}

async function collect(chunks: string[], toolsEnabled: boolean, signal?: AbortSignal): Promise<ChatStreamPart[]> {
  const out: ChatStreamPart[] = [];
  for await (const part of parseChatStream(from(chunks), { toolsEnabled, signal })) {
    out.push(part);
  }
  return out;
}

const text = (parts: ChatStreamPart[]) =>
  parts.flatMap((p) => (p.kind === "text" ? [p.text] : [])).join("");
const delta = (d: object) => ({ choices: [{ delta: d }] });
const timings = { prompt_n: 12, prompt_ms: 100, predicted_n: 5, predicted_ms: 250, cache_n: 30 };

describe("parseChatStream", () => {
  it("completes consecutive tool turns when an SSE connection cannot serve another POST", async () => {
    const seen = new Set<http.IncomingMessage["socket"]>();
    let rejectedReuse = 0;
    let accepted = 0;
    const server = http.createServer((req, res) => {
      if (seen.has(req.socket)) {
        rejectedReuse++;
        req.socket.destroy();
        return;
      }
      seen.add(req.socket);
      req.resume();
      req.on("end", () => {
        accepted++;
        res.writeHead(200, { "Content-Type": "text/event-stream", Connection: "keep-alive" });
        res.end('data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n');
      });
    });
    await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
    try {
      const endpoint = `http://127.0.0.1:${(server.address() as import("node:net").AddressInfo).port}`;
      for (let turn = 0; turn < 2; turn++) {
        const response = await openChatStream(endpoint, { model: "test", messages: [], stream: true });
        const parts: ChatStreamPart[] = [];
        for await (const part of parseChatStream(response, { toolsEnabled: true })) parts.push(part);
        assert.equal(text(parts), "ok");
      }
      assert.equal(accepted, 2);
      assert.equal(rejectedReuse, 0);
    } finally {
      server.closeAllConnections();
      await new Promise<void>(resolve => server.close(() => resolve()));
    }
  });
  for (const legacyDuplicate of [false, true]) {
    it(`separates inline thoughts without duplicating legacy reasoning (duplicate=${legacyDuplicate})`, async () => {
      const events: ChatStreamPart[] = [];
      for await (const event of parseChatStream(from(sse([
        ...(legacyDuplicate ? [delta({ reasoning_content: "Check it." })] : []),
        delta({ content: "<thi" }), delta({ content: "nk>Check it.</think>Answer" }),
      ])), { toolsEnabled: false, separateReasoning: true })) events.push(event);
      assert.equal(text(events), "Answer");
      assert.deepEqual(events.filter(p => p.kind === "reasoning"), [{ kind: "reasoning", text: "Check it." }]);
    });
  }

  it("keeps an unfinished thinking block out of answer text", async () => {
    const events: ChatStreamPart[] = [];
    for await (const event of parseChatStream(from(sse([delta({ content: "<think>Still working" })])),
      { toolsEnabled: false, separateReasoning: true })) events.push(event);
    assert.equal(text(events), "");
    assert.deepEqual(events.filter(p => p.kind === "reasoning"), [{ kind: "reasoning", text: "Still working" }]);
  });
  for (const toolsEnabled of [false, true]) {
    it(`keeps separate reasoning out of final text and tool parsing (tools=${toolsEnabled})`, async () => {
      const events: ChatStreamPart[] = [];
      for await (const event of parseChatStream(from(sse([
        delta({ reasoning_content: "I should inspect the file." }),
        delta({ content: "Answer" }),
        { choices: [{ message: { reasoning_content: "I should inspect the file.", content: "Answer" } }] },
      ])), { toolsEnabled, separateReasoning: true })) events.push(event);
      assert.equal(text(events), "Answer");
      assert.deepEqual(events.filter(p => p.kind === "reasoning"), [{ kind: "reasoning", text: "I should inspect the file." }]);
    });
  }

  it("emits a reasoning-only message once without manufacturing an answer", async () => {
    const events: ChatStreamPart[] = [];
    for await (const event of parseChatStream(from(sse([
      { choices: [{ message: { reasoning_content: "Still working." } }] },
    ])), { toolsEnabled: true, separateReasoning: true })) events.push(event);
    assert.equal(text(events), "");
    assert.deepEqual(events.filter(p => p.kind === "reasoning"), [{ kind: "reasoning", text: "Still working." }]);
  });
  for (const toolsEnabled of [false, true]) {
    it(`emits message-style content and reasoning exactly once (tools=${toolsEnabled})`, async () => {
      for (const message of [{ content: "Hello" }, { reasoning_content: "Thinking hard." }]) {
        const parts = await collect(sse([{ choices: [{ message }] }, { choices: [{ message }] }, "[DONE]"]), toolsEnabled);
        assert.equal(text(parts), "content" in message ? message.content : message.reasoning_content);
      }
      const parts = await collect(sse([
        delta({ content: "Hello" }),
        { choices: [{ message: { content: "Hello" } }] },
      ]), toolsEnabled);
      assert.equal(text(parts), "Hello", "a final message must not repeat streamed content");
    });
  }

  it("streams plain content and ends with server stats and a trace", async () => {
    const parts = await collect(
      sse([delta({ content: "Hel" }), delta({ content: "lo" }), { choices: [{ delta: {}, finish_reason: "stop" }], timings }, "[DONE]"]),
      false
    );
    assert.equal(text(parts), "Hello");
    const stats = parts.filter((p) => p.kind === "stats").at(-1);
    assert.ok(stats && stats.kind === "stats");
    assert.equal(stats.promptTokens, 42, "cache_n + prompt_n");
    assert.equal(stats.completionTokens, 5);
    assert.equal(stats.cachedPromptTokens, 30);
    assert.equal(stats.genTokPerSec, 20);
    const trace = parts.at(-1);
    assert.ok(trace && trace.kind === "trace");
    assert.equal(trace.visibleText, "Hello");
  });

  it("assembles streamed structured tool calls with one id for event and trace", async () => {
    const parts = await collect(
      sse([
        delta({ tool_calls: [{ index: 0, id: "call_a", function: { name: "read_file", arguments: '{"pa' } }] }),
        delta({ tool_calls: [{ index: 0, function: { arguments: 'th":"/x"}' } }] }),
        delta({ tool_calls: [{ index: 1, function: { name: "list_dir", arguments: "{}" } }] }),
      ]),
      true
    );
    const calls = parts.filter((p) => p.kind === "tool_call");
    assert.deepEqual(
      calls.map((c) => c.kind === "tool_call" && [c.name, c.input]),
      [
        ["read_file", { path: "/x" }],
        ["list_dir", {}],
      ]
    );
    const trace = parts.at(-1);
    assert.ok(trace && trace.kind === "trace");
    assert.deepEqual(
      trace.structuredToolCalls.map((c) => c.id),
      calls.map((c) => c.kind === "tool_call" && c.callId)
    );
    assert.equal(calls[0]?.kind === "tool_call" && calls[0].callId, "call_a");
  });

  it("turns Qwen XML tool calls in content into tool calls without leaking the XML", async () => {
    const xml = "<tool_call>\n<function=read_file>\n<parameter=target_file>\n/tmp/a.ts\n</parameter>\n</function>\n</tool_call>";
    const parts = await collect(sse([delta({ content: "Reading it.\n" }), delta({ content: xml })], 5), true);
    assert.equal(text(parts).trim(), "Reading it.");
    assert.ok(!text(parts).includes("<tool_call"));
    const call = parts.find((p) => p.kind === "tool_call");
    assert.ok(call && call.kind === "tool_call");
    assert.equal(call.name, "read_file");
    assert.deepEqual(call.input, { target_file: "/tmp/a.ts" });
  });

  it("shows reasoning-only output live without tools, and at the end with tools", async () => {
    const events = [delta({ reasoning_content: "Thinking " }), delta({ reasoning_content: "hard." })];
    assert.equal(text(await collect(sse(events), false)), "Thinking hard.");
    const withTools = await collect(sse(events), true);
    assert.equal(text(withTools), "Thinking hard.");
    assert.equal(withTools.filter((p) => p.kind === "text").length, 1, "surfaced once, after the stream");
  });

  it("throws SseStreamError on an error event, mid-stream", async () => {
    await assert.rejects(
      () => collect(sse([delta({ content: "partial" }), { error: { message: "context overflow", code: 400 } }]), false),
      (e: unknown) => e instanceof SseStreamError && /context overflow/.test(e.message)
    );
    await assert.rejects(
      () => collect(["data: ", JSON.stringify(delta({ content: "a" })), "\n\nerror: {\"message\":\"boom\"}\n\n"], false),
      (e: unknown) => e instanceof SseStreamError && /boom/.test(e.message)
    );
  });

  it("returns quietly with no stats or trace after an abort", async () => {
    const ac = new AbortController();
    const out: ChatStreamPart[] = [];
    for await (const part of parseChatStream(from(sse([delta({ content: "a" }), delta({ content: "b" })], 4)), {
      toolsEnabled: false,
      signal: ac.signal,
    })) {
      out.push(part);
      ac.abort();
    }
    assert.ok(out.every((p) => p.kind !== "trace"));
  });
});

describe("withSystemHints", () => {
  it("merges hints into the leading system message", () => {
    const msgs = [
      { role: "system", content: "You are Copilot." },
      { role: "user", content: "hi" },
    ];
    const out = withSystemHints(msgs, ["Hint A", " ", "Hint B"]);
    assert.equal(out.length, 2);
    assert.equal(out[0]?.content, "You are Copilot.\n\nHint A\n\nHint B");
    assert.equal(msgs[0]?.content, "You are Copilot.", "input is not mutated");
  });

  it("adds a system message at the front when there is none, and is a no-op without hints", () => {
    const msgs = [{ role: "user", content: "hi" }];
    assert.deepEqual(withSystemHints(msgs, ["Hint"]), [{ role: "system", content: "Hint" }, ...msgs]);
    assert.equal(withSystemHints(msgs, []), msgs);
  });
});

describe("streamChatCompletion (TUI client) on the shared parser", () => {
  let server: http.Server;
  let endpoint = "";
  before(async () => {
    server = http.createServer((_req, res) => {
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      res.end(sse([delta({ content: "Hi " }), delta({ content: "there" }), { choices: [], timings }, "[DONE]"], 64).join(""));
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const addr = server.address();
    if (!addr || typeof addr === "string") {
      throw new Error("no port");
    }
    endpoint = `http://127.0.0.1:${addr.port}`;
  });
  after(async () => {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  });

  it("yields text, one final stats event, then done", async () => {
    const events: ChatStreamEvent[] = [];
    for await (const ev of streamChatCompletion({ endpoint, messages: [{ role: "user", content: "hi" }] })) {
      events.push(ev);
    }
    assert.deepEqual(
      events.map((e) => e.kind),
      ["text", "text", "stats", "done"]
    );
    const stats = events[2];
    assert.ok(stats?.kind === "stats");
    assert.equal(stats.completionTokens, 5);
  });
});
