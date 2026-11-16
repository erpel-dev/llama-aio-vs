import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { createServer } from "node:http";
import { setServerApiKeyProvider } from "../src/serverHttp";
import {
  discoverLoopbackLlamaServers,
  localListenerName,
  parseLsofListeners,
  parseNetstatListeners,
  parseSsListeners,
  probeLocalBigMoeOnEdgeServer,
  probeLocalKoboldCppServer,
  probeLocalLmStudioServer,
  probeLocalOllamaServer,
  probeLocalSglangServer,
  probeLocalTabbyApiServer,
  probeLocalVllmServer,
  probeLocalProjectMayaServer,
  probeLocalRustingStrataServer,
  probeLocalStrataServer,
  strataListenerRole,
} from "../src/localServers";

const SS = `
State  Recv-Q Send-Q Local Address:Port Peer Address:PortProcess
LISTEN 0      128        127.0.0.1:8080      0.0.0.0:*    users:(("llama-server",pid=42,fd=3))
LISTEN 0      128            [::1]:8080         [::]:*    users:(("llama-server",pid=42,fd=4))
LISTEN 0      128          0.0.0.0:8742      0.0.0.0:*    users:(("llama-server",pid=7,fd=5))
LISTEN 0      128             [::]:8742         [::]:*    users:(("llama-server",pid=7,fd=6))
LISTEN 0      128        127.0.0.1:3000      0.0.0.0:*    users:(("node",pid=9,fd=3))
LISTEN 0      128     192.168.1.5:8081      0.0.0.0:*    users:(("llama-server",pid=11,fd=3))
LISTEN 0      128                *:22             *:*
LISTEN 0      128            [::1]:9090         [::]:*    users:(("llama-server",pid=3,fd=1))
`;

const CMDS: Record<number, string> = {
  42: "/opt/llama/llama-server -m model.gguf --port 8080",
  7: "llama-server -m other.gguf --host 0.0.0.0 --port 8742",
  9: "node /app/server.js --port 3000",
  11: "/opt/llama/llama-server -m lan.gguf --host 192.168.1.5",
  3: "/opt/llama/llama-server --host ::1 --port 9090",
  15: "/home/timo/Strata/.venv/bin/python /home/timo/Strata/serve/server.py --engine strata --port 8080",
};

describe("parseSsListeners", () => {
  it("reads pid, bind, and port, and drops lines with no pid", () => {
    const sockets = parseSsListeners(SS);
    assert.ok(sockets.some((s) => s.pid === 42 && s.bind === "127.0.0.1" && s.port === 8080));
    assert.ok(sockets.some((s) => s.pid === 42 && s.bind === "::1" && s.port === 8080));
    assert.equal(sockets.some((s) => s.port === 22), false);
  });
});

describe("discoverLoopbackLlamaServers", () => {
  const commandLine = (pid: number) => CMDS[pid];

  it("offers loopback llama-server processes and skips everything else", () => {
    const found = discoverLoopbackLlamaServers({ sockets: parseSsListeners(SS), commandLine });
    assert.deepEqual(
      found.map((s) => `${s.name} ${s.url}`),
      [
        "llama-server http://127.0.0.1:8080",
        "llama-server http://127.0.0.1:8742",
        "llama-server http://[::1]:9090",
      ]
    );
  });

  it("does not list a pid this extension already owns", () => {
    const found = discoverLoopbackLlamaServers({
      sockets: parseSsListeners(SS),
      commandLine,
      skipPids: [7],
    });
    assert.deepEqual(
      found.map((s) => s.port),
      [8080, 9090]
    );
  });

  it("offers a Strata listener without treating it as llama-server", () => {
    const sockets = parseSsListeners(
      'LISTEN 0 128 127.0.0.1:8080 0.0.0.0:* users:(("python",pid=15,fd=7))\n'
    );
    const found = discoverLoopbackLlamaServers({ sockets, commandLine });
    assert.deepEqual(found, [{ url: "http://127.0.0.1:8080", port: 8080, pid: 15, name: "Strata" }]);
    assert.equal(localListenerName("node /app/server.js"), undefined);
    assert.equal(localListenerName("/opt/llama/llama-server -m x.gguf"), "llama-server");
    assert.equal(localListenerName("/usr/local/bin/bmoe-server -m model.gguf -c 32768"), "BigMoeOnEdge");
    assert.equal(localListenerName("C:\\Program Files\\BigMoeOnEdge\\bmoe-server.exe --port 8765"), "BigMoeOnEdge");
    assert.equal(localListenerName("/Applications/BigMoeOnEdge.app/Contents/MacOS/bmoe-desktop"), undefined, "the app window does not serve the API");
    assert.equal(localListenerName("/x/bmoe-server-tests --gtest"), undefined);
    assert.equal(localListenerName("/home/me/RustingStrata/target/release/rustingstrata serve model.gguf"), "RustingStrata");
    assert.equal(localListenerName("C:\\tools\\rustingstrata.exe"), "RustingStrata");
    assert.equal(localListenerName("/home/me/models/mistral-7b-instruct.llamafile --server --port 8080"), "llamafile");
    assert.equal(localListenerName("llamafile -m model.gguf --server"), "llamafile");
    assert.equal(localListenerName("/x/llamafile-tests"), undefined);
    assert.equal(strataListenerRole("/home/me/RustingStrata/target/release/rustingstrata"), undefined, "not Strata itself");
  });

  it("discovers the new default port and custom ports without a --port argument", () => {
    for (const port of [8095, 18095]) {
      const sockets = parseSsListeners(
        `LISTEN 0 128 0.0.0.0:${port} 0.0.0.0:* users:(("python",pid=15,fd=7))\n`
      );
      for (const cmd of ["python -m serve.server --engine strata", "/opt/Strata/python/python -m serve.server"]) {
        assert.deepEqual(discoverLoopbackLlamaServers({ sockets, commandLine: () => cmd }), [
          { url: `http://127.0.0.1:${port}`, port, pid: 15, name: "Strata" },
        ]);
      }
    }
  });

  it("lists only the API host of a Strata Desktop install", () => {
    // One install starts the API host, an Electron window helper, a web harness,
    // and an engine for the loaded model. Only the host is a chat target.
    const install = "/tmp/.mount_StrataKhGCjC/usr/lib/Strata";
    const desktop: Record<number, string> = {
      8267: `${install}/node ${install}/host/server/index.mjs`,
      8291: "/tmp/.mount_StrataKhGCjC/usr/bin/strata-desktop --window-child",
      15450: `${install}/node ${install}/harness/lib/bin.js --profile web --port 0 --no-open`,
      32782: `${install}/control/python/bin/python3.12 -m serve.server --engine strata --config /x.json --host 127.0.0.1 --port 18500`,
    };
    const sockets = parseSsListeners(
      [
        'LISTEN 0 511 127.0.0.1:47831 0.0.0.0:* users:(("MainThread",pid=8267,fd=23))',
        'LISTEN 0 128 127.0.0.1:34987 0.0.0.0:* users:(("strata-desktop",pid=8291,fd=19))',
        'LISTEN 0 511 127.0.0.1:43943 0.0.0.0:* users:(("MainThread",pid=15450,fd=21))',
        'LISTEN 0 256 127.0.0.1:18500 0.0.0.0:* users:(("python3.12",pid=32782,fd=11))',
      ].join("\n")
    );
    assert.deepEqual(discoverLoopbackLlamaServers({ sockets, commandLine: (pid) => desktop[pid] }), [
      { url: "http://127.0.0.1:47831", port: 47831, pid: 8267, name: "Strata" },
    ]);
  });

  it("still offers a Strata engine that runs on its own", () => {
    const sockets = parseSsListeners(
      'LISTEN 0 128 127.0.0.1:47831 0.0.0.0:* users:(("python",pid=15,fd=7))\n'
    );
    assert.deepEqual(
      discoverLoopbackLlamaServers({ sockets, commandLine: () => "/usr/local/bin/strata --serve --port 47831" }),
      [{ url: "http://127.0.0.1:47831", port: 47831, pid: 15, name: "Strata" }]
    );
  });

  it("classifies Strata processes by the role they play", () => {
    assert.equal(strataListenerRole("/x/Strata/host/server/index.mjs"), "host");
    assert.equal(strataListenerRole("C:\\Strata\\host\\server\\index.mjs"), "host");
    assert.equal(strataListenerRole("/x/strata-desktop --window-child"), "internal");
    assert.equal(strataListenerRole("/x/Strata/node /x/Strata/harness/lib/bin.js --profile web"), "internal");
    assert.equal(strataListenerRole("python -m serve.server --engine strata"), "engine");
    assert.equal(strataListenerRole("node /app/server.js"), undefined);
    assert.equal(localListenerName("/x/strata-desktop --window-child"), undefined);
  });

  it("returns nothing when the socket list could not be read", () => {
    assert.deepEqual(discoverLoopbackLlamaServers({ sockets: undefined, commandLine }), []);
  });
});

describe("Strata health discovery without process metadata", () => {
  for (const [body, expected] of [
    ['{"status":"ok","service":"strata","loaded":true}', true],
    ['{"status":"ok","service":"strata","loaded":false,"api_key":true}', true],
    ['{"status":"ok"}', false],
    ['{"service":"another-app"}', false],
    ["<html>Other application</html>", false],
  ] as const) {
    it(`identifies ${body}`, async () => {
      setServerApiKeyProvider(() => "private-key-for-another-server");
      const server = createServer((req, res) => {
        assert.equal(req.headers.authorization, undefined);
        if (req.url !== "/health") {
          res.writeHead(404);
          res.end();
          return;
        }
        res.end(body);
      });
      await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
      try {
        const addr = server.address();
        assert.ok(addr && typeof addr !== "string");
        assert.equal(await probeLocalStrataServer(`http://127.0.0.1:${addr.port}`), expected);
      } finally {
        setServerApiKeyProvider(undefined);
        await new Promise<void>((resolve, reject) => server.close(err => err ? reject(err) : resolve()));
      }
    });
  }

  it("does not probe non-loopback hosts", async () => {
    assert.equal(await probeLocalStrataServer("http://192.0.2.1:8095"), false);
  });

  it("handles invalid URLs", async () => {
    assert.equal(await probeLocalStrataServer("invalid"), false);
  });

  for (const [marker, expected] of [["x-strata-client", true], ["x-other-client", false]]) {
    it(`identifies the Desktop host via /api/health with ${marker}`, async () => {
      const server = createServer((req, res) => {
        assert.equal(req.headers.authorization, undefined);
        if (req.url === "/health") {
          res.end("<html>Strata Desktop UI</html>");
          return;
        }
        assert.equal(req.url, "/api/health");
        res.setHeader("access-control-allow-headers", `content-type, authorization, ${marker}`);
        res.end(JSON.stringify({ ok: true, uptimeSec: 1, backend: { kind: "strata", name: "Strata" }, engine: { status: "stopped" }, clients: 0 }));
      });
      await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
      try {
        const addr = server.address();
        assert.ok(addr && typeof addr !== "string");
        assert.equal(await probeLocalStrataServer(`http://127.0.0.1:${addr.port}`), expected);
      } finally {
        await new Promise<void>((resolve) => server.close(() => resolve()));
      }
    });
  }
});

describe("parseLsofListeners", () => {
  it("uses the pid column, not the truncated command name", () => {
    const text = [
      "COMMAND     PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME",
      "llama-ser   42 timo    3u  IPv4  0x1      0t0  TCP 127.0.0.1:8080 (LISTEN)",
      "node         9 timo    5u  IPv4  0x3      0t0  TCP *:3000 (LISTEN)",
    ].join("\n");
    assert.deepEqual(parseLsofListeners(text), [
      { bind: "127.0.0.1", port: 8080, pid: 42 },
      { bind: "*", port: 3000, pid: 9 },
    ]);
  });
});

describe("parseNetstatListeners", () => {
  it("reads Windows LISTENING rows", () => {
    const text = [
      "  TCP    127.0.0.1:8080         0.0.0.0:0              LISTENING       42",
      "  TCP    [::1]:8080             [::]:0                 LISTENING       42",
      "  TCP    0.0.0.0:8742           0.0.0.0:0              LISTENING       7",
    ].join("\n");
    const found = discoverLoopbackLlamaServers({ sockets: parseNetstatListeners(text), commandLine: (pid) => CMDS[pid] });
    assert.deepEqual(
      found.map((s) => s.url),
      ["http://127.0.0.1:8080", "http://127.0.0.1:8742"]
    );
  });
});

describe("BigMoeOnEdge discovery without process metadata", () => {
  const info = { version: "0.9.0", overlap_available: true, planner_available: true, host: {}, state: "ready", error: "", model: {}, generating: false, calibrating: false };
  const cases: Array<[string, number, string, boolean]> = [
    ["its /api/info", 200, JSON.stringify(info), true],
    ["its /api/info with no model loaded", 200, JSON.stringify({ ...info, state: "empty", model: null }), true],
    ["its API-key refusal", 401, '{"error":{"message":"an API key is required: send it as Authorization: Bearer <key> (or X-Api-Key: <key>)"}}', true],
    ["another server's /api/info", 200, '{"version":"1.2.3","state":"ready"}', false],
    ["another server's 401", 401, '{"error":"unauthorized"}', false],
    ["a web page", 200, "<html>Other application</html>", false],
  ];
  for (const [what, status, body, expected] of cases) {
    it(`${expected ? "identifies" : "ignores"} ${what}`, async () => {
      setServerApiKeyProvider(() => "private-key-for-another-server");
      const server = createServer((req, res) => {
        assert.equal(req.headers.authorization, undefined, "never sends the managed server's key");
        if (req.url !== "/api/info") {
          res.writeHead(404);
          res.end();
          return;
        }
        res.writeHead(status, { "content-type": "application/json" });
        res.end(body);
      });
      await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
      try {
        const addr = server.address();
        assert.ok(addr && typeof addr !== "string");
        assert.equal(await probeLocalBigMoeOnEdgeServer(`http://127.0.0.1:${addr.port}`), expected);
      } finally {
        setServerApiKeyProvider(undefined);
        await new Promise<void>((resolve) => server.close(() => resolve()));
      }
    });
  }

  it("does not probe non-loopback hosts or invalid URLs", async () => {
    assert.equal(await probeLocalBigMoeOnEdgeServer("http://192.0.2.1:8765"), false);
    assert.equal(await probeLocalBigMoeOnEdgeServer("invalid"), false);
  });
});

/** Serve one canned reply per path on loopback and run `fn` against its URL. */
async function withReplies(replies: Record<string, [number, string]>, fn: (url: string) => Promise<void>) {
  setServerApiKeyProvider(() => "private-key-for-another-server");
  const server = createServer((req, res) => {
    assert.equal(req.headers.authorization, undefined, "never sends the managed server's key");
    const [status, body] = replies[req.url || ""] ?? [404, "{}"];
    res.writeHead(status, { "content-type": "application/json" });
    res.end(body);
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const addr = server.address();
    assert.ok(addr && typeof addr !== "string");
    await fn(`http://127.0.0.1:${addr.port}`);
  } finally {
    setServerApiKeyProvider(undefined);
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}

describe("Project Maya and RustingStrata discovery without process metadata", () => {
  // Project Maya serve/server.py /health (v1.1.65); PolyStrata and Strata answer with service: "strata".
  const maya = { status: "ok", max_context: 131072, model: "glm-5.3-flash", max_text_file: 1, model_label: "GLM-5.3-Flash · Maya-S24",
    images: false, api_key: false, defaults: {}, videos: false, version: "1.1.65", mcp_tools: false, unloaded: false,
    history_thinking: true, ablate: null };
  const polystrata = { status: "ok", max_context: 131072, model: "qwen3.8", images: false, api_key: false,
    image_generation: false, api_monitor: false, loaded: true, service: "strata" };
  const mayaCases: Array<[string, Record<string, [number, string]>, boolean]> = [
    ["Project Maya's /health", { "/health": [200, JSON.stringify(maya)] }, true],
    ["PolyStrata's /health", { "/health": [200, JSON.stringify(polystrata)] }, false],
    ["llama-server's /health", { "/health": [200, '{"status":"ok"}'] }, false],
    ["a page", { "/health": [200, "<html></html>"] }, false],
  ];
  for (const [what, replies, expected] of mayaCases) {
    it(`Project Maya: ${expected ? "identifies" : "ignores"} ${what}`, () =>
      withReplies(replies, async (url) => assert.equal(await probeLocalProjectMayaServer(url), expected)));
  }

  // RustingStrata src/main.rs /health (v1.2.4).
  const rustCases: Array<[string, Record<string, [number, string]>, boolean]> = [
    ["a loaded model", { "/health": [200, '{"status":"ok","waiting":0,"slots":1}'] }, true],
    ["no model", { "/health": [503, '{"status":"no model"}'] }, true],
    ["a model loading", { "/health": [503, '{"status":"loading"}'] }, true],
    ["llama-server ready", { "/health": [200, '{"status":"ok"}'] }, false],
    ["llama-server loading", { "/health": [503, '{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}'] }, false],
    ["extra fields", { "/health": [200, '{"status":"ok","waiting":0,"slots":1,"service":"strata"}'] }, false],
    ["Project Maya", { "/health": [200, JSON.stringify(maya)] }, false],
  ];
  for (const [what, replies, expected] of rustCases) {
    it(`RustingStrata: ${expected ? "identifies" : "ignores"} ${what}`, () =>
      withReplies(replies, async (url) => assert.equal(await probeLocalRustingStrataServer(url), expected)));
  }

  it("does not probe non-loopback hosts", async () => {
    assert.equal(await probeLocalProjectMayaServer("http://192.0.2.1:8080"), false);
    assert.equal(await probeLocalRustingStrataServer("http://192.0.2.1:8080"), false);
  });
});

describe("other common local servers, identified by their own replies", () => {
  const models = (owner: string) => JSON.stringify({ object: "list", data: [{ id: "m", object: "model", owned_by: owner }] });
  type Case = [string, (url: string) => Promise<boolean>, Record<string, [number, string]>, boolean];
  const cases: Case[] = [
    ["Ollama", probeLocalOllamaServer, { "/": [200, "Ollama is running"], "/api/version": [200, '{"version":"0.12.3"}'] }, true],
    ["Ollama look-alike without /api/version", probeLocalOllamaServer, { "/": [200, "Ollama is running"] }, false],
    ["a page on Ollama's check", probeLocalOllamaServer, { "/": [200, "<html>Ollama is running</html>"] }, false],
    ["LM Studio 0.4 (/api/v1/models)", probeLocalLmStudioServer, { "/api/v1/models": [200, JSON.stringify({ models: [
      { type: "llm", publisher: "qwen", key: "qwen/qwen3.5-2b", display_name: "Qwen3.5 2B", loaded_instances: [] }] })] }, true],
    ["LM Studio 0.4 with no models", probeLocalLmStudioServer, { "/api/v1/models": [200, '{"models":[]}'] }, true],
    ["LM Studio 0.3 (/api/v0/models)", probeLocalLmStudioServer, { "/api/v0/models": [200, JSON.stringify({ object: "list", data: [
      { id: "qwen3.5-2b", object: "model", type: "llm", publisher: "qwen", arch: "qwen35", compatibility_type: "gguf", quantization: "Q4_K_M", state: "not-loaded", max_context_length: 32768 }] })] }, true],
    ["an OpenAI model list on LM Studio's port", probeLocalLmStudioServer, { "/api/v0/models": [200, models("organization_owner")] }, false],
    ["vLLM", probeLocalVllmServer, { "/v1/models": [200, models("vllm")] }, true],
    ["llama-server's model list as vLLM", probeLocalVllmServer, { "/v1/models": [200, models("llamacpp")] }, false],
    ["SGLang", probeLocalSglangServer, { "/v1/models": [200, models("sglang")] }, true],
    ["TabbyAPI", probeLocalTabbyApiServer, { "/v1/models": [200, models("tabbyAPI")] }, true],
    ["vLLM's list as TabbyAPI", probeLocalTabbyApiServer, { "/v1/models": [200, models("vllm")] }, false],
    ["KoboldCpp", probeLocalKoboldCppServer, { "/api/extra/version": [200, '{"result":"KoboldCpp","version":"1.103","protected":false}'] }, true],
    ["another /api/extra/version", probeLocalKoboldCppServer, { "/api/extra/version": [200, '{"result":"Other","version":"1"}'] }, false],
  ];
  for (const [what, probe, replies, expected] of cases) {
    it(`${expected ? "identifies" : "ignores"} ${what}`, () =>
      withReplies(replies, async (url) => assert.equal(await probe(url), expected)));
  }

  it("does not probe non-loopback hosts", async () => {
    for (const probe of [probeLocalOllamaServer, probeLocalLmStudioServer, probeLocalVllmServer, probeLocalKoboldCppServer]) {
      assert.equal(await probe("http://192.0.2.1:11434"), false);
    }
  });
});
