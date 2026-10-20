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
  probeLocalStrataServer,
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
