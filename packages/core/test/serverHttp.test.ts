import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import { describe, it, afterEach } from "node:test";
import {
  ConnectionError,
  isPublicPlainHttp,
  joinServerUrl,
  localEndpoint,
  normalizeServerUrl,
  serverJson,
  setServerApiKeyProvider,
} from "../src/index";

describe("normalizeServerUrl", () => {
  it("adds http and drops a query string", () => {
    assert.equal(normalizeServerUrl("gpu-box.lan:8080/llama/?x=1"), "http://gpu-box.lan:8080/llama");
  });

  it("rejects credentials in the URL", () => {
    assert.throws(() => normalizeServerUrl("http://user:secret@host:8080"), /API key/);
  });

  it("rejects non-http schemes", () => {
    assert.throws(() => normalizeServerUrl("ftp://host/file"), /http/);
  });
});

describe("joinServerUrl", () => {
  it("keeps a reverse-proxy prefix", () => {
    assert.equal(joinServerUrl("https://example.test/llama/", "/v1/models"), "https://example.test/llama/v1/models");
  });
});

describe("isPublicPlainHttp", () => {
  it("is true only for plain http outside private and loopback space", () => {
    assert.equal(isPublicPlainHttp("http://203.0.113.10:8080"), true);
    assert.equal(isPublicPlainHttp("http://192.168.0.5:8080"), false);
    assert.equal(isPublicPlainHttp("http://127.0.0.1:8080"), false);
    assert.equal(isPublicPlainHttp("https://203.0.113.10"), false);
    assert.equal(isPublicPlainHttp("http://gpu-box.lan:8080"), false);
  });
});

describe("localEndpoint", () => {
  it("maps a wildcard bind to loopback", () => {
    assert.equal(localEndpoint("0.0.0.0", 8080), "http://127.0.0.1:8080");
    assert.equal(localEndpoint("::", 8080), "http://[::1]:8080");
  });
});

describe("serverJson", () => {
  let server: Server;
  let base = "";
  const seen: string[] = [];

  afterEach(async () => {
    setServerApiKeyProvider(undefined);
    seen.length = 0;
    if (server) {
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });

  async function listen(handler: (req: import("node:http").IncomingMessage, res: import("node:http").ServerResponse) => void) {
    server = createServer(handler);
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
    const addr = server.address();
    if (!addr || typeof addr === "string") {
      throw new Error("no port");
    }
    base = `http://127.0.0.1:${addr.port}`;
  }

  it("sends the bearer key registered for the origin and keeps the response body", async () => {
    setServerApiKeyProvider(() => "top-secret");
    await listen((req, res) => {
      seen.push(`${req.url} ${req.headers.authorization || ""}`);
      if (req.url === "/llama/props") {
        res.writeHead(401, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: "unauthorized" } }));
        return;
      }
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ ok: true }));
    });
    const ok = await serverJson<{ ok: boolean }>(joinServerUrl(`${base}/llama`, "/health"));
    assert.equal(ok.ok, true);
    assert.match(seen[0] || "", /Bearer top-secret/);
    await assert.rejects(
      () => serverJson(joinServerUrl(`${base}/llama`, "/props")),
      (err: unknown) => {
        assert.ok(err instanceof ConnectionError);
        assert.equal(err.kind, "auth");
        assert.match(err.body || "", /unauthorized/);
        return true;
      }
    );
  });

  it("classifies a loading 503", async () => {
    await listen((_req, res) => {
      res.writeHead(503, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: { message: "Loading model" } }));
    });
    await assert.rejects(
      () => serverJson(`${base}/health`),
      (err: unknown) => err instanceof ConnectionError && err.kind === "loading"
    );
  });

  it("reports connection refused", async () => {
    await assert.rejects(
      () => serverJson("http://127.0.0.1:1/health", { timeoutMs: 2000 }),
      (err: unknown) => err instanceof ConnectionError && err.kind === "refused"
    );
  });
});
