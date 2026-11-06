import assert from "node:assert/strict";
import * as http from "node:http";
import { after, before, describe, it } from "node:test";
import { gatewayRefusal, WebUiGateway } from "../src/webUiGateway";

describe("gatewayRefusal", () => {
  it("allows the gateway's own page and plain navigation", () => {
    assert.equal(gatewayRefusal({ host: "127.0.0.1:9742" }, true), undefined);
    assert.equal(gatewayRefusal({ host: "localhost:9742", origin: "http://localhost:9742", secFetchSite: "same-origin" }, true), undefined);
    assert.equal(gatewayRefusal({ host: "[::1]:9742" }, true), undefined);
  });
  it("refuses DNS rebinding (non-loopback Host) on a desktop editor", () => {
    assert.match(gatewayRefusal({ host: "evil.example:9742" }, true) || "", /Host/);
    assert.equal(gatewayRefusal({ host: "codeserver.example" }, false), undefined, "editor-forwarded port");
  });
  it("refuses other sites, even with a loopback Host", () => {
    assert.match(gatewayRefusal({ host: "127.0.0.1:9742", origin: "https://evil.example" }, true) || "", /another site/);
    assert.match(gatewayRefusal({ host: "127.0.0.1:9742", secFetchSite: "cross-site" }, true) || "", /cross-site/);
    assert.match(gatewayRefusal({ host: "127.0.0.1:9742", origin: "null" }, true) || "", /opaque/);
    assert.match(gatewayRefusal({ host: "codeserver.example", origin: "https://evil.example" }, false) || "", /another site/);
  });
});

describe("WebUiGateway", () => {
  let upstream: http.Server;
  let upstreamUrl = "";
  const seen: Array<{ auth?: string; host?: string; origin?: string }> = [];
  let gateway: WebUiGateway;
  let base = "";
  before(async () => {
    upstream = http.createServer((req, res) => {
      seen.push({ auth: req.headers.authorization, host: req.headers.host, origin: req.headers.origin as string | undefined });
      if (req.url === "/stream") {
        res.writeHead(200, { "content-type": "text/event-stream", "access-control-allow-origin": "*" });
        res.write("data: one\n\n");
        setTimeout(() => res.end("data: two\n\n"), 20);
        return;
      }
      res.writeHead(req.headers.authorization === "Bearer k1" ? 200 : 401, { "access-control-allow-origin": "*" });
      res.end(req.headers.authorization === "Bearer k1" ? "ok" : "denied");
    });
    await new Promise<void>((r) => upstream.listen(0, "127.0.0.1", () => r()));
    upstreamUrl = `http://127.0.0.1:${(upstream.address() as { port: number }).port}`;
    gateway = new WebUiGateway({ upstream: () => upstreamUrl, apiKey: () => "k1", strictHost: () => true });
    const port = await gateway.start(0);
    base = `http://127.0.0.1:${port}`;
  });
  after(() => {
    gateway.close();
    upstream.close();
  });

  const get = (path: string, headers: Record<string, string> = {}) =>
    new Promise<{ status: number; body: string; headers: http.IncomingHttpHeaders }>((resolve, reject) => {
      const req = http.request(base, { headers, path }, (res) => {
        let body = "";
        res.on("data", (c) => (body += c));
        res.on("end", () => resolve({ status: res.statusCode || 0, body, headers: res.headers }));
      });
      req.on("error", reject);
      req.end();
    });

  it("adds the managed key, replaces any key the page sent, and strips CORS grants", async () => {
    const r = await get("/props", { authorization: "Bearer stale-from-browser" });
    assert.equal(r.status, 200);
    assert.equal(r.body, "ok");
    assert.equal(seen.at(-1)?.auth, "Bearer k1");
    assert.equal(r.headers["access-control-allow-origin"], undefined);
    assert.ok(!r.body.includes("k1"));
  });

  it("streams SSE through unchanged", async () => {
    const r = await get("/stream");
    assert.equal(r.body, "data: one\n\ndata: two\n\n");
  });

  it("rejects absolute, network-path and backslash targets without forwarding credentials", async () => {
    const before = seen.length;
    for (const target of [upstreamUrl + "/props", "//" + new URL(upstreamUrl).host + "/props",
      "/" + String.fromCharCode(92) + new URL(upstreamUrl).host + "/props"]) {
      const r = await get(target);
      assert.equal(r.status, 400, target);
      assert.ok(!r.body.includes("k1"));
    }
    assert.equal(seen.length, before);
  });

  it("refuses nonlocal or unsupported upstreams before constructing a credentialed request", async () => {
    for (const target of ["", "http://example.invalid", "https://127.0.0.1:1"]) {
      let keyReads = 0;
      const g = new WebUiGateway({ upstream: () => target, apiKey: () => { keyReads++; return "k1"; }, strictHost: () => true });
      try {
        const port = await g.start();
        const status = await new Promise<number>((resolve, reject) => {
          http.get(`http://127.0.0.1:${port}/props`, res => { res.resume(); res.on("end", () => resolve(res.statusCode!)); }).on("error", reject);
        });
        assert.equal(status, 502);
        assert.equal(keyReads, 0);
      } finally { g.close(); }
    }
  });

  it("refuses a cross-site request before it reaches the server", async () => {
    const before = seen.length;
    const r = await get("/v1/chat/completions", { origin: "https://evil.example" });
    assert.equal(r.status, 403);
    assert.equal(seen.length, before);
    assert.ok(!r.body.includes("k1"));
  });

  it("answers 502 with a hint when llama-server is down", async () => {
    const g = new WebUiGateway({ upstream: () => "http://127.0.0.1:1", apiKey: () => "k1", strictHost: () => true });
    const port = await g.start(0);
    const r = await new Promise<{ status: number; body: string }>((resolve) => {
      http.get(`http://127.0.0.1:${port}/`, (res) => {
        let body = "";
        res.on("data", (c) => (body += c));
        res.on("end", () => resolve({ status: res.statusCode || 0, body }));
      });
    });
    g.close();
    assert.equal(r.status, 502);
    assert.match(r.body, /not reachable/);
  });

  it("falls back to a free port when the preferred one is busy", async () => {
    const busyPort = (upstream.address() as { port: number }).port;
    const g = new WebUiGateway({ upstream: () => upstreamUrl, apiKey: () => "k1", strictHost: () => true });
    const port = await g.start(busyPort);
    g.close();
    assert.notEqual(port, busyPort);
  });
});
