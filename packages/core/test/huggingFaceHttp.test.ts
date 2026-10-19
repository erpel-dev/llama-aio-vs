import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import * as http from "node:http";
import https = require("node:https");
import { Readable } from "node:stream";
import { describe, it, type TestContext } from "node:test";
import { HuggingFaceClient } from "../src/huggingFace";
import type { SettingsStore } from "../src/settings";

const origin = "https://huggingface.co";
const client = () => new HuggingFaceClient({
  getConfig: () => ({ get: () => "SYNTHETIC_TEST_TOKEN" }),
} as unknown as SettingsStore);

type Reply = { status?: number; headers?: http.IncomingHttpHeaders; body?: unknown };
function intercept(t: TestContext, reply: (url: URL, headers: http.OutgoingHttpHeaders) => Reply): void {
  t.mock.method(https, "get", (url: string, options: https.RequestOptions, cb: (res: http.IncomingMessage) => void) => {
    const parsed = new URL(url); // Match Node's synchronous request validation.
    if (parsed.protocol !== "https:") throw new Error("unsupported protocol");
    const response = reply(parsed, (options.headers || {}) as http.OutgoingHttpHeaders);
    const req = new EventEmitter();
    queueMicrotask(() => {
      const res = Readable.from([Buffer.from(JSON.stringify(response.body ?? []))]) as http.IncomingMessage;
      res.statusCode = response.status ?? 200;
      res.headers = response.headers || {};
      cb(res);
    });
    return req;
  });
}

describe("Hugging Face metadata transport", { timeout: 5000 }, () => {
  for (const kind of ["search", "tree"] as const) {
    const request = (c: HuggingFaceClient) => kind === "search" ? c.searchGgufModels("test") : c.listGgufFiles("old/model");

    it(`resolves relative ${kind} redirects and scopes tokens to the original origin`, async (t) => {
      const seen: Array<[string, unknown]> = [];
      intercept(t, (url, headers) => {
        seen.push([url.origin, headers.Authorization]);
        if (seen.length === 1) return { status: 307, headers: { location: "/renamed" } };
        if (url.pathname === "/renamed") return { status: 302, headers: { location: "https://other-origin.invalid/final" } };
        return {};
      });
      await request(client());
      assert.deepEqual(seen, [
        [origin, "Bearer SYNTHETIC_TEST_TOKEN"],
        [origin, "Bearer SYNTHETIC_TEST_TOKEN"],
        ["https://other-origin.invalid", undefined],
      ]);
    });

    for (const location of ["http://other-origin.invalid/", "https://[invalid"]) {
      it(`rejects an unsafe or invalid ${kind} redirect (${location})`, async (t) => {
        intercept(t, () => ({ status: 302, headers: { location } }));
        await assert.rejects(request(client()), /HTTPS|Invalid URL/);
      });
    }

    it(`bounds ${kind} redirect loops separately from page count`, async (t) => {
      let hops = 0;
      intercept(t, () => {
        assert.ok(++hops <= 20, "redirect loop was not bounded");
        return { status: 302, headers: { location: "/loop" } };
      });
      await assert.rejects(request(client()), /Too many redirects/);
    });

    it(`handles synchronous request-construction errors after a ${kind} redirect`, async (t) => {
      intercept(t, (url) => {
        if (url.pathname === "/throws") throw new Error("request construction failed");
        return { status: 302, headers: { location: `${origin}/throws` } };
      });
      await assert.rejects(request(client()), /request construction failed/);
    });
  }

  it("resolves relative page links and strips credentials from cross-origin pages", async (t) => {
    const seen: unknown[] = [];
    intercept(t, (url, headers) => {
      seen.push(headers.Authorization);
      if (seen.length === 1) return {
        headers: { link: '<?cursor=2>; rel="next"' },
        body: [{ path: "a.gguf", type: "file", size: 1 }],
      };
      if (url.search === "?cursor=2") return {
        headers: { link: '<https://other-origin.invalid/page3>; rel="next"' },
        body: [{ path: "b.gguf", type: "file", size: 2 }],
      };
      return { body: [{ path: "c.gguf", type: "file", size: 3 }] };
    });
    const files = await client().listGgufFiles("org/model");
    assert.deepEqual(files.map((f) => f.path), ["a.gguf", "b.gguf", "c.gguf"]);
    assert.deepEqual(seen, ["Bearer SYNTHETIC_TEST_TOKEN", "Bearer SYNTHETIC_TEST_TOKEN", undefined]);
  });

  it("rejects malformed pagination URLs as a handled error", async (t) => {
    intercept(t, () => ({ headers: { link: '<https://[invalid>; rel="next"' } }));
    await assert.rejects(client().listGgufFiles("org/model"), /Invalid URL/);
  });
});
