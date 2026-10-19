import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as http from "node:http";
import * as os from "node:os";
import * as path from "node:path";
import { after, before, describe, it } from "node:test";
import {
  downloadHttpFile,
  DownloadVerifyError,
  GatedDownloadError,
  normalizeSha256,
  resolveRedirect,
  sendsTokenTo,
} from "../src/httpDownload";

describe("normalizeSha256", () => {
  it("accepts a hex digest with or without the lfs prefix", () => {
    const hex = "a".repeat(64);
    assert.equal(normalizeSha256(`sha256:${hex}`), hex);
    assert.equal(normalizeSha256(hex.toUpperCase()), hex);
    assert.equal(normalizeSha256("nope"), undefined);
  });
});

describe("redirect helpers", () => {
  it("resolves a relative Location against the sending URL", () => {
    assert.equal(
      resolveRedirect("https://huggingface.co/old/m/resolve/main/a.gguf", "/new/m/resolve/main/a.gguf"),
      "https://huggingface.co/new/m/resolve/main/a.gguf"
    );
  });

  it("refuses an HTTPS to HTTP downgrade", () => {
    assert.throws(() => resolveRedirect("https://huggingface.co/a", "http://cdn.example/a"), /HTTPS to HTTP/);
    assert.equal(resolveRedirect("http://127.0.0.1:1/a", "http://127.0.0.1:2/b"), "http://127.0.0.1:2/b");
  });

  it("sends the token only to the original origin", () => {
    assert.equal(sendsTokenTo("https://huggingface.co/a", "https://huggingface.co/b"), true);
    assert.equal(sendsTokenTo("https://huggingface.co/a", "https://cdn-lfs.hf.co/b"), false);
    assert.equal(sendsTokenTo("https://huggingface.co/a", "http://huggingface.co/b"), false);
  });
});

describe("downloadHttpFile", () => {
  for (const mode of ["headers", "redirect", "body", "progress"] as const) {
    it(`covers ${mode} waits with the idle timeout`, { timeout: 3000 }, async (t) => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-idle-"));
      const server = http.createServer((req, res) => {
        if (mode === "redirect" && req.url === "/start") {
          res.writeHead(302, { Location: "/silent" }); res.end();
        } else if (mode === "body") {
          res.writeHead(200); res.write("a");
        } else if (mode === "progress") {
          let sent = 0;
          const timer = setInterval(() => {
            res.write("a");
            if (++sent === 8) { clearInterval(timer); res.end(); }
          }, 20);
          res.on("close", () => clearInterval(timer));
        }
      });
      await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
      t.after(async () => {
        server.closeAllConnections();
        await new Promise<void>((resolve) => server.close(() => resolve()));
        fs.rmSync(dir, { recursive: true, force: true });
      });
      const dest = path.join(dir, "idle.gguf");
      const started = Date.now();
      const result = downloadHttpFile({
        url: `http://127.0.0.1:${(server.address() as { port: number }).port}/start`,
        dest, idleTimeoutMs: 100, signal: AbortSignal.timeout(1000),
      });
      if (mode === "progress") {
        await result;
        assert.equal(fs.readFileSync(dest, "utf8"), "a".repeat(8));
      } else {
        await assert.rejects(result, /idle timeout/);
        assert.ok(Date.now() - started < 900, "timeout must precede the cleanup abort");
      }
    });
  }

  const payload = Buffer.from("hello-llama-aio-download-manager-payload");
  const sha = createHash("sha256").update(payload).digest("hex");
  let server: http.Server;
  let base = "";
  let dropAfter = 0;
  let hits = 0;
  let cdn: http.Server;
  let cdnBase = "";
  const authSeen: Record<string, string | undefined> = {};

  before(async () => {
    server = http.createServer((req, res) => {
      hits += 1;
      const url = req.url || "/";
      authSeen[url] = req.headers.authorization;
      if (url.startsWith("/renamed")) {
        res.writeHead(307, { Location: "/file" });
        res.end();
        return;
      }
      if (url.startsWith("/to-cdn")) {
        res.writeHead(302, { Location: `${cdnBase}/file` });
        res.end();
        return;
      }
      if (url.startsWith("/gated")) {
        res.writeHead(401, { "Content-Type": "text/plain" });
        res.end("gated");
        return;
      }
      if (url.startsWith("/missing")) {
        res.writeHead(404);
        res.end("nope");
        return;
      }
      const range = req.headers.range;
      let start = 0;
      if (range) {
        const m = /^bytes=(\d+)-/.exec(range);
        start = m ? Number(m[1]) : 0;
      }
      const slice = payload.subarray(start);
      if (dropAfter > 0 && start === 0) {
        res.writeHead(200, { "Content-Length": payload.length, "Content-Type": "application/octet-stream" });
        res.write(payload.subarray(0, dropAfter));
        res.destroy();
        return;
      }
      if (start > 0) {
        res.writeHead(206, {
          "Content-Length": slice.length,
          "Content-Range": `bytes ${start}-${payload.length - 1}/${payload.length}`,
        });
      } else {
        res.writeHead(200, { "Content-Length": payload.length });
      }
      res.end(slice);
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const addr = server.address();
    if (!addr || typeof addr === "string") {
      throw new Error("no listen port");
    }
    base = `http://127.0.0.1:${addr.port}`;

    cdn = http.createServer((req, res) => {
      authSeen[`cdn:${req.url}`] = req.headers.authorization;
      res.writeHead(200, { "Content-Length": payload.length });
      res.end(payload);
    });
    await new Promise<void>((resolve) => cdn.listen(0, "127.0.0.1", resolve));
    const cdnAddr = cdn.address();
    if (!cdnAddr || typeof cdnAddr === "string") {
      throw new Error("no listen port");
    }
    cdnBase = `http://127.0.0.1:${cdnAddr.port}`;
  });

  after(async () => {
    await new Promise<void>((resolve) => server.close(() => resolve()));
    await new Promise<void>((resolve) => cdn.close(() => resolve()));
  });

  const tmp = () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-dl-"));
    return {
      dest: path.join(dir, "model.gguf"),
      dir,
    };
  };

  it("writes the file and checks size + sha256", async () => {
    const { dest, dir } = tmp();
    try {
      await downloadHttpFile({
        url: `${base}/file`,
        dest,
        expectedSize: payload.length,
        expectedSha256: sha,
      });
      assert.equal(fs.readFileSync(dest).toString(), payload.toString());
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it("resumes a partial after the connection drops", async () => {
    const { dest, dir } = tmp();
    hits = 0;
    try {
      fs.writeFileSync(dest + ".partial", payload.subarray(0, 10));
      await downloadHttpFile({
        url: `${base}/file`,
        dest,
        expectedSize: payload.length,
        expectedSha256: sha,
      });
      assert.equal(fs.readFileSync(dest).equals(payload), true);
      assert.ok(hits >= 1);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it("does not treat a too-small dest as complete", async () => {
    const { dest, dir } = tmp();
    try {
      fs.writeFileSync(dest, "tiny");
      await downloadHttpFile({
        url: `${base}/file`,
        dest,
        expectedSize: payload.length,
        expectedSha256: sha,
      });
      assert.equal(fs.statSync(dest).size, payload.length);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it("rejects a sha256 mismatch", async () => {
    const { dest, dir } = tmp();
    try {
      await assert.rejects(
        () =>
          downloadHttpFile({
            url: `${base}/file`,
            dest,
            expectedSize: payload.length,
            expectedSha256: "b".repeat(64),
          }),
        (e: unknown) => e instanceof DownloadVerifyError
      );
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it("maps 401 to a gated error with a page URL", async () => {
    const { dest, dir } = tmp();
    try {
      await assert.rejects(
        () =>
          downloadHttpFile({
            url: `${base}/gated`,
            dest,
            gatedPageUrl: "https://huggingface.co/org/gated-model",
          }),
        (e: unknown) =>
          e instanceof GatedDownloadError && e.pageUrl.includes("gated-model")
      );
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it("follows a relative redirect (renamed repo)", async () => {
    const { dest, dir } = tmp();
    try {
      await downloadHttpFile({ url: `${base}/renamed`, dest, expectedSha256: sha, token: "hf_secret" });
      assert.equal(fs.readFileSync(dest).equals(payload), true);
      assert.equal(authSeen["/file"], "Bearer hf_secret");
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it("drops the token on a cross-origin redirect", async () => {
    const { dest, dir } = tmp();
    try {
      await downloadHttpFile({ url: `${base}/to-cdn`, dest, expectedSha256: sha, token: "hf_secret" });
      assert.equal(fs.readFileSync(dest).equals(payload), true);
      assert.equal(authSeen["/to-cdn"], "Bearer hf_secret");
      assert.equal(authSeen["cdn:/file"], undefined);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});
