import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as http from "node:http";
import * as os from "node:os";
import * as path from "node:path";
import { after, before, describe, it } from "node:test";
import { DownloadManager } from "../src/downloadManager";
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
  it("protects another process's direct download from cleanup and duplicate writers", { timeout: 5000 }, async (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-cross-window-download-"));
    const dest = path.join(dir, "model.gguf");
    let response: http.ServerResponse | undefined;
    const server = http.createServer((_req, res) => {
      response = res; res.writeHead(200, { "Content-Length": 2 }); res.write("a");
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const url = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
    const script = `require(process.argv[1]).downloadHttpFile({ url: process.argv[2], dest: process.argv[3] }).catch(err => { console.error(err); process.exitCode = 1; });`;
    const child = spawn(process.execPath, ["-e", script, require.resolve("../src/httpDownload"), url, dest], { stdio: ["ignore", "ignore", "pipe"] });
    let errors = "";
    child.stderr?.on("data", (part) => { errors += part; });
    const done = new Promise<void>((resolve, reject) => {
      child.on("error", reject);
      child.on("exit", (code) => code === 0 ? resolve() : reject(new Error(errors)));
    });
    done.catch(() => undefined);
    t.after(async () => {
      child.kill(); server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
      fs.rmSync(dir, { recursive: true, force: true });
    });
    const until = Date.now() + 2000;
    while (!fs.existsSync(dest + ".partial") || fs.statSync(dest + ".partial").size !== 1) {
      assert.ok(Date.now() < until, "child download did not start");
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    const mgr = new DownloadManager();
    assert.equal(await mgr.removePartialIfIdle(dest + ".partial", async () => assert.fail("foreign writer must be protected")), false);
    await assert.rejects(downloadHttpFile({ url, dest }), (err: unknown) => (err as NodeJS.ErrnoException).code === "ELOCKED");
    response!.end("b"); await done;
    assert.equal(fs.readFileSync(dest, "utf8"), "ab");
    assert.equal(fs.existsSync(dest + ".partial.lock"), false);
  });

  for (const range of [undefined, "bytes 0-4/10", "bytes 5-9/11", "bytes 5-8/10", "bytes 5-9/*", "bytes 5-9/9007199254740993"]) {
    it(`rejects an inconsistent resumed range: ${range}`, async (t) => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-range-"));
      const dest = path.join(dir, "model.gguf");
      fs.writeFileSync(dest + ".partial", "01234");
      const server = http.createServer((req, res) => {
        assert.equal(req.headers.range, "bytes=5-");
        res.writeHead(206, { "Content-Length": 5, ...(range ? { "Content-Range": range } : {}) });
        res.end("ABCDE");
      });
      await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
      t.after(async () => {
        server.closeAllConnections();
        await new Promise<void>((resolve) => server.close(() => resolve()));
        fs.rmSync(dir, { recursive: true, force: true });
      });
      await assert.rejects(downloadHttpFile({
        dest, expectedSize: 10, url: `http://127.0.0.1:${(server.address() as { port: number }).port}`,
      }), DownloadVerifyError);
      assert.equal(fs.existsSync(dest), false);
      assert.equal(fs.readFileSync(dest + ".partial", "utf8"), "01234");
    });
  }

  for (const complete of [true, false]) {
    it(`checks chunked ranged bodies before publishing (complete=${complete})`, async (t) => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-range-body-"));
      const dest = path.join(dir, "model.gguf");
      fs.writeFileSync(dest + ".partial", "01234");
      const server = http.createServer((_req, res) => {
        res.writeHead(206, { "Content-Range": "bytes 5-9/10", "Transfer-Encoding": "chunked" });
        res.end(complete ? "56789" : "56");
      });
      await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
      t.after(async () => {
        await new Promise<void>((resolve) => server.close(() => resolve()));
        fs.rmSync(dir, { recursive: true, force: true });
      });
      const result = downloadHttpFile({ dest, url: `http://127.0.0.1:${(server.address() as { port: number }).port}` });
      if (complete) {
        await result;
        assert.equal(fs.readFileSync(dest, "utf8"), "0123456789");
      } else {
        await assert.rejects(result, DownloadVerifyError);
        assert.equal(fs.existsSync(dest), false);
      }
    });
  }

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
