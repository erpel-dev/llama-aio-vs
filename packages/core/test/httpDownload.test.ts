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
} from "../src/httpDownload";

describe("normalizeSha256", () => {
  it("accepts a hex digest with or without the lfs prefix", () => {
    const hex = "a".repeat(64);
    assert.equal(normalizeSha256(`sha256:${hex}`), hex);
    assert.equal(normalizeSha256(hex.toUpperCase()), hex);
    assert.equal(normalizeSha256("nope"), undefined);
  });
});

describe("downloadHttpFile", () => {
  const payload = Buffer.from("hello-llama-aio-download-manager-payload");
  const sha = createHash("sha256").update(payload).digest("hex");
  let server: http.Server;
  let base = "";
  let dropAfter = 0;
  let hits = 0;

  before(async () => {
    server = http.createServer((req, res) => {
      hits += 1;
      const url = req.url || "/";
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
  });

  after(async () => {
    await new Promise<void>((resolve) => server.close(() => resolve()));
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
});
