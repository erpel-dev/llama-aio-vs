import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as http from "node:http";
import * as os from "node:os";
import * as path from "node:path";
import { after, before, describe, it } from "node:test";
import { DownloadManager } from "../src/downloadManager";

describe("DownloadManager", () => {
  const payload = Buffer.from("manager-payload-0123456789");
  const sha = createHash("sha256").update(payload).digest("hex");
  let server: http.Server;
  let base = "";

  before(async () => {
    server = http.createServer((_req, res) => {
      res.writeHead(200, { "Content-Length": payload.length });
      res.end(payload);
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

  it("runs a job to done and reports progress", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-mgr-"));
    const dest = path.join(dir, "a.gguf");
    const mgr = new DownloadManager(1);
    try {
      const { done, job } = mgr.enqueue({
        label: "a.gguf",
        dest,
        url: `${base}/a`,
        expectedSize: payload.length,
        expectedSha256: sha,
      });
      assert.equal(job.state === "queued" || job.state === "running", true);
      const out = await done;
      assert.equal(out, dest);
      assert.equal(mgr.list()[0]?.state, "done");
      assert.equal(fs.readFileSync(dest).equals(payload), true);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it("cancel rejects the waiter", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-mgr-"));
    const dest = path.join(dir, "b.gguf");
    const mgr = new DownloadManager(1);
    const { done, job } = mgr.enqueue({
      label: "b.gguf",
      dest,
      url: `${base}/b`,
    });
    mgr.cancel(job.id);
    await assert.rejects(done);
    assert.equal(mgr.list()[0]?.state, "cancelled");
    fs.rmSync(dir, { recursive: true, force: true });
  });
});
