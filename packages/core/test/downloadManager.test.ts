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

  it("revalidates completed files after deletion or replacement", async (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-redownload-"));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const mgr = new DownloadManager();
    const spec = { dest: path.join(dir, "model.gguf"), label: "model", url: base, expectedSize: payload.length, expectedSha256: sha };
    const first = mgr.enqueue(spec); await first.done;
    fs.unlinkSync(spec.dest);
    const second = mgr.enqueue(spec); await second.done;
    assert.notEqual(first.job.id, second.job.id);
    assert.equal(fs.readFileSync(spec.dest).equals(payload), true);
    fs.writeFileSync(spec.dest, Buffer.alloc(payload.length));
    await mgr.enqueue(spec).done;
    assert.equal(fs.readFileSync(spec.dest).equals(payload), true, "a same-sized corrupt file is checked again");
  });
});

async function waitFor(check: () => boolean): Promise<void> {
  const deadline = Date.now() + 2000;
  while (!check()) {
    assert.ok(Date.now() < deadline, "condition did not settle");
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

describe("download queue regressions", () => {
  it("reserves cleanup destinations and excludes pending, paused and closing jobs", async (t) => {
    const transport = require("../src/httpDownload") as typeof import("../src/httpDownload");
    const attempts: Array<(error: Error) => void> = [];
    t.mock.method(transport, "downloadHttpFile", () => new Promise<string>((_resolve, reject) => attempts.push(reject)));
    const mgr = new DownloadManager(1);
    const dest = path.join(os.tmpdir(), "reserved.gguf");
    let release!: () => void;
    const cleaning = mgr.removePartialIfIdle(dest + ".partial", () => new Promise<void>((resolve) => { release = resolve; }));
    const job = mgr.enqueue({ dest, label: "reserved", url: "https://example.invalid" });
    assert.equal(job.job.state, "queued");
    assert.equal(attempts.length, 0);
    assert.equal(await mgr.removePartialIfIdle(dest + ".partial", async () => assert.fail("already reserved")), false);
    await waitFor(() => !!release);
    release(); await cleaning;
    assert.equal(attempts.length, 1);
    mgr.pause(job.job.id);
    assert.equal(mgr.isPartialInUse(dest + ".partial"), true);
    mgr.cancel(job.job.id); mgr.clearFinished();
    assert.equal(await mgr.removePartialIfIdle(dest + ".partial", async () => assert.fail("writer still closing")), false);
    attempts[0]!(new transport.DownloadAbortError("pause"));
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(await mgr.removePartialIfIdle(dest + ".partial", async () => undefined), true);
  });

  for (const concurrency of [1, 2]) {
    it(`drains queued downloads within ${concurrency} slots`, { timeout: 5000 }, async (t) => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-queue-"));
      const waiting: http.ServerResponse[] = [];
      const server = http.createServer((_req, res) => {
        res.writeHead(200, { "Content-Length": 2 });
        res.write("a");
        waiting.push(res);
      });
      await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
      const port = (server.address() as { port: number }).port;
      const mgr = new DownloadManager(concurrency);
      const pump = t.mock.method(mgr as unknown as { pump: () => void }, "pump");
      t.after(async () => {
        mgr.list().forEach((j) => mgr.cancel(j.id));
        server.closeAllConnections();
        await new Promise<void>((resolve) => server.close(() => resolve()));
        fs.rmSync(dir, { recursive: true, force: true });
      });
      const jobs = Array.from({ length: 5 }, (_, i) => mgr.enqueue({
        label: String(i), dest: path.join(dir, `${i}.gguf`), url: `http://127.0.0.1:${port}/${i}`,
      }));
      await waitFor(() => waiting.length === concurrency);
      assert.equal(mgr.list().filter((j) => j.state === "running").length, concurrency);
      for (let i = 0; i < jobs.length; i++) {
        await waitFor(() => waiting.length > i);
        waiting[i]!.end("b");
      }
      await Promise.all(jobs.map((j) => j.done));
      assert.ok(mgr.list().every((j) => j.state === "done"));
      assert.ok(pump.mock.callCount() < 100, "a full queue must not recursively reschedule itself");
      for (const job of jobs) assert.equal(fs.readFileSync(job.job.dest, "utf8"), "ab");
    });
  }

  for (const concurrency of [1, 2]) for (const viaSubscriber of [false, true]) {
    it(`honors immediate resume${viaSubscriber ? " from a subscriber" : ""} (slots=${concurrency})`, { timeout: 5000 }, async (t) => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-resume-"));
      let hits = 0;
      const server = http.createServer((_req, res) => {
        hits++;
        res.writeHead(200, { "Content-Length": 2 });
        if (hits === 1) res.write("a");
        else res.end("ab");
      });
      await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
      const mgr = new DownloadManager(concurrency);
      let writers = 0;
      let maxWriters = 0;
      const createWriteStream = fs.createWriteStream;
      t.mock.method(require("fs"), "createWriteStream", (...args: Parameters<typeof fs.createWriteStream>) => {
        const stream = createWriteStream(...args);
        maxWriters = Math.max(maxWriters, ++writers);
        stream.once("close", () => writers--);
        return stream;
      });
      t.after(async () => {
        mgr.list().forEach((j) => mgr.cancel(j.id));
        server.closeAllConnections();
        await new Promise<void>((resolve) => server.close(() => resolve()));
        fs.rmSync(dir, { recursive: true, force: true });
      });
      const dest = path.join(dir, "resume.gguf");
      const { job, done } = mgr.enqueue({ label: "resume", dest,
        url: `http://127.0.0.1:${(server.address() as { port: number }).port}/file` });
      await waitFor(() => fs.existsSync(dest + ".partial") && fs.statSync(dest + ".partial").size === 1);
      if (viaSubscriber) mgr.subscribe((jobs) => {
        if (jobs[0]?.state === "paused") mgr.resume(job.id);
      });
      mgr.pause(job.id);
      mgr.resume(job.id);
      await done;
      assert.equal(hits, 2);
      assert.equal(mgr.list()[0]?.state, "done");
      assert.equal(fs.readFileSync(dest, "utf8"), "ab");
      assert.equal(maxWriters, 1, "the old writer must close before resume opens the partial");
    });
  }

  it("keeps a cancelled destination locked through cleanup and clearFinished", async (t) => {
    const transport = require("../src/httpDownload") as typeof import("../src/httpDownload");
    const attempts: Array<{ resolve: (dest: string) => void; reject: (err: Error) => void }> = [];
    t.mock.method(transport, "downloadHttpFile", () => new Promise<string>((resolve, reject) => {
      attempts.push({ resolve, reject });
    }));
    const mgr = new DownloadManager(2);
    const spec = { dest: path.join(os.tmpdir(), "cancelled-retry.gguf"), label: "test", url: "https://example.invalid" };
    const first = mgr.enqueue(spec);
    mgr.pause(first.job.id);
    mgr.resume(first.job.id);
    mgr.cancel(first.job.id);
    mgr.clearFinished();
    const retry = mgr.enqueue(spec);
    assert.equal(attempts.length, 1, "cancelled transfer still owns the destination until cleanup settles");
    attempts[0]!.reject(new transport.DownloadAbortError("pause"));
    await assert.rejects(first.done, /cancelled/);
    await waitFor(() => attempts.length === 2);
    assert.equal(mgr.list()[0]?.state, "running");
    attempts[1]!.resolve(spec.dest);
    await retry.done;
    assert.equal(mgr.list()[0]?.state, "done");
  });

  it("keeps a second pause requested before the first abort settles", async (t) => {
    const transport = require("../src/httpDownload") as typeof import("../src/httpDownload");
    const attempts: Array<{ resolve: (dest: string) => void; reject: (err: Error) => void }> = [];
    t.mock.method(transport, "downloadHttpFile", () => new Promise<string>((resolve, reject) => {
      attempts.push({ resolve, reject });
    }));
    const mgr = new DownloadManager(2);
    const spec = { dest: path.join(os.tmpdir(), "paused-retry.gguf"), label: "test", url: "https://example.invalid" };
    const job = mgr.enqueue(spec);
    mgr.pause(job.job.id);
    mgr.resume(job.job.id);
    mgr.pause(job.job.id);
    attempts[0]!.reject(new transport.DownloadAbortError("pause"));
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(attempts.length, 1);
    assert.equal(mgr.list()[0]?.state, "paused");
    mgr.resume(job.job.id);
    assert.equal(attempts.length, 2);
    attempts[1]!.resolve(spec.dest);
    await job.done;
    assert.equal(mgr.list()[0]?.state, "done");
  });

  for (const status of [500, 401]) {
    it(`deduplicates retries after HTTP ${status}`, { timeout: 5000 }, async (t) => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-retry-"));
      const pending: http.ServerResponse[] = [];
      const server = http.createServer((req, res) => {
        if (req.url === "/fail") { res.writeHead(status); res.end(); }
        else { res.writeHead(200, { "Content-Length": 2 }); res.write("a"); pending.push(res); }
      });
      await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
      const mgr = new DownloadManager(2);
      t.after(async () => {
        mgr.list().forEach((j) => mgr.cancel(j.id));
        server.closeAllConnections();
        await new Promise<void>((resolve) => server.close(() => resolve()));
        fs.rmSync(dir, { recursive: true, force: true });
      });
      const base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
      const dest = path.join(dir, "retry.gguf");
      for (let i = 0; i < 3; i++) {
        await assert.rejects(mgr.enqueue({ label: "fail", dest, url: `${base}/fail` }).done);
      }
      const retry = mgr.enqueue({ label: "retry", dest, url: `${base}/ok` });
      const duplicate = mgr.enqueue({ label: "duplicate", dest: `${dir}${path.sep}.${path.sep}retry.gguf`, url: `${base}/ok` });
      assert.equal(duplicate.job.id, retry.job.id);
      assert.equal(duplicate.done, retry.done);
      await waitFor(() => pending.length === 1);
      pending[0]!.end("b");
      await retry.done;
      assert.equal(fs.readFileSync(dest, "utf8"), "ab");
    });
  }
});
