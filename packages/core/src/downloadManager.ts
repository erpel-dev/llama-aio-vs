/**
 * In-process download job list (pause / resume / cancel) for the M5 panel.
 */
import { downloadHttpFile, DownloadAbortError, GatedDownloadError } from "./httpDownload";

export type DownloadJobState =
  | "queued"
  | "running"
  | "paused"
  | "verifying"
  | "done"
  | "error"
  | "gated"
  | "cancelled";

export interface DownloadJobSnapshot {
  id: string;
  label: string;
  dest: string;
  url: string;
  modelId?: string;
  received: number;
  total: number;
  bytesPerSec: number;
  etaSeconds?: number;
  state: DownloadJobState;
  error?: string;
  pageUrl?: string;
  resumable: boolean;
}

export interface EnqueueDownload {
  label: string;
  dest: string;
  url: string;
  token?: string;
  expectedSize?: number;
  expectedSha256?: string;
  gatedPageUrl?: string;
  modelId?: string;
  headers?: Record<string, string>;
}

type Listener = (jobs: DownloadJobSnapshot[]) => void;

interface InternalJob extends EnqueueDownload {
  id: string;
  received: number;
  total: number;
  bytesPerSec: number;
  state: DownloadJobState;
  error?: string;
  pageUrl?: string;
  controller?: AbortController;
  done: Promise<string>;
  resolve: (dest: string) => void;
  reject: (err: Error) => void;
}

export class DownloadManager {
  private readonly jobs = new Map<string, InternalJob>();
  private readonly listeners = new Set<Listener>();
  private seq = 0;
  private running = 0;
  private readonly concurrency: number;

  constructor(concurrency = 1) {
    this.concurrency = Math.max(1, concurrency);
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    listener(this.list());
    return () => {
      this.listeners.delete(listener);
    };
  }

  list(): DownloadJobSnapshot[] {
    return [...this.jobs.values()].map((j) => this.snapshot(j));
  }

  hasActive(): boolean {
    return [...this.jobs.values()].some((j) => j.state === "running" || j.state === "queued" || j.state === "paused");
  }

  enqueue(spec: EnqueueDownload): { job: DownloadJobSnapshot; done: Promise<string> } {
    const existing = [...this.jobs.values()].find((j) => j.dest === spec.dest && j.state !== "cancelled");
    if (existing && (existing.state === "done" || existing.state === "running" || existing.state === "queued")) {
      return { job: this.snapshot(existing), done: existing.done };
    }
    if (existing && existing.state === "paused") {
      this.resume(existing.id);
      return { job: this.snapshot(existing), done: existing.done };
    }

    let resolve!: (dest: string) => void;
    let reject!: (err: Error) => void;
    const done = new Promise<string>((res, rej) => {
      resolve = res;
      reject = rej;
    });
    // Avoid unhandled rejection if nobody awaits.
    done.catch(() => undefined);

    const id = `dl-${++this.seq}-${Date.now().toString(36)}`;
    const job: InternalJob = {
      ...spec,
      id,
      received: 0,
      total: spec.expectedSize || 0,
      bytesPerSec: 0,
      state: "queued",
      pageUrl: spec.gatedPageUrl,
      done,
      resolve,
      reject,
    };
    this.jobs.set(id, job);
    this.emit();
    void this.pump();
    return { job: this.snapshot(job), done };
  }

  pause(id: string): void {
    const job = this.jobs.get(id);
    if (!job || (job.state !== "running" && job.state !== "queued")) {
      return;
    }
    job.state = "paused";
    job.controller?.abort("pause");
    this.emit();
    void this.pump();
  }

  resume(id: string): void {
    const job = this.jobs.get(id);
    if (!job || job.state !== "paused") {
      return;
    }
    job.state = "queued";
    job.error = undefined;
    this.emit();
    void this.pump();
  }

  cancel(id: string): void {
    const job = this.jobs.get(id);
    if (!job || job.state === "done") {
      return;
    }
    job.state = "cancelled";
    job.controller?.abort("cancel");
    job.reject(new DownloadAbortError("cancel"));
    this.emit();
    void this.pump();
  }

  clearFinished(): void {
    for (const [id, job] of this.jobs) {
      if (job.state === "done" || job.state === "cancelled" || job.state === "error" || job.state === "gated") {
        this.jobs.delete(id);
      }
    }
    this.emit();
  }

  private snapshot(job: InternalJob): DownloadJobSnapshot {
    const remaining = job.total > job.received ? job.total - job.received : 0;
    const etaSeconds =
      job.bytesPerSec > 1024 && remaining > 0 ? Math.round(remaining / job.bytesPerSec) : undefined;
    return {
      id: job.id,
      label: job.label,
      dest: job.dest,
      url: job.url,
      modelId: job.modelId,
      received: job.received,
      total: job.total,
      bytesPerSec: job.bytesPerSec,
      etaSeconds,
      state: job.state,
      error: job.error,
      pageUrl: job.pageUrl,
      resumable: job.state === "paused" || job.state === "error",
    };
  }

  private emit(): void {
    const snap = this.list();
    for (const listener of this.listeners) {
      try {
        listener(snap);
      } catch {
        // UI listeners must not break the queue
      }
    }
  }

  private async pump(): Promise<void> {
    while (this.running < this.concurrency) {
      const next = [...this.jobs.values()].find((j) => j.state === "queued");
      if (!next) {
        return;
      }
      this.running += 1;
      next.state = "running";
      next.controller = new AbortController();
      this.emit();
      try {
        const dest = await downloadHttpFile({
          url: next.url,
          dest: next.dest,
          token: next.token,
          expectedSize: next.expectedSize,
          expectedSha256: next.expectedSha256,
          gatedPageUrl: next.gatedPageUrl,
          headers: next.headers,
          signal: next.controller.signal,
          onProgress: (info) => {
            next.received = info.received;
            next.total = info.total || next.total;
            next.bytesPerSec = info.bytesPerSec;
            if (info.received >= (info.total || 0) && info.total > 0 && next.expectedSha256) {
              next.state = "verifying";
            }
            this.emit();
          },
        });
        next.state = "done";
        next.received = next.total || next.received;
        next.resolve(dest);
        this.emit();
      } catch (e) {
        if (e instanceof DownloadAbortError && e.reason === "pause") {
          next.state = "paused";
          this.emit();
        } else if (e instanceof DownloadAbortError) {
          next.state = "cancelled";
          this.emit();
        } else if (e instanceof GatedDownloadError) {
          next.state = "gated";
          next.error = e.message;
          next.pageUrl = e.pageUrl;
          next.reject(e);
          this.emit();
        } else {
          next.state = "error";
          next.error = e instanceof Error ? e.message : String(e);
          next.reject(e instanceof Error ? e : new Error(String(e)));
          this.emit();
        }
      } finally {
        this.running -= 1;
        next.controller = undefined;
      }
    }
    if ([...this.jobs.values()].some((j) => j.state === "queued")) {
      void this.pump();
    }
  }
}

export const downloadManager = new DownloadManager(1);
