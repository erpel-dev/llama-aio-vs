/**
 * Resumable HTTPS download with idle timeout, size check, and optional sha256.
 */
import { createHash, type Hash } from "crypto";
import * as fs from "fs";
import * as http from "http";
import * as https from "https";
import * as path from "path";
import { ensureDirs } from "./paths";

export class DownloadAbortError extends Error {
  constructor(public readonly reason: "cancel" | "pause") {
    super(reason === "pause" ? "Download paused" : "Download cancelled");
    this.name = "DownloadAbortError";
  }
}

export class GatedDownloadError extends Error {
  constructor(
    public readonly pageUrl: string,
    public readonly status: number
  ) {
    super(
      `HTTP ${status}: this repository is gated. Accept the licence on the model page and add a Hugging Face token.`
    );
    this.name = "GatedDownloadError";
  }
}

export class DownloadVerifyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DownloadVerifyError";
  }
}

export interface HttpDownloadProgress {
  received: number;
  total: number;
  bytesPerSec: number;
}

export interface HttpDownloadOptions {
  url: string;
  dest: string;
  token?: string;
  expectedSize?: number;
  /** Hex sha256 (with or without a `sha256:` prefix). */
  expectedSha256?: string;
  headers?: Record<string, string>;
  /** Model page to open when HF returns 401/403. */
  gatedPageUrl?: string;
  idleTimeoutMs?: number;
  signal?: AbortSignal;
  onProgress?: (info: HttpDownloadProgress) => void;
}

const DEFAULT_IDLE_MS = 45_000;
const MAX_REDIRECTS = 8;

export function normalizeSha256(value: string | undefined): string | undefined {
  if (!value) {
    return undefined;
  }
  const hex = value.replace(/^sha256:/i, "").trim().toLowerCase();
  return /^[0-9a-f]{64}$/.test(hex) ? hex : undefined;
}

export function partialDownloadPath(dest: string): string {
  return dest.endsWith(".partial") ? dest : `${dest}.partial`;
}

function request(
  url: string,
  options: https.RequestOptions,
  cb: (res: http.IncomingMessage) => void
): http.ClientRequest {
  return url.startsWith("http://") ? http.get(url, options, cb) : https.get(url, options, cb);
}

function fileSha256(filePath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const hash = createHash("sha256");
    const stream = fs.createReadStream(filePath);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", () => resolve(hash.digest("hex")));
  });
}

async function existingIsComplete(
  dest: string,
  expectedSize?: number,
  expectedSha256?: string
): Promise<boolean> {
  if (!fs.existsSync(dest)) {
    return false;
  }
  const st = fs.statSync(dest);
  if (!st.isFile() || st.size <= 0) {
    return false;
  }
  if (expectedSize && st.size !== expectedSize) {
    return false;
  }
  if (expectedSha256) {
    const got = await fileSha256(dest);
    return got === expectedSha256;
  }
  return !!expectedSize || st.size > 0;
}

/**
 * Download `url` to `dest`. Writes `dest.partial` until size (and sha256, when
 * known) check out, then renames. A dropped connection leaves the partial so
 * the next call can send `Range`.
 */
export async function downloadHttpFile(options: HttpDownloadOptions): Promise<string> {
  const dest = options.dest;
  const expectedSha256 = normalizeSha256(options.expectedSha256);
  const expectedSize =
    options.expectedSize && options.expectedSize > 0 ? options.expectedSize : undefined;

  if (await existingIsComplete(dest, expectedSize, expectedSha256)) {
    return dest;
  }
  if (fs.existsSync(dest) && expectedSize) {
    try {
      fs.unlinkSync(dest);
    } catch {
      // replace via rename later
    }
  }

  const partial = partialDownloadPath(dest);
  ensureDirs(path.dirname(partial));

  let existing = 0;
  try {
    const st = fs.statSync(partial);
    if (st.isFile()) {
      existing = st.size;
    }
  } catch {
    existing = 0;
  }

  if (expectedSize && existing > expectedSize) {
    fs.unlinkSync(partial);
    existing = 0;
  }

  const signal = options.signal;
  if (signal?.aborted) {
    throw new DownloadAbortError("cancel");
  }

  await new Promise<void>((resolve, reject) => {
    let settled = false;
    let idle: NodeJS.Timeout | undefined;
    let req: http.ClientRequest | undefined;
    let out: fs.WriteStream | undefined;
    const idleMs = options.idleTimeoutMs ?? DEFAULT_IDLE_MS;
    const startedAt = Date.now();
    let received = existing;
    let lastTick = startedAt;
    let lastBytes = existing;
    let hash: Hash | undefined;
    let hashReady = Promise.resolve();

    const fail = (err: Error) => {
      if (settled) {
        return;
      }
      settled = true;
      if (idle) {
        clearTimeout(idle);
      }
      signal?.removeEventListener("abort", onAbort);
      req?.destroy();
      if (out) {
        out.destroy();
      }
      reject(err);
    };

    const ok = () => {
      if (settled) {
        return;
      }
      settled = true;
      if (idle) {
        clearTimeout(idle);
      }
      signal?.removeEventListener("abort", onAbort);
      resolve();
    };

    const onAbort = () => {
      const reason = (signal as AbortSignal & { reason?: string }).reason;
      fail(new DownloadAbortError(reason === "pause" ? "pause" : "cancel"));
    };
    signal?.addEventListener("abort", onAbort);

    const bumpIdle = () => {
      if (idle) {
        clearTimeout(idle);
      }
      idle = setTimeout(() => fail(new Error("Download stalled (idle timeout)")), idleMs);
    };

    const armHash = async () => {
      if (!expectedSha256) {
        return;
      }
      hash = createHash("sha256");
      if (existing > 0) {
        await new Promise<void>((res, rej) => {
          const stream = fs.createReadStream(partial);
          stream.on("data", (chunk) => hash!.update(chunk));
          stream.on("error", rej);
          stream.on("end", () => res());
        });
      }
    };

    const go = (url: string, hops: number) => {
      if (hops > MAX_REDIRECTS) {
        fail(new Error("Too many redirects"));
        return;
      }
      const headers: Record<string, string> = {
        "User-Agent": "llama-aio-vs",
        Accept: "*/*",
        ...(options.headers || {}),
      };
      if (options.token) {
        headers.Authorization = `Bearer ${options.token}`;
      }
      if (existing > 0) {
        headers.Range = `bytes=${existing}-`;
      }
      req = request(url, { headers }, (res) => {
        const status = res.statusCode || 0;
        if (status >= 300 && status < 400 && res.headers.location) {
          res.resume();
          go(res.headers.location, hops + 1);
          return;
        }
        if (status === 401 || status === 403) {
          res.resume();
          fail(new GatedDownloadError(options.gatedPageUrl || url, status));
          return;
        }
        if (status === 416 && existing > 0) {
          res.resume();
          try {
            fs.unlinkSync(partial);
          } catch {
            // ignore
          }
          existing = 0;
          received = 0;
          go(options.url, hops + 1);
          return;
        }
        if (status < 200 || status >= 300) {
          res.resume();
          fail(new Error(`Download failed: HTTP ${status}`));
          return;
        }

        const restart = status === 200 && existing > 0;
        if (restart) {
          try {
            fs.unlinkSync(partial);
          } catch {
            // ignore
          }
          existing = 0;
          received = 0;
        }

        const length = Number(res.headers["content-length"] || 0);
        const total =
          status === 206 && length > 0
            ? existing + length
            : length > 0
              ? length
              : expectedSize || 0;

        hashReady
          .then(() => armHash())
          .then(() => {
            out = fs.createWriteStream(partial, { flags: existing > 0 ? "a" : "w" });
            bumpIdle();
            res.on("error", fail);
            out.on("error", fail);
            res.on("data", (chunk: Buffer) => {
              received += chunk.length;
              hash?.update(chunk);
              bumpIdle();
              const now = Date.now();
              const dt = Math.max(1, now - lastTick);
              if (now - lastTick >= 250) {
                const bytesPerSec = ((received - lastBytes) * 1000) / dt;
                lastTick = now;
                lastBytes = received;
                options.onProgress?.({ received, total, bytesPerSec });
              }
            });
            res.pipe(out);
            out.on("finish", () => {
              out?.close((closeErr) => {
                if (closeErr) {
                  fail(closeErr);
                  return;
                }
                if (expectedSha256 && hash) {
                  const got = hash.digest("hex");
                  if (got !== expectedSha256) {
                    fail(new DownloadVerifyError(`sha256 mismatch (got ${got.slice(0, 12)}…)`));
                    return;
                  }
                }
                ok();
              });
            });
          })
          .catch(fail);
      });
      req.on("error", fail);
    };

    hashReady = Promise.resolve();
    go(options.url, 0);
  });

  const st = fs.statSync(partial);
  if (expectedSize && st.size !== expectedSize) {
    throw new DownloadVerifyError(
      `Downloaded size ${st.size} did not match expected ${expectedSize}`
    );
  }
  fs.renameSync(partial, dest);
  return dest;
}
