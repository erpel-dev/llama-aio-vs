/**
 * Loopback gateway for the managed llama-server's own web UI.
 *
 * The bundled llama.cpp UI keeps its API key only in that browser's settings
 * and has no supported way to receive one, so opening it directly ends at an
 * "Enter API Key" screen. This gateway serves the same UI on 127.0.0.1 and adds
 * the managed key to every request, so the page works without pasting a key.
 *
 * Threat model: same-machine access is acceptable (the server already runs as
 * this user); the key exists to keep *other* parties out. So the gateway:
 * - listens on loopback only;
 * - refuses a Host that is not loopback (DNS rebinding: a web page that
 *   resolves its own name to 127.0.0.1) unless the editor forwards ports;
 * - refuses cross-site requests (Origin of another site, Sec-Fetch-Site:
 *   cross-site), so web pages open in the browser cannot use or read it;
 * - strips CORS headers from responses and never logs or echoes the key.
 */
import * as http from "http";

export interface GatewayRequestInfo {
  host?: string;
  origin?: string;
  secFetchSite?: string;
}

const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]", "::1"]);

function hostnameOf(hostHeader: string): string {
  const h = hostHeader.trim().toLowerCase();
  if (h.startsWith("[")) {
    const end = h.indexOf("]");
    return end > 0 ? h.slice(0, end + 1) : h;
  }
  const colon = h.lastIndexOf(":");
  return colon > 0 ? h.slice(0, colon) : h;
}

/**
 * Why a request must not pass, or undefined to forward it.
 * `strictHost` is false only when the editor itself forwards the port (a
 * browser-hosted or remote VS Code), where the Host is the editor's address.
 */
export function gatewayRefusal(req: GatewayRequestInfo, strictHost: boolean): string | undefined {
  const host = (req.host || "").trim();
  if (!host) {
    return "missing Host header";
  }
  if (strictHost && !LOOPBACK_HOSTS.has(hostnameOf(host))) {
    return "Host is not this machine";
  }
  const site = (req.secFetchSite || "").toLowerCase();
  if (site === "cross-site") {
    return "cross-site request";
  }
  const origin = (req.origin || "").trim();
  if (origin && origin !== "null") {
    let originHost = "";
    try {
      originHost = new URL(origin).host.toLowerCase();
    } catch {
      return "malformed Origin";
    }
    if (originHost !== host.toLowerCase()) {
      return "request from another site";
    }
  } else if (origin === "null") {
    return "opaque Origin";
  }
  return undefined;
}

const HOP_BY_HOP = new Set([
  "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
  "transfer-encoding", "upgrade", "host", "authorization", "origin", "referer", "cookie",
]);

export interface WebUiGatewayOptions {
  /** Managed llama-server base URL, e.g. http://127.0.0.1:8742 */
  upstream: () => string;
  /** Current managed key (read fresh so a rotated key just works). */
  apiKey: () => string | undefined;
  strictHost: () => boolean;
}

export class WebUiGateway {
  private server?: http.Server;
  private port = 0;

  constructor(private readonly options: WebUiGatewayOptions) {}

  get running(): boolean {
    return !!this.server?.listening;
  }

  /** Loopback URL of the gateway (only valid after `start`). */
  url(): string {
    return `http://127.0.0.1:${this.port}/`;
  }

  /**
   * Listen on 127.0.0.1. `preferredPort` keeps the page's origin, and with it
   * the web UI's saved conversations, stable across restarts; a busy port
   * falls back to any free one.
   */
  async start(preferredPort = 0): Promise<number> {
    if (this.server?.listening) {
      return this.port;
    }
    const server = http.createServer((req, res) => this.handle(req, res));
    const listen = (port: number) =>
      new Promise<number>((resolve, reject) => {
        const onError = (err: Error) => {
          server.off("listening", onListening);
          reject(err);
        };
        const onListening = () => {
          server.off("error", onError);
          resolve((server.address() as { port: number }).port);
        };
        server.once("error", onError);
        server.once("listening", onListening);
        server.listen(port, "127.0.0.1");
      });
    try {
      this.port = await listen(preferredPort);
    } catch (err) {
      if (!preferredPort || (err as NodeJS.ErrnoException).code !== "EADDRINUSE") throw err;
      this.port = await listen(0);
    }
    this.server = server;
    return this.port;
  }

  close(): void {
    this.server?.close();
    this.server = undefined;
  }

  private handle(req: http.IncomingMessage, res: http.ServerResponse): void {
    const refusal = gatewayRefusal(
      {
        host: req.headers.host,
        origin: typeof req.headers.origin === "string" ? req.headers.origin : undefined,
        secFetchSite: typeof req.headers["sec-fetch-site"] === "string" ? req.headers["sec-fetch-site"] : undefined,
      },
      this.options.strictHost()
    );
    if (refusal) {
      res.writeHead(403, { "content-type": "text/plain; charset=utf-8" });
      res.end(`Llama AIO web UI gateway: refused (${refusal}).`);
      return;
    }
    let target: URL;
    try {
      target = new URL(req.url || "/", this.options.upstream());
    } catch {
      res.writeHead(502, { "content-type": "text/plain; charset=utf-8" });
      res.end("Llama AIO web UI gateway: no local server address.");
      return;
    }
    const headers: http.OutgoingHttpHeaders = {};
    for (const [name, value] of Object.entries(req.headers)) {
      if (value !== undefined && !HOP_BY_HOP.has(name.toLowerCase()) && !name.toLowerCase().startsWith("sec-fetch-")) {
        headers[name] = value;
      }
    }
    const key = this.options.apiKey();
    if (key) {
      headers.authorization = `Bearer ${key}`;
    }
    const upstream = http.request(
      {
        protocol: target.protocol,
        hostname: target.hostname,
        port: target.port,
        method: req.method,
        path: target.pathname + target.search,
        headers,
      },
      (up) => {
        const out: http.OutgoingHttpHeaders = {};
        for (const [name, value] of Object.entries(up.headers)) {
          const n = name.toLowerCase();
          // No CORS grants: only the gateway's own page may read responses.
          if (value !== undefined && !n.startsWith("access-control-") && n !== "connection" && n !== "keep-alive") {
            out[name] = value;
          }
        }
        res.writeHead(up.statusCode || 502, out);
        up.pipe(res);
      }
    );
    upstream.on("error", () => {
      if (!res.headersSent) {
        res.writeHead(502, { "content-type": "text/plain; charset=utf-8" });
      }
      res.end("Llama AIO web UI gateway: the local llama-server is not reachable. Start it from the sidebar.");
    });
    // Stop streaming generation upstream when the page goes away.
    res.on("close", () => upstream.destroy());
    req.pipe(upstream);
  }
}
