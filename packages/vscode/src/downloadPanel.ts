import * as path from "path";
import * as vscode from "vscode";
import {
  downloadManager,
  formatBytes,
  type DownloadJobSnapshot,
  type DownloadManager,
  type SettingsStore,
} from "@llama-aio/core";

function formatSpeed(bytesPerSec: number): string {
  if (!bytesPerSec || bytesPerSec < 1) {
    return "";
  }
  return `${formatBytes(bytesPerSec)}/s`;
}

function formatEta(seconds?: number): string {
  if (!seconds || seconds < 0 || !Number.isFinite(seconds)) {
    return "";
  }
  if (seconds < 60) {
    return `${seconds}s left`;
  }
  if (seconds < 3600) {
    return `${Math.round(seconds / 60)} min left`;
  }
  return `${(seconds / 3600).toFixed(1)} h left`;
}

export class DownloadPanel {
  public static current: DownloadPanel | undefined;
  private panel: vscode.WebviewPanel | undefined;
  private readonly unsub: () => void;

  constructor(
    private readonly store: SettingsStore,
    private readonly manager: DownloadManager = downloadManager
  ) {
    this.unsub = this.manager.subscribe((jobs) => this.postJobs(jobs));
  }

  dispose(): void {
    this.unsub();
    this.panel?.dispose();
    if (DownloadPanel.current === this) {
      DownloadPanel.current = undefined;
    }
  }

  static show(store: SettingsStore, manager: DownloadManager = downloadManager): DownloadPanel {
    if (!DownloadPanel.current) {
      DownloadPanel.current = new DownloadPanel(store, manager);
    }
    DownloadPanel.current.reveal();
    return DownloadPanel.current;
  }

  reveal(): void {
    if (this.panel) {
      this.panel.reveal(vscode.ViewColumn.Beside);
      this.postJobs(this.manager.list());
      return;
    }
    this.panel = vscode.window.createWebviewPanel(
      "llamaAio.downloads",
      "Downloads",
      vscode.ViewColumn.Beside,
      { enableScripts: true, retainContextWhenHidden: true }
    );
    this.panel.webview.html = this.html();
    this.panel.webview.onDidReceiveMessage((msg) => void this.onMessage(msg));
    this.panel.onDidDispose(() => {
      this.panel = undefined;
    });
    this.postJobs(this.manager.list());
  }

  private postJobs(jobs: DownloadJobSnapshot[]): void {
    this.panel?.webview.postMessage({ type: "jobs", jobs });
  }

  private async onMessage(msg: { type?: string; id?: string; url?: string }): Promise<void> {
    const type = msg.type || "";
    const id = msg.id || "";
    if (type === "pause") {
      this.manager.pause(id);
    } else if (type === "resume") {
      this.manager.resume(id);
    } else if (type === "cancel") {
      this.manager.cancel(id);
    } else if (type === "clear") {
      this.manager.clearFinished();
    } else if (type === "openPage" && msg.url) {
      await vscode.env.openExternal(vscode.Uri.parse(msg.url));
    } else if (type === "setToken") {
      const token = await vscode.window.showInputBox({
        title: "Hugging Face access token",
        prompt: "Paste a token with access to gated repos (stored in Llama AIO config).",
        password: true,
        ignoreFocusOut: true,
      });
      if (token !== undefined) {
        await this.store.getConfig().update("hfToken", token.trim());
        vscode.window.setStatusBarMessage("Llama AIO: Hugging Face token saved", 4000);
      }
    } else if (type === "reveal" && id) {
      const job = this.manager.list().find((j) => j.id === id);
      if (job?.dest) {
        await vscode.commands.executeCommand("revealFileInOS", vscode.Uri.file(path.dirname(job.dest)));
      }
    }
  }

  private html(): string {
    return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8" />
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline';" />
<style>
  :root {
    --fg: var(--vscode-foreground);
    --muted: var(--vscode-descriptionForeground);
    --border: var(--vscode-widget-border, var(--vscode-panel-border));
    --card: var(--vscode-editorWidget-background);
    --bar: var(--vscode-progressBar-background, #3d9cd6);
    --bar-dim: #777;
    --err: var(--vscode-errorForeground);
    --ok: var(--vscode-testing-iconPassed, #3d9c5c);
  }
  body { font-family: var(--vscode-font-family); color: var(--fg); margin: 0; padding: 16px 18px 24px; }
  h1 { font-size: 13px; font-weight: 600; margin: 0 0 12px; letter-spacing: .02em; text-transform: uppercase; opacity: .85; }
  .empty { color: var(--muted); font-size: 13px; padding: 24px 4px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; }
  .row { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }
  .name { font-weight: 600; font-size: 13px; word-break: break-all; }
  .dim { color: var(--muted); font-size: 11px; }
  .gated { color: var(--err); font-size: 12px; font-weight: 600; }
  .progress { height: 6px; background: color-mix(in srgb, var(--fg) 12%, transparent); border-radius: 99px; overflow: hidden; margin: 8px 0; }
  .progress > span { display: block; height: 100%; background: var(--bar); border-radius: 99px; }
  .progress.paused > span { background: var(--bar-dim); }
  .actions { display: flex; gap: 6px; flex-wrap: wrap; }
  button { font: inherit; font-size: 11px; padding: 3px 9px; border-radius: 4px; border: 1px solid var(--border); background: transparent; color: var(--fg); cursor: pointer; }
  button.pri { background: var(--vscode-button-background); color: var(--vscode-button-foreground); border-color: transparent; }
  .toolbar { display: flex; justify-content: flex-end; margin-bottom: 10px; }
</style>
</head>
<body>
  <div class="toolbar"><button id="clear">Clear finished</button></div>
  <h1>Downloads</h1>
  <div id="list"><div class="empty">No downloads yet. Use Download from Hugging Face in the sidebar.</div></div>
  <script>
    const vscode = acquireVsCodeApi();
    const list = document.getElementById('list');
    document.getElementById('clear').addEventListener('click', () => vscode.postMessage({ type: 'clear' }));

    function fmtBytes(n) {
      n = Number(n) || 0;
      if (n >= 1073741824) return (n / 1073741824).toFixed(n >= 10737418240 ? 1 : 2) + ' GiB';
      if (n >= 1048576) return (n / 1048576).toFixed(1) + ' MiB';
      if (n >= 1024) return (n / 1024).toFixed(0) + ' KiB';
      return n + ' B';
    }
    function eta(s) {
      s = Number(s) || 0;
      if (s <= 0) return '';
      if (s < 60) return s + 's left';
      if (s < 3600) return Math.round(s / 60) + ' min left';
      return (s / 3600).toFixed(1) + ' h left';
    }
    function btn(label, type, id, extra, pri) {
      const b = document.createElement('button');
      b.textContent = label;
      if (pri) b.className = 'pri';
      b.addEventListener('click', () => vscode.postMessage({ type, id, ...extra }));
      return b;
    }
    function render(jobs) {
      list.innerHTML = '';
      if (!jobs.length) {
        list.innerHTML = '<div class="empty">No downloads yet. Use Download from Hugging Face in the sidebar.</div>';
        return;
      }
      for (const j of jobs) {
        const card = document.createElement('div');
        card.className = 'card';
        const top = document.createElement('div');
        top.className = 'row';
        const name = document.createElement('span');
        name.className = 'name';
        name.textContent = j.label;
        const right = document.createElement('span');
        if (j.state === 'gated') {
          right.className = 'gated';
          right.textContent = 'gated';
        } else if (j.state === 'paused') {
          right.className = 'dim';
          right.textContent = 'paused';
        } else if (j.state === 'done') {
          right.className = 'dim';
          right.textContent = 'done';
        } else if (j.state === 'error' || j.state === 'cancelled') {
          right.className = 'gated';
          right.textContent = j.state;
        } else {
          right.className = 'dim';
          right.textContent = j.total ? (fmtBytes(j.received) + ' / ' + fmtBytes(j.total)) : fmtBytes(j.received);
        }
        top.appendChild(name);
        top.appendChild(right);
        card.appendChild(top);

        if (j.state !== 'gated' && j.state !== 'error' && j.state !== 'cancelled') {
          const bar = document.createElement('div');
          bar.className = 'progress' + (j.state === 'paused' ? ' paused' : '');
          const fill = document.createElement('span');
          const pct = j.total > 0 ? Math.min(100, Math.round((j.received / j.total) * 100)) : (j.state === 'done' ? 100 : 8);
          fill.style.width = pct + '%';
          bar.appendChild(fill);
          card.appendChild(bar);
        }

        const bot = document.createElement('div');
        bot.className = 'row dim';
        const meta = document.createElement('span');
        const bits = [];
        if (j.state === 'running' || j.state === 'verifying') {
          if (j.bytesPerSec) bits.push(fmtBytes(j.bytesPerSec) + '/s');
          if (j.etaSeconds) bits.push(eta(j.etaSeconds));
          bits.push(j.state === 'verifying' ? 'sha256 verifying' : 'sha256 verifying on the fly');
        } else if (j.state === 'paused') {
          bits.push((j.total ? fmtBytes(j.received) + ' / ' + fmtBytes(j.total) + ' · ' : '') + 'resumable');
        } else if (j.state === 'gated') {
          bits.push(j.error || 'Accept the licence on huggingface.co and add a token.');
        } else if (j.error) {
          bits.push(j.error);
        } else if (j.state === 'done') {
          bits.push('saved · ' + (j.dest || ''));
        }
        meta.textContent = bits.join(' · ');
        const actions = document.createElement('span');
        actions.className = 'actions';
        if (j.state === 'running' || j.state === 'queued' || j.state === 'verifying') {
          actions.appendChild(btn('Pause', 'pause', j.id));
          actions.appendChild(btn('Cancel', 'cancel', j.id));
        } else if (j.state === 'paused' || j.state === 'error') {
          actions.appendChild(btn('Resume', 'resume', j.id, null, true));
          actions.appendChild(btn('Cancel', 'cancel', j.id));
        } else if (j.state === 'gated') {
          if (j.pageUrl) actions.appendChild(btn('Open page', 'openPage', j.id, { url: j.pageUrl }));
          actions.appendChild(btn('Set token', 'setToken', j.id));
        }
        bot.appendChild(meta);
        bot.appendChild(actions);
        card.appendChild(bot);
        list.appendChild(card);
      }
    }
    window.addEventListener('message', (e) => {
      if (e.data && e.data.type === 'jobs') render(e.data.jobs || []);
    });
  </script>
</body>
</html>`;
  }
}

export function formatDownloadStatusLine(jobs: DownloadJobSnapshot[]): string {
  const active = jobs.filter((j) => j.state === "running" || j.state === "queued" || j.state === "paused");
  if (!active.length) {
    return "";
  }
  const run = active.find((j) => j.state === "running") || active[0]!;
  const pct = run.total > 0 ? Math.round((run.received / run.total) * 100) : 0;
  const speed = formatSpeed(run.bytesPerSec);
  const left = formatEta(run.etaSeconds);
  return `$(cloud-download) ${run.label} ${pct}%${speed ? " · " + speed : ""}${left ? " · " + left : ""}`;
}
