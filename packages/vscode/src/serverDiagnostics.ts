import * as fs from "fs";
import * as vscode from "vscode";
import { getLogPath, LaunchCancelledError, ProcessManager, TerminalLaunchError } from "@llama-aio/core";

let channel: vscode.OutputChannel | undefined;
let processManager: ProcessManager | undefined;

export function initServerDiagnostics(pm: ProcessManager): vscode.Disposable {
  processManager = pm;
  channel = vscode.window.createOutputChannel("Llama AIO");
  return channel;
}

function output(): vscode.OutputChannel {
  if (!channel) {
    channel = vscode.window.createOutputChannel("Llama AIO");
  }
  return channel;
}

export function showServerOutput(): void {
  output().show(true);
}

/** Open ~/.llama-aio-vs/runtime/llama-server.log in the editor. */
export async function openServerLog(): Promise<void> {
  const logPath = getLogPath();
  if (!fs.existsSync(logPath)) {
    void vscode.window.showWarningMessage(
      `No server log yet (${logPath}). Start the server once to create it.`
    );
    return;
  }
  const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(logPath));
  await vscode.window.showTextDocument(doc, { preview: true });
}

/** Copy the argv the next start would use. */
export async function copyServerCommandLine(): Promise<void> {
  if (!processManager) {
    void vscode.window.showErrorMessage("Llama AIO is not ready to build a command line yet.");
    return;
  }
  try {
    const line = processManager.describeCommandLine();
    await vscode.env.clipboard.writeText(line);
    output().appendLine(`# command line\n${line}`);
    void vscode.window.showInformationMessage("Copied the llama-server command line.");
  } catch (e) {
    void vscode.window.showErrorMessage(
      `Could not copy the command line: ${e instanceof Error ? e.message : String(e)}`
    );
  }
}

/**
 * Write a failed start/reload to the Llama AIO output channel, including the
 * last 20 log lines, and show an error toast that can open that channel.
 * The channel is not forced into view (an auto-start failure at window open
 * should not steal focus). A start cancelled by Stop is not an error.
 */
/**
 * `retryInBackground` switches the launch mode to Background and starts again;
 * it is offered when an external terminal never ran llama-server. The prompt is
 * not awaited there, so the failed launch releases its claim before the retry.
 */
export async function reportLaunchFailure(
  prefix: string,
  error: unknown,
  retryInBackground?: () => Promise<unknown>
): Promise<void> {
  if (error instanceof LaunchCancelledError) {
    vscode.window.setStatusBarMessage("Llama AIO: server start cancelled", 4000);
    return;
  }
  const message = error instanceof Error ? error.message : String(error);
  const out = output();
  out.appendLine("");
  out.appendLine(`# ${new Date().toISOString()} ${prefix}`);
  out.appendLine(message);
  const recent = processManager?.readRecentLogLines(20) || "";
  if (recent && !message.includes(recent)) {
    out.appendLine("");
    out.appendLine("# last 20 log lines");
    out.appendLine(recent);
  }
  if (error instanceof TerminalLaunchError && retryInBackground) {
    void vscode.window.showErrorMessage(`${prefix}: ${message}`, "Start in Background mode", "Show output").then(async (pick) => {
      if (pick === "Start in Background mode") {
        await retryInBackground();
      } else if (pick === "Show output") {
        out.show(true);
      }
    });
    return;
  }
  const choice = await vscode.window.showErrorMessage(`${prefix}: ${message}`, "Show output");
  if (choice === "Show output") {
    out.show(true);
  }
}
