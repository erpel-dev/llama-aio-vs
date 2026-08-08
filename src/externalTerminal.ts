import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { spawn, ChildProcess } from "child_process";
import { ensureDirs, whichOnPath } from "./paths";

export type LaunchMode = "externalTerminal" | "background";

function shQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

function powershellSingleQuote(value: string): string {
  return `'${value.replace(/'/g, "''")}'`;
}

function resolveLinuxTerminal(): { bin: string; argsPrefix: string[] } | undefined {
  const configured = (process.env.TERMINAL || "").trim();
  const candidates = [
    configured,
    "gnome-terminal",
    "konsole",
    "xfce4-terminal",
    "mate-terminal",
    "tilix",
    "kitty",
    "alacritty",
    "xterm",
    "x-terminal-emulator",
  ].filter(Boolean);

  for (const name of candidates) {
    const bin = whichOnPath(name);
    if (!bin) {
      continue;
    }
    const base = path.basename(name);
    if (base.includes("gnome-terminal") || base.includes("mate-terminal")) {
      return { bin, argsPrefix: ["--title=Llama AIO · llama-server", "--"] };
    }
    if (base.includes("xfce4-terminal") || base.includes("tilix")) {
      return { bin, argsPrefix: ["--title=Llama AIO · llama-server", "-e"] };
    }
    if (base === "konsole") {
      return { bin, argsPrefix: ["--title", "Llama AIO · llama-server", "-e"] };
    }
    if (base === "kitty" || base === "alacritty") {
      return { bin, argsPrefix: ["-e"] };
    }
    // xterm / x-terminal-emulator
    return { bin, argsPrefix: ["-T", "Llama AIO · llama-server", "-e"] };
  }
  return undefined;
}

/**
 * Launch llama-server in a visible OS terminal window.
 * Closing that window kills the server process tree.
 * Returns the launcher PID (terminal/shell), not necessarily llama-server.
 */
export function spawnInExternalTerminal(options: {
  /** Path shown in the terminal banner / used for LD_LIBRARY_PATH. */
  binary: string;
  args: string[];
  env: NodeJS.ProcessEnv;
  logPath: string;
  /**
   * Optional launcher (e.g. steam-run on NixOS). When set, the shell runs
   * `command …prefixArgs …args` instead of invoking `binary` directly.
   * Callers should put `binary` inside `prefixArgs` when using a wrapper.
   */
  command?: string;
  prefixArgs?: string[];
}): ChildProcess {
  const { binary, args, env, logPath, command, prefixArgs } = options;
  ensureDirs(path.dirname(logPath));

  if (process.platform === "win32") {
    return spawnWindows(binary, args, env, logPath);
  }
  if (process.platform === "darwin") {
    return spawnMac(binary, args, env, logPath);
  }
  return spawnLinux(binary, args, env, logPath, command, prefixArgs);
}

function spawnLinux(
  binary: string,
  args: string[],
  env: NodeJS.ProcessEnv,
  logPath: string,
  command?: string,
  prefixArgs?: string[]
): ChildProcess {
  const term = resolveLinuxTerminal();
  if (!term) {
    throw new Error(
      "No external terminal found (tried gnome-terminal, konsole, kitty, xterm, …). " +
        "Install one, set $TERMINAL, or set llamaAio.launchMode to \"background\"."
    );
  }

  const libDir = path.dirname(binary);
  const launchArgv = command
    ? [command, ...(prefixArgs || []), ...args]
    : [binary, ...args];
  const launchLine = launchArgv.map(shQuote).join(" ");
  const cmd = [
    `echo "Llama AIO · llama-server"`,
    `echo "Binary: ${shQuote(binary)}"`,
    command ? `echo "Launcher: ${shQuote(command)}"` : `true`,
    `echo "Log also mirrored to: ${shQuote(logPath)}"`,
    `echo`,
    `export LD_LIBRARY_PATH=${shQuote(libDir)}\${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}`,
    `${launchLine} 2>&1 | tee -a ${shQuote(logPath)}`,
    `code=$?`,
    `echo`,
    `echo "llama-server exited with code $code"`,
    `echo "Press Enter to close this window…"`,
    `read -r _ || true`,
    `exit $code`,
  ].join("; ");

  const child = spawn(term.bin, [...term.argsPrefix, "bash", "-lc", cmd], {
    detached: true,
    stdio: "ignore",
    env: { ...process.env, ...env },
  });
  child.unref();
  return child;
}

function spawnMac(
  binary: string,
  args: string[],
  env: NodeJS.ProcessEnv,
  logPath: string
): ChildProcess {
  const libDir = path.dirname(binary);
  const script = [
    `echo "Llama AIO · llama-server"`,
    `export DYLD_LIBRARY_PATH=${shQuote(libDir)}\${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}`,
    `${shQuote(binary)} ${args.map(shQuote).join(" ")} 2>&1 | tee -a ${shQuote(logPath)}`,
    `code=$?`,
    `echo`,
    `echo "llama-server exited with code $code"`,
    `echo "Press Enter to close…"`,
    `read -r _ || true`,
  ].join("; ");

  const child = spawn(
    "osascript",
    ["-e", `tell application "Terminal" to do script ${shQuote(script)}`, "-e", 'tell application "Terminal" to activate'],
    {
      detached: true,
      stdio: "ignore",
      env: { ...process.env, ...env },
    }
  );
  child.unref();
  return child;
}

function spawnWindows(
  binary: string,
  args: string[],
  env: NodeJS.ProcessEnv,
  logPath: string
): ChildProcess {
  const binDir = path.dirname(binary);
  // PowerShell script: live console + Tee-Object into the extension log (needed for
  // fatal-line detection), PATH for sibling DLLs, self-delete when done.
  const ps1Path = path.join(os.tmpdir(), `llama-aio-vs-${Date.now()}.ps1`);
  const q = powershellSingleQuote;
  const script = [
    "$ErrorActionPreference = 'Continue'",
    `$env:PATH = ${q(binDir + ";")} + $env:PATH`,
    "Write-Host 'Llama AIO · llama-server'",
    `Write-Host ('Binary: ' + ${q(binary)})`,
    `Write-Host ('Log also mirrored to: ' + ${q(logPath)})`,
    "Write-Host ''",
    `Set-Location -LiteralPath ${q(binDir)}`,
    `& ${q(binary)} ${args.map(q).join(" ")} 2>&1 | Tee-Object -FilePath ${q(logPath)} -Append`,
    "$code = $LASTEXITCODE",
    "Write-Host ''",
    "Write-Host (\"llama-server exited with code $code\")",
    "Write-Host 'Press Enter to close this window…'",
    "try { Read-Host | Out-Null } catch {}",
    "try { Remove-Item -LiteralPath $PSCommandPath -Force -ErrorAction SilentlyContinue } catch {}",
    "exit $code",
    "",
  ].join("\r\n");
  fs.writeFileSync(ps1Path, script, "utf8");

  const psArgs = [
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    ps1Path,
  ];

  // Prefer Windows Terminal if available; start in the binary dir for DLL lookup.
  const wt = whichOnPath("wt.exe") || whichOnPath("wt");
  if (wt) {
    const child = spawn(wt, ["-d", binDir, "powershell.exe", ...psArgs], {
      detached: true,
      stdio: "ignore",
      windowsHide: false,
      cwd: binDir,
      env: { ...process.env, ...env },
    });
    child.unref();
    return child;
  }

  // No `/D binDir`: the spawn cwd below already puts cmd (and therefore the
  // started process) in the binary directory, and dropping the flag avoids
  // `start` mis-parsing an install path that contains spaces.
  const child = spawn(
    "cmd.exe",
    ["/c", "start", "Llama AIO - llama-server", "powershell.exe", ...psArgs],
    {
      detached: true,
      stdio: "ignore",
      windowsHide: false,
      cwd: binDir,
      env: { ...process.env, ...env },
      shell: false,
    }
  );
  child.unref();
  return child;
}

/** Resolve launch mode from settings string. */
export function resolveLaunchMode(value: string | undefined): LaunchMode {
  return value === "background" ? "background" : "externalTerminal";
}
