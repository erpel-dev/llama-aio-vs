import * as path from "path";
import { spawn, ChildProcess } from "child_process";
import { ensureDirs, whichOnPath } from "./paths";

export type LaunchMode = "externalTerminal" | "background";

export type ExternalLaunchPlan = {
  command: string;
  argv: string[];
  cwd: string;
};

function shQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

/** Ensure llama-server also writes the extension log (no shell tee on Windows). */
export function withLogFileArg(args: string[], logPath: string): string[] {
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--log-file" || args[i] === "-lf") {
      return args;
    }
  }
  return ["--log-file", logPath, ...args];
}

/**
 * argv for a visible Windows console running llama-server.exe directly.
 * Never PowerShell: a temp `.ps1` + `-ExecutionPolicy Bypass` looks like a dropper.
 */
export function buildWindowsExternalLaunch(options: {
  binary: string;
  args: string[];
  logPath: string;
  windowsTerminal?: string;
}): ExternalLaunchPlan {
  const { binary, args, logPath, windowsTerminal } = options;
  const cwd = path.dirname(binary);
  const launchArgs = withLogFileArg(args, logPath);
  if (windowsTerminal) {
    return {
      command: windowsTerminal,
      argv: ["-d", cwd, "--title", "Llama AIO - llama-server", binary, ...launchArgs],
      cwd,
    };
  }
  return {
    command: "cmd.exe",
    argv: ["/c", "start", "Llama AIO - llama-server", binary, ...launchArgs],
    cwd,
  };
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
        'Install one, set $TERMINAL, or set launchMode to "background" in ~/.llama-aio-vs/config.json.'
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
  const wt = whichOnPath("wt.exe") || whichOnPath("wt");
  const plan = buildWindowsExternalLaunch({
    binary,
    args,
    logPath,
    windowsTerminal: wt,
  });
  const child = spawn(plan.command, plan.argv, {
    detached: true,
    stdio: "ignore",
    windowsHide: false,
    cwd: plan.cwd,
    env: { ...process.env, ...env, PATH: `${plan.cwd};${env.PATH || process.env.PATH || ""}` },
    shell: false,
  });
  child.unref();
  return child;
}

/** Resolve launch mode from settings string. */
export function resolveLaunchMode(value: string | undefined): LaunchMode {
  return value === "background" ? "background" : "externalTerminal";
}
