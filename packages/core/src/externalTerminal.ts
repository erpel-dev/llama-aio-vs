import * as fs from "fs";
import * as path from "path";
import { spawn, spawnSync, ChildProcess } from "child_process";
import { ensureDirs, whichOnPath } from "./paths";

export type LaunchMode = "externalTerminal" | "background";

export type ExternalLaunchPlan = {
  command: string;
  argv: string[];
  cwd: string;
};

/** How to spawn a Linux terminal (possibly via Flatpak's host portal). */
export type LinuxTerminalLauncher = {
  command: string;
  /** Args before `bash -lc <script>`. */
  prefix: string[];
  terminal: string;
  viaFlatpakHost: boolean;
};

export type LinuxTerminalResolveHooks = {
  env?: NodeJS.ProcessEnv;
  lookup?: (command: string) => string | undefined;
  hostLookup?: (command: string) => string | undefined;
  isFlatpak?: boolean;
  flatpakSpawn?: string;
};

const LINUX_TERMINAL_CANDIDATES = [
  "gnome-terminal",
  "kgx",
  "ptyxis",
  "konsole",
  "xfce4-terminal",
  "mate-terminal",
  "tilix",
  "kitty",
  "alacritty",
  "wezterm",
  "ghostty",
  "foot",
  "qterminal",
  "xterm",
  "x-terminal-emulator",
];

const HOST_BIN_DIRS = ["/usr/bin", "/usr/local/bin", "/bin"];

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

export function isFlatpakSandbox(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.FLATPAK_ID) || fs.existsSync("/.flatpak-info");
}

export function linuxTerminalArgsPrefix(binPath: string): string[] {
  const base = path.basename(binPath);
  if (
    base.includes("gnome-terminal") ||
    base.includes("mate-terminal") ||
    base === "kgx" ||
    base === "ptyxis"
  ) {
    return ["--title=Llama AIO · llama-server", "--"];
  }
  if (base.includes("xfce4-terminal") || base.includes("tilix") || base === "qterminal") {
    return ["--title=Llama AIO · llama-server", "-e"];
  }
  if (base === "konsole") {
    return ["--title", "Llama AIO · llama-server", "-e"];
  }
  if (base === "wezterm") {
    return ["start", "--"];
  }
  if (base === "kitty" || base === "alacritty" || base === "ghostty") {
    return ["-e"];
  }
  if (base === "foot") {
    return [];
  }
  return ["-T", "Llama AIO · llama-server", "-e"];
}

function isExecutableFile(filePath: string): boolean {
  try {
    return fs.existsSync(filePath) && fs.statSync(filePath).isFile();
  } catch {
    return false;
  }
}

function lookupSandboxTerminal(name: string, lookup: (command: string) => string | undefined): string | undefined {
  const fromPath = lookup(name);
  if (fromPath) {
    return fromPath;
  }
  if (path.isAbsolute(name) && isExecutableFile(name)) {
    return name;
  }
  if (!name.includes("/") && !name.includes("\\")) {
    for (const dir of HOST_BIN_DIRS) {
      const full = path.join(dir, name);
      if (isExecutableFile(full)) {
        return full;
      }
    }
  }
  return undefined;
}

function findFlatpakSpawn(lookup: (command: string) => string | undefined): string | undefined {
  return lookup("flatpak-spawn") || (isExecutableFile("/usr/bin/flatpak-spawn") ? "/usr/bin/flatpak-spawn" : undefined);
}

function hostWhich(flatpakSpawn: string, name: string): string | undefined {
  try {
    const result = spawnSync(flatpakSpawn, ["--host", "sh", "-c", `command -v ${shQuote(name)}`], {
      encoding: "utf8",
      timeout: 4000,
    });
    const found = (result.stdout || "").trim().split(/\n/)[0];
    if (result.status === 0 && found) {
      return found;
    }
  } catch {
    // Portal missing, or flatpak-spawn cannot reach the host.
  }
  return undefined;
}

export function resolveLinuxTerminalLauncher(
  hooks: LinuxTerminalResolveHooks = {}
): LinuxTerminalLauncher | undefined {
  const env = hooks.env || process.env;
  const lookup = hooks.lookup || whichOnPath;
  const flatpak = hooks.isFlatpak ?? isFlatpakSandbox(env);
  const configured = (env.TERMINAL || "").trim();
  const candidates = [configured, ...LINUX_TERMINAL_CANDIDATES].filter(Boolean);

  if (flatpak) {
    const flatpakSpawn = hooks.flatpakSpawn || findFlatpakSpawn(lookup);
    const hostLookup = hooks.hostLookup || (flatpakSpawn ? (name: string) => hostWhich(flatpakSpawn, name) : undefined);
    if (flatpakSpawn && hostLookup) {
      for (const name of candidates) {
        const terminal = hostLookup(name);
        if (!terminal) {
          continue;
        }
        return {
          command: flatpakSpawn,
          prefix: ["--host", terminal, ...linuxTerminalArgsPrefix(terminal)],
          terminal,
          viaFlatpakHost: true,
        };
      }
    }
    return undefined;
  }

  for (const name of candidates) {
    const bin = lookupSandboxTerminal(name, lookup);
    if (!bin) {
      continue;
    }
    return {
      command: bin,
      prefix: linuxTerminalArgsPrefix(bin),
      terminal: bin,
      viaFlatpakHost: false,
    };
  }
  return undefined;
}

export function missingLinuxTerminalMessage(opts?: { flatpak?: boolean; flatpakId?: string }): string {
  const flatpak = opts?.flatpak ?? isFlatpakSandbox();
  const id = opts?.flatpakId || process.env.FLATPAK_ID || "com.visualstudio.code";
  if (flatpak) {
    return (
      "No host terminal found from the Flatpak sandbox (tried konsole, gnome-terminal, kitty, …). " +
      `Allow host commands (\`flatpak override --user --talk-name=org.freedesktop.Flatpak ${id}\`), ` +
      "set $TERMINAL to a host terminal name, or set launchMode to \"background\" in ~/.llama-aio-vs/config.json."
    );
  }
  return (
    "No external terminal found (tried gnome-terminal, konsole, kitty, xterm, …). " +
    'Install one, set $TERMINAL, or set launchMode to "background" in ~/.llama-aio-vs/config.json.'
  );
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
  const term = resolveLinuxTerminalLauncher();
  if (!term) {
    throw new Error(missingLinuxTerminalMessage());
  }

  const libDir = path.dirname(binary);
  const launchArgv = command
    ? [command, ...(prefixArgs || []), ...args]
    : [binary, ...args];
  const launchLine = launchArgv.map(shQuote).join(" ");
  const extraExports = Object.entries(env)
    .filter(([key, value]) => value !== undefined && process.env[key] !== value)
    .map(([key, value]) => `export ${key}=${shQuote(value as string)}`);
  const cmd = [
    `echo "Llama AIO · llama-server"`,
    `echo "Binary: ${shQuote(binary)}"`,
    command ? `echo "Launcher: ${shQuote(command)}"` : `true`,
    `echo "Log also mirrored to: ${shQuote(logPath)}"`,
    `echo`,
    ...extraExports,
    `export LD_LIBRARY_PATH=${shQuote(libDir)}\${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}`,
    `${launchLine} 2>&1 | tee -a ${shQuote(logPath)}`,
    `code=$?`,
    `echo`,
    `echo "llama-server exited with code $code"`,
    `echo "Press Enter to close this window…"`,
    `read -r _ || true`,
    `exit $code`,
  ].join("; ");

  // flatpak-spawn --host must keep the sandbox env (portal). Host env for
  // llama-server is applied in the bash script above, not on this process.
  const childEnv = term.viaFlatpakHost ? process.env : { ...process.env, ...env };
  const child = spawn(term.command, [...term.prefix, "bash", "-lc", cmd], {
    detached: true,
    stdio: "ignore",
    env: childEnv,
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
