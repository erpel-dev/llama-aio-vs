import assert from "node:assert/strict";
import path from "node:path";
import { describe, it } from "node:test";
import {
  appleScriptString,
  appleScriptTerminalLaunch,
  buildWindowsExternalLaunch,
  linuxTerminalArgsPrefix,
  missingLinuxTerminalMessage,
  resolveLinuxTerminalLauncher,
  withLogFileArg,
} from "../src/externalTerminal";

describe("withLogFileArg", () => {
  it("prepends --log-file when missing", () => {
    assert.deepEqual(withLogFileArg(["--port", "8742"], "C:\\logs\\s.log"), [
      "--log-file",
      "C:\\logs\\s.log",
      "--port",
      "8742",
    ]);
  });

  it("leaves an existing --log-file or -lf alone", () => {
    const withLong = ["--log-file", "a.log", "--port", "1"];
    const withShort = ["-lf", "a.log", "--port", "1"];
    assert.equal(withLogFileArg(withLong, "b.log"), withLong);
    assert.equal(withLogFileArg(withShort, "b.log"), withShort);
  });
});

describe("buildWindowsExternalLaunch", () => {
  const binary = path.join("C:", "Users", "me", ".llama-aio-vs", "llama.cpp", "vulkan", "llama-server.exe");
  const logPath = path.join("C:", "Users", "me", ".llama-aio-vs", "runtime", "llama-server.log");
  const args = ["-m", "model.gguf", "--port", "8742"];

  it("spawns llama-server.exe directly, never a shell", () => {
    const plan = buildWindowsExternalLaunch({ binary, args, logPath });
    assert.equal(plan.command, binary);
    assert.deepEqual(plan.argv, ["--log-file", logPath, ...args]);
    const joined = plan.argv.join(" ").toLowerCase();
    assert.equal(joined.includes("cmd.exe"), false);
    assert.equal(joined.includes("powershell"), false);
    assert.equal(joined.includes("executionpolicy"), false);
    assert.equal(joined.includes(".ps1"), false);
    assert.equal(plan.cwd, path.dirname(binary));
  });

  it("keeps shell metacharacters inside a single argv element", () => {
    const nasty = path.join("C:", "models", "evil", "model&calc&.gguf");
    const plan = buildWindowsExternalLaunch({
      binary,
      args: ["-m", nasty, "--port", "8742"],
      logPath,
    });
    // No shell sits between spawn and llama-server, so `&` cannot split the command.
    assert.equal(plan.command, binary);
    assert.equal(plan.argv.filter((arg) => arg === nasty).length, 1);
    assert.equal(plan.argv.includes("-m"), true);
  });
});

describe("resolveLinuxTerminalLauncher", () => {
  it("uses a PATH hit with the matching argv prefix", () => {
    const plan = resolveLinuxTerminalLauncher({
      isFlatpak: false,
      env: {},
      lookup: (name) => (name === "konsole" ? "/usr/bin/konsole" : undefined),
    });
    assert.ok(plan);
    assert.equal(plan.viaFlatpakHost, false);
    assert.equal(plan.command, "/usr/bin/konsole");
    assert.deepEqual(plan.prefix, linuxTerminalArgsPrefix("/usr/bin/konsole"));
  });

  it("honors $TERMINAL before the built-in list", () => {
    const plan = resolveLinuxTerminalLauncher({
      isFlatpak: false,
      env: { TERMINAL: "alacritty" },
      lookup: (name) => (name === "alacritty" ? "/usr/bin/alacritty" : undefined),
    });
    assert.ok(plan);
    assert.equal(plan.terminal, "/usr/bin/alacritty");
    assert.deepEqual(plan.prefix, ["-e"]);
  });

  it("wraps the host terminal in flatpak-spawn --host inside a Flatpak", () => {
    const plan = resolveLinuxTerminalLauncher({
      isFlatpak: true,
      env: { FLATPAK_ID: "com.visualstudio.code" },
      lookup: (name) => (name === "flatpak-spawn" ? "/usr/bin/flatpak-spawn" : undefined),
      hostLookup: (name) => (name === "konsole" ? "/usr/bin/konsole" : undefined),
    });
    assert.ok(plan);
    assert.equal(plan.viaFlatpakHost, true);
    assert.equal(plan.command, "/usr/bin/flatpak-spawn");
    assert.equal(plan.terminal, "/usr/bin/konsole");
    assert.deepEqual(plan.prefix, [
      "--host",
      "/usr/bin/konsole",
      "--title",
      "Llama AIO · llama-server",
      "-e",
    ]);
  });

  it("returns undefined in Flatpak when the host portal cannot see a terminal", () => {
    const plan = resolveLinuxTerminalLauncher({
      isFlatpak: true,
      lookup: (name) => (name === "flatpak-spawn" ? "/usr/bin/flatpak-spawn" : undefined),
      hostLookup: () => undefined,
    });
    assert.equal(plan, undefined);
    assert.match(missingLinuxTerminalMessage({ flatpak: true, flatpakId: "com.visualstudio.code" }), /Flatpak/);
  });
});

describe("appleScriptTerminalLaunch", () => {
  it("runs a script file and escapes backslash and quote for AppleScript", () => {
    const plain = appleScriptTerminalLaunch("/tmp/llama-aio/launch.sh");
    assert.equal(
      plain,
      'tell application "Terminal" to do script "bash " & quoted form of "/tmp/llama-aio/launch.sh"'
    );
    assert.equal(plain.includes("echo "), false);

    const tricky = '/tmp/llama "aio"\\launch.sh';
    const quoted = appleScriptString(tricky);
    assert.equal(quoted, '"/tmp/llama \\"aio\\"\\\\launch.sh"');
    const launched = appleScriptTerminalLaunch(tricky);
    assert.match(launched, /quoted form of "/);
    assert.equal(launched.includes("'\''"), false);
  });
});
