import assert from "node:assert/strict";
import path from "node:path";
import { describe, it } from "node:test";
import { buildWindowsExternalLaunch, withLogFileArg } from "../src/externalTerminal";

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

  it("starts llama-server.exe via cmd start, never PowerShell", () => {
    const plan = buildWindowsExternalLaunch({ binary, args, logPath });
    assert.equal(plan.command, "cmd.exe");
    assert.equal(plan.argv[0], "/c");
    assert.equal(plan.argv[1], "start");
    assert.equal(plan.argv[2], "Llama AIO - llama-server");
    assert.equal(plan.argv[3], binary);
    assert.equal(plan.argv[4], "--log-file");
    assert.equal(plan.argv[5], logPath);
    const joined = plan.argv.join(" ").toLowerCase();
    assert.equal(joined.includes("powershell"), false);
    assert.equal(joined.includes("executionpolicy"), false);
    assert.equal(joined.includes(".ps1"), false);
    assert.equal(plan.cwd, path.dirname(binary));
  });

  it("uses Windows Terminal as the console host when available", () => {
    const wt = path.join("C:", "Program Files", "Windows Terminal", "wt.exe");
    const plan = buildWindowsExternalLaunch({
      binary,
      args,
      logPath,
      windowsTerminal: wt,
    });
    assert.equal(plan.command, wt);
    assert.deepEqual(plan.argv.slice(0, 5), [
      "-d",
      path.dirname(binary),
      "--title",
      "Llama AIO - llama-server",
      binary,
    ]);
    assert.equal(plan.argv.includes("powershell.exe"), false);
  });
});
