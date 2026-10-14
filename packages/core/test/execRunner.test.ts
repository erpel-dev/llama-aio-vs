import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { runStepsAsync, runStepsSync, type ExecSteps } from "../src/execRunner";

const node = process.execPath;

function* probe(): ExecSteps<string[]> {
  const out: string[] = [];
  const a = yield { command: node, args: ["-e", "process.stdout.write('one')"], timeoutMs: 10_000 };
  out.push(`${a.ok}:${a.stdout}`);
  const b = yield { command: node, args: ["-e", "process.exit(3)"], timeoutMs: 10_000 };
  out.push(`${b.ok}:${b.status}`);
  const c = yield {
    command: node,
    args: ["-e", "process.stderr.write('listed'); process.exit(1)"],
    timeoutMs: 10_000,
    captureStderr: true,
  };
  out.push(`${c.ok}:${c.stderr}`);
  const d = yield { command: "llama-aio-no-such-tool", args: [], timeoutMs: 10_000 };
  out.push(`${d.ok}`);
  return out;
}

describe("execRunner", () => {
  it("drives the same steps synchronously and asynchronously", async () => {
    const expected = ["true:one", "false:3", "false:listed", "false"];
    assert.deepEqual(runStepsSync(probe()), expected);
    assert.deepEqual(await runStepsAsync(probe()), expected);
  });

  it("returns raw bytes for binary requests", async () => {
    function* bytes(): ExecSteps<number[]> {
      const r = yield {
        command: node,
        args: ["-e", "process.stdout.write(Buffer.from([0xff, 0xfe, 0x41, 0x00]))"],
        timeoutMs: 10_000,
        binary: true,
      };
      return [...(r.buffer || [])];
    }
    assert.deepEqual(runStepsSync(bytes()), [0xff, 0xfe, 0x41, 0x00]);
    assert.deepEqual(await runStepsAsync(bytes()), [0xff, 0xfe, 0x41, 0x00]);
  });
});
