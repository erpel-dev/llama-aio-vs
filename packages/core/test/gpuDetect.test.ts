import assert from "node:assert/strict";
import { it } from "node:test";
import { linuxHasGpuDevice } from "../src/llamaInstaller";

it("linuxHasGpuDevice: render node or NVIDIA device means a GPU; neither means CPU-only", () => {
  const fake = (tree: Record<string, string[]>) => (dir: string) => tree[dir] ?? [];
  assert.equal(linuxHasGpuDevice(fake({ "/dev/dri": ["card0", "renderD128"] })), true);
  assert.equal(linuxHasGpuDevice(fake({ "/dev": ["null", "nvidia0", "nvidiactl"] })), true);
  assert.equal(linuxHasGpuDevice(fake({ "/dev/dri": ["card0"], "/dev": ["null", "tty"] })), false);
  assert.equal(linuxHasGpuDevice(fake({})), false);
});
