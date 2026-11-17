import assert from "node:assert/strict";
import * as http from "http";
import type { AddressInfo } from "net";
import { after, before, describe, it } from "node:test";
import { loadSettings } from "./helpers";
import { ProcessManager } from "../src/processManager";
import type { LoraAdapterSetting } from "../src/types";

/** A stand-in for llama-server's /lora-adapters endpoint. */
describe("applying LoRA strengths to the running server", () => {
  let server: http.Server;
  let endpoint = "";
  let adapters: Array<{ id: number; path: string; scale: number }> = [];
  let posts: unknown[] = [];
  let failPost = false;
  let holdNextGet: ((reply: () => void) => void) | undefined;

  before(async () => {
    server = http.createServer((req, res) => {
      let body = "";
      req.on("data", d => (body += d));
      req.on("end", () => {
        res.setHeader("content-type", "application/json");
        if (req.method === "POST") {
          if (failPost) { res.statusCode = 500; res.end(JSON.stringify({ error: { message: "boom" } })); return; }
          const wanted = JSON.parse(body) as Array<{ id: number; scale: number }>;
          posts.push(wanted);
          adapters = adapters.map(a => ({ ...a, scale: wanted.find(w => w.id === a.id)?.scale ?? 0 }));
          res.end(JSON.stringify({ success: true }));
        } else {
          const hold = holdNextGet;
          holdNextGet = undefined;
          const reply = () => res.end(JSON.stringify(adapters));
          if (hold) hold(reply);
          else reply();
        }
      });
    });
    await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
    endpoint = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  });
  after(() => { server.closeAllConnections(); server.close(); });

  function manager(loaded: LoraAdapterSetting[], wanted: LoraAdapterSetting[]) {
    const store = {
      isRemote: () => false,
      getEndpoint: () => endpoint,
      getState: () => ({ selectedModelPath: "/m/model.gguf", loadSettings: loadSettings({ loraAdapters: [...wanted] }) }),
    };
    const pm = new ProcessManager(store as never);
    const stopped: boolean[] = [];
    Object.assign(pm, {
      getLaunchedConfig: () => ({ modelPath: "/m/model.gguf", loadSettings: loadSettings({ loraAdapters: loaded }), startedAt: "" }),
      stopServer: async () => { stopped.push(true); },
    });
    adapters = loaded.map((a, id) => ({ id, path: a.path, scale: 1 }));
    posts = [];
    return { pm, stopped };
  }
  const a = { path: "/l/a.gguf", scale: 0.8, enabled: true };
  const b = { path: "/l/b.gguf", scale: 1, enabled: true };

  it("sets saved strengths after a start, including adapters saved as off", async () => {
    const { pm } = manager([a, { ...b, enabled: false }], [a, { ...b, enabled: false }]);
    await pm.applyLoraScales();
    assert.deepEqual(posts, [[{ id: 0, scale: 0.8 }, { id: 1, scale: 0 }]]);
  });

  it("applies overlapping edits in order and records the final live strength for Discard", async () => {
    const wanted = [{ ...a, scale: 0.2 }];
    const { pm } = manager([a], wanted);
    const store = (pm as unknown as { store: { getState(): unknown } }).store;
    const getState = store.getState;
    let snapshots = 0;
    store.getState = () => { snapshots++; return getState(); };
    let lock = { modelPath: "/m/model.gguf", startedAt: "", loadSettings: loadSettings({ loraAdapters: [a] }) };
    Object.assign(pm, {
      getLaunchedConfig: () => lock,
      readLock: () => lock,
      lockPidIsOurs: () => true,
      writeLock: (value: typeof lock) => { lock = value; },
    });
    const pendingGet = new Promise<() => void>(resolve => { holdNextGet = resolve; });
    const first = pm.applyLoraScales();
    const reply = await pendingGet;
    wanted[0] = { ...a, scale: 0.8 };
    const second = pm.applyLoraScales();
    await new Promise<void>(resolve => setImmediate(resolve));
    const snapshotsBeforeRelease = snapshots;
    reply();
    await Promise.all([first, second]);
    assert.equal(snapshotsBeforeRelease, 1, "the second edit must wait for the first POST");
    assert.deepEqual(posts, [[{ id: 0, scale: 0.2 }], [{ id: 0, scale: 0.8 }]]);
    assert.equal(adapters[0].scale, 0.8);
    assert.equal(lock.loadSettings.loraAdapters[0].scale, 0.8);
  });

  it("still reaches the server when an added file waits for a reload", async () => {
    const { pm } = manager([a], [{ ...a, enabled: false }, b]);
    await pm.applyLoraScales();
    assert.deepEqual(posts, [[{ id: 0, scale: 0 }]]);
  });

  it("still reaches the server when a removed file waits for a reload", async () => {
    const { pm } = manager([a, b], [{ ...a, scale: 0.3 }]);
    await pm.applyLoraScales();
    assert.deepEqual(posts, [[{ id: 0, scale: 0.3 }, { id: 1, scale: 1 }]]);
  });

  it("makes no request without loaded adapters", async () => {
    const { pm } = manager([], [a]);
    await pm.applyLoraScales();
    assert.deepEqual(posts, []);
  });

  it("rejects instead of reporting success when the server refuses", async t => {
    failPost = true;
    t.after(() => { failPost = false; });
    const { pm } = manager([a], [{ ...a, enabled: false }]);
    await assert.rejects(pm.applyLoraScales());
  });

  it("fails the start and stops the server when strengths cannot be applied", async t => {
    failPost = true;
    t.after(() => { failPost = false; });
    const { pm, stopped } = manager([a], [{ ...a, enabled: false }]);
    await assert.rejects(
      (pm as unknown as { applyLoraScalesOrStop(): Promise<void> }).applyLoraScalesOrStop(),
      /did not accept the LoRA adapter strengths, so it was stopped/
    );
    assert.equal(stopped.length, 1);
    assert.equal(adapters[0].scale, 1, "the stand-in server still has the adapter at full strength");
  });

  it("recovers when a retry succeeds during start", async () => {
    const { pm, stopped } = manager([a], [{ ...a, enabled: false }]);
    failPost = true;
    setTimeout(() => { failPost = false; }, 300);
    await (pm as unknown as { applyLoraScalesOrStop(): Promise<void> }).applyLoraScalesOrStop();
    assert.equal(stopped.length, 0);
    assert.equal(adapters[0].scale, 0);
  });
});
