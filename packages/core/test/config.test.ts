import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, it } from "node:test";
import { ConfigFile } from "../src/config";

describe("ConfigFile", () => {
  it("announces a peer's edit that a local write absorbed, and keeps it", async () => {
    const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "llama-aio-config-")), "config.json");
    const config = new ConfigFile(file);
    await config.setPreferences({ port: 8742 } as never);

    // The TUI writes a new port; this window writes something else before its watcher fires.
    const peer = JSON.parse(fs.readFileSync(file, "utf8"));
    peer.app.port = 9000;
    fs.writeFileSync(file, JSON.stringify(peer, null, 2));

    const external: number[] = [];
    config.onDidChangeExternally((c) => external.push(c.app.port as number));
    await config.setPreferences({ launchMode: "background" } as never);

    assert.deepEqual(external, [9000], "the absorbed peer edit is announced once");
    assert.equal(config.getPreferences().port, 9000, "peer edit kept");
    assert.equal(config.getPreferences().launchMode, "background", "local edit applied");

    // A plain local write with nothing new on disk is not external.
    await config.setPreferences({ launchMode: "externalTerminal" } as never);
    assert.equal(external.length, 1);
    config.dispose();
  });
});
