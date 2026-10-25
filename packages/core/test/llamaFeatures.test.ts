import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, it } from "node:test";
import { NO_SERVER_FEATURES, parseLlamaServerFeatures, probeLlamaServerFeatures,
  probeLlamaServerFeaturesSync } from "../src/llamaFeatures";

// Definitions from ggml-org/llama.cpp 344be0a60074e6395da3c4a4f300f7361df5a67d.
const help = `--moe-cache-mib N                       GPU cache size in MiB
--spec-draft-n-ubatch N                 physical maximum batch size of the draft context
--spec-draft-sampling {greedy,probabilistic}
                                       how the draft is sampled
`;
const all = { probed: true, moeCache: true, draftSampling: true, draftUbatch: true };

describe("optional llama.cpp feature detection", () => {
  it("recognizes actual help definitions, aliases and ANSI colors", () => {
    assert.deepEqual(parseLlamaServerFeatures(help), all);
    assert.deepEqual(parseLlamaServerFeatures(help.replace("--moe-cache-mib", "-mc, --moe-cache-mib")), all);
    assert.deepEqual(parseLlamaServerFeatures("\x1b[32m" + help + "\x1b[0m"), all);
  });
  it("does not infer support from a build number, descriptions, prefixes or a greedy-only enum", () => {
    const parsed = parseLlamaServerFeatures(`version: 999999
Use --moe-cache-mib if a newer binary supports it
--spec-draft-n-ubatch-extra N
--spec-draft-sampling {greedy}
`);
    assert.deepEqual(parsed, { ...NO_SERVER_FEATURES, probed: true });
  });

  function fake(t: { after: (fn: () => void) => void }, body: string) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llama-help-"));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const script = path.join(dir, "llama-server.cjs");
    fs.writeFileSync(script, body);
    return { script, launch: { binary: script, command: process.execPath, prefixArgs: [script],
      env: { ...process.env, LLAMA_HELP_TEST: "wrapper environment" } } };
  }
  const deviceCheck = `if (process.argv.includes('--list-devices')) { console.log('CUDA0: Test GPU (24576 MiB)'); process.exit(0); }\n`;

  it("uses wrapper args/env, captures stderr, shares concurrent probes and caches subsequent sync calls", async (t) => {
    const { script, launch } = fake(t, `
const fs = require('fs');
if (!['--help', '--list-devices'].includes(process.argv[2]) || process.env.LLAMA_HELP_TEST !== 'wrapper environment') process.exit(2);
fs.appendFileSync(__filename + '.calls', 'x');
${deviceCheck}
process.stderr.write(${JSON.stringify(help)});
`);
    const result = await Promise.all(Array.from({ length: 5 }, () => probeLlamaServerFeatures(launch)));
    result.forEach((f) => assert.deepEqual(f, all));
    assert.deepEqual(probeLlamaServerFeaturesSync(launch), all);
    assert.equal(fs.readFileSync(script + ".calls", "utf8"), "xx", "one help and one GPU check, shared by all callers");
    result[0].moeCache = false;
    assert.deepEqual(await probeLlamaServerFeatures(launch), all, "callers cannot mutate the cached result");
  });
  it("detects both upgrades/backports and downgrades at the same binary path", async (t) => {
    const { script, launch } = fake(t, `process.stdout.write('--ctx-size N');`);
    assert.deepEqual(await probeLlamaServerFeatures(launch), { ...NO_SERVER_FEATURES, probed: true });
    fs.writeFileSync(script, deviceCheck + `process.stdout.write(${JSON.stringify(help)});`);
    assert.deepEqual(probeLlamaServerFeaturesSync(launch), all);
    fs.writeFileSync(script, `process.stdout.write('--threads N');`);
    assert.deepEqual(await probeLlamaServerFeatures(launch), { ...NO_SERVER_FEATURES, probed: true });
  });
  it("fails closed on a missing executable or partial help from a failed process", async (t) => {
    const { script, launch } = fake(t, `process.stdout.write(${JSON.stringify(help)}); process.exit(2);`);
    assert.deepEqual(await probeLlamaServerFeatures(launch), NO_SERVER_FEATURES);
    fs.unlinkSync(script);
    assert.deepEqual(probeLlamaServerFeaturesSync(launch), NO_SERVER_FEATURES);
  });
  it("refuses expert caching on a CPU-only PATH binary even when its help advertises the flag", async (t) => {
    const { launch } = fake(t, `if (process.argv.includes('--list-devices')) console.log('Available devices:\\n (none)');
else process.stdout.write(${JSON.stringify(help)});`);
    assert.deepEqual(await probeLlamaServerFeatures(launch), { ...all, moeCache: false });
  });
});
