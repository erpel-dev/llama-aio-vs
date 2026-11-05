import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  explainLocalModelModes,
  explainModelModes,
  explainRemoteModelModes,
  resolveModelModes,
  resolveRemoteModelModes,
} from "../src/modelModes";

describe("explain*ModelModes (#35 profile visibility)", () => {
  it("reports how a profile matched, consistently with resolve*", () => {
    const byName = explainModelModes(undefined, "Spark-X2.6-1.7B-Q4_K_M.gguf");
    assert.equal(byName.via, "name");
    assert.equal(byName.profile, resolveModelModes(undefined, "Spark-X2.6-1.7B-Q4_K_M.gguf"));
    const byArch = explainModelModes({ architecture: "qwen35", path: "/m/opaque.gguf" } as never, "/m/opaque.gguf");
    assert.equal(byArch.via, "architecture");
    assert.ok(byArch.profile);
    assert.equal(explainModelModes(undefined, "mmproj-F16.gguf").via, "excluded");
    assert.deepEqual(explainModelModes(undefined, "totally-unknown-7b.gguf"), { via: "generic" });
  });

  it("names the advertised identity that matched on a remote server", () => {
    const hf = explainRemoteModelModes({ modelId: "opaque", modelAlias: "unsloth/Spark-X2.6-4B-GGUF" });
    assert.equal(hf.identityField, "alias");
    assert.equal(hf.profile, resolveRemoteModelModes({ modelId: "opaque", modelAlias: "unsloth/Spark-X2.6-4B-GGUF" }));
    const fromAliases = explainRemoteModelModes({ modelId: "opaque", modelAliases: ["gpt-oss-20b"] });
    assert.equal(fromAliases.identityField, "aliases");
    assert.equal(fromAliases.profile?.familyLabel, "GPT-OSS");
    const unknown = explainRemoteModelModes({ modelId: "my-private-model" });
    assert.equal(unknown.profile, undefined);
    assert.equal(unknown.via, "generic");
    assert.equal(unknown.identity, "my-private-model");
  });

  it("contradicting aliases fall back explicitly instead of picking the first", () => {
    const m = explainRemoteModelModes({ modelId: "opaque", modelAliases: ["gpt-oss-20b", "glm-5.3-flash"] });
    assert.equal(m.profile, undefined);
    assert.equal(m.via, "unverified");
  });

  it("a pending local selection reports the running model, not the selection", () => {
    const caps = { architecture: "qwen35", path: "/m/new-selection.gguf" } as never;
    const running = explainLocalModelModes(caps, "/m/LFM3-1.7B-Q4_K_M.gguf", "/m/LFM3-1.7B-Q4_K_M.gguf");
    assert.notEqual(running.via, "architecture");
  });
});
