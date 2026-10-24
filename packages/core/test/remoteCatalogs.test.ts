import assert from "node:assert/strict";
import { describe, it } from "node:test";
import catalogs from "./fixtures/remote-model-catalogs.json";
import { remoteModelIdentity } from "../src/remoteServer";
import { applyModeToRequestBody, resolveModelModes, resolveModeParams, resolveRemoteModelModes } from "../src/modelModes";

describe("pinned Strata and Project Maya catalog identities", () => {
  assert.equal(catalogs.rows.length, 39);
  for (const row of catalogs.rows) {
    it(`selects ${row.provider}/${row.installId} from standalone and proxied identities`, () => {
      const standalone = resolveRemoteModelModes(remoteModelIdentity({ id: row.servedId }));
      assert.equal(standalone?.familyLabel, row.profile);
      const proxied = resolveRemoteModelModes(remoteModelIdentity({ id: "my-private-model", alias_of: row.servedId }));
      assert.equal(proxied?.familyLabel, row.profile);
      assert.equal(resolveRemoteModelModes({ modelId: "my-private-model", modelAliases: [row.servedId] })?.familyLabel, row.profile);
      if (row.provider === "strata") {
        const label = "label" in row ? row.label : undefined;
        const desktop = resolveRemoteModelModes(remoteModelIdentity({
          id: `${row.installId}--2`, meta: { label, aliases: [row.servedId] },
        }));
        assert.equal(desktop?.familyLabel.replace(/ 1017$/, ""), row.profile);
        assert.deepEqual(desktop?.modes, standalone?.modes);
      }
      assert.ok(standalone);
      for (const mode of Object.keys(standalone.modes)) {
        const body: Record<string, unknown> = { model: row.servedId };
        applyModeToRequestBody(body, resolveModeParams(standalone, { reasoningEffort: mode }).params);
        assert.equal(body.model, row.servedId);
        if (row.family === "deepseek" || row.family === "deepseek41") {
          assert.equal(body.top_p, 1);
          assert.equal(body.temperature, 1);
          const kwargs = body.chat_template_kwargs as Record<string, unknown>;
          assert.equal(kwargs.enable_thinking, mode !== "No Think");
          assert.equal(kwargs.thinking_mode, undefined);
          if (mode !== "No Think") assert.ok([50, 75, 100].includes(kwargs.reasoning_effort as number));
        }
      }
    });
  }
  it("recognizes the three official IST-DAS tags without confusing arbitrary bpw filenames", () => {
    for (const bits of ["1.75", "2.00", "2.25"]) {
      assert.equal(resolveModelModes(undefined, `deepseek-v4.1-flash-ist-das-${bits}bpw`)?.familyLabel, "DeepSeek V4.1 Flash 1017");
    }
    for (const id of ["q2_0", "iq3_s", "Maya-M", "maya-m", "model-2.00bpw", "deepseek-v4.1-flash-ist-das-9.00bpw"]) {
      assert.equal(resolveModelModes(undefined, id), undefined, id);
    }
  });
  it("does not guess among conflicting aliases or override a canonical model identity", () => {
    const modelAliases = ["qwen3.8-flash-next", "glm-5.3-flash"];
    assert.equal(resolveRemoteModelModes({ modelId: "opaque", modelAliases }), undefined);
    assert.equal(resolveRemoteModelModes({ modelId: "glm-5.3-flash", modelAliases })?.familyLabel, "GLM-5.3-Flash");
  });

  for (const row of catalogs.legacyRows) {
    it(`also resolves legacy real-engine entry ${row.id} from its published family label`, () => {
      assert.equal(resolveRemoteModelModes({ modelId: row.id, modelName: row.label })?.familyLabel, row.profile);
    });
  }
});
