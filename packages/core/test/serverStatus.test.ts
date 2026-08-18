import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { effectiveServerUiState } from "../src/types";

describe("effectiveServerUiState", () => {
  it("shows starting only while boot is in progress and HTTP is down", () => {
    assert.deepEqual(effectiveServerUiState({ starting: true }), {
      starting: true,
      ready: false,
    });
  });

  it("treats a live server as ready even if starting was left set", () => {
    assert.deepEqual(
      effectiveServerUiState({ starting: true, running: true }),
      { starting: false, ready: true }
    );
    assert.deepEqual(
      effectiveServerUiState({ starting: true, httpReady: true }),
      { starting: false, ready: true }
    );
  });

  it("is stopped when nothing is up", () => {
    assert.deepEqual(effectiveServerUiState({}), { starting: false, ready: false });
  });
});
