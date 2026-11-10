import assert from "node:assert/strict";
import { it } from "node:test";
import { estimateRemoteFit } from "../src/modelPickerInfo";

const GiB = 1024 ** 3;

it("estimateRemoteFit: file size plus allowance against capacity", () => {
  assert.equal(estimateRemoteFit(4.7 * GiB, 16 * GiB)?.fit, "fits");
  assert.equal(estimateRemoteFit(13.5 * GiB, 16 * GiB)?.fit, "tight"); // 14.85 needed: inside the 1.5 GiB headroom
  assert.equal(estimateRemoteFit(19.8 * GiB, 16 * GiB)?.fit, "wont-fit");
  assert.equal(estimateRemoteFit(1.2 * GiB, 16 * GiB)?.needBytes, 2.2 * GiB); // 1 GiB minimum allowance
  assert.equal(estimateRemoteFit(0, 16 * GiB), undefined);
  assert.equal(estimateRemoteFit(4 * GiB, 0), undefined, "unknown capacity gives no verdict");
});
