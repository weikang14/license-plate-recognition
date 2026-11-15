import { test } from "node:test";
import assert from "node:assert/strict";
import { render } from "../../web/js/pages/about.js";

const records = [
  { plate: "JHB2149", match: true },
  { plate: "", match: false },
  { plate: "", match: false },
  { plate: "JHB2149", match: true },
];
const summary = { videos: 2, vehicles: 5, captures: 4, matched: 2, match_rate: 0.5 };
const metrics = { final: "f", runs: [{ key: "f", label: "YOLOv8s (final)", best: { map50: 0.974, precision: 0.951, recall: 0.957 } }] };

test("about states the failure numbers the data actually shows", () => {
  const root = { innerHTML: "" };
  render(root, { records, summary, metrics });
  assert.match(root.innerHTML, /returned no text on 2 of 4 captures/);
  assert.match(root.innerHTML, /All 2 verified captures are the same plate, JHB 2149/);
  assert.doesNotMatch(root.innerHTML, /main reason/);
});

test("about explains tracking and reports the vehicle count", () => {
  const root = { innerHTML: "" };
  render(root, { records, summary, metrics });
  assert.match(root.innerHTML, /tracked 5 vehicles/);
  assert.match(root.innerHTML, /ByteTrack/);
  assert.match(root.innerHTML, /photo of a plate/);
});

test("about lists each verified plate once", () => {
  const root = { innerHTML: "" };
  const recs = ["JHB2149", "VEW6482", "JHB2149"].map((plate) => ({ plate, match: true }));
  render(root, { records: recs, summary: { ...summary, captures: 3, matched: 3 }, metrics });
  assert.match(root.innerHTML, /The 3 verified captures are 2 plates, JHB 2149 and VEW 6482\./);
});

test("about explains the in-order cross-check", () => {
  const root = { innerHTML: "" };
  render(root, { records, summary, metrics });
  assert.match(root.innerHTML, /left-to-right order/);
  assert.doesNotMatch(root.innerHTML, /compared without order|Known shortcut/);
});
