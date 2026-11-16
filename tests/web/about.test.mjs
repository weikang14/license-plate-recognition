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

test("about says who built what", () => {
  const root = { innerHTML: "" };
  render(root, { records, summary, metrics });
  assert.match(root.innerHTML, /model training and the recognition pipeline: weikang14's final year project/);
  assert.match(root.innerHTML, /built with Claude Code/);
});

test("about counts vehicles with at least one verified reading", () => {
  const root = { innerHTML: "" };
  const recs = [
    { video: "a", car_id: 1, plate: "JHB2149", match: true },
    { video: "a", car_id: 1, plate: "", match: false },
    { video: "a", car_id: 2, plate: "", match: false },
    { video: "b", car_id: 1, plate: "VEW6482", match: true },
  ];
  render(root, { records: recs, summary: { ...summary, captures: 4, matched: 2 }, metrics });
  assert.match(root.innerHTML, /2 of the 3 vehicles with a plate capture got at least one verified reading/);
});

test("about says the vehicle detector also keeps buses and trucks", () => {
  const root = { innerHTML: "" };
  render(root, { records, summary, metrics });
  assert.match(root.innerHTML, /car, bus and truck are used/);
});

test("about lists three next steps, led by a single recognition model", () => {
  const root = { innerHTML: "" };
  render(root, { records, summary, metrics });
  const steps = root.innerHTML.match(/<ol class="next-steps stack-list">([\s\S]*?)<\/ol>/);
  assert.ok(steps, "next steps list is missing");
  assert.equal(steps[1].match(/<li>/g).length, 3);
  assert.match(steps[1], /single recognition model/);
  assert.match(steps[1], /two-row plates/);
  assert.doesNotMatch(root.innerHTML, /YOLOv9/);
});
