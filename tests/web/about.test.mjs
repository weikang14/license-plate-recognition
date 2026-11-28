import { test } from "node:test";
import assert from "node:assert/strict";
import { render } from "../../web/js/pages/about.js";

const records = [
  { plate: "JHB2149", agreed: "JHB2149", match: true },
  { plate: "", agreed: "", match: false },
  { plate: "", agreed: "JHB2149", match: true },  // YOLO and OXR agreed, Tesseract read nothing
  { plate: "JHB2149", agreed: "JHB2149", match: true },
];
const summary = { videos: 2, vehicles: 5, captures: 4, matched: 2, match_rate: 0.5 };
const metrics = { final: "f", runs: [{ key: "f", label: "YOLOv8s (final)", best: { map50: 0.974, precision: 0.951, recall: 0.957 } }] };

test("about states the failure numbers the data actually shows", () => {
  const root = { innerHTML: "" };
  render(root, { records, summary, metrics });
  assert.match(root.innerHTML, /returned no text on 2 of 4 captures/);
  assert.match(root.innerHTML, /Every confirmed plate is JHB 2149\./);
  assert.doesNotMatch(root.innerHTML, /main reason/);
});

test("about explains tracking and reports the vehicle count", () => {
  const root = { innerHTML: "" };
  render(root, { records, summary, metrics });
  assert.match(root.innerHTML, /tracked 5 vehicles/);
  assert.match(root.innerHTML, /ByteTrack/);
  assert.match(root.innerHTML, /photo of a plate/);
});

test("about lists each confirmed plate once, and not plates verified only once", () => {
  const root = { innerHTML: "" };
  const recs = [["a", "JHB2149"], ["a", "JHB2149"], ["b", "VEW6482"], ["b", "VEW6482"], ["c", "JHB2149"], ["c", "JHB2149"], ["d", "VEH6482"]]
    .map(([video, agreed]) => ({ video, car_id: 1, plate: "", agreed, match: true }));
  render(root, { records: recs, summary: { ...summary, captures: 7, matched: 7 }, metrics });
  assert.match(root.innerHTML, /The confirmed plates are JHB 2149 and VEW 6482\./);
  assert.doesNotMatch(root.innerHTML, /confirmed plates are[^.]*VEH/);
});

test("about explains the in-order cross-check", () => {
  const root = { innerHTML: "" };
  render(root, { records, summary, metrics });
  assert.match(root.innerHTML, /left-to-right order/);
  assert.doesNotMatch(root.innerHTML, /compared without order|Known shortcut/);
});

test("about explains the 2-of-3 cross-check and credits SoMark OXR", () => {
  const root = { innerHTML: "" };
  render(root, { records, summary, metrics });
  assert.match(root.innerHTML, /two of the three/);
  assert.match(root.innerHTML, /SoMark OXR/);
  assert.match(root.innerHTML, /github\.com\/SoMarkAI\/OXR/);
});

test("about says who built what", () => {
  const root = { innerHTML: "" };
  render(root, { records, summary, metrics });
  assert.match(root.innerHTML, /model training and the recognition pipeline: weikang14's final year project/);
  assert.match(root.innerHTML, /built with Claude Code/);
});

test("about counts vehicles whose plate was confirmed (the same reading verified twice)", () => {
  const root = { innerHTML: "" };
  const recs = [
    { video: "a", car_id: 1, agreed: "JHB2149", match: true },
    { video: "a", car_id: 1, agreed: "", match: false },
    { video: "a", car_id: 1, agreed: "JHB2149", match: true },
    { video: "a", car_id: 2, agreed: "", match: false },
    { video: "b", car_id: 1, agreed: "VEW6482", match: true },
    { video: "b", car_id: 1, agreed: "VEH6482", match: true },
  ];
  render(root, { records: recs, summary: { ...summary, captures: 6, matched: 4 }, metrics });
  assert.match(root.innerHTML, /1 of the 3 vehicles with a plate capture had their plate confirmed/);
});

test("about says the vehicle detector also keeps buses and trucks", () => {
  const root = { innerHTML: "" };
  render(root, { records, summary, metrics });
  assert.match(root.innerHTML, /car, bus and truck are used/);
});

test("about lists four next steps, led by a single recognition model and a faster OXR", () => {
  const root = { innerHTML: "" };
  render(root, { records, summary, metrics });
  const steps = root.innerHTML.match(/<ol class="next-steps stack-list">([\s\S]*?)<\/ol>/);
  assert.ok(steps, "next steps list is missing");
  assert.equal(steps[1].match(/<li>/g).length, 4);
  assert.match(steps[1], /single recognition model/);
  assert.match(steps[1], /vLLM/);
  assert.match(steps[1], /two-row plates/);
  assert.doesNotMatch(root.innerHTML, /YOLOv9/);
});
