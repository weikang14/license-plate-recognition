import { test } from "node:test";
import assert from "node:assert/strict";
import { render } from "../../web/js/pages/about.js";

const records = [
  { plate: "JHB2149", match: true },
  { plate: "", match: false },
  { plate: "", match: false },
  { plate: "JHB2149", match: true },
];
const summary = { videos: 2, captures: 4, matched: 2, match_rate: 0.5 };
const metrics = { final: "f", runs: [{ key: "f", label: "YOLOv8s (final)", best: { map50: 0.974, precision: 0.951, recall: 0.957 } }] };

test("about states the failure numbers the data actually shows", () => {
  const root = { innerHTML: "" };
  render(root, { records, summary, metrics });
  assert.match(root.innerHTML, /returned no text on 2 of 4 captures/);
  assert.match(root.innerHTML, /All 2 verified captures are the same plate, JHB 2149/);
  assert.doesNotMatch(root.innerHTML, /main reason/);
});
