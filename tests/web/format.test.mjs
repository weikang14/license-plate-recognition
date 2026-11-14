import { test } from "node:test";
import assert from "node:assert/strict";
import { filterRecords, formatPct, formatPlate, formatTime } from "../../web/js/format.js";

test("formatPlate spaces letter/digit groups", () => {
  assert.equal(formatPlate("VW2468"), "VW 2468");
  assert.equal(formatPlate("WXY9123A"), "WXY 9123 A");
});

test("formatPlate shows a dash for empty values", () => {
  assert.equal(formatPlate(""), "—");
  assert.equal(formatPlate(null), "—");
  assert.equal(formatPlate(undefined), "—");
});

test("formatTime is m:ss", () => {
  assert.equal(formatTime(41.13), "0:41");
  assert.equal(formatTime(75), "1:15");
});

test("formatPct keeps one decimal", () => {
  assert.equal(formatPct(0.667), "66.7%");
  assert.equal(formatPct(0), "0.0%");
});

const records = [
  { id: "a", plate: "VW2468", video: "test6", match: true },
  { id: "b", plate: "JHB2149", video: "test3", match: false },
  { id: "c", plate: "", video: "test6", match: false },
];
const ids = (rs) => rs.map((r) => r.id);

test("filterRecords searches plates ignoring case and spaces", () => {
  assert.deepEqual(ids(filterRecords(records, { q: "vw 24" })), ["a"]);
  assert.deepEqual(ids(filterRecords(records, {})), ["a", "b", "c"]);
});

test("filterRecords filters by video and result", () => {
  assert.deepEqual(ids(filterRecords(records, { video: "test6" })), ["a", "c"]);
  assert.deepEqual(ids(filterRecords(records, { result: "mismatch" })), ["b", "c"]);
  assert.deepEqual(ids(filterRecords(records, { result: "matched", video: "test3" })), []);
});
