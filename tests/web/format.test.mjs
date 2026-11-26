import { test } from "node:test";
import assert from "node:assert/strict";
import { confirmedCaptures, filterRecords, formatPct, formatPlate, formatTime, reelCaptures } from "../../web/js/format.js";

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

test("reelCaptures finds each capture in the reel: segment start plus frames into it, at the reel's fps", () => {
  const reel = { fps: 25, segments: [{ video: "A", from: 0, to: 74, at: 0 }, { video: "B", from: 100, to: 149, at: 75 }] };
  const records = [
    { id: "b", video: "B", frame: 111 },  // frame index 110: 10 frames into B's segment -> (75 + 10) / 25
    { id: "a", video: "A", frame: 26 },   // frame index 25 -> 1 s
    { id: "x", video: "B", frame: 300 },  // cut from the reel
    { id: "y", video: "C", frame: 5 },    // not in this reel
  ];
  assert.deepEqual(reelCaptures(reel, records).map((c) => [c.r.id, c.t]), [["a", 1], ["b", 3.4]]);
});

test("confirmedCaptures returns the capture where a car's same reading was verified for the second time", () => {
  const rec = (id, video, car_id, plate, match = true) => ({ id, video, car_id, plate, match });
  const records = [
    rec("a1", "A", 1, "JX706"),              // verified once only: never confirmed
    rec("b1", "A", 2, "VEH6482"),            // a wrong reading, verified once
    rec("b2", "A", 2, "VEW6482"),
    rec("b3", "A", 2, "VEW6482", false),     // a mismatch does not count
    rec("b4", "A", 2, "VEW6482"),            // second verified VEW6482: car 2 confirmed here
    rec("c1", "B", 2, "VEW6482"),            // same car id in another video is another car
    rec("c2", "B", 2, "VEW6482"),
    rec("c3", "B", 2, "VEW6482"),            // already confirmed
  ];
  assert.deepEqual(confirmedCaptures(records).map((r) => r.id), ["b4", "c2"]);
});
