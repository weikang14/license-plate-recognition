import { test } from "node:test";
import assert from "node:assert/strict";
import { letterPanel, speedPanel } from "../../web/js/pages/models.js";

const speed = {
  cpu: { hardware: "Laptop CPU", frame: "1920x1080", torch: "2.3.1", ultralytics: "8.1.27", vehicle_ms: 70, plate_ms: 140, characters_ms: 50, tesseract_ms: 150 },
  gpu: { hardware: "Laptop GPU", frame: "1920x1080", torch: "2.7.0", ultralytics: "8.3.127", vehicle_ms: 10, plate_ms: 14, characters_ms: 14, oxr_ms: 812 },
};

test("speed panel lists each stage per device", () => {
  const html = speedPanel(speed);
  assert.match(html, /Laptop CPU/);
  assert.match(html, /Laptop GPU/);
  assert.match(html, /70.0 ms/);  // one decimal, as in the README (12.0 ms, not 12 ms)
  assert.match(html, /150.0 ms/);
});

test("speed panel turns stage times into frames per second (vehicles every frame, plates on 7 of 30 frames)", () => {
  const html = speedPanel(speed);
  assert.match(html, /10 fps/);  // 70 + 140 * 7 / 30 = 102.7 ms
  assert.match(html, /75 fps/);  // 10 + 14 * 7 / 30 = 13.3 ms
});

test("speed panel says where Tesseract and OXR were not timed", () => {
  assert.match(speedPanel(speed), /CPU only/);   // Tesseract has no GPU version
  assert.match(speedPanel(speed), /GPU only/);   // OXR is far too slow on a CPU
  assert.match(speedPanel(speed), /812.0 ms/);
});

test("letter panel shows the character model's validation scores", () => {
  const html = letterPanel({ precision: 0.949, recall: 0.904, map50: 0.96, map50_95: 0.789 });
  assert.match(html, /0\.949/);
  assert.match(html, /0\.960/);
  assert.match(html, /0\.789/);
});

test("letter panel shows the training curves and a detection from the FYP report", () => {
  const html = letterPanel({ precision: 0.949, recall: 0.904, map50: 0.96, map50_95: 0.789 });
  assert.match(html, /src="img\/letter-training\.png"/);
  assert.match(html, /src="img\/letter-detection\.png"/);
  assert.match(html, /FYP report/);
  assert.match(html, /about 500 epochs/);
});
