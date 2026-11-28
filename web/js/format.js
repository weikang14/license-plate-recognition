// Small display helpers shared by every page.

export const DASH = "—";

/** "VW2468" -> "VW 2468"; empty -> "—". */
export function formatPlate(s) {
  if (!s) return DASH;
  return s.replace(/([A-Za-z])(?=\d)|(\d)(?=[A-Za-z])/g, "$1$2 ");
}

/** Seconds -> "m:ss". */
export function formatTime(sec) {
  const s = Math.floor(sec);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** 0.667 -> "66.7%". */
export function formatPct(x) {
  return `${(x * 100).toFixed(1)}%`;
}

const squash = (s) => (s || "").replace(/\s+/g, "").toUpperCase();

const readings = (r) => [r.plate, (r.yolo_chars || []).join(""), r.oxr, r.agreed];

/** q matches any reader's reading ignoring case/spaces; result is "all" | "matched" | "mismatch". */
export function filterRecords(records, { q = "", video = "all", result = "all" } = {}) {
  const needle = squash(q);
  return records.filter(
    (r) =>
      (!needle || readings(r).some((s) => squash(s).includes(needle))) &&
      (video === "all" || r.video === video) &&
      (result === "all" || (result === "matched") === r.match),
  );
}

// Captures shown in a highlight reel, in reel order, each with its time t (seconds) in the reel.
export function reelCaptures(reel, records) {
  const out = [];
  for (const g of reel.segments) {
    for (const r of records) {
      const n = r.frame - 1;  // records count frames from 1, segments from 0
      if (r.video === g.video && n >= g.from && n <= g.to) out.push({ r, t: (g.at + n - g.from) / reel.fps });
    }
  }
  return out.sort((a, b) => a.t - b.t);
}

// The capture that confirmed each car's plate: its same agreed reading verified for the second time (the pipeline's rule).
export function confirmedCaptures(records) {
  const once = new Set(), done = new Set(), out = [];
  for (const r of records) {
    const car = `${r.video}#${r.car_id}`, key = `${car}|${r.agreed}`;
    if (!r.match || done.has(car)) continue;
    if (once.has(key)) { done.add(car); out.push(r); } else once.add(key);
  }
  return out;
}
