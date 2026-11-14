// Small display helpers shared by every page.

export const DASH = "—";

/** "VW2468" -> "VW 2468"; empty -> "—". */
export function formatPlate(s) {
  if (!s) return DASH;
  return s.replace(/(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])/g, " ");
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

/** q matches the plate ignoring case/spaces; result is "all" | "matched" | "mismatch". */
export function filterRecords(records, { q = "", video = "all", result = "all" } = {}) {
  const needle = squash(q);
  return records.filter(
    (r) =>
      (!needle || squash(r.plate).includes(needle)) &&
      (video === "all" || r.video === video) &&
      (result === "all" || (result === "matched") === r.match),
  );
}
