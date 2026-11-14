// Shared rendering bits: escaping, the plate chip, and Chart.js lifecycle.
import { DASH, formatPlate } from "./format.js";

const ENTITIES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ENTITIES[c]);

/** Malaysian-style plate: white characters on black. */
export function plateChip(plate) {
  return plate
    ? `<span class="plate">${esc(formatPlate(plate))}</span>`
    : `<span class="plate plate-empty" title="OCR read no characters">${DASH}</span>`;
}

// Mirrors the CSS tokens in css/app.css.
export const COLORS = { sign: "#1F5A94", muted: "#A9BCCD", ok: "#1E7B4F", warn: "#B4690E", ink2: "#5A6875", rule: "#DCE2E8" };

const charts = [];

/** Create a Chart.js chart; if the CDN didn't load, say so instead of breaking the page. */
export function makeChart(canvas, config) {
  if (typeof Chart === "undefined") {
    canvas.replaceWith(Object.assign(document.createElement("p"), {
      className: "state", textContent: "Chart unavailable: Chart.js could not be loaded.",
    }));
    return null;
  }
  Chart.defaults.font.family = "Barlow, system-ui, sans-serif";
  Chart.defaults.color = COLORS.ink2;
  const chart = new Chart(canvas, { ...config, options: { responsive: true, maintainAspectRatio: false, ...config.options } });
  charts.push(chart);
  return chart;
}

export function destroyCharts() {
  while (charts.length) charts.pop().destroy();
}
