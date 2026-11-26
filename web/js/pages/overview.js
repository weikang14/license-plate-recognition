// Overview: headline numbers, per-video captures, confidence spread, the confirmed plates.
import { confirmedCaptures, formatPct, formatTime } from "../format.js";
import { COLORS, esc, makeChart, plateChip } from "../ui.js";

export function render(root, { records, summary }) {
  const s = summary;
  const kpis = [
    ["Videos processed", s.videos],
    ["Vehicles tracked", s.vehicles],
    ["Plate captures", s.captures],
    ["Verified captures", s.matched, "ok"],
    ["Verification rate", formatPct(s.match_rate)],
    ["Final model mAP50", s.map50.toFixed(3)],
  ];
  // one row per confirmed plate, from its latest confirmation; a single verified capture can still be wrong
  const confirmed = confirmedCaptures(records);
  const recent = [...new Map(confirmed.map((r) => [r.plate, r])).values()].reverse();
  const cars = (plate) => confirmed.filter((r) => r.plate === plate).length;

  root.innerHTML = `
    <div class="stack">
      <section class="kpis" aria-label="Key figures">
        ${kpis.map(([label, value, cls]) => `
          <div class="kpi"><span class="kpi-value ${cls || ""}">${esc(value)}</span><span class="kpi-label">${label}</span></div>`).join("")}
      </section>
      <div class="grid-2">
        <section class="panel">
          <h2>Captures per video <small>verified vs. not verified</small></h2>
          <div class="chart-box"><canvas id="c-videos" aria-label="Plate captures per video"></canvas></div>
        </section>
        <section class="panel">
          <h2>Detection confidence <small>of captured plates</small></h2>
          <div class="chart-box"><canvas id="c-scores" aria-label="Distribution of plate detection confidence"></canvas></div>
        </section>
      </div>
      <section class="panel">
        <h2>Confirmed plates <small>the same reading verified twice on one vehicle</small></h2>
        ${recent.length ? `<ul class="recent">${recent.map((r) => `
          <li><a href="#/records?id=${encodeURIComponent(r.id)}">
            <img src="${esc(r.crop)}" alt="Plate ${esc(r.plate)}" loading="lazy">
            ${plateChip(r.plate)}
            <span class="muted">Vehicle #${r.car_id} in ${esc(r.video)} at ${formatTime(r.time_s)}${cars(r.plate) > 1 ? `, ${cars(r.plate)} vehicles in all` : ""}</span>
            <span class="muted">${r.score.toFixed(2)}</span>
          </a></li>`).join("")}</ul>`
        : `<p class="state">No vehicle had the same reading verified twice yet. See <a href="#/records">all captures</a>.</p>`}
      </section>
    </div>`;

  const per = s.per_video;
  makeChart(root.querySelector("#c-videos"), {
    type: "bar",
    data: {
      labels: per.map((p) => p.video),
      datasets: [
        { label: "Verified", data: per.map((p) => p.matched), backgroundColor: COLORS.ok },
        { label: "Not verified", data: per.map((p) => p.captures - p.matched), backgroundColor: COLORS.muted },
      ],
    },
    options: {
      scales: { x: { stacked: true, grid: { display: false } }, y: { stacked: true, beginAtZero: true, ticks: { precision: 0 } } },
      plugins: { legend: { position: "bottom" } },
    },
  });

  const e = s.score_hist.edges;
  makeChart(root.querySelector("#c-scores"), {
    type: "bar",
    data: {
      labels: s.score_hist.counts.map((_, i) => `${e[i].toFixed(2)}–${e[i + 1].toFixed(2)}`),
      datasets: [{ label: "Captures", data: s.score_hist.counts, backgroundColor: COLORS.sign }],
    },
    options: {
      scales: { x: { grid: { display: false } }, y: { beginAtZero: true, ticks: { precision: 0 } } },
      plugins: { legend: { display: false } },
    },
  });
}
