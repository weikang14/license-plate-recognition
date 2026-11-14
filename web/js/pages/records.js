// Recognition records: every plate capture, searchable, with a detail view comparing OCR and YOLO.
import { DASH, filterRecords, formatTime } from "../format.js";
import { esc, plateChip } from "../ui.js";

const PAGE_SIZE = 25;
const resultTag = (r) =>
  r.match ? '<span class="tag tag-ok">Verified</span>' : '<span class="tag tag-warn">Mismatch</span>';
const sortedChars = (chars) => (chars.length ? esc([...chars].sort().join(" ")) : DASH);

export function render(root, { records, summary }, params) {
  const playable = new Set(summary.per_video.filter((p) => p.playback).map((p) => p.video));
  const state = { q: "", video: "all", result: "all", page: 0 };

  root.innerHTML = `
    <section class="panel">
      <form class="filters" role="search">
        <label>Plate<input type="search" name="q" placeholder="e.g. VEW 2468" autocomplete="off"></label>
        <label>Video<select name="video"><option value="all">All videos</option>
          ${summary.per_video.map((p) => `<option value="${esc(p.video)}">${esc(p.video)}</option>`).join("")}</select></label>
        <label>Result<select name="result"><option value="all">All results</option>
          <option value="matched">Verified</option><option value="mismatch">Mismatch</option></select></label>
      </form>
      <p class="muted" aria-live="polite" id="rec-count"></p>
      <div class="table-wrap">
        <table class="records">
          <thead><tr><th>Snapshot</th><th>Plate (OCR)</th><th>Video</th><th>Time</th><th>Confidence</th><th>Characters</th><th>Result</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
      <nav class="pager" aria-label="Pages">
        <button type="button" data-step="-1">Previous</button><span></span><button type="button" data-step="1">Next</button>
      </nav>
    </section>
    <dialog class="detail" aria-labelledby="detail-title"></dialog>`;

  const form = root.querySelector(".filters");
  const tbody = root.querySelector("tbody");
  const pager = root.querySelector(".pager");
  const dialog = root.querySelector("dialog");
  const byId = new Map(records.map((r) => [r.id, r]));

  function draw() {
    const rows = filterRecords(records, state);
    const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
    state.page = Math.min(state.page, pages - 1);
    const shown = rows.slice(state.page * PAGE_SIZE, (state.page + 1) * PAGE_SIZE);
    root.querySelector("#rec-count").textContent =
      `${rows.length} of ${records.length} captures${rows.length ? "" : ". Try a different search or filter."}`;
    tbody.innerHTML = shown.map((r) => `
      <tr tabindex="0" data-id="${esc(r.id)}" aria-label="Open capture ${esc(r.plate || "with no OCR text")}">
        <td><img src="${esc(r.crop)}" alt="" loading="lazy"></td>
        <td>${plateChip(r.plate)}</td>
        <td>${esc(r.video)}</td>
        <td>${formatTime(r.time_s)}</td>
        <td>${r.score.toFixed(2)}</td>
        <td>${r.n_chars}</td>
        <td>${resultTag(r)}</td>
      </tr>`).join("");
    pager.querySelector("span").textContent = `Page ${state.page + 1} of ${pages}`;
    pager.querySelector('[data-step="-1"]').disabled = state.page === 0;
    pager.querySelector('[data-step="1"]').disabled = state.page >= pages - 1;
  }

  function open(r) {
    const link = playable.has(r.video)
      ? `<a class="button" href="#/videos?v=${encodeURIComponent(r.video)}&t=${r.time_s}">Open in video playback</a>`
      : `<p class="muted">No playback video was published for ${esc(r.video)}.</p>`;
    dialog.innerHTML = `
      <div class="detail-head">
        <h2 id="detail-title">${plateChip(r.plate)} ${resultTag(r)}</h2>
        <form method="dialog"><button type="submit" class="close">Close</button></form>
      </div>
      <img class="detail-img" src="${esc(r.crop)}" alt="Captured plate ${esc(r.plate)}">
      <dl class="compare">
        <div><dt>Tesseract OCR text</dt><dd>${r.plate ? esc(r.plate) : DASH}</dd></div>
        <div><dt>OCR characters, sorted</dt><dd>${sortedChars([...(r.plate || "")])}</dd></div>
        <div><dt>YOLO characters, sorted</dt><dd>${sortedChars(r.yolo_chars)}</dd></div>
      </dl>
      <p class="muted">A plate is verified only when both sorted lists are identical.</p>
      <dl class="facts">
        <div><dt>Video</dt><dd>${esc(r.video)}</dd></div>
        <div><dt>Time</dt><dd>${formatTime(r.time_s)} (frame ${r.frame})</dd></div>
        <div><dt>Detection confidence</dt><dd>${r.score.toFixed(3)}</dd></div>
      </dl>
      ${link}`;
    dialog.showModal();
  }

  form.addEventListener("input", (e) => {
    state[e.target.name] = e.target.value;
    state.page = 0;
    draw();
  });
  form.addEventListener("submit", (e) => e.preventDefault());
  pager.addEventListener("click", (e) => {
    const step = Number(e.target.dataset?.step);
    if (step) { state.page += step; draw(); }
  });
  tbody.addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-id]");
    if (tr) open(byId.get(tr.dataset.id));
  });
  tbody.addEventListener("keydown", (e) => {
    const tr = e.target.closest("tr[data-id]");
    if (tr && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); open(byId.get(tr.dataset.id)); }
  });
  dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });

  draw();
  const wanted = byId.get(params.get("id"));
  if (wanted) open(wanted);
}
