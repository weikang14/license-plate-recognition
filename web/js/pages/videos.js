// Video playback: annotated videos with their captures as a clickable timeline.
import { formatTime } from "../format.js";
import { esc, plateChip } from "../ui.js";

export function render(root, { records, summary }, params) {
  const list = summary.per_video.filter((p) => p.playback);
  if (!list.length) {
    root.innerHTML = '<p class="state">No playback videos were published. Run tools/build_site_data.py with --playback.</p>';
    return;
  }

  root.innerHTML = `
    <div class="video-layout">
      <section class="panel">
        <div class="video-tabs" role="tablist" aria-label="Videos">
          ${list.map((p) => `<button type="button" role="tab" data-video="${esc(p.video)}">${esc(p.video)}</button>`).join("")}
        </div>
        <video controls preload="metadata" playsinline></video>
        <p class="muted">Green boxes are tracked cars with their ids, red boxes are plates on those cars. The counters in the corner are drawn by the pipeline itself.</p>
      </section>
      <section class="panel">
        <h2>Captures in this video <small></small></h2>
        <ol class="timeline"></ol>
      </section>
    </div>`;

  const video = root.querySelector("video");
  const timeline = root.querySelector(".timeline");
  let mine = [];

  function load(p, t) {
    for (const b of root.querySelectorAll(".video-tabs button")) {
      b.setAttribute("aria-selected", String(b.dataset.video === p.video));
    }
    mine = records.filter((r) => r.video === p.video);
    root.querySelector("h2 small").textContent =
      `${p.vehicles} vehicles tracked, ${mine.length} captures, ${mine.filter((r) => r.match).length} verified`;
    timeline.innerHTML = mine.length
      ? mine.map((r, i) => `
        <li><button type="button" data-i="${i}">
          <span class="t">${formatTime(r.time_s)}</span><span class="car">#${r.car_id}</span>${plateChip(r.plate)}
          ${r.match ? '<span class="tag tag-ok">Verified</span>' : '<span class="tag tag-warn">Mismatch</span>'}
        </button></li>`).join("")
      : '<li class="muted">No captures in this video.</li>';
    video.src = p.playback;
    if (t) video.addEventListener("loadedmetadata", () => { video.currentTime = t; }, { once: true });
  }

  root.querySelector(".video-tabs").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-video]");
    if (b) load(list.find((p) => p.video === b.dataset.video));
  });
  timeline.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-i]");
    if (!b) return;
    video.currentTime = mine[Number(b.dataset.i)].time_s;
    video.play().catch(() => {});
  });
  video.addEventListener("timeupdate", () => {
    for (const b of timeline.querySelectorAll("button[data-i]")) {
      b.classList.toggle("now", Math.abs(mine[Number(b.dataset.i)].time_s - video.currentTime) <= 0.5);
    }
  });

  load(list.find((p) => p.video === params.get("v")) || list[0], Number(params.get("t")) || 0);
}
