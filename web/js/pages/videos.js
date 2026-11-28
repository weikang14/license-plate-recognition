// Highlight reels: each source's annotated videos cut to the stretches where plates are read, with a clickable timeline.
import { formatTime, reelCaptures } from "../format.js";
import { esc, plateChip } from "../ui.js";

export function render(root, { records, summary }, params) {
  const reels = summary.reels || [];
  if (!reels.length) {
    root.innerHTML = '<p class="state">No highlight videos were published. Run tools/merge_site_data.py.</p>';
    return;
  }

  root.innerHTML = `
    <div class="video-layout">
      <section class="panel">
        <div class="video-tabs" role="tablist" aria-label="Highlight videos">
          ${reels.map((g, i) => `<button type="button" role="tab" data-reel="${i}">${esc(g.label)}</button>`).join("")}
        </div>
        <video controls preload="metadata" playsinline></video>
        <p><a class="reel-download" download>Download this video (MP4)</a></p>
        <p class="muted">Each video with a verified plate is cut to the stretches where plates are being read, <span class="cut-rule"></span>. The video's name is in the bottom-left corner. Green boxes are tracked cars with their ids, followed by the plate once it is confirmed (the same reading verified twice); a confirmed car is not read again. Red boxes are plates on those cars. The counters in the corner are drawn by the pipeline itself; Count is the number of confirmed cars in that video.</p>
      </section>
      <section class="panel">
        <h2>Captures in this video <small></small></h2>
        <ol class="timeline"></ol>
      </section>
    </div>`;

  const video = root.querySelector("video");
  const timeline = root.querySelector(".timeline");
  let mine = [];

  function load(i, id) {
    for (const b of root.querySelectorAll(".video-tabs button")) b.setAttribute("aria-selected", String(b.dataset.reel === String(i)));
    mine = reelCaptures(reels[i], records);
    const s = (n) => `${n} second${n === 1 ? "" : "s"}`, cut = reels[i].cut;
    root.querySelector(".cut-rule").textContent =
      `from ${s(cut.before)} before to ${s(cut.after)} after each capture; anything over ${s(cut.gap)} without a capture is cut`;
    root.querySelector("h2 small").textContent =
      `${new Set(mine.map((c) => c.r.video)).size} videos, ${mine.length} captures, ${mine.filter((c) => c.r.match).length} verified`;
    timeline.innerHTML = mine.length
      ? mine.map((c, k) => `${k && mine[k - 1].r.video === c.r.video ? "" : `<li class="tl-video">${esc(c.r.video)}</li>`}
        <li><button type="button" data-k="${k}">
          <span class="t">${formatTime(c.t)}</span><span class="car">#${c.r.car_id}</span>${plateChip(c.r.agreed || c.r.plate)}
          ${c.r.match ? '<span class="tag tag-ok">Verified</span>' : '<span class="tag tag-warn">Mismatch</span>'}
        </button></li>`).join("")
      : '<li class="muted">No captures in this video.</li>';
    video.src = root.querySelector(".reel-download").href = reels[i].src;
    const k = mine.findIndex((c) => c.r.id === id);
    if (k < 0) return;
    video.addEventListener("loadedmetadata", () => { video.currentTime = mine[k].t; }, { once: true });
    const row = timeline.querySelector(`[data-k="${k}"]`);
    timeline.scrollTop = row.offsetTop - timeline.clientHeight / 2;  // .timeline is position: relative, so it is the offsetParent
  }

  root.querySelector(".video-tabs").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-reel]");
    if (b) load(Number(b.dataset.reel));
  });
  timeline.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-k]");
    if (!b) return;
    video.currentTime = mine[Number(b.dataset.k)].t;
    video.play().catch(() => {});
  });
  video.addEventListener("timeupdate", () => {
    for (const b of timeline.querySelectorAll("button[data-k]")) {
      b.classList.toggle("now", Math.abs(mine[Number(b.dataset.k)].t - video.currentTime) <= 0.5);
    }
  });

  const id = params.get("id");
  const found = reels.findIndex((g) => reelCaptures(g, records).some((c) => c.r.id === id));
  load(Math.max(found, 0), id);
}
