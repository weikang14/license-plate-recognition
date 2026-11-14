// Model performance: the three plate-detector training runs, their curves and the final run's plots.
import { COLORS, esc, makeChart } from "../ui.js";

const LINE_COLORS = [COLORS.sign, COLORS.warn, COLORS.muted];

export function render(root, { metrics }) {
  const runs = metrics.runs;
  const final = runs.find((r) => r.key === metrics.final);
  const p = metrics.plots;
  const f3 = (x) => x.toFixed(3);

  root.innerHTML = `
    <div class="stack">
      <section class="panel">
        <h2>Plate detector training runs <small>best epoch = highest validation mAP50-95</small></h2>
        <div class="table-wrap">
          <table class="runs">
            <thead><tr><th>Run</th><th>Epochs trained</th><th>Best epoch</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>mAP50-95</th></tr></thead>
            <tbody>${runs.map((r) => `
              <tr${r.key === metrics.final ? ' class="final"' : ""}>
                <th scope="row">${esc(r.label)}</th><td>${r.epochs_trained}</td><td>${r.best.epoch}</td>
                <td>${f3(r.best.precision)}</td><td>${f3(r.best.recall)}</td><td>${f3(r.best.map50)}</td><td>${f3(r.best.map50_95)}</td>
              </tr>`).join("")}</tbody>
          </table>
        </div>
        <p class="muted">All three runs detect a single plate class and were trained from scratch (model .yaml, no pre-trained weights) on the same Roboflow dataset. The highlighted run is the model used by the pipeline.</p>
      </section>
      <div class="grid-2">
        <section class="panel">
          <h2>Validation mAP50-95 per epoch</h2>
          <div class="chart-box"><canvas id="c-map" aria-label="Validation mAP50-95 per epoch for each run"></canvas></div>
        </section>
        <section class="panel">
          <h2>Box loss <small>${esc(final.label)}</small></h2>
          <div class="chart-box"><canvas id="c-loss" aria-label="Training and validation box loss per epoch"></canvas></div>
        </section>
      </div>
      <section class="panel">
        <h2>Evaluation plots <small>${esc(final.label)}, validation set</small></h2>
        <div class="plots">
          ${[["Confusion matrix", p.confusion_matrix], ["Precision–recall curve", p.pr_curve], ["F1–confidence curve", p.f1_curve]]
            .map(([cap, src]) => `<figure><a href="${esc(src)}" target="_blank" rel="noopener"><img src="${esc(src)}" alt="${cap}" loading="lazy"></a><figcaption>${cap}</figcaption></figure>`).join("")}
        </div>
      </section>
      <section class="panel">
        <h2>Validation predictions <small>open an image to see it full size</small></h2>
        <div class="gallery">
          ${p.val_pred.map((src, i) => `<a href="${esc(src)}" target="_blank" rel="noopener"><img src="${esc(src)}" alt="Validation batch ${i} with predicted plate boxes" loading="lazy"></a>`).join("")}
        </div>
      </section>
    </div>`;

  const points = (r, key) => r.curves.epoch.map((x, i) => ({ x, y: r.curves[key][i] }));
  const line = (label, data, color, dash) => ({ label, data, borderColor: color, backgroundColor: color, borderWidth: 2, pointRadius: 0, borderDash: dash });
  const xy = { x: { type: "linear", title: { display: true, text: "Epoch" }, grid: { display: false } }, y: { beginAtZero: true } };

  makeChart(root.querySelector("#c-map"), {
    type: "line",
    data: { datasets: runs.map((r, i) => line(r.label, points(r, "map50_95"), LINE_COLORS[i % 3])) },
    options: { scales: xy, plugins: { legend: { position: "bottom" } }, interaction: { mode: "nearest", intersect: false } },
  });
  makeChart(root.querySelector("#c-loss"), {
    type: "line",
    data: { datasets: [line("Training", points(final, "train_box_loss"), COLORS.sign), line("Validation", points(final, "val_box_loss"), COLORS.warn, [5, 4])] },
    options: { scales: xy, plugins: { legend: { position: "bottom" } }, interaction: { mode: "nearest", intersect: false } },
  });
}
