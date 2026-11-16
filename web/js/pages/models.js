// Model performance: the plate-detector training runs, the character model's scores, speed, and the final run's plots.
import { COLORS, esc, makeChart } from "../ui.js";

const LINE_COLORS = [COLORS.sign, COLORS.warn, COLORS.muted];
const f3 = (x) => x.toFixed(3);

const STAGES = [
  ["vehicle_ms", "Vehicle detection (YOLOv8n)", "every frame"],
  ["plate_ms", "Plate detection (YOLOv8s)", "every 7th frame"],
  ["characters_ms", "Character detection (YOLOv8n)", "per capture"],
  ["tesseract_ms", "Tesseract OCR", "per capture, twice if psm 13 is needed"],
];

export function letterPanel(s) {
  return `
      <section class="panel">
        <h2>Character detector <small>plate_letter.pt, YOLOv8n fine-tuned from COCO weights</small></h2>
        <div class="table-wrap">
          <table class="runs">
            <thead><tr><th>Precision</th><th>Recall</th><th>mAP50</th><th>mAP50-95</th></tr></thead>
            <tbody><tr><td>${f3(s.precision)}</td><td>${f3(s.recall)}</td><td>${f3(s.map50)}</td><td>${f3(s.map50_95)}</td></tr></tbody>
          </table>
        </div>
        <p class="muted">Validation scores stored in the model file the pipeline uses, from a 100-epoch run saved at epoch 24. It detects 0–9 and A–Y without I, O and Z, plus a back_car class. The curves below come from a longer run of about 500 epochs, shown in the FYP report; its weights were not kept.</p>
        <div class="plots letter-figs">
          <figure><a href="img/letter-training.png" target="_blank" rel="noopener"><img src="img/letter-training.png" alt="Character model training curves over about 500 epochs: losses, precision, recall, mAP50 and mAP50-95" loading="lazy"></a><figcaption>Training curves, about 500 epochs (FYP report, figure 39)</figcaption></figure>
          <figure><a href="img/letter-detection.png" target="_blank" rel="noopener"><img src="img/letter-detection.png" alt="Each character of plate JHB 2149 boxed and labelled by the character model" loading="lazy"></a><figcaption>Characters detected on a plate crop (FYP report, figure 40)</figcaption></figure>
        </div>
      </section>`;
}

export function speedPanel(speed) {
  const devs = Object.values(speed);
  const cell = (ms) => (ms == null ? "CPU only" : `${ms} ms`);
  const fps = (d) => Math.round(1000 / (d.vehicle_ms + d.plate_ms / 7));
  return `
      <section class="panel">
        <h2>Speed <small>one 1080p frame, measured with tools/benchmark.py</small></h2>
        <div class="table-wrap">
          <table class="runs">
            <thead><tr><th>Stage</th><th>Runs</th>${devs.map((d) => `<th>${esc(d.hardware)}</th>`).join("")}</tr></thead>
            <tbody>${STAGES.map(([key, label, when]) => `
              <tr><th scope="row">${label}</th><td>${when}</td>${devs.map((d) => `<td>${cell(d[key])}</td>`).join("")}</tr>`).join("")}
              <tr class="final"><th scope="row">Frames per second</th><td>vehicles + plates</td>${devs.map((d) => `<td>${fps(d)} fps</td>`).join("")}</tr>
            </tbody>
          </table>
        </div>
        <p class="muted">Frames per second leaves out video decoding and the per-capture steps, which run at most once every 1.5 seconds. Tesseract has no GPU version. ${devs.map((d) => `${esc(d.hardware)}: PyTorch ${esc(d.torch)}, Ultralytics ${esc(d.ultralytics)}.`).join(" ")}</p>
      </section>`;
}

export function render(root, { metrics, speed }) {
  const runs = metrics.runs;
  const final = runs.find((r) => r.key === metrics.final);
  const p = metrics.plots;

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
      </section>${metrics.letter ? letterPanel(metrics.letter) : ""}${speed ? speedPanel(speed) : ""}
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
