// About: the project in plain words (from the FYP report), the pipeline, the stack and the models.
import { formatPct } from "../format.js";

const REPO = "https://github.com/weikang14/license-plate-recognition";

const STEPS = [
  ["Find vehicles", "YOLOv8n (COCO) marks cars in every frame."],
  ["Find the plate", "A YOLOv8s plate detector trained for this project. Plates at 0.75 confidence or higher are captured, at most once every 1.5 seconds."],
  ["Straighten it", "The crop is converted to grayscale, resized and perspective-corrected so the characters sit level."],
  ["Read it twice", "Tesseract OCR reads the text, and a second YOLOv8 model detects each character on its own."],
  ["Cross-check", "The plate is verified only when both readers find the same characters. The OCR text is kept because it preserves character order."],
];

export function render(root, { summary, metrics }) {
  const final = metrics.runs.find((r) => r.key === metrics.final);

  root.innerHTML = `
    <div class="about">
      <section class="panel prose">
        <h2>What this project does</h2>
        <p>This system reads Malaysian license plates from video. It finds a vehicle, finds its plate, and reads the plate with two independent methods. A plate is only recorded as verified when both methods agree. It was built as a final year project to automate the plate logging that guards still do by hand at the gates of housing areas, malls and office buildings.</p>

        <h3>Background</h3>
        <p>Malaysia has about 22 million registered vehicles for about 34 million people, roughly 66 vehicles for every 100 residents and the second-highest rate in Southeast Asia. Writing down every entry and exit by hand is slow and labour-intensive, and plates are the one feature that identifies a vehicle reliably.</p>

        <h3>Objective</h3>
        <p>Apply computer vision to detect and read license plates automatically, so that vehicle entry and exit can be logged without a person watching the camera.</p>

        <h3>Results</h3>
        <p>The plate detector used by the pipeline, a YOLOv8s trained from scratch, reached a validation mAP50 of ${final.best.map50.toFixed(3)}, with precision ${final.best.precision.toFixed(3)} and recall ${final.best.recall.toFixed(3)}. Across ${summary.videos} recorded test videos, the pipeline made ${summary.captures} plate captures, and ${summary.matched} of them (${formatPct(summary.match_rate)}) passed the OCR and YOLO cross-check. Every number on this site comes from that run.</p>

        <h3>What the experiments taught</h3>
        <p>Fine-tuning YOLOv8s and YOLOv8m from pre-trained weights on more than 13,000 images gave excellent training scores but poor detection on real footage. Training from scratch from the model definition (.yaml) worked as expected, so all final plate models were trained that way.</p>

        <h3>Limitations and next steps</h3>
        <p>Detection confidence drops when the camera is not level with the plate. The perspective correction uses fixed points, so plates filmed at close range leave too little margin after correction, and OCR misreads them. This is the main reason many captures end as a mismatch. The camera also has no zoom. Next steps are a camera with zoom, testing newer YOLO versions, better handling of lighting and occlusion, and connecting a live camera feed.</p>
      </section>

      <section class="panel">
        <h2>How a plate is read</h2>
        <ol class="pipeline">
          ${STEPS.map(([t, d]) => `<li><strong>${t}</strong><span>${d}</span></li>`).join("")}
        </ol>
      </section>

      <div class="grid-2">
        <section class="panel">
          <h2>Models</h2>
          <div class="table-wrap">
            <table class="runs">
              <thead><tr><th>File</th><th>Job</th><th>Classes</th></tr></thead>
              <tbody>
                <tr><th scope="row">plate.pt</th><td>Plate detection (YOLOv8s)</td><td>plate</td></tr>
                <tr><th scope="row">plate_letter.pt</th><td>Character detection</td><td>0–9, A–Y without I, O and Z, plus back_car</td></tr>
                <tr><th scope="row">yolov8n.pt</th><td>Vehicle detection (COCO)</td><td>only car is used</td></tr>
              </tbody>
            </table>
          </div>
        </section>
        <section class="panel">
          <h2>Built with</h2>
          <ul class="stack-list">
            <li>Python, PyTorch and Ultralytics YOLOv8 (8.1.27)</li>
            <li>OpenCV for video and image processing</li>
            <li>Tesseract OCR 5 through pytesseract</li>
            <li>This dashboard: HTML, CSS, JavaScript and Chart.js 4.4.1, published with GitHub Pages</li>
          </ul>
          <p><a class="button" href="${REPO}" target="_blank" rel="noopener">View the source on GitHub</a></p>
        </section>
      </div>
    </div>`;
}
