// About: the project in plain words (from the FYP report), the pipeline, the stack and the models.
import { formatPct, formatPlate } from "../format.js";

const REPO = "https://github.com/weikang14/license-plate-recognition";

const STEPS = [
  ["Find vehicles", "YOLOv8n (COCO) marks cars in every frame."],
  ["Find the plate", "A YOLOv8s plate detector trained for this project. Plates at 0.75 confidence or higher are captured, at most once every 1.5 seconds."],
  ["Straighten it", "The crop is converted to grayscale, resized and perspective-corrected so the characters sit level."],
  ["Read it twice", "Tesseract OCR reads the text, and a second YOLOv8 model detects each character on its own."],
  ["Cross-check", "The plate is verified only when both readers find the same characters, compared without order. The OCR text is kept as the result."],
];

export function render(root, { records, summary, metrics }) {
  const final = metrics.runs.find((r) => r.key === metrics.final);
  const noText = records.filter((r) => !r.plate).length;
  const verified = [...new Set(records.filter((r) => r.match).map((r) => r.plate))];
  const verifiedLine = verified.length === 1
    ? `All ${summary.matched} verified captures are the same plate, ${formatPlate(verified[0])}.`
    : `The ${summary.matched} verified captures cover ${verified.length} different plates.`;

  root.innerHTML = `
    <div class="about">
      <section class="panel prose">
        <h2>What this project does</h2>
        <p>This system reads Malaysian license plates from video. It finds a vehicle, finds its plate, and reads the plate with two independent methods. A plate is only recorded as verified when both methods agree, and the results are saved to a CSV file. It was built as a final year project to automate the plate logging that security staff still do by hand at the entrances of housing areas, malls and office buildings.</p>

        <h3>Background</h3>
        <p>Malaysia has about 22 million registered vehicles for about 34 million people, roughly 66 vehicles for every 100 residents and the second-highest rate in Southeast Asia after Brunei. Recording every entry and exit by hand is slow and inefficient, and plates are easier to identify a vehicle by than its make, model or colour. With an ageing population and younger workers less willing to take repetitive jobs, automating this work matters more.</p>

        <h3>Objective</h3>
        <p>Examine how license plate detection, built on computer vision and artificial intelligence, can automate the recording of vehicles entering and leaving, and look at the trends and prospects of the technology.</p>

        <h3>Results</h3>
        <p>The plate detector used by the pipeline, a YOLOv8s trained from scratch, reached a validation mAP50 of ${final.best.map50.toFixed(3)}, with precision ${final.best.precision.toFixed(3)} and recall ${final.best.recall.toFixed(3)}. Across ${summary.videos} recorded test videos, the pipeline made ${summary.captures} plate captures, and ${summary.matched} of them (${formatPct(summary.match_rate)}) passed the OCR and YOLO cross-check. Every number on this site comes from that run.</p>

        <h3>What the experiments taught</h3>
        <p>Training started from pre-trained YOLOv8n weights, which gave stable results. Fine-tuning YOLOv8s and YOLOv8m from pre-trained weights on more than 13,000 images then gave excellent training scores but poor detection on real footage. Training from scratch from the model definition (.yaml) worked as expected, so the final plate detectors were trained that way. For data that differs a lot from what the pre-trained weights saw, training from scratch is often the better choice.</p>

        <h3>Limitations and next steps</h3>
        <p>On these test videos both readers fail often: Tesseract returned no text on ${noText} of ${summary.captures} captures, and the character model frequently misses a character, so most captures end as a mismatch. ${verifiedLine} The project report adds that detection confidence drops when the camera is not level with the vehicle and plate, and that plates filmed at close range leave too little space after correction for OCR to read the letters. The camera also has no zoom. Next steps are reading the plate directly from the character model by ordering its boxes, a camera with zoom, trying YOLOv9, better handling of lighting, occlusion and real-time processing, and connecting a live camera feed.</p>
      </section>

      <section class="panel">
        <h2>How a plate is read</h2>
        <ol class="pipeline">
          ${STEPS.map(([t, d]) => `<li><strong>${t}</strong><span>${d}</span></li>`).join("")}
        </ol>
        <p class="pipeline-note"><strong>Known shortcut in step 5.</strong> The character model returns the characters it finds in a scrambled order, because YOLO lists detections by confidence rather than by position. During the project no way was found to put them back in reading order, so Tesseract supplies the order and the cross-check only compares which characters were found, not their order. Sorting the character boxes by their horizontal position would fix this, and is listed under next steps.</p>
      </section>

      <div class="grid-2">
        <section class="panel">
          <h2>Models</h2>
          <div class="table-wrap">
            <table class="runs">
              <thead><tr><th>File</th><th>Job</th><th>Classes</th></tr></thead>
              <tbody>
                <tr><th scope="row">plate.pt</th><td>Plate detection (YOLOv8s, trained from scratch)</td><td>plate</td></tr>
                <tr><th scope="row">plate_letter.pt</th><td>Character detection (YOLOv8n, fine-tuned)</td><td>0–9, A–Y without I, O and Z, plus back_car</td></tr>
                <tr><th scope="row">yolov8n.pt</th><td>Vehicle detection (COCO YOLOv8n, named car.pt in the project code)</td><td>only car is used</td></tr>
              </tbody>
            </table>
          </div>
        </section>
        <section class="panel">
          <h2>Built with</h2>
          <ul class="stack-list">
            <li>Python in an Anaconda environment</li>
            <li>Ultralytics YOLOv8 on PyTorch, trained with CUDA on a GPU (models trained with Ultralytics 8.1 and 8.2)</li>
            <li>OpenCV for video and image processing</li>
            <li>Tesseract OCR 5 through pytesseract</li>
            <li>This dashboard: HTML, CSS, JavaScript and Chart.js 4.4.1, published with GitHub Pages</li>
          </ul>
          <p><a class="button" href="${REPO}" target="_blank" rel="noopener">View the source on GitHub</a></p>
        </section>
      </div>
    </div>`;
}
