// About: the project in plain words (from the FYP report), the pipeline, the stack and the models.
import { formatPct, formatPlate } from "../format.js";

const REPO = "https://github.com/weikang14/license-plate-recognition";

const STEPS = [
  ["Find and track vehicles", "YOLOv8n (COCO) finds vehicles in every frame (cars, plus buses and trucks, because a car close to the camera is often labelled truck), and ByteTrack (supervision) gives each one an id that follows it from frame to frame and survives gaps of up to 3 seconds."],
  ["Find the plate", "A YOLOv8s plate detector trained for this project. A plate only counts when it sits on a tracked car, so a photo of a plate held up to the camera is ignored. Plates at 0.75 confidence or higher are captured, at most once every 1.5 seconds."],
  ["Prepare it", "The crop is converted to grayscale and enlarged to 800 pixels wide. The project's fixed perspective correction is no longer used (see What the experiments taught)."],
  ["Read it twice", "Tesseract OCR reads the text, and a second YOLOv8 model detects each character on its own. If the two readings differ, Tesseract reads the plate once more as a single line of text."],
  ["Cross-check", "Both readings are put in left-to-right order, and the plate is verified only when they are identical."],
];

function describeVerified(plates, n) {
  const names = [...new Set(plates)].map(formatPlate);
  if (!names.length) return "No capture passed the cross-check.";
  if (names.length === 1) return `All ${n} verified captures are the same plate, ${names[0]}.`;
  return `The ${n} verified captures are ${names.length} plates, ${names.slice(0, -1).join(", ")} and ${names.at(-1)}.`;
}

export function render(root, { records, summary, metrics }) {
  const final = metrics.runs.find((r) => r.key === metrics.final);
  const noText = records.filter((r) => !r.plate).length;
  const verifiedLine = describeVerified(records.filter((r) => r.match).map((r) => r.plate), summary.matched);
  const vehicles = (rs) => new Set(rs.map((r) => `${r.video}#${r.car_id}`)).size;
  const perVehicle = `Counted per vehicle, ${vehicles(records.filter((r) => r.match))} of the ${vehicles(records)} vehicles with a plate capture got at least one verified reading.`;

  root.innerHTML = `
    <div class="about">
      <section class="panel prose">
        <h2>What this project does</h2>
        <p>This system reads Malaysian license plates from video. It finds a vehicle, keeps tracking it, finds the plate on it, and reads the plate with two independent methods. A plate is only recorded as verified when both methods agree, and the results are saved to a CSV file. It was built as a final year project to automate the plate logging that security staff still do by hand at the entrances of housing areas, malls and office buildings.</p>

        <h3>Background</h3>
        <p>Malaysia has about 22 million registered vehicles for about 34 million people, roughly 66 vehicles for every 100 residents and the second-highest rate in Southeast Asia after Brunei. Recording every entry and exit by hand is slow and inefficient, and plates are easier to identify a vehicle by than its make, model or colour. With an ageing population and younger workers less willing to take repetitive jobs, automating this work matters more.</p>

        <h3>Objective</h3>
        <p>Examine how license plate detection, built on computer vision and artificial intelligence, can automate the recording of vehicles entering and leaving, and look at the trends and prospects of the technology.</p>

        <h3>Results</h3>
        <p>The plate detector used by the pipeline, a YOLOv8s trained from scratch, reached a validation mAP50 of ${final.best.map50.toFixed(3)}, with precision ${final.best.precision.toFixed(3)} and recall ${final.best.recall.toFixed(3)}. Across ${summary.videos} recorded test videos, the pipeline tracked ${summary.vehicles} vehicles and made ${summary.captures} plate captures, and ${summary.matched} of them (${formatPct(summary.match_rate)}) passed the OCR and YOLO cross-check. ${perVehicle} Every number on this site comes from that run.</p>

        <h3>What the experiments taught</h3>
        <p>Training started from pre-trained YOLOv8n weights, which gave stable results. Fine-tuning YOLOv8s and YOLOv8m from pre-trained weights on more than 13,000 images then gave excellent training scores but poor detection on real footage. Training from scratch from the model definition (.yaml) worked as expected, so the final plate detectors were trained that way. For data that differs a lot from what the pre-trained weights saw, training from scratch is often the better choice.</p>
        <p>After the project, the reading step was tested on the saved plate snapshots. The original perspective correction used four fixed points that suit one plate position, so on most plates it cropped or skewed characters for both readers. Rerunning all 15 test videos without it raised verified captures from 10 to 47 (6.7% to 31.5%), so it was removed. Comparing the two readings in left-to-right order, with the character model at confidence 0.4 and a second Tesseract pass that reads the plate as a single line, then raised them to 60 (40.3%), with no plate passing out of order. Limiting OCR to the letters used on Malaysian plates (no I, O or Z) made no difference to the cross-check.</p>

        <h3>Limitations and next steps</h3>
        <p>On these test videos both readers fail often: Tesseract returned no text on ${noText} of ${summary.captures} captures, and the character model frequently misses a character, so most captures end as a mismatch. ${verifiedLine} The project report adds that detection confidence drops when the camera is not level with the vehicle and plate, and that plates filmed at close range left too little space after correction for OCR to read the letters; that correction has since been removed. The tracker matches vehicles by position only, so when one car passes close in front of a parked one it can take over the parked car's id. The camera also has no zoom. Next steps are a stronger OCR step, which is now the weaker reader, support for two-row plates, a camera with zoom, trying YOLOv9, better handling of lighting, occlusion and real-time processing, and connecting a live camera feed.</p>
      </section>

      <section class="panel">
        <h2>How a plate is read</h2>
        <ol class="pipeline">
          ${STEPS.map(([t, d]) => `<li><strong>${t}</strong><span>${d}</span></li>`).join("")}
        </ol>
        <p class="pipeline-note"><strong>Fixed after the project: reading order.</strong> The character model returns its characters in confidence order, not reading order, and during the project no way was found to put them back in order. The original cross-check therefore only compared which characters were found, so a reading like 2149JHB could pass for JHB 2149. Both readings are now put in left-to-right order, the character boxes by their horizontal position and Tesseract's words by theirs, and compared exactly, with the character model at confidence 0.4 instead of 0.5. This works for single-row plates; two-row plates would need the boxes grouped by row first.</p>
      </section>

      <div class="grid-2">
        <section class="panel">
          <h2>Models</h2>
          <div class="table-wrap">
            <table class="runs wrap">
              <thead><tr><th>File</th><th>Job</th><th>Classes</th></tr></thead>
              <tbody>
                <tr><th scope="row">plate.pt</th><td>Plate detection (YOLOv8s, trained from scratch)</td><td>plate</td></tr>
                <tr><th scope="row">plate_letter.pt</th><td>Character detection (YOLOv8n, fine-tuned)</td><td>0–9, A–Y without I, O and Z, plus back_car</td></tr>
                <tr><th scope="row">yolov8n.pt</th><td>Vehicle detection (COCO YOLOv8n, named car.pt in the project code)</td><td>car, bus and truck are used</td></tr>
              </tbody>
            </table>
          </div>
        </section>
        <section class="panel">
          <h2>Built with</h2>
          <ul class="stack-list">
            <li>Python in an Anaconda environment</li>
            <li>Ultralytics YOLOv8 on PyTorch, trained with CUDA on a GPU (models trained with Ultralytics 8.1 and 8.2)</li>
            <li>supervision (ByteTrack) for vehicle tracking</li>
            <li>OpenCV for video and image processing</li>
            <li>Tesseract OCR 5 through pytesseract</li>
            <li>This dashboard: HTML, CSS, JavaScript and Chart.js 4.4.1, published with GitHub Pages</li>
          </ul>
          <h2>Who built what</h2>
          <ul class="stack-list">
            <li>Dataset, model training and the recognition pipeline: weikang14's final year project</li>
            <li>This dashboard, the tests and the changes made after the project: built with Claude Code, an AI coding assistant</li>
          </ul>
          <p><a class="button" href="${REPO}" target="_blank" rel="noopener">View the source on GitHub</a></p>
        </section>
      </div>
    </div>`;
}
