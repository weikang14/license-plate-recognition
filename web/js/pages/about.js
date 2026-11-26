// About: the project in plain words (from the FYP report), the pipeline, the stack and the models.
import { confirmedCaptures, formatPct, formatPlate } from "../format.js";

const REPO = "https://github.com/weikang14/license-plate-recognition";

const STEPS = [
  ["Find and track vehicles", "YOLOv8n (COCO) finds vehicles in every frame (cars, plus buses and trucks, because a car close to the camera is often labelled truck), and ByteTrack (supervision) gives each one an id that follows it from frame to frame and survives gaps of up to 3 seconds."],
  ["Find the plate", "A YOLOv8s plate detector trained for this project. A plate only counts when it sits on a tracked car, so a photo of a plate held up to the camera is ignored. Plates at 0.75 confidence or higher are captured 7 times a second, one per car, until the same reading of the car's plate has been verified twice; the tracked car then keeps that reading and is not read again."],
  ["Prepare it", "The crop is converted to grayscale and enlarged to 800 pixels wide. The project's fixed perspective correction is no longer used (see What the experiments taught)."],
  ["Read it twice", "Tesseract OCR reads the text, and a second YOLOv8 model detects each character on its own. If the two readings differ, Tesseract reads the plate once more as a single line of text."],
  ["Cross-check", "Both readings are put in left-to-right order, and the capture is verified only when they are identical and hold both letters and a number, as Malaysian plates do, including the series sold at auction (W 1, MALAYSIA 8). A vehicle's plate is confirmed when the same reading is verified twice, because both readers sometimes drop the same characters."],
];

function describeConfirmed(plates) {
  const names = [...new Set(plates)].map(formatPlate);
  if (!names.length) return "No vehicle had the same reading verified twice, so no plate was confirmed.";
  if (names.length === 1) return `Every confirmed plate is ${names[0]}.`;
  return `The confirmed plates are ${names.slice(0, -1).join(", ")} and ${names.at(-1)}.`;
}

export function render(root, { records, summary, metrics }) {
  const final = metrics.runs.find((r) => r.key === metrics.final);
  const noText = records.filter((r) => !r.plate).length;
  const confirmed = confirmedCaptures(records);
  const verifiedLine = describeConfirmed(confirmed.map((r) => r.plate));
  const vehicles = new Set(records.map((r) => `${r.video}#${r.car_id}`)).size;
  const perVehicle = `Counted per vehicle, ${confirmed.length} of the ${vehicles} vehicles with a plate capture had their plate confirmed.`;

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
        <p>The plate detector used by the pipeline, a YOLOv8s trained from scratch, reached a validation mAP50 of ${final.best.map50.toFixed(3)}, with precision ${final.best.precision.toFixed(3)} and recall ${final.best.recall.toFixed(3)}. Across ${summary.videos} recorded videos (test videos and dashcam clips), the pipeline tracked ${summary.vehicles} vehicles and made ${summary.captures} plate captures, and ${summary.matched} of them (${formatPct(summary.match_rate)}) passed the OCR and YOLO cross-check. ${perVehicle} Every number on this site comes from that run.</p>

        <h3>What the experiments taught</h3>
        <p>Training started from pre-trained YOLOv8n weights, which gave stable results. Fine-tuning YOLOv8s and YOLOv8m from pre-trained weights on more than 13,000 images then gave excellent training scores but poor detection on real footage. Training from scratch from the model definition (.yaml) worked as expected, so the final plate detectors were trained that way. For data that differs a lot from what the pre-trained weights saw, training from scratch is often the better choice.</p>
        <p>After the project, the reading step was tested on the saved plate snapshots. The original perspective correction used four fixed points that suit one plate position, so on most plates it cropped or skewed characters for both readers. Rerunning all 15 test videos without it raised verified captures from 10 to 47 (6.7% to 31.5%), so it was removed. Comparing the two readings in left-to-right order, with the character model at confidence 0.4 and a second Tesseract pass that reads the plate as a single line, then raised them to 60 (40.3%), with no plate passing out of order. Keeping each vehicle's id through gaps of up to 3 seconds then lowered the vehicle count from 223 to 184, because a car that stays in view is no longer counted twice. Reading each tracked car only until its plate is verified then cut captures from 156 to 67 with the same 34 vehicles verified, because a verified car had been read again every 1.5 seconds while it stayed in view. Dropping the 1.5-second gap and reading plates 7 times a second (the original code read every 7th frame, about 4 times a second) then raised the vehicles with a verified plate from 34 to 42 of 48. With more reads, both readers once agreed on a wrong reading (VEH 6482 for VEW 6482), so a plate is now confirmed only when the same reading is verified twice; all 42 confirmed plates are correct. Requiring the agreed reading to hold both letters and a number changed nothing on these videos, where every verified reading already was one; it is there for longer dashcam footage, where both readers agreed on a car badge read as 7 and on a plate read without its letters. Limiting OCR to the letters used on Malaysian plates (no I, O or Z) made no difference to the cross-check.</p>

        <h3>Limitations and next steps</h3>
        <p>On these videos both readers fail often: Tesseract returned no text on ${noText} of ${summary.captures} captures, and the character model frequently misses a character, so many captures end as a mismatch. ${verifiedLine} The project report adds that detection confidence drops when the camera is not level with the vehicle and plate, and that plates filmed at close range left too little space after correction for OCR to read the letters; that correction has since been removed. The tracker matches vehicles by position only, so when one car passes close in front of a parked one it can take over the parked car's id. The camera also has no zoom.</p>
        <p>Next steps:</p>
        <ol class="next-steps stack-list">
          <li>Replace Tesseract, the character model and the cross-check with a single recognition model that reads the whole plate in one pass. Running two readers on every capture, and Tesseract a second time when they disagree, is the slowest part of reading a plate, and Tesseract cannot run on a GPU.</li>
          <li>Support two-row plates. The left-to-right ordering only works for plates written on one line.</li>
          <li>Stop a car that passes in front of a parked one from taking over the parked car's id.</li>
        </ol>
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
