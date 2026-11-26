# License Plate Recognition (YOLOv8 + Tesseract)

**Live demo: https://weikang14.github.io/license-plate-recognition/**

A final year project that reads Malaysian license plates from video, plus a dashboard that shows what the pipeline found on 15 recorded test videos and 14 dashcam clips. Every number, snapshot and video on the site comes from running the pipeline on that footage.

![Overview](docs/screenshots/overview.png)

| Recognition records | Highlight videos |
|---|---|
| ![Records](docs/screenshots/records.png) | ![Highlight videos](docs/screenshots/videos.png) |

![Model performance](docs/screenshots/models.png)

## How it works

1. **Find and track vehicles**: YOLOv8n (COCO) finds vehicles in every frame (cars, plus buses and trucks, because a car close to the camera is often labelled truck), and ByteTrack ([supervision](https://github.com/roboflow/supervision)) gives each one an id that follows it from frame to frame and survives gaps of up to 3 seconds, so a car that stays in view is not counted twice.
2. **Find the plate**: a YOLOv8s plate detector trained for this project. A plate only counts when it sits on a tracked car, so a photo of a plate held up to the camera is ignored. Plates with confidence 0.75 or higher are captured 7 times a second, one per car, until the same reading of the car's plate has been verified twice; the tracked car then keeps that reading and is not read again.
3. **Prepare it**: the crop is converted to grayscale and enlarged to 800 pixels wide. The project's original fixed-point perspective correction was removed after testing: its fixed points crop or skew characters on most plates, and without it verified captures on the test videos rose from 10 to 47.
4. **Read it twice**: Tesseract OCR reads the text (page mode 6), and a second YOLOv8 model detects each character at confidence 0.4. If the two readings differ, Tesseract reads the plate once more as a single raw line (page mode 13).
5. **Cross-check**: both readings are put in left-to-right order, Tesseract's words and the character boxes by their horizontal position, and the capture is verified only when they are identical and hold both letters and a number, as Malaysian plates do, including the series sold at auction (W 1, MALAYSIA 8). A vehicle's plate is confirmed when the same reading is verified twice: with several reads per second, both readers sometimes drop the same characters and agree on a wrong reading once. The original project compared the characters without order, because the character model lists them by confidence, not position, so a reading like 2149JHB could pass for JHB 2149. This works for single-row plates; two-row plates would need the boxes grouped by row first.

## Results

Plate detector (YOLOv8s, trained from scratch): mAP50 0.974, mAP50-95 0.651, precision 0.951, recall 0.957.

| | Test videos | Dashcam | All |
|---|---|---|---|
| Videos | 15 | 14 | 29 |
| Vehicles tracked (counted per video) | 185 | 3223 | 3408 |
| Plate captures (plates on a tracked car) | 321, from 48 vehicles | 2200, from 195 vehicles | 2521, from 243 vehicles |
| Verified by the OCR + YOLO cross-check | 85 (26.5%), from 42 vehicles | 24 (1.1%), from 15 vehicles | 109 (4.3%), from 57 vehicles |
| Vehicles with a confirmed plate (same reading verified twice) | 42 of 48 | 9 of 195 | 51 of 243 |
| Average detection confidence | 0.796 | 0.773 | 0.775 |

The plate detector is reliable; reading the plate is the weak step. Tesseract returned no text on 891 of 2521 captures, and the character model often misses a character, so the strict cross-check rejects about three in four captures on the test videos and almost all on the dashcam clips; each vehicle is read again until its plate is confirmed, so most captures are retries. On the test videos the confirmed plates are two plates, JHB 2149 and VEW 6482, seen in 12 of the 15 videos. The dashcam clips, added after the project and run with the same pipeline and settings, are ordinary driving: most plates are small, far away or pass quickly, so only 9 of the 195 vehicles with a plate capture had their plate confirmed, all 9 correctly (checked by eye against the snapshots). The site's highlight videos show every video with a verified plate, cut to the stretches where plates are being read. Close-up shots where the whole car is not in view are skipped on purpose: without a tracked car there is nothing to prove the plate is on a real vehicle. On the test videos, removing the fixed perspective correction raised verified captures from 10 (6.7%) to 47 (31.5%); the project report had already noted that it left too little margin around plates filmed at close range. Comparing the readings in left-to-right order, with the character model at confidence 0.4 and a second Tesseract pass as a single line, then raised them to 60 (40.3%) and removed the one capture that had passed out of order (2149JHB). Keeping each vehicle's id through gaps of up to 3 seconds then lowered the vehicle count from 223 to 184, because a car that stays in view is no longer counted twice. Reading each tracked car only until its plate is verified then cut captures from 156 to 67 with the same 34 vehicles verified, because a verified car had been read again every 1.5 seconds while it stayed in view. Dropping the 1.5-second gap and reading plates 7 times a second (the original code read every 7th frame, about 4 times a second) then raised the vehicles with a verified plate from 34 to 42 of 48. With more reads, both readers once agreed on a wrong reading (VEH 6482 for VEW 6482), so a plate is now confirmed only when the same reading is verified twice; all 42 confirmed plates are correct. Requiring the agreed reading to hold both letters and a number changed nothing on these videos, where every verified reading already was one; it is there for longer dashcam footage, where both readers agreed on a car badge read as 7 and on a plate read without its letters. The published test-video run, built on the GPU, has 185 vehicles (one more than on the CPU, from float rounding in one video), 321 captures, 85 verified (26.5%) and 42 vehicles with a confirmed plate. Every mismatch is listed on the site with both readings side by side.

## Speed

Time per stage on one 1080p frame, measured with `tools/benchmark.py` (also on the site's Model performance page):

| Stage | Runs | Laptop CPU (Core i5-12450H) | Laptop GPU (RTX 4060) |
|---|---|---|---|
| Vehicle detection (YOLOv8n) | every frame | 75.8 ms | 10.6 ms |
| Plate detection (YOLOv8s) | 7 frames a second | 120.7 ms | 10.4 ms |
| Character detection (YOLOv8n) | per capture | 58.4 ms | 13.8 ms |
| Tesseract OCR | per capture, twice if psm 13 is needed | 152.1 ms | CPU only |
| **Frames per second** (vehicles + plates, 30 fps video) | | **10 fps** | **77 fps** |

On the CPU the pipeline runs at about 10 fps, below the cameras' 25-30 fps, so the site data is built on the GPU. There the detectors run at about 77 fps one frame at a time (the pipeline sends them 8 frames per call, about 2.5 times faster in a quick test), and Tesseract, which has no GPU version and runs 7 times a second for each car not yet verified, becomes the slowest step. CPU: PyTorch 2.3.1 with Ultralytics 8.1.27; GPU: PyTorch 2.7.0 with Ultralytics 8.3.127.

## Next steps

1. **One recognition model instead of two readers.** Replace Tesseract, the character model and the cross-check with a single model that reads the whole plate in one pass. Running two readers on every capture, and Tesseract a second time when they disagree, is the slowest part of reading a plate (see Speed above), and Tesseract cannot run on a GPU.
2. **Two-row plates.** The left-to-right ordering only works for plates written on one line.
3. **Keep vehicle ids through occlusion.** The tracker matches vehicles by position only, so a car that passes close in front of a parked one can take over the parked car's id.

## Who built what

- **weikang14, final year project:** the dataset, training the plate and character detectors, and the recognition pipeline: vehicle, plate, two readers and the cross-check (`code/Final_License_Plate_Detection.ipynb`).
- **Built with [Claude Code](https://claude.com/claude-code), an AI coding assistant, after the project:** this dashboard, moving the notebook's pipeline into `tools/` with tests and the speed benchmark, and the post-project changes described above (vehicle tracking, removing the fixed perspective correction, the ordered cross-check). Earlier commits in this repository show "Claude Code" as the author name for this reason.

## Repository layout

```
code/Final_License_Plate_Detection.ipynb   original FYP notebook
models/                                    plate.pt, plate_letter.pt, yolov8n.pt
tools/lpr_pipeline.py                      the notebook's pipeline as a module
tools/build_site_data.py                   runs the pipeline on a folder of videos, writes the site data for it
tools/merge_site_data.py                   combines those folders into web/data/ and cuts the highlight videos
tools/site_data.py                         metrics and summary for the dashboard
tools/benchmark.py                         times each stage on CPU or GPU, writes web/speed.json
tools/serve.py                             local static server (supports video seeking)
web/                                       the dashboard (plain HTML, CSS, JS and Chart.js)
tests/                                     pytest and node tests
```

## Run the dashboard locally

```
python tools/serve.py
```

Then open http://127.0.0.1:8765. (`python -m http.server` also works, but it can't seek inside videos.)

## Regenerate the site data

Needs Python 3.11 with the packages in `requirements.txt` (`pip install -r requirements.txt`), plus [Tesseract OCR](https://github.com/UB-Mannheim/tesseract) and [ffmpeg](https://ffmpeg.org/). Point `TESSERACT_CMD` and `FFMPEG` at them if they are not on `PATH`. The raw videos (`videos/`: test videos and dashcam clips) and the training runs (`training_results/`) are kept locally and are not in this repository; tests that need them are skipped when they are missing.

```
python tools/build_site_data.py --videos videos/testing_video --out videos/built/testing_video --playback 15
python tools/build_site_data.py --videos videos/raw_video --out videos/built/raw_video --playback 14
python tools/merge_site_data.py --out web/data --cut "Test videos=5,8,15" "Test videos=videos/built/testing_video" "Dashcam=videos/built/raw_video"
python tools/benchmark.py --device cpu      # or --device 0 for the first CUDA GPU
python -m pytest tests
node --test "tests/web/*.test.mjs"
```
