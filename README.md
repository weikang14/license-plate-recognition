# License Plate Recognition (YOLOv8 + Tesseract)

**Live demo: https://weikang14.github.io/license-plate-recognition/**

A final year project that reads Malaysian license plates from video, plus a dashboard that shows what the pipeline found on 15 recorded test videos. Every number, snapshot and video on the site comes from running the pipeline on that footage.

![Overview](docs/screenshots/overview.png)

| Recognition records | Video playback |
|---|---|
| ![Records](docs/screenshots/records.png) | ![Playback](docs/screenshots/videos.png) |

![Model performance](docs/screenshots/models.png)

## How it works

1. **Find and track vehicles**: YOLOv8n (COCO) finds vehicles in every frame (cars, plus buses and trucks, because a car close to the camera is often labelled truck), and ByteTrack ([supervision](https://github.com/roboflow/supervision)) gives each one an id that follows it from frame to frame and survives gaps of up to 3 seconds, so a car that stays in view is not counted twice.
2. **Find the plate**: a YOLOv8s plate detector trained for this project. A plate only counts when it sits on a tracked car, so a photo of a plate held up to the camera is ignored. Plates with confidence 0.75 or higher are captured, at most once every 1.5 seconds.
3. **Prepare it**: the crop is converted to grayscale and enlarged to 800 pixels wide. The project's original fixed-point perspective correction was removed after testing: its fixed points crop or skew characters on most plates, and without it verified captures on the test videos rose from 10 to 47.
4. **Read it twice**: Tesseract OCR reads the text (page mode 6), and a second YOLOv8 model detects each character at confidence 0.4. If the two readings differ, Tesseract reads the plate once more as a single raw line (page mode 13).
5. **Cross-check**: both readings are put in left-to-right order, Tesseract's words and the character boxes by their horizontal position, and the plate is verified only when they are identical. The original project compared the characters without order, because the character model lists them by confidence, not position, so a reading like 2149JHB could pass for JHB 2149. This works for single-row plates; two-row plates would need the boxes grouped by row first.

## Results

| | |
|---|---|
| Plate detector (YOLOv8s, trained from scratch) | mAP50 0.974, mAP50-95 0.651, precision 0.951, recall 0.957 |
| Test videos processed | 15 |
| Vehicles tracked (counted per video) | 223 |
| Plate captures (plates on a tracked car) | 149, from 45 vehicles |
| Verified by the OCR + YOLO cross-check | 60 (40.3%), from 33 vehicles |
| Average detection confidence | 0.825 |

The plate detector is reliable; reading the plate is the weak step. Tesseract returned no text on 33 of 149 captures, and the character model often misses a character, so the strict cross-check still rejects about three in five captures. The verified captures are two plates, JHB 2149 and VEW 6482, seen in 10 of the 15 videos. Close-up shots where the whole car is not in view are skipped on purpose: without a tracked car there is nothing to prove the plate is on a real vehicle. Removing the fixed perspective correction raised verified captures from 10 (6.7%) to 47 (31.5%) on the same footage; the project report had already noted that it left too little margin around plates filmed at close range. Comparing the readings in left-to-right order, with the character model at confidence 0.4 and a second Tesseract pass as a single line, then raised them to 60 (40.3%) and removed the one capture that had passed out of order (2149JHB). Every mismatch is listed on the site with both readings side by side.

## Repository layout

```
code/Final_License_Plate_Detection.ipynb   original FYP notebook
models/                                    plate.pt, plate_letter.pt, yolov8n.pt
tools/lpr_pipeline.py                      the notebook's pipeline as a module
tools/build_site_data.py                   runs the pipeline on test videos, writes web/data/
tools/site_data.py                         metrics and summary for the dashboard
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

Needs Python with `ultralytics`, `supervision`, `opencv-python` and `pytesseract`, plus [Tesseract OCR](https://github.com/UB-Mannheim/tesseract) and [ffmpeg](https://ffmpeg.org/). Point `TESSERACT_CMD` and `FFMPEG` at them if they are not on `PATH`. The raw test videos (`videos/`) and the training runs (`training_results/`) are kept locally and are not in this repository; tests that need them are skipped when they are missing.

```
python tools/build_site_data.py --videos path/to/test/videos --out web/data --playback 4
python -m pytest tests
node --test "tests/web/*.test.mjs"
```
