# License Plate Recognition (YOLOv8 + Tesseract)

**Live demo: https://weikang14.github.io/license-plate-recognition/**

A final year project that reads Malaysian license plates from video, plus a dashboard that shows what the pipeline found on 15 recorded test videos. Every number, snapshot and video on the site comes from running the pipeline on that footage.

![Overview](docs/screenshots/overview.png)

| Recognition records | Video playback |
|---|---|
| ![Records](docs/screenshots/records.png) | ![Playback](docs/screenshots/videos.png) |

![Model performance](docs/screenshots/models.png)

## How it works

1. **Find and track vehicles**: YOLOv8n (COCO) finds cars in every frame, and ByteTrack ([supervision](https://github.com/roboflow/supervision)) gives each car an id that follows it from frame to frame.
2. **Find the plate**: a YOLOv8s plate detector trained for this project. A plate only counts when it sits on a tracked car, so a photo of a plate held up to the camera is ignored. Plates with confidence 0.75 or higher are captured, at most once every 1.5 seconds.
3. **Prepare it**: the crop is converted to grayscale and enlarged to 800 pixels wide. The project's original fixed-point perspective correction was removed after testing: its fixed points crop or skew characters on most plates, and without it verified captures on the test videos rose from 10 to 47.
4. **Read it twice**: Tesseract OCR reads the text, and a second YOLOv8 model detects each character.
5. **Cross-check**: a plate is verified only when both readers find the same characters, compared without order, and the OCR text is kept as the result. Known shortcut: the character model returns its characters in a scrambled order (YOLO lists detections by confidence, not position), and during the project no way was found to put them back in reading order, so Tesseract supplies the order. Sorting the character boxes by horizontal position would fix this.

## Results

| | |
|---|---|
| Plate detector (YOLOv8s, trained from scratch) | mAP50 0.974, mAP50-95 0.651, precision 0.951, recall 0.957 |
| Test videos processed | 15 |
| Vehicles tracked (counted per video) | 223 |
| Plate captures (plates on a tracked car) | 149, from 45 vehicles |
| Verified by the OCR + YOLO cross-check | 47 (31.5%), from 32 vehicles |
| Average detection confidence | 0.825 |

The plate detector is reliable; reading the plate is the weak step. Tesseract returned no text on 35 of 149 captures, and the character model often misses a character, so the strict cross-check still rejects about two thirds of captures. The verified captures are two plates, JHB 2149 and VEW 6482, seen in 10 of the 15 videos. One capture of JHB 2149 passed as 2149JHB, because the cross-check ignores character order (the known shortcut in step 5). Close-up shots where the whole car is not in view are skipped on purpose: without a tracked car there is nothing to prove the plate is on a real vehicle. Removing the fixed perspective correction raised verified captures from 10 (6.7%) to 47 (31.5%) on the same footage; the project report had already noted that it left too little margin around plates filmed at close range. Every mismatch is listed on the site with both readings side by side.

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
