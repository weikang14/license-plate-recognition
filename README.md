# License Plate Recognition (YOLOv8 + Tesseract)

**Live demo: https://weikang14.github.io/license-plate-recognition/**

A final year project that reads Malaysian license plates from video, plus a dashboard that shows what the pipeline found on 15 recorded test videos. Every number, snapshot and video on the site comes from running the pipeline on that footage.

![Overview](docs/screenshots/overview.png)

| Recognition records | Video playback |
|---|---|
| ![Records](docs/screenshots/records.png) | ![Playback](docs/screenshots/videos.png) |

![Model performance](docs/screenshots/models.png)

## How it works

1. **Find vehicles**: YOLOv8n (COCO) marks cars in each frame.
2. **Find the plate**: a YOLOv8s plate detector trained for this project. Plates detected with confidence 0.75 or higher are captured, at most once every 1.5 seconds.
3. **Straighten it**: the crop is converted to grayscale, resized and perspective-corrected.
4. **Read it twice**: Tesseract OCR reads the text, and a second YOLOv8 model detects each character.
5. **Cross-check**: a plate is verified only when both readers find the same characters. The OCR text is kept because it preserves character order.

## Results

| | |
|---|---|
| Plate detector (YOLOv8s, trained from scratch) | mAP50 0.974, mAP50-95 0.651, precision 0.951, recall 0.957 |
| Test videos processed | 15 |
| Plate captures | 181 |
| Verified by the OCR + YOLO cross-check | 11 (6.1%) |
| Average detection confidence | 0.82 |

The plate detector is reliable; reading the plate is the weak step. Tesseract returned no text on 67 of 181 captures, and the character model frequently misses a character, so the strict cross-check rejects most captures. All 11 verified captures are the same plate, JHB 2149, seen in four videos. The project report notes that the fixed perspective correction leaves too little margin around plates filmed at close range, which hurts OCR. Every mismatch is listed on the site with both readings side by side.

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

Needs Python with `ultralytics`, `opencv-python` and `pytesseract`, plus [Tesseract OCR](https://github.com/UB-Mannheim/tesseract) and [ffmpeg](https://ffmpeg.org/). Point `TESSERACT_CMD` and `FFMPEG` at them if they are not on `PATH`. The raw test videos (`videos/`) and the training runs (`training_results/`) are kept locally and are not in this repository; tests that need them are skipped when they are missing.

```
python tools/build_site_data.py --videos path/to/test/videos --out web/data --playback 4
python -m pytest tests
node --test "tests/web/*.test.mjs"
```
