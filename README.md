# License Plate Recognition (YOLOv8 + Tesseract + OXR)

**Live demo: https://weikang14.github.io/license-plate-recognition/**

A final year project that reads Malaysian license plates from video, plus a dashboard that shows what the pipeline found on 15 recorded test videos and 14 dashcam clips. Every number, snapshot and video on the site comes from running the pipeline on that footage.

![Overview](docs/screenshots/overview.png)

| Recognition records | Highlight videos |
|---|---|
| ![Records](docs/screenshots/records.png) | ![Highlight videos](docs/screenshots/videos.png) |

![Model performance](docs/screenshots/models.png)

Download the highlight videos (MP4, 1280x720): [test videos](https://github.com/weikang14/license-plate-recognition/raw/main/web/data/videos/reel-1.mp4) (28 MB) and [dashcam clips](https://github.com/weikang14/license-plate-recognition/raw/main/web/data/videos/reel-2.mp4) (31 MB). To run the project yourself, clone this repository or [download it as a zip](https://github.com/weikang14/license-plate-recognition/archive/refs/heads/main.zip) (about 430 MB: code, models and five test videos) and see [Download and run](#download-and-run).

## How it works

1. **Find and track vehicles**: YOLOv8n (COCO) finds vehicles in every frame (cars, plus buses and trucks, because a car close to the camera is often labelled truck), and ByteTrack ([supervision](https://github.com/roboflow/supervision)) gives each one an id that follows it from frame to frame and survives gaps of up to 3 seconds, so a car that stays in view is not counted twice.
2. **Find the plate**: a YOLOv8s plate detector trained for this project. A plate only counts when it sits on a tracked car, so a photo of a plate held up to the camera is ignored. The plate detector runs on 7 evenly spaced frames in every second of video (about every 4th frame), and plates with confidence 0.75 or higher are captured on those frames, one per car, until the same reading of the car's plate has been verified twice; the tracked car then keeps that reading and is not read again.
3. **Prepare it**: for Tesseract and the character model, the crop is converted to grayscale and enlarged to 800 pixels wide; OXR gets the colour crop. The project's original fixed-point perspective correction was removed after testing: its fixed points crop or skew characters on most plates, and without it verified captures on the test videos rose from 10 to 47.
4. **Read it three times**: Tesseract OCR reads the text (page mode 6), and a second YOLOv8 model, trained for this project, detects each character at confidence 0.4. Since the project, a third reader has been added: [OXR-1.0](https://github.com/SoMarkAI/OXR), a 1.3-billion-parameter document-reading model from SoMark, used as published. If Tesseract agrees with neither of the others, it reads the plate once more as a single raw line (page mode 13).
5. **Cross-check**: the readings are put in left-to-right order (Tesseract's words and the character boxes by their horizontal position; OXR returns text in reading order), and the capture is verified when two of the three are identical and hold both letters and a number, as Malaysian plates do, including the series sold at auction (W 1, MALAYSIA 8). A vehicle's plate is confirmed when the same reading is verified twice: with several reads per second, two readers sometimes make the same mistake once. The original project compared the characters without order, because the character model lists them by confidence, not position, so a reading like 2149JHB could pass for JHB 2149. This works for single-row plates; the character model would need its boxes grouped by row first for two-row plates, which OXR already reads.

## Results

Plate detector (YOLOv8s, trained from scratch): mAP50 0.974, mAP50-95 0.651, precision 0.951, recall 0.957.

| | Test videos | Dashcam | All |
|---|---|---|---|
| Videos | 15 | 14 | 29 |
| Vehicles tracked (counted per video) | 184 | 3219 | 3403 |
| Plate captures (plates on a tracked car) | 163, from 48 vehicles | 1621, from 195 vehicles | 1784, from 243 vehicles |
| Verified by the cross-check (2 of 3 readers agree) | 90 (55.2%), from 45 vehicles | 69 (4.3%), from 38 vehicles | 159 (8.9%), from 83 vehicles |
| Vehicles with a confirmed plate (same reading verified twice) | 44 of 48 | 26 of 195 | 70 of 243 |
| Average detection confidence | 0.783 | 0.773 | 0.774 |

The plate detector is reliable; reading the plate is the weak step. Tesseract returned no text on 812 of 1784 captures, and the character model often misses a character, so about half the captures on the test videos fail the cross-check, and almost all on the dashcam clips; each vehicle is read again until its plate is confirmed, so most captures are retries. On the test videos the confirmed plates are two plates, JHB 2149 and VEW 6482, seen in 12 of the 15 videos. The dashcam clips, added after the project and run with the same pipeline and settings, are ordinary driving: most plates are small, far away or pass quickly, so only 26 of the 195 vehicles with a plate capture had their plate confirmed, all 26 correctly (checked by eye against the snapshots). The site's highlight videos show every video with a verified plate, cut to the stretches where plates are being read. Close-up shots where the whole car is not in view are skipped on purpose: without a tracked car there is nothing to prove the plate is on a real vehicle. On the test videos, removing the fixed perspective correction raised verified captures from 10 (6.7%) to 47 (31.5%); the project report had already noted that it left too little margin around plates filmed at close range. Comparing the readings in left-to-right order, with the character model at confidence 0.4 and a second Tesseract pass as a single line, then raised them to 60 (40.3%) and removed the one capture that had passed out of order (2149JHB). Keeping each vehicle's id through gaps of up to 3 seconds then lowered the vehicle count from 223 to 184, because a car that stays in view is no longer counted twice. Reading each tracked car only until its plate is verified then cut captures from 156 to 67 with the same 34 vehicles verified, because a verified car had been read again every 1.5 seconds while it stayed in view. Dropping the 1.5-second gap and reading plates 7 times a second (the original code read every 7th frame, about 4 times a second) then raised the vehicles with a verified plate from 34 to 42 of 48. With more reads, both readers once agreed on a wrong reading (VEH 6482 for VEW 6482), so a plate is now confirmed only when the same reading is verified twice; all 42 confirmed plates are correct. Requiring the agreed reading to hold both letters and a number changed nothing on these videos, where every verified reading already was one; it is there for longer dashcam footage, where both readers agreed on a car badge read as 7 and on a plate read without its letters. With those two readers, the published test-video run had 321 captures, 85 verified (26.5%) and 42 vehicles with a confirmed plate, and the dashcam clips 9. Adding OXR as a third reader, with a capture verified when two of the three agree, raised verified test-video captures to 90 (55.2%) and the vehicles with a confirmed plate to 44 of 48, all correct, while halving the captures to 163, because cars are confirmed sooner and then no longer read; on the dashcam clips the confirmed vehicles rose from 9 to 26. On the captures of the confirmed vehicles, OXR read the whole plate exactly in 78% of them, Tesseract in 37% and the character model in 15%; Tesseract and OXR agree in 137 of the 159 verified captures, and the character model in 62. Six times, two readers agreed once on a wrong reading (such as WA 8605 P for WA 6605 P), and none of those was confirmed, because the same reading must pass twice. Every mismatch is listed on the site with all three readings side by side.

## Speed

Time per stage on one 1080p frame, measured with `tools/benchmark.py` (also on the site's Model performance page):

| Stage | Runs | Laptop CPU (Core i5-12450H) | Laptop GPU (RTX 4060) |
|---|---|---|---|
| Vehicle detection (YOLOv8n) | every frame | 43.7 ms | 11.6 ms |
| Plate detection (YOLOv8s) | 7 frames per second of video | 93.7 ms | 12.0 ms |
| Character detection (YOLOv8n) | per capture | 32.6 ms | 14.4 ms |
| Tesseract OCR | per capture, twice if psm 13 is needed | 140.3 ms | CPU only |
| OXR-1.0 (SoMark OXR) | per capture | GPU only | 487.2 ms |
| **Frames per second, detectors only** (vehicles + plates, no reading, 30 fps video) | | **15 fps** | **69 fps** |

Two rates are easy to mix up here. How often a plate is read is set by the pipeline: on 7 frames in every second of video, for each car whose plate is not yet confirmed. How fast the laptop gets through the video is a separate number, and it depends on what is in the video. The site data is built offline from recorded video, not live, so the two do not have to match.

On the CPU the detectors alone run at about 15 fps, below the cameras' 25-30 fps, so the site data is built on the GPU. There they run at about 69 fps one frame at a time (the pipeline sends them 8 frames per call, about 2.5 times faster in a quick test). Reading the plate is the slow part: one capture takes about 0.64 s (OXR 487 ms, Tesseract 140 ms on the CPU, the character model 14 ms), so a second of video in which one car is being read takes about 4.5 s to process, and the build's progress bar drops to a few frames a second there. Stretches with no unconfirmed plate go through at 40-70 frames a second. Overall, in the build that made the site data, the 18 minutes of test videos were read in 10 minutes and the 66 minutes of dashcam clips in 54 minutes (not counting cutting the highlight videos); the busiest clip, 5 minutes of traffic with 508 captures, took 11 minutes. Live, the pipeline would keep up with the camera only while no car is being read. OXR takes 34 times as long as the character model because it is a general document model, read one crop at a time with transformers, and Tesseract has no GPU version. Both columns: PyTorch 2.11.0 (CUDA 13.0 build) with Ultralytics 8.3.127.

## Next steps

1. **One recognition model instead of three readers.** Train a single model on Malaysian plates that reads the whole plate in one pass, replacing Tesseract, the character model, OXR and the cross-check. OXR shows how much better reading the whole crop works here than Tesseract, but it is a general document model, far larger and slower than this job needs, and Tesseract cannot run on a GPU.
2. **Make OXR faster until then.** It reads one crop at a time with transformers; reading all the crops of a frame in one batch, or serving it with vLLM on Linux as SoMark does, would cut the wait.
3. **Two-row plates in the character model.** Its left-to-right ordering only works for plates written on one line; OXR already reads two-row plates.
4. **Keep vehicle ids through occlusion.** The tracker matches vehicles by position only, so a car that passes close in front of a parked one can take over the parked car's id.

## Who built what

- **weikang14, final year project:** the dataset, training the plate and character detectors, and the recognition pipeline: vehicle, plate, two readers and the cross-check (`code/Final_License_Plate_Detection.ipynb`).
- **Built with [Claude Code](https://claude.com/claude-code), an AI coding assistant, after the project:** this dashboard, moving the notebook's pipeline into `tools/` with tests and the speed benchmark, and the post-project changes described above (vehicle tracking, removing the fixed perspective correction, the ordered cross-check, adding OXR as a third reader). Earlier commits in this repository show "Claude Code" as the author name for this reason.
- **The third reader, [OXR-1.0](https://github.com/SoMarkAI/OXR) by SoMark (SoMark OXR):** used as published, under Apache 2.0 with SoMark's additional terms; not trained here.

## Repository layout

```
code/Final_License_Plate_Detection.ipynb   original FYP notebook
models/                                    plate.pt, plate_letter.pt, yolov8n.pt
sample_videos/                             five test videos with no people in view, to try the pipeline on
tools/lpr_pipeline.py                      the notebook's pipeline as a module
tools/build_site_data.py                   runs the pipeline on a folder of videos, writes the site data for it
tools/merge_site_data.py                   combines those folders into web/data/ and cuts the highlight videos
tools/site_data.py                         metrics and summary for the dashboard
tools/benchmark.py                         times each stage on CPU or GPU, writes web/speed.json
tools/serve.py                             local static server (supports video seeking)
tools/setup_env.py                         sets up .venv, the OXR weights and local.env (see Download and run)
web/                                       the dashboard (plain HTML, CSS, JS and Chart.js)
tests/                                     pytest and node tests
```

## Download and run

Everything needed is in this repository: `git clone https://github.com/weikang14/license-plate-recognition.git`, or [download it as a zip](https://github.com/weikang14/license-plate-recognition/archive/refs/heads/main.zip) (about 430 MB). `sample_videos/` holds five of the fifteen test videos, the ones with no people in view (test2, test3, test4, MOVI0022, MOVI0024), without sound; test3 and test4 are cut to 36 and 46 seconds to stay under GitHub's 100 MB file limit, and the cut parts hold no plate captures. The other ten show drivers' faces or passers-by and are not published. Running the pipeline needs an NVIDIA GPU (OXR runs on CUDA) and Python 3.12. In the project folder, on Windows:

```
python tools/setup_env.py
.venv\Scripts\python -m pytest tests
.venv\Scripts\python tools/build_site_data.py --videos sample_videos --out videos/built/samples --playback 3
.venv\Scripts\python tools/merge_site_data.py --out web/data "Sample videos=videos/built/samples"
.venv\Scripts\python tools/serve.py
```

`setup_env.py` makes `.venv` (PyTorch with CUDA, and `requirements.txt`), downloads the OXR-1.0 weights (3.1 GB) into `oxr_model/` and writes `local.env`, which tells the tools where OXR, Tesseract and ffmpeg are. If Tesseract or ffmpeg is missing, it prints the command that installs it (or set `TESSERACT_CMD`, `FFMPEG` and `FFPROBE` to their paths); `--cuda cu126` picks an older CUDA build for older drivers. On Linux, use `.venv/bin/python`. Run the tests before the build: some of them check the published data against this README, and the build replaces that data in your copy with the results for these five videos (about 5 minutes on an RTX 4060 laptop GPU; `git checkout web/data` or a fresh download brings the published data back). `--videos` can point at any folder of .mp4 files.

## Run the dashboard locally

```
python tools/serve.py
```

Then open http://127.0.0.1:8765. (`python -m http.server` also works, but it can't seek inside videos.)

## Regenerate the site data

Needs Python 3.12 with the packages in `requirements.txt` (`pip install -r requirements.txt`, after the CUDA build of PyTorch from pytorch.org), a CUDA GPU for OXR, plus [Tesseract OCR](https://github.com/UB-Mannheim/tesseract) and [ffmpeg](https://ffmpeg.org/). Point `TESSERACT_CMD`, `FFMPEG` and `FFPROBE` (it comes with ffmpeg) at them if they are not on `PATH`. Download the OXR-1.0 weights from [Hugging Face](https://huggingface.co/SoMarkAI/OXR-1.0) and set `OXR_MODEL` to that folder; the build stops if it is not set, and the benchmark times OXR on the GPU when it is. The raw videos (`videos/`: test videos and dashcam clips) and the training runs (`training_results/`) are kept locally and are not in this repository; tests that need them are skipped when they are missing. Without `training_results/`, the build reuses the training scores and plots already in `web/data`, which belong to the same model files in `models/`.

The site's data was built with:

```
python tools/build_site_data.py --videos videos/testing_video --out videos/built/testing_video --playback 15
python tools/build_site_data.py --videos videos/raw_video --out videos/built/raw_video --playback 14
python tools/merge_site_data.py --out web/data --cut "Test videos=5,8,15" "Test videos=videos/built/testing_video" "Dashcam=videos/built/raw_video"
python tools/benchmark.py --device cpu      # or --device 0 for the first CUDA GPU
python -m pytest tests
node --test "tests/web/*.test.mjs"
```
