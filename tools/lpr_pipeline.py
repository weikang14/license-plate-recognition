"""Recognition pipeline, ported from code/Final_License_Plate_Detection.ipynb.

Same models, thresholds and plate processing as the notebook. Changes:
plates are read 7 times a second of video (the notebook's `fps = 7` read every 7th frame, about 4 a second),
mismatched captures are recorded too, each record keeps its frame number and time in the video, and plates
are cropped from the undrawn frame. After the FYP, cars are tracked with ByteTrack and a
plate only counts when it sits on a tracked car (the original plan, so a plate photo held up
to the camera is ignored; the final notebook ran the plate detector on the whole frame).
The notebook's fixed-point perspective correction was also dropped after testing (see prepare_plate),
and both readers now read left to right, so the cross-check compares the two readings in order
(the notebook compared sorted characters) with the character model at conf 0.4. The agreed reading must also hold
letters and a number, as Malaysian plates do (auctioned ones such as W 1 and MALAYSIA 8 too), so a badge read as "7"
is not verified. When Tesseract's block reading (psm 6) disagrees, the plate is read again as one raw line (psm 13). The tracker counts buses
and trucks as vehicles (a car close to the camera is often labelled truck) and keeps a lost car's id for 3 s,
so a car that stays in view is not counted again. Each tracked car is read on every capture frame until the same
reading has passed the cross-check twice (once can be both readers dropping the same characters), then keeps that
reading and is not read again (the notebook read one plate every 1.5 s, verified or not).
After the portfolio went up, OXR-1.0 (SoMark's 1.3B document model) was added as a third reader of the colour crop,
and a capture is verified when any two of the three readings agree; without OXR_MODEL set, only Tesseract and YOLO read.
"""
from __future__ import annotations  # `int | None` hints on Python 3.9

import hashlib
import importlib
import os
import sys
import types
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import supervision as sv
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

CAPTURES_PER_S = 7     # plate reads per second of video, spread evenly over the frames
LETTER_CONF = 0.4      # character model; 0.4 verified the most captures on the test videos (0.5 in the notebook)
VEHICLE_CLASSES = {2, 5, 7}  # COCO car, bus, truck; the notebook used car only
TRACK_BUFFER_S = 3     # a lost car keeps its id this long (ByteTrack's default is 1 s)
OCR_PSMS = (6, 13)     # notebook's psm 6 first; psm 13 (raw line) only when psm 6 agrees with neither other reader
DETECT_BATCH = 8       # frames per detector call; one at a time leaves a GPU mostly idle (8 was ~2.5x faster on an RTX 4060)


def clean_ocr_text(text: str) -> str:
    """Notebook's OCR clean-up, kept verbatim (digits and upper-case letters)."""
    return "".join(e for e in text if e.isalnum() and e.isdigit() or e.isupper())


def chars_match(ocr_chars: list[str], yolo_chars: list[str]) -> bool:
    """A plate counts only when OCR and YOLO read the same characters in the same order, letters and a number among them.

    Malaysian plates, the auctioned series too (W 1, MALAYSIA 8, G1M 1), have both; a badge read as "7" does not.
    """
    return (list(ocr_chars) == list(yolo_chars) and any(c.isalpha() for c in yolo_chars)
            and any(c.isdigit() for c in yolo_chars))


def agreed_reading(*readings: str) -> str:
    """The reading at least two of the readers (Tesseract, YOLO characters, OXR) agree on, by chars_match; else ""."""
    return next((a for a, b in combinations(readings, 2) if chars_match(list(a), list(b))), "")


def reading_order(boxes: list[list[float]], names: dict) -> list[str]:
    """Character boxes (x1, y1, x2, y2, score, cls) -> characters left to right; back_car is not a character.

    ponytail: single-row plates only; two-row plates need grouping by row first.
    """
    return [names[int(c)] for x1, _, x2, _, _, c in sorted(boxes, key=lambda b: b[0] + b[2]) if len(names[int(c)]) == 1]


def ocr_text_left_to_right(data: dict) -> str:
    """pytesseract image_to_data output -> cleaned words joined left to right (Tesseract may put them on separate lines)."""
    return "".join(clean_ocr_text(w) for _, w in sorted(zip(data["left"], data["text"])))


def vehicle_boxes(rows: list[list[float]]) -> list[tuple]:
    """COCO detections (x1, y1, x2, y2, score, cls) -> vehicle boxes (x1, y1, x2, y2, score) for the tracker."""
    return [(x1, y1, x2, y2, s) for x1, y1, x2, y2, s, c in rows if int(c) in VEHICLE_CLASSES]


def car_under(plate: tuple, cars: list[tuple]) -> int | None:
    """Id of the tracked car (x1, y1, x2, y2, id) whose box holds the plate's centre, else None."""
    cx, cy = (plate[0] + plate[2]) / 2, (plate[1] + plate[3]) / 2
    return next((car_id for x1, y1, x2, y2, car_id in cars if x1 <= cx <= x2 and y1 <= cy <= y2), None)


class CarTracker:
    """ByteTrack (supervision) over the car detections, so each car keeps one id across frames."""

    def __init__(self, fps: float):
        # supervision counts the buffer in 30 fps frames and rescales it to the video's frame rate
        self._tracker = sv.ByteTrack(lost_track_buffer=TRACK_BUFFER_S * 30, frame_rate=max(1, round(fps)))

    def update(self, boxes: list[tuple]) -> list[tuple]:
        """boxes: (x1, y1, x2, y2, score) for this frame -> tracked (x1, y1, x2, y2, id)."""
        if boxes:
            arr = np.asarray(boxes, dtype=float)
            dets = sv.Detections(xyxy=arr[:, :4], confidence=arr[:, 4], class_id=np.full(len(arr), 2))
        else:
            dets = sv.Detections.empty()  # still advance the tracker so lost cars age out
        tracked = self._tracker.update_with_detections(dets)
        return [(*map(float, xyxy), int(tid)) for xyxy, tid in zip(tracked.xyxy, tracked.tracker_id)]


def is_capture_frame(frame_no: int, fps: float, per_s: float = CAPTURES_PER_S) -> bool:
    """frame_no is 1-based, as in the notebook. True on per_s evenly spread frames a second (every frame below per_s fps)."""
    return int(frame_no * per_s / fps) > int((frame_no - 1) * per_s / fps)


def prepare_plate(plate_bgr: np.ndarray) -> np.ndarray:
    """Gray and resized to width 800, as in the notebook, without its fixed perspective warp.

    The notebook's warp used four fixed points tuned to one plate; it crops or skews characters for
    both readers, and dropping it raised verified captures on the 15 test videos from 10 to 47.
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (800, int(gray.shape[0] * (800 / gray.shape[1]))), interpolation=cv2.INTER_CUBIC)


class OxrReader:
    """OXR-1.0 (SoMark OXR, github.com/SoMarkAI/OXR) reading a plate crop, with transformers on a CUDA GPU.

    ponytail: one crop per generate call, about 0.5 s on an RTX 4060 (tools/benchmark.py); batching the crops of a frame, or vLLM on
    Linux (OXR's own setup), would be the speed-up.
    """

    PROMPT = "Recognize the content of the image."  # OXR's own text-recognition prompt

    def __init__(self, generate):
        self._generate = generate  # RGB PIL image -> text
        self._read: dict[tuple, str] = {}  # annotate mode re-reads the fast-mode crops: same pixels, same text, no wait

    def __call__(self, plate_bgr: np.ndarray) -> str:
        key = (plate_bgr.shape, hashlib.sha1(plate_bgr.tobytes()).digest())
        if key not in self._read:
            img = Image.fromarray(cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB))
            self._read[key] = clean_ocr_text(self._generate(img).upper())
        return self._read[key]

    @classmethod
    def load(cls, model_dir: str) -> "OxrReader":
        import torch
        # transformers 5 imports the model's own code twice under two names, so its config classes stop matching;
        # importing the weights folder once as a package avoids that.
        pkg = types.ModuleType("oxr_weights")
        pkg.__path__ = [str(model_dir)]
        sys.modules["oxr_weights"] = pkg
        m = importlib.import_module("oxr_weights.modeling_oxr")
        model = m.OXR.from_pretrained(str(model_dir), config=m.OXRConfig.from_pretrained(str(model_dir)),
                                      dtype=torch.bfloat16)
        # transformers 5 leaves the buffers OXR's own code builds in __init__ (not in the weights file) as uninitialised
        # memory: rebuild them as that code does. Left as they are, the image-marker ids hold leftover values (a
        # device-side assert when out of range, the wrong markers when not) and the vision encoder's rotary
        # frequencies are garbage.
        v = model.config.visual_vocab_size
        model.indicator_token_indices.copy_(torch.arange(v - len(m.INDICATOR_IDS), v))
        for r in model.modules():
            if isinstance(r, m.VisionRotaryEmbedding):
                r.inv_freq.copy_(1.0 / (r.theta ** (torch.arange(0, r.dim, 2, dtype=torch.float) / r.dim)))
        model = model.to("cuda").eval()
        tok = model.text_tokenizer

        def generate(img: Image.Image) -> str:
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": cls.PROMPT}]}]
            with torch.inference_mode():
                ids, pix, grid = model.preprocess_inputs(messages)
                out = model.generate(ids.cuda(), pixel_values=pix.cuda().to(torch.bfloat16), grid_thws=grid.cuda(),
                                     max_new_tokens=32, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id,
                                     eos_token_id=[151643, 151645])  # end-of-text and end-of-turn: else it runs to 32
            return tok.decode(out[0], skip_special_tokens=True)
        return cls(generate)


@dataclass
class Models:
    car: YOLO     # COCO yolov8n, only car, bus and truck are used
    plate: YOLO
    letter: YOLO  # 0-9, A-Y (no I/O/Z), back_car
    oxr: OxrReader | None = None  # third reader; None: Tesseract and YOLO must agree

    @classmethod
    def load(cls, models_dir: Path) -> "Models":
        """The YOLO models in models_dir, and OXR from the folder OXR_MODEL names (if set)."""
        d, oxr = Path(models_dir), os.environ.get("OXR_MODEL")
        return cls(YOLO(str(d / "yolov8n.pt")), YOLO(str(d / "plate.pt")), YOLO(str(d / "plate_letter.pt")),
                   OxrReader.load(oxr) if oxr else None)


def ocr_until_match(read, others: list[list[str]], psms: tuple[int, ...] = OCR_PSMS) -> str:
    """read(psm) -> OCR text. Try each psm until one agrees with another reader; if none does, keep the first reading."""
    first = None
    for psm in psms:
        text = read(psm)
        if any(chars_match(list(text), o) for o in others):
            return text
        first = text if first is None else first
    return first


def _boxes(model: YOLO, imgs: np.ndarray | list[np.ndarray], conf: float = 0.5, **kw) -> list[list[list[float]]]:
    """One image or a batch -> per image, rows of (x1, y1, x2, y2, score, cls)."""
    return [r.boxes.data.tolist() for r in model(imgs, conf=conf, verbose=False, **kw)]


def _read_plate(plate_bgr: np.ndarray, models: Models) -> tuple[str, list[str], str]:
    """Notebook's double reading, Tesseract on the inverted plate and YOLO characters on the plate, plus OXR on the
    colour crop ("" without OXR); all left to right."""
    letter = prepare_plate(plate_bgr)
    chars = reading_order(_boxes(models.letter, cv2.cvtColor(letter, cv2.COLOR_GRAY2RGB), conf=LETTER_CONF)[0],
                          models.letter.names)
    oxr = models.oxr(plate_bgr) if models.oxr else ""
    inverted = cv2.bitwise_not(letter)
    read = lambda psm: ocr_text_left_to_right(pytesseract.image_to_data(
        inverted, lang="eng", config=f"--psm {psm}", output_type=pytesseract.Output.DICT))
    return ocr_until_match(read, [chars, list(oxr)]), chars, oxr


def _draw_overlay(frame: np.ndarray, o: dict) -> None:
    """The notebook's five status lines."""
    lines = [f"Count: {o['count']}", f"Plate: {o['plate']}", f"Plate Letter: {o['letter']}",
             f"Plate Character: {o['chars']}", f"Score: {o['score']:.2f}"]
    for text, y in zip(lines, (60, 95, 130, 165, 195)):
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


def _detect_in_batches(cap, models: Models, fps: float, plates_every_frame: bool, max_frames: int | None, batch: int):
    """Yield (frame_no, frame, car rows, plate rows or None) per frame, running the detectors `batch` frames at a time."""
    done = 0
    while True:
        chunk = []
        while len(chunk) < batch and (max_frames is None or done + len(chunk) < max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            chunk.append(frame)
        if not chunk:
            return
        nos = range(done + 1, done + len(chunk) + 1)
        cars = _boxes(models.car, chunk, agnostic_nms=True)  # one box per vehicle, not car + truck
        # Capture frames get their own call in both modes: on a GPU the batch changes float rounding, and
        # annotate mode must read the same plates as fast mode. Other frames only have their plates drawn.
        plates = {}
        for want in ([i for i, no in enumerate(nos) if is_capture_frame(no, fps)],
                     [i for i, no in enumerate(nos) if plates_every_frame and not is_capture_frame(no, fps)]):
            if want:
                plates.update(zip(want, _boxes(models.plate, [chunk[i] for i in want])))
        yield from ((no, f, c, plates.get(i)) for i, (no, f, c) in enumerate(zip(nos, chunk, cars)))
        done += len(chunk)


def process_video(video_path: Path, models: Models, crops_dir: Path | None, *, annotate_to: Path | None = None,
                  max_frames: int | None = None, batch: int = DETECT_BATCH, progress: bool = False) -> tuple[list[dict], dict]:
    """Run the pipeline over a video.

    Cars are detected and tracked on every frame in both modes. Fast mode (annotate_to=None) runs
    the plate detector only on capture frames; annotate mode runs it on every frame and writes the
    boxed video at the source fps. Both give the same records. Frames go to the detectors `batch`
    at a time; the tracker and the plate reading still go frame by frame. On the CPU the batch size
    does not change the records. On a GPU it changes float rounding, as moving from CPU to GPU does:
    on the 15 test videos the plates and verified captures stayed the same, but one car's id shifted
    by one (MOVI0023). progress=True draws a progress bar (frames, elapsed and remaining time) on stderr.
    """
    pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_CMD", "tesseract")
    video_path = Path(video_path)
    stem = video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = None
    if annotate_to is not None:
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(str(annotate_to), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if crops_dir is not None:
        Path(crops_dir).mkdir(parents=True, exist_ok=True)

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bar = tqdm(total=min(n, max_frames or n), desc=f"{stem} ({'drawing' if writer else 'reading'})", unit="frame",
               disable=not progress, ascii=True, mininterval=1, file=sys.stderr)
    tracker, records, frames = CarTracker(fps), [], 0
    car_ids: set[int] = set()
    confirmed: dict[int, str] = {}  # car id -> its plate, once the same agreed reading passed twice; not read again
    passed: set[tuple[int, str]] = set()  # (car id, agreed reading) that passed the cross-check once
    overlay = {"count": 0, "plate": 0, "letter": None, "chars": 0, "score": 0.0}  # notebook's final_* values
    try:
        for frame_no, frame, car_rows, plate_rows in _detect_in_batches(cap, models, fps, writer is not None,
                                                                         max_frames, batch):
            frames = frame_no
            bar.update()
            capture = is_capture_frame(frame_no, fps)
            clean = frame.copy() if writer is not None else frame  # crops come from the undrawn frame
            cars = tracker.update(vehicle_boxes(car_rows))
            car_ids.update(c[4] for c in cars)
            if writer is not None:
                for x1, y1, x2, y2, car_id in cars:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"#{car_id} {confirmed.get(car_id, '')}".rstrip(), (int(x1), max(int(y1) - 8, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 0), 2, cv2.LINE_AA)
            if writer is None and not capture:
                continue  # fast mode: plates are only read on capture frames

            plate_box, read_now = 0, set()
            t_s = (frame_no - 1) / fps
            for x1, y1, x2, y2, score, _ in plate_rows:
                car_id = car_under((x1, y1, x2, y2), cars)
                if car_id is None:
                    continue  # the plan: a plate only counts when it sits on a tracked car
                if writer is not None:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                if car_id in confirmed:
                    continue  # same tracked car: keep its confirmed plate, don't read it again
                if not capture or car_id in read_now or score < 0.75:
                    continue  # ponytail: one plate per car per frame, the detector's most confident one
                region = clean[int(y1):int(y2), int(x1):int(x2)]
                if region.size == 0:
                    continue
                read_now.add(car_id)
                text, chars, oxr = _read_plate(region, models)
                agreed = agreed_reading(text, "".join(chars), oxr)
                match = bool(agreed)
                rid = f"{stem}-{frame_no:06d}-{car_id}"
                records.append({"id": rid, "video": stem, "frame": frame_no, "time_s": round(t_s, 2),
                                "car_id": car_id, "plate": text, "yolo_chars": chars, "n_chars": len(chars),
                                "oxr": oxr, "agreed": agreed, "score": round(score, 3), "match": match,
                                "crop": f"{rid}.jpg"})
                if crops_dir is not None:
                    cv2.imwrite(str(Path(crops_dir) / f"{rid}.jpg"), region, [cv2.IMWRITE_JPEG_QUALITY, 100])
                if match and (car_id, agreed) in passed:
                    confirmed[car_id] = agreed
                    plate_box += 1
                    overlay = {"count": overlay["count"] + 1, "plate": plate_box, "letter": agreed,
                               "chars": len(chars), "score": score}
                elif match:
                    passed.add((car_id, agreed))

            if writer is not None:
                _draw_overlay(frame, overlay)
                writer.write(frame)
    finally:
        bar.close()
        cap.release()
        if writer is not None:
            writer.release()
    return records, {"video": stem, "fps": fps, "frames": frames, "duration_s": round(frames / fps, 2),
                     "vehicles": len(car_ids)}
