"""Recognition pipeline, ported from code/Final_License_Plate_Detection.ipynb.

Same models, thresholds and plate processing as the notebook. Changes:
the 1.5 s capture gap is measured in video time (not wall-clock), mismatched captures
are recorded too, each record keeps its frame number and time in the video, and plates
are cropped from the undrawn frame. After the FYP, cars are tracked with ByteTrack and a
plate only counts when it sits on a tracked car (the original plan, so a plate photo held up
to the camera is ignored; the final notebook ran the plate detector on the whole frame).
The notebook's fixed-point perspective correction was also dropped after testing (see prepare_plate).
"""
import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import supervision as sv
from ultralytics import YOLO

CAPTURE_STRIDE = 7     # notebook: fps = 7 -> try a capture every 7th frame
CAPTURE_GAP_S = 1.5    # notebook: timedelta(seconds=1.5)


def clean_ocr_text(text: str) -> str:
    """Notebook's OCR clean-up, kept verbatim (digits and upper-case letters)."""
    return "".join(e for e in text if e.isalnum() and e.isdigit() or e.isupper())


def chars_match(ocr_chars: list[str], yolo_chars: list[str]) -> bool:
    """A plate counts only when OCR and YOLO saw the same characters."""
    return sorted(ocr_chars) == sorted(yolo_chars) and len(yolo_chars) > 0


def car_under(plate: tuple, cars: list[tuple]) -> int | None:
    """Id of the tracked car (x1, y1, x2, y2, id) whose box holds the plate's centre, else None."""
    cx, cy = (plate[0] + plate[2]) / 2, (plate[1] + plate[3]) / 2
    return next((car_id for x1, y1, x2, y2, car_id in cars if x1 <= cx <= x2 and y1 <= cy <= y2), None)


class CarTracker:
    """ByteTrack (supervision) over the car detections, so each car keeps one id across frames."""

    def __init__(self, fps: float):
        self._tracker = sv.ByteTrack(frame_rate=max(1, round(fps)))

    def update(self, boxes: list[tuple]) -> list[tuple]:
        """boxes: (x1, y1, x2, y2, score) for this frame -> tracked (x1, y1, x2, y2, id)."""
        if boxes:
            arr = np.asarray(boxes, dtype=float)
            dets = sv.Detections(xyxy=arr[:, :4], confidence=arr[:, 4], class_id=np.full(len(arr), 2))
        else:
            dets = sv.Detections.empty()  # still advance the tracker so lost cars age out
        tracked = self._tracker.update_with_detections(dets)
        return [(*map(float, xyxy), int(tid)) for xyxy, tid in zip(tracked.xyxy, tracked.tracker_id)]


def is_capture_frame(frame_no: int, stride: int = CAPTURE_STRIDE) -> bool:
    """frame_no is 1-based, as in the notebook."""
    return frame_no % stride == 0


class TimeGate:
    """At most one capture per min_gap_s of video time; only a pass moves the clock."""

    def __init__(self, min_gap_s: float = CAPTURE_GAP_S):
        self.min_gap_s = min_gap_s
        self.last = -math.inf

    def allow(self, t_s: float) -> bool:
        if t_s - self.last < self.min_gap_s:
            return False
        self.last = t_s
        return True


def prepare_plate(plate_bgr: np.ndarray) -> np.ndarray:
    """Gray and resized to width 800, as in the notebook, without its fixed perspective warp.

    The notebook's warp used four fixed points tuned to one plate; it crops or skews characters for
    both readers, and dropping it raised verified captures on the 15 test videos from 10 to 47.
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (800, int(gray.shape[0] * (800 / gray.shape[1]))), interpolation=cv2.INTER_CUBIC)


@dataclass
class Models:
    car: YOLO     # COCO yolov8n, only class 2 (car) is used
    plate: YOLO
    letter: YOLO  # 0-9, A-Y (no I/O/Z), back_car

    @classmethod
    def load(cls, models_dir: Path) -> "Models":
        d = Path(models_dir)
        return cls(YOLO(str(d / "yolov8n.pt")), YOLO(str(d / "plate.pt")), YOLO(str(d / "plate_letter.pt")))


def _boxes(model: YOLO, img: np.ndarray) -> list[list[float]]:
    return model(img, conf=0.5, verbose=False)[0].boxes.data.tolist()


def _read_plate(plate_bgr: np.ndarray, models: Models) -> tuple[str, list[str]]:
    """Notebook's double reading: Tesseract on the inverted plate, YOLO characters on the plate."""
    letter = prepare_plate(plate_bgr)
    text = clean_ocr_text(pytesseract.image_to_string(cv2.bitwise_not(letter), lang="eng", config="--psm 6"))
    chars = [models.letter.names[int(c)] for *_, c in _boxes(models.letter, cv2.cvtColor(letter, cv2.COLOR_GRAY2RGB))]
    return text, chars


def _draw_overlay(frame: np.ndarray, o: dict) -> None:
    """The notebook's five status lines."""
    lines = [f"Count: {o['count']}", f"Plate: {o['plate']}", f"Plate Letter: {o['letter']}",
             f"Plate Character: {o['chars']}", f"Score: {o['score']:.2f}"]
    for text, y in zip(lines, (60, 95, 130, 165, 195)):
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


def process_video(video_path: Path, models: Models, crops_dir: Path | None, *,
                  annotate_to: Path | None = None, max_frames: int | None = None) -> tuple[list[dict], dict]:
    """Run the pipeline over a video.

    Cars are detected and tracked on every frame in both modes. Fast mode (annotate_to=None) runs
    the plate detector only on capture frames; annotate mode runs it on every frame and writes the
    boxed video at the source fps. Both give the same records.
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

    gate, tracker, records, frames = TimeGate(), CarTracker(fps), [], 0
    car_ids: set[int] = set()
    overlay = {"count": 0, "plate": 0, "letter": None, "chars": 0, "score": 0.0}  # notebook's final_* values
    try:
        while max_frames is None or frames < max_frames:
            frame_no = frames + 1
            capture = is_capture_frame(frame_no)
            ok, frame = cap.read()
            if not ok:
                break
            frames = frame_no
            clean = frame.copy() if writer is not None else frame  # crops come from the undrawn frame
            cars = tracker.update([(x1, y1, x2, y2, s) for x1, y1, x2, y2, s, cls in _boxes(models.car, clean)
                                   if int(cls) == 2])
            car_ids.update(c[4] for c in cars)
            if writer is not None:
                for x1, y1, x2, y2, car_id in cars:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"#{car_id}", (int(x1), max(int(y1) - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 0), 2, cv2.LINE_AA)
            if writer is None and not capture:
                continue  # fast mode: plates are only read on capture frames

            plate_box = 0
            t_s = (frame_no - 1) / fps
            for x1, y1, x2, y2, score, _ in _boxes(models.plate, clean):
                car_id = car_under((x1, y1, x2, y2), cars)
                if car_id is None:
                    continue  # the plan: a plate only counts when it sits on a tracked car
                if writer is not None:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                if not capture or not gate.allow(t_s) or score < 0.75:
                    continue
                region = clean[int(y1):int(y2), int(x1):int(x2)]
                if region.size == 0:
                    continue
                text, chars = _read_plate(region, models)
                match = chars_match(list(text), chars)
                rid = f"{stem}-{frame_no:06d}"
                records.append({"id": rid, "video": stem, "frame": frame_no, "time_s": round(t_s, 2),
                                "car_id": car_id, "plate": text, "yolo_chars": chars, "n_chars": len(chars),
                                "score": round(score, 3), "match": match, "crop": f"{rid}.jpg"})
                if crops_dir is not None:
                    cv2.imwrite(str(Path(crops_dir) / f"{rid}.jpg"), region, [cv2.IMWRITE_JPEG_QUALITY, 100])
                if match:
                    plate_box += 1
                    overlay = {"count": overlay["count"] + 1, "plate": plate_box, "letter": text,
                               "chars": len(chars), "score": score}

            if writer is not None:
                _draw_overlay(frame, overlay)
                writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return records, {"video": stem, "fps": fps, "frames": frames, "duration_s": round(frames / fps, 2),
                     "vehicles": len(car_ids)}
