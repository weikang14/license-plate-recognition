from types import SimpleNamespace

import cv2
import numpy as np

import tools.lpr_pipeline as lpr
from tools.lpr_pipeline import (CarTracker, Models, car_under, chars_match, clean_ocr_text, is_capture_frame,
                                ocr_text_left_to_right, ocr_until_match, prepare_plate, process_video, reading_order,
                                vehicle_boxes)


def test_car_under_uses_the_plate_centre():
    cars = [(100, 100, 400, 300, 7), (500, 100, 800, 300, 9)]
    assert car_under((350, 250, 450, 280), cars) == 7    # centre (400, 265) on the first car's edge
    assert car_under((600, 270, 700, 320), cars) == 9    # pokes out below, centre (650, 295) still inside
    assert car_under((600, 280, 700, 330), cars) is None  # centre (650, 305) is below the car
    assert car_under((420, 200, 480, 230), cars) is None  # between the two cars
    assert car_under((150, 150, 250, 200), []) is None    # no car tracked


def test_car_tracker_keeps_ids_across_frames():
    tracker = CarTracker(fps=30)
    first = tracker.update([(100, 100, 300, 250, 0.9), (600, 100, 800, 260, 0.8)])
    ids = sorted(t[4] for t in first)
    for k in range(1, 6):  # both cars drift a little each frame
        moved = tracker.update([(100 + 4 * k, 100, 300 + 4 * k, 250, 0.9), (600, 100 + 2 * k, 800, 260 + 2 * k, 0.8)])
        assert sorted(t[4] for t in moved) == ids
    # A car that appears mid-stream is confirmed on its second frame (ByteTrack's activation), then gets a fresh id.
    three = [(124, 100, 324, 250, 0.9), (600, 112, 800, 272, 0.8), (1200, 400, 1500, 600, 0.9)]
    tracker.update(three)
    new = tracker.update([(128, 100, 328, 250, 0.9), (600, 114, 800, 274, 0.8), (1202, 400, 1502, 600, 0.9)])
    assert len({t[4] for t in new}) == 3 and set(ids) < {t[4] for t in new}


def test_car_tracker_keeps_the_id_through_a_short_gap_only():
    def stay_then_vanish(gap_frames):
        tracker = CarTracker(fps=30)
        for _ in range(5):
            seen = tracker.update([(100, 100, 300, 250, 0.9)])
        for _ in range(gap_frames):  # the car is not detected (dip in confidence, labelled truck, occluded)
            tracker.update([])
        tracker.update([(100, 100, 300, 250, 0.9)])
        back = tracker.update([(100, 100, 300, 250, 0.9)])
        return seen[0][4], back[0][4]

    old, new = stay_then_vanish(60)   # 2 s at 30 fps
    assert new == old
    old, new = stay_then_vanish(150)  # 5 s: a different visit
    assert new != old


def test_vehicle_boxes_counts_buses_and_trucks_as_vehicles():
    rows = [[0, 0, 9, 9, 0.9, 2], [1, 1, 9, 9, 0.8, 7], [2, 2, 9, 9, 0.7, 5], [3, 3, 9, 9, 0.9, 0], [4, 4, 9, 9, 0.9, 3]]
    assert vehicle_boxes(rows) == [(0, 0, 9, 9, 0.9), (1, 1, 9, 9, 0.8), (2, 2, 9, 9, 0.7)]  # car, truck, bus; not person, motorcycle


def test_car_tracker_handles_frames_without_cars():
    assert CarTracker(fps=25).update([]) == []


def test_clean_ocr_text():
    assert clean_ocr_text("VW 2468\n") == "VW2468"
    assert clean_ocr_text("ab-12 C|") == "12C"


def test_chars_match_compares_in_reading_order():
    assert chars_match(list("VW2"), ["V", "W", "2"]) is True
    assert chars_match(list("2149JHB"), list("JHB2149")) is False  # same characters, wrong order
    assert chars_match([], []) is False
    assert chars_match(list("WA6605P"), list("WA6605P")) is True     # letters, digits, an optional letter
    for auctioned in ("W1", "MALAYSIA8", "G1M1", "1M4U1234"):          # JPJ special series sold at auction
        assert chars_match(list(auctioned), list(auctioned)) is True
    assert chars_match(list("48860"), list("48860")) is False        # agreed, but no letters: not a plate
    assert chars_match(["7"], ["7"]) is False                         # a badge, not a plate


def test_reading_order_sorts_boxes_left_to_right_and_skips_back_car():
    names = {0: "J", 1: "H", 2: "B", 3: "back_car"}
    boxes = [[300, 10, 380, 90, 0.9, 2], [0, 0, 800, 100, 0.6, 3], [20, 10, 100, 90, 0.8, 0], [160, 10, 240, 90, 0.95, 1]]
    assert reading_order(boxes, names) == ["J", "H", "B"]


def test_ocr_text_reads_words_left_to_right():
    data = {"text": ["2149", "", "JHB", "|"], "left": [400, 0, 30, 700]}  # pytesseract image_to_data columns
    assert ocr_text_left_to_right(data) == "JHB2149"


def test_capture_frames_are_7_a_second_spread_evenly():
    assert [n for n in range(1, 31) if is_capture_frame(n, 30)] == [5, 9, 13, 18, 22, 26, 30]
    assert sum(is_capture_frame(n, 25) for n in range(1, 251)) == 70  # 10 s at 25 fps
    assert all(is_capture_frame(n, 5) for n in range(1, 11))       # under 7 fps: every frame


def test_prepare_plate_is_gray_800_wide_and_keeps_aspect():
    out = prepare_plate(np.random.randint(0, 256, (60, 200, 3), dtype=np.uint8))
    assert out.shape == (240, 800) and out.dtype == np.uint8


def test_ocr_until_match_stops_at_the_first_reading_that_agrees_with_yolo():
    calls = []
    def reader(readings):
        def read(psm):
            calls.append(psm)
            return readings[psm]
        return read
    assert ocr_until_match(reader({6: "JHB2149", 13: "X"}), list("JHB2149")) == "JHB2149" and calls == [6]
    assert ocr_until_match(reader({6: "JHB214", 13: "JHB2149"}), list("JHB2149")) == "JHB2149"
    assert ocr_until_match(reader({6: "JHB214", 13: ""}), list("JHB2149")) == "JHB214"  # no agreement: first reading kept


class FakeDetector:
    """Stands in for a YOLO model: finds nothing and records which frames (told apart by grey level) each call got."""

    def __init__(self):
        self.calls = []

    def __call__(self, imgs, **kw):
        self.calls.append([round(float(im.mean()) / 12) + 1 for im in imgs])
        return [SimpleNamespace(boxes=SimpleNamespace(data=np.zeros((0, 6)))) for _ in imgs]


class BoxDetector(FakeDetector):
    """A YOLO stand-in that finds the same boxes (x1, y1, x2, y2, score, cls) on every frame."""

    def __init__(self, rows):
        super().__init__()
        self.rows = np.asarray(rows, dtype=float)

    def __call__(self, imgs, **kw):
        return [SimpleNamespace(boxes=SimpleNamespace(data=self.rows)) for _ in imgs]


def grey_video(tmp_path, frames):
    """A 64x48 25 fps video; frame i + 1 has grey level i * 12."""
    video = tmp_path / "grey.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    for i in range(frames):
        w.write(np.full((48, 64, 3), i * 12 % 256, np.uint8))
    w.release()
    return video


def test_a_car_is_read_once_on_every_capture_frame_until_the_same_plate_is_verified_twice(tmp_path, monkeypatch):
    readings = iter([("JHB214", list("JHB2149")), ("JHB2149", list("JHB2149")), ("VEH6482", list("VEH6482")),
                     ("", list("JHB2149")), ("JHB2149", list("JHB2149"))])
    monkeypatch.setattr(lpr, "_read_plate", lambda region, models: next(readings))
    car = BoxDetector([[0, 0, 64, 48, 0.9, 2]])
    plates = BoxDetector([[20, 30, 44, 40, 0.9, 0], [20, 5, 44, 15, 0.8, 0]])  # two plate boxes on the one car
    records, _ = process_video(grey_video(tmp_path, 40), Models(car, plates, FakeDetector()), None)
    assert [(r["frame"], r["plate"], r["match"]) for r in records] == [  # 25 fps: 7 a second; a 6th read would fail
        (4, "JHB214", False), (8, "JHB2149", True), (11, "VEH6482", True), (15, "", False), (18, "JHB2149", True)]


def test_cars_in_the_same_frame_are_each_read(tmp_path, monkeypatch):
    monkeypatch.setattr(lpr, "_read_plate", lambda region, models: ("AB1", list("AB1")))
    cars = BoxDetector([[0, 0, 30, 48, 0.9, 2], [34, 0, 64, 48, 0.9, 2]])
    plates = BoxDetector([[5, 30, 25, 40, 0.9, 0], [39, 30, 59, 40, 0.9, 0]])
    records, _ = process_video(grey_video(tmp_path, 20), Models(cars, plates, FakeDetector()), tmp_path / "crops")
    assert [r["frame"] for r in records] == [4, 4, 8, 8] and len({r["car_id"] for r in records}) == 2
    assert sorted(p.name for p in (tmp_path / "crops").iterdir()) == sorted(r["crop"] for r in records)


def test_capture_frames_reach_the_plate_detector_in_the_same_batches_in_both_modes(tmp_path):
    # On a GPU the batch changes float rounding, so annotate mode must detect capture frames exactly as fast mode does.
    video = grey_video(tmp_path, 20)

    def plate_calls(**kw):
        plate = FakeDetector()
        process_video(video, Models(FakeDetector(), plate, FakeDetector()), None, **kw)
        return plate.calls

    fast = plate_calls()
    assert fast == [[4, 8], [11, 15], [18]]  # capture frames only (7 a second at 25 fps), one call per batch of 8 frames
    annotated = plate_calls(annotate_to=tmp_path / "boxed.mp4")
    assert all(c in annotated for c in fast), annotated
