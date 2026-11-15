import numpy as np

from tools.lpr_pipeline import (CarTracker, TimeGate, car_under, chars_match, clean_ocr_text, is_capture_frame,
                                ocr_text_left_to_right, ocr_until_match, prepare_plate, reading_order, vehicle_boxes)


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


def test_reading_order_sorts_boxes_left_to_right_and_skips_back_car():
    names = {0: "J", 1: "H", 2: "B", 3: "back_car"}
    boxes = [[300, 10, 380, 90, 0.9, 2], [0, 0, 800, 100, 0.6, 3], [20, 10, 100, 90, 0.8, 0], [160, 10, 240, 90, 0.95, 1]]
    assert reading_order(boxes, names) == ["J", "H", "B"]


def test_ocr_text_reads_words_left_to_right():
    data = {"text": ["2149", "", "JHB", "|"], "left": [400, 0, 30, 700]}  # pytesseract image_to_data columns
    assert ocr_text_left_to_right(data) == "JHB2149"


def test_is_capture_frame():
    assert is_capture_frame(7) and is_capture_frame(14)
    assert not is_capture_frame(1) and not is_capture_frame(8)


def test_time_gate():
    gate = TimeGate()
    assert [gate.allow(t) for t in (0.0, 1.0, 1.49, 1.5, 2.0)] == [True, False, False, True, False]


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
