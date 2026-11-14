import numpy as np

from tools.lpr_pipeline import (CarTracker, TimeGate, car_under, chars_match, clean_ocr_text, is_capture_frame,
                                prepare_plate)


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


def test_car_tracker_handles_frames_without_cars():
    assert CarTracker(fps=25).update([]) == []


def test_clean_ocr_text():
    assert clean_ocr_text("VW 2468\n") == "VW2468"
    assert clean_ocr_text("ab-12 C|") == "12C"


def test_chars_match():
    assert chars_match(["V", "W", "2"], ["2", "W", "V"]) is True
    assert chars_match([], []) is False
    assert chars_match(["V"], ["V", "back_car"]) is False


def test_is_capture_frame():
    assert is_capture_frame(7) and is_capture_frame(14)
    assert not is_capture_frame(1) and not is_capture_frame(8)


def test_time_gate():
    gate = TimeGate()
    assert [gate.allow(t) for t in (0.0, 1.0, 1.49, 1.5, 2.0)] == [True, False, False, True, False]


def test_prepare_plate_is_gray_800_wide_and_keeps_aspect():
    out = prepare_plate(np.random.randint(0, 256, (60, 200, 3), dtype=np.uint8))
    assert out.shape == (240, 800) and out.dtype == np.uint8
