import numpy as np

from tools.lpr_pipeline import TimeGate, chars_match, clean_ocr_text, in_any_car, is_capture_frame, warp_plate


def test_in_any_car_uses_the_plate_centre():
    cars = [(100, 100, 400, 300), (500, 100, 800, 300)]
    assert in_any_car((350, 250, 450, 280), cars)        # centre (400, 265) on the first car's edge
    assert in_any_car((600, 270, 700, 320), cars)        # pokes out below, centre (650, 295) still inside
    assert not in_any_car((600, 280, 700, 330), cars)    # centre (650, 305) is below the car
    assert not in_any_car((420, 200, 480, 230), cars)    # between the two cars
    assert not in_any_car((150, 150, 250, 200), [])      # no car detected


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


def test_warp_plate_shape():
    out = warp_plate(np.random.randint(0, 256, (60, 200, 3), dtype=np.uint8))
    assert out.shape == (310, 630) and out.dtype == np.uint8
