import numpy as np

from tools.lpr_pipeline import TimeGate, chars_match, clean_ocr_text, is_capture_frame, warp_plate


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
