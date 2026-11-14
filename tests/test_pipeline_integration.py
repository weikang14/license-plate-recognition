"""Runs the real models + Tesseract on the first 700 frames of test6 (a 0.87 plate appears near frame 240)."""
import os
from pathlib import Path

import cv2
import pytest

from tools.lpr_pipeline import Models, process_video

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "videos" / "testing_video" / "test6.mp4"
FIELDS = {"id", "video", "frame", "time_s", "car_id", "plate", "yolo_chars", "n_chars", "score", "match", "crop"}

pytestmark = [
    pytest.mark.skipif(not Path(os.environ.get("TESSERACT_CMD", "")).is_file(), reason="Tesseract not found: set TESSERACT_CMD"),
    pytest.mark.skipif(not VIDEO.is_file(), reason="test videos are local-only (not in the repository)"),
]


@pytest.fixture(scope="module")
def models():
    return Models.load(ROOT / "models")


@pytest.fixture(scope="module")
def fast(models, tmp_path_factory):
    crops = tmp_path_factory.mktemp("crops")
    records, info = process_video(VIDEO, models, crops, max_frames=700)
    return records, info, crops


def test_fast_mode_records(fast):
    records, info, crops = fast
    assert info["video"] == "test6" and info["frames"] == 700 and info["vehicles"] >= 1
    assert len(records) >= 1
    for r in records:
        assert set(r) == FIELDS
        assert r["score"] >= 0.75 and r["frame"] % 7 == 0 and isinstance(r["car_id"], int)
        assert (crops / r["crop"]).is_file()
    times = [r["time_s"] for r in records]
    assert all(b - a >= 1.5 - 0.01 for a, b in zip(times, times[1:]))


def test_annotate_matches_fast(models, fast, tmp_path):
    out = tmp_path / "annotated.mp4"
    records, info = process_video(VIDEO, models, None, annotate_to=out, max_frames=700)
    strip = lambda rs: [{k: v for k, v in r.items() if k != "crop"} for r in rs]
    assert strip(records) == strip(fast[0])
    cap = cv2.VideoCapture(str(out))
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(info["fps"], abs=0.01)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 700
