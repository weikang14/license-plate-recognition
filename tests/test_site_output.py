"""Checks the generated web/data against the site's contract. Skipped until build_site_data.py has run."""
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DATA = WEB / "data"
SOURCE_VIDEOS = ROOT / "videos" / "testing_video"
FIELDS = {"id", "video", "frame", "time_s", "car_id", "plate", "yolo_chars", "n_chars", "score", "match", "crop"}
MB = 1024 * 1024

pytestmark = pytest.mark.skipif(not (DATA / "summary.json").is_file(), reason="run tools/build_site_data.py first")


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def probe(path):
    out = subprocess.run([os.environ.get("FFPROBE", "ffprobe"), "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=codec_name,pix_fmt,avg_frame_rate,duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def rate(fraction):
    num, den = fraction.split("/")
    return int(num) / int(den)


def test_records_schema():
    records = load("records.json")
    assert len({r["id"] for r in records}) == len(records)
    for r in records:
        assert set(r) == FIELDS
        assert r["crop"].startswith("data/crops/") and (WEB / r["crop"]).is_file()


def test_summary_consistent():
    records, s = load("records.json"), load("summary.json")
    assert s["captures"] == len(records)
    assert s["matched"] == sum(r["match"] for r in records)
    assert len(s["per_video"]) == 15
    assert s["vehicles"] == sum(p["vehicles"] for p in s["per_video"])
    assert all(isinstance(r["car_id"], int) for r in records)
    for p in s["per_video"]:
        if p["playback"]:
            assert (WEB / p["playback"]).is_file()


def test_videos_are_browser_playable():
    videos = sorted((DATA / "videos").glob("*.mp4"))
    assert videos
    for v in videos:
        info = probe(v)
        assert (info["codec_name"], info["pix_fmt"]) == ("h264", "yuv420p"), v.name
        assert v.stat().st_size <= 25 * MB, v.name
        # Some sources are variable frame rate (nominal 25/1, real ~24.5); records use the real average,
        # so the web video must keep the source's average rate and duration for seeking to line up.
        # The source videos are local-only, so this part only runs where they exist.
        if not (SOURCE_VIDEOS / v.name).is_file():
            continue
        src = probe(SOURCE_VIDEOS / v.name)
        assert rate(info["avg_frame_rate"]) == pytest.approx(rate(src["avg_frame_rate"]), abs=0.01), v.name
        assert float(info["duration"]) == pytest.approx(float(src["duration"]), abs=0.05), v.name


def test_site_budget():
    files = [p for p in WEB.rglob("*") if p.is_file()]
    assert sum(p.stat().st_size for p in files) <= 200 * MB
    assert all(p.stat().st_size < 100 * MB for p in files)


def test_metrics_final():
    m = load("metrics.json")
    assert m["final"] == "yolov8s_yaml(3)"
    final = next(r for r in m["runs"] if r["key"] == m["final"])
    assert final["best"]["map50"] == 0.974
