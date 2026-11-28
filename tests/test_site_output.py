"""Checks the generated web/data against the site's contract. Skipped until build_site_data.py has run."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DATA = WEB / "data"
FIELDS = {"id", "video", "frame", "time_s", "car_id", "plate", "yolo_chars", "n_chars", "score", "match", "crop", "oxr", "agreed"}
MB = 1024 * 1024
FFPROBE = os.environ.get("FFPROBE", "ffprobe")

pytestmark = pytest.mark.skipif(not (DATA / "summary.json").is_file(), reason="run tools/build_site_data.py first")
needs_ffprobe = pytest.mark.skipif(shutil.which(FFPROBE) is None, reason="ffprobe not found: set FFPROBE (it comes with ffmpeg)")


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def probe(path, entries="codec_name,pix_fmt,avg_frame_rate,duration"):
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                          "-show_entries", f"stream={entries}", "-of", "json", str(path)],
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
    sources = [p["source"] for p in s["per_video"]]
    assert (sources.count("Test videos"), sources.count("Dashcam")) == (15, 14) and s["videos"] == 29
    assert s["vehicles"] == sum(p["vehicles"] for p in s["per_video"])
    assert all(isinstance(r["car_id"], int) for r in records)


@needs_ffprobe
def test_each_source_has_a_reel_of_its_videos_with_a_verified_plate():
    records, s = load("records.json"), load("summary.json")
    assert [g["label"] for g in s["reels"]] == ["Test videos", "Dashcam"]
    for g in s["reels"]:
        mine = {p["video"] for p in s["per_video"] if p["source"] == g["label"]}
        assert {x["video"] for x in g["segments"]} == {p["video"] for p in s["per_video"] if p["video"] in mine and p["matched"]}
        in_reel = [r for r in records if any(x["video"] == r["video"] and x["from"] <= r["frame"] - 1 <= x["to"]
                                             for x in g["segments"])]
        assert len(in_reel) == sum(p["captures"] for p in s["per_video"] if p["video"] in mine and p["matched"])
        # The timeline maps a capture to frame `at + offset` of the reel, so the reel must hold exactly these frames.
        frames = sum(x["to"] - x["from"] + 1 for x in g["segments"])
        assert int(probe(WEB / g["src"], "nb_frames")["nb_frames"]) == frames, g["label"]


@needs_ffprobe
def test_videos_are_browser_playable():
    videos = sorted((DATA / "videos").glob("*.mp4"))
    assert [v.name for v in videos] == ["reel-1.mp4", "reel-2.mp4"]
    for v in videos:
        info = probe(v)
        assert (info["codec_name"], info["pix_fmt"]) == ("h264", "yuv420p"), v.name
        assert rate(info["avg_frame_rate"]) == 25, v.name
        assert v.stat().st_size <= 45 * MB, v.name


def test_site_budget():
    files = [p for p in WEB.rglob("*") if p.is_file()]
    assert sum(p.stat().st_size for p in files) <= 200 * MB
    assert all(p.stat().st_size < 100 * MB for p in files)


def test_metrics_final():
    m = load("metrics.json")
    assert m["final"] == "yolov8s_yaml(3)"
    final = next(r for r in m["runs"] if r["key"] == m["final"])
    assert final["best"]["map50"] == 0.974
    assert m["letter"] == {"precision": 0.949, "recall": 0.904, "map50": 0.96, "map50_95": 0.789}


def test_letter_figures_are_published():
    for name in ("letter-training.png", "letter-detection.png"):
        assert (WEB / "img" / name).stat().st_size > 10_000
