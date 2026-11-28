import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tools import merge_site_data as m

FFMPEG, FFPROBE = os.environ.get("FFMPEG", "ffmpeg"), os.environ.get("FFPROBE", "ffprobe")


def test_reel_keeps_1_s_before_and_2_s_after_each_capture_and_cuts_gaps_over_3_s():
    # 25 fps, 1-based capture frames; ranges are 0-based and inclusive
    assert m.reel_segments([10, 20, 200], 300) == [(0, 69), (174, 249)]  # 10 and 20 merge; 200 is 105 frames later
    assert m.reel_segments([100, 250], 300) == [(74, 299)]                 # a gap of exactly 3 s (75 frames) is kept
    assert m.reel_segments([290], 300) == [(264, 299)]                     # never past the last frame
    assert m.reel_segments([], 300) == []
    # a looser cut for the fixed-camera test videos: 5 s before, 8 s after, cut gaps over 15 s
    assert m.reel_segments([200, 600], 1000, cut=(5, 8, 15)) == [(74, 799)]   # 3 s apart after padding: kept
    assert m.reel_segments([100, 900], 1000, cut=(5, 8, 15)) == [(0, 299), (774, 999)]


def built(tmp_path, name, videos, captures):
    """A minimal build_site_data.py output: 10 s test-pattern playback videos and one record per capture frame."""
    d = tmp_path / name
    (d / "videos").mkdir(parents=True)
    (d / "crops").mkdir()
    (d / "plots").mkdir()
    records = []
    for v in videos:
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25:duration=10",
                        "-pix_fmt", "yuv420p", str(d / "videos" / f"{v}.mp4")], check=True)
        for f in captures:
            rid = f"{v}-{f:06d}-1"
            (d / "crops" / f"{rid}.jpg").write_bytes(b"jpg")
            records.append({"id": rid, "video": v, "frame": f, "time_s": round((f - 1) / 25, 2), "car_id": 1,
                            "plate": "AB1", "yolo_chars": ["A", "B", "1"], "n_chars": 3, "score": 0.9, "match": True,
                            "crop": f"data/crops/{rid}.jpg"})
    per_video = [{"video": v, "vehicles": 1, "captures": len(captures), "matched": len(captures), "duration_s": 10.0,
                  "playback": f"data/videos/{v}.mp4"} for v in videos]
    (d / "records.json").write_text(json.dumps(records), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps({"generated": "2026-11-26", "map50": 0.974, "per_video": per_video}),
                                    encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps({"final": "f"}), encoding="utf-8")
    return d


def frames_in(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=nb_read_frames,width,height", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


@pytest.mark.skipif(shutil.which(FFMPEG) is None, reason="ffmpeg not found: set FFMPEG")
def test_merge_combines_the_sources_and_cuts_one_reel_each(tmp_path):
    test = built(tmp_path, "test", ["A", "B"], [51, 200])  # 2 s before, 3 s after: frames 0-125 and 149-249, 24 apart
    dash = built(tmp_path, "dash", ["C"], [100])            # frames 74-149
    m.main(["--out", str(tmp_path / "site"), "--cut", "Test videos=2,3,4", f"Test videos={test}", f"Dashcam={dash}"])

    site = tmp_path / "site"
    records = json.loads((site / "records.json").read_text(encoding="utf-8"))
    s = json.loads((site / "summary.json").read_text(encoding="utf-8"))
    assert len(records) == 5 and all((site / r["crop"].removeprefix("data/")).is_file() for r in records)
    assert [(p["video"], p["source"], p["playback"]) for p in s["per_video"]] == [
        ("A", "Test videos", None), ("B", "Test videos", None), ("C", "Dashcam", None)]
    assert (s["videos"], s["captures"], s["matched"], s["map50"]) == (3, 5, 5, 0.974)
    assert (site / "metrics.json").is_file() and not (site / "videos" / "A.mp4").exists()

    first, second = s["reels"]
    assert (first["label"], first["src"], first["fps"]) == ("Test videos", "data/videos/reel-1.mp4", 25)
    assert first["cut"] == {"before": 2, "after": 3, "gap": 4} and second["cut"] == {"before": 1, "after": 2, "gap": 3}
    assert first["segments"] == [{"video": "A", "from": 0, "to": 249, "at": 0}, {"video": "B", "from": 0, "to": 249, "at": 250}]
    assert second["segments"] == [{"video": "C", "from": 74, "to": 149, "at": 0}]
    reel = frames_in(site / "videos" / "reel-1.mp4")
    assert int(reel["nb_read_frames"]) == 500  # 2 segments of 250 frames: the timeline's mapping relies on this
    assert int(frames_in(site / "videos" / "reel-2.mp4")["nb_read_frames"]) == 76


def test_merge_refuses_two_sources_with_the_same_video(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({"per_video": [{"video": "X"}]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="X"):
        m.check_sources([("one", a), ("two", b)])


def test_reel_cutting_never_reads_the_console(tmp_path, monkeypatch):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        if not kw.get("capture_output"):  # ffmpeg writes its last argument; ffprobe reports 250 frames
            Path(cmd[-1]).write_bytes(b"x")
        return type("Done", (), {"stdout": "250\n"})
    monkeypatch.setattr(m.subprocess, "run", run)
    m.cut_reel([("A", tmp_path / "A.mp4")], [{"video": "A", "frame": 30}], tmp_path / "reel.mp4", "ffmpeg", "ffprobe")
    ffmpeg = [c for c in calls if "ffprobe" not in c[0]]  # transcode_h264 uses the FFMPEG path
    assert len(ffmpeg) == 2 and all("-nostdin" in c for c in ffmpeg)  # the cut, then the transcode
