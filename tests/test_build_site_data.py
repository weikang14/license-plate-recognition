"""Safety checks that run before the hours-long build touches anything."""
import sys
from pathlib import Path

import pytest

from tools import build_site_data as b

EXE = sys.executable  # any executable that exists stands in for ffmpeg


@pytest.fixture
def runs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.csv").write_text("epoch\n")
    return [("k", "label", run)]


def test_preflight_refuses_to_wipe_an_unrelated_folder(tmp_path, runs):
    out = tmp_path / "out"
    out.mkdir()
    (out / "precious.txt").write_text("keep me")
    with pytest.raises(SystemExit):
        b.preflight(out, EXE, runs)
    assert (out / "precious.txt").is_file()


def test_preflight_accepts_previous_output_or_new_folder(tmp_path, runs):
    old = tmp_path / "old"
    old.mkdir()
    (old / "summary.json").write_text("{}")
    b.preflight(old, EXE, runs)
    b.preflight(tmp_path / "fresh", EXE, runs)


def test_preflight_needs_ffmpeg(tmp_path, runs):
    with pytest.raises(SystemExit):
        b.preflight(tmp_path / "out", "no-such-ffmpeg-binary", runs)


def test_preflight_needs_training_runs_or_the_published_metrics(tmp_path):
    missing, pub = [("k", "label", tmp_path / "missing")], tmp_path / "pub"
    pub.mkdir()
    with pytest.raises(SystemExit):
        b.preflight(tmp_path / "out", EXE, missing, pub)
    (pub / "metrics.json").write_text("{}")
    b.preflight(tmp_path / "out", EXE, missing, pub)  # a fresh clone has no training_results/


def test_training_metrics_falls_back_to_the_published_copy(tmp_path):
    pub = tmp_path / "pub"
    (pub / "plots").mkdir(parents=True)
    (pub / "plots" / "F1_curve.png").write_bytes(b"png")
    (pub / "metrics.json").write_text('{"final": "k"}')
    plots = tmp_path / "out" / "plots"
    assert b.training_metrics(plots, [("k", "label", tmp_path / "missing")], pub) == {"final": "k"}
    assert (plots / "F1_curve.png").read_bytes() == b"png"


def test_preflight_refuses_leftover_staging(tmp_path, runs):
    (tmp_path / "out.new").mkdir()
    with pytest.raises(SystemExit):
        b.preflight(tmp_path / "out", EXE, runs)


def test_publish_swaps_staging_into_place(tmp_path):
    out, staging = tmp_path / "out", tmp_path / "out.new"
    out.mkdir()
    (out / "old.json").write_text("old")
    staging.mkdir()
    (staging / "new.json").write_text("new")
    b.publish(staging, out)
    assert [p.name for p in out.iterdir()] == ["new.json"] and not staging.exists()


def test_ffmpeg_never_reads_the_console(tmp_path, monkeypatch):
    # A key pressed in the build window once left ffmpeg waiting at its "Enter command:" prompt, stalling the build.
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"x")
    monkeypatch.setattr(b.subprocess, "run", run)
    b.transcode_h264(tmp_path / "in.mp4", tmp_path / "out.mp4")
    assert calls and all("-nostdin" in c for c in calls)
