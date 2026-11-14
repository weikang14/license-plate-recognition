from pathlib import Path

import pytest

from tools.site_data import build_metrics, load_run, select_playback, summarize

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "training_results" / "Results_yolov8s" / "yolov8s_yaml(3)"
NANO = ROOT / "training_results" / "Results_yolov8n" / "yolov8n_yaml"
needs_runs = pytest.mark.skipif(not (FINAL.is_dir() and NANO.is_dir()),
                                reason="training_results/ is local-only (not in the repository)")


def rec(video, score, match):
    return {"video": video, "score": score, "match": match}


@needs_runs
def test_load_run_final_metrics():
    run = load_run(FINAL)
    assert run["epochs_trained"] == 375
    assert run["best"] == {"epoch": 275, "precision": 0.951, "recall": 0.957, "map50": 0.974, "map50_95": 0.651}
    assert len(run["curves"]["epoch"]) == 375


@needs_runs
def test_load_run_yolov8n():
    run = load_run(NANO)
    assert run["epochs_trained"] == 198
    assert run["best"]["epoch"] == 98
    assert run["best"]["map50"] == 0.970
    assert run["best"]["map50_95"] == 0.637


@needs_runs
def test_build_metrics_copies_plots(tmp_path):
    m = build_metrics([("final", "YOLOv8s (final)", FINAL), ("nano", "YOLOv8n", NANO)], "final", tmp_path)
    names = {"confusion_matrix.png", "PR_curve.png", "F1_curve.png",
             "val_batch0_pred.jpg", "val_batch1_pred.jpg", "val_batch2_pred.jpg"}
    assert names <= {p.name for p in tmp_path.iterdir()}
    assert m["final"] == "final" and [r["key"] for r in m["runs"]] == ["final", "nano"]
    assert len(m["plots"]["val_pred"]) == 3
    paths = [m["plots"]["confusion_matrix"], m["plots"]["pr_curve"], m["plots"]["f1_curve"], *m["plots"]["val_pred"]]
    assert all(p.startswith("data/plots/") for p in paths)


def test_summarize_counts():
    records = [rec("a", 0.80, False), rec("a", 1.00, True), rec("b", 0.90, True)]
    infos = [{"video": "a", "duration_s": 10.0, "vehicles": 3}, {"video": "b", "duration_s": 20.0, "vehicles": 1}]
    s = summarize(records, infos, 0.974, {"a"})
    assert (s["videos"], s["vehicles"], s["captures"], s["matched"]) == (2, 4, 3, 2)
    assert s["match_rate"] == 0.667
    assert s["avg_score"] == 0.9
    assert s["map50"] == 0.974
    assert s["score_hist"]["edges"] == [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    assert s["score_hist"]["counts"] == [0, 1, 0, 1, 1]
    a, b = s["per_video"]
    assert a == {"video": "a", "vehicles": 3, "captures": 2, "matched": 1, "duration_s": 10.0,
                 "playback": "data/videos/a.mp4"}
    assert b["playback"] is None


def test_summarize_zero_capture_video():
    infos = [{"video": "a", "duration_s": 5.0, "vehicles": 1}, {"video": "c", "duration_s": 7.0, "vehicles": 0}]
    s = summarize([rec("a", 0.8, True)], infos, 0.974, set())
    c = s["per_video"][1]
    assert (c["video"], c["captures"], c["matched"]) == ("c", 0, 0)

    empty = summarize([], infos, 0.974, set())
    assert empty["match_rate"] == 0 and empty["avg_score"] == 0


def test_select_playback():
    per_video = [{"video": f"v{i}", "matched": m, "captures": c}
                 for i, (m, c) in enumerate(zip([5, 0, 5, 9, 1, 2], [6, 3, 8, 9, 1, 2]))]
    assert select_playback(per_video, k=4) == ["v3", "v2", "v0", "v5"]
