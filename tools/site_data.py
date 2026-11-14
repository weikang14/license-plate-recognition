"""Pure data shaping for the dashboard: training metrics, recognition summary, playback choice."""
import csv
import shutil
from bisect import bisect_right
from pathlib import Path

PLOTS = {
    "confusion_matrix": "confusion_matrix.png",
    "pr_curve": "PR_curve.png",
    "f1_curve": "F1_curve.png",
}
VAL_PRED = ["val_batch0_pred.jpg", "val_batch1_pred.jpg", "val_batch2_pred.jpg"]
SCORE_EDGES = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]  # captures need score >= 0.75


def load_run(run_dir: Path) -> dict:
    """Read an Ultralytics results.csv; best epoch = highest mAP50-95."""
    with open(Path(run_dir) / "results.csv", newline="") as f:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    best = max(rows, key=lambda r: r["metrics/mAP50-95(B)"])
    col = lambda key: [round(r[key], 4) for r in rows]
    return {
        "epochs_trained": len(rows),
        "best": {
            "epoch": int(best["epoch"]),
            "precision": round(best["metrics/precision(B)"], 3),
            "recall": round(best["metrics/recall(B)"], 3),
            "map50": round(best["metrics/mAP50(B)"], 3),
            "map50_95": round(best["metrics/mAP50-95(B)"], 3),
        },
        "curves": {
            "epoch": [int(r["epoch"]) for r in rows],
            "train_box_loss": col("train/box_loss"),
            "val_box_loss": col("val/box_loss"),
            "map50": col("metrics/mAP50(B)"),
            "map50_95": col("metrics/mAP50-95(B)"),
        },
    }


def build_metrics(runs: list[tuple[str, str, Path]], final_key: str, plots_out: Path) -> dict:
    """Load every run and copy the final run's plot images into plots_out."""
    plots_out = Path(plots_out)
    plots_out.mkdir(parents=True, exist_ok=True)
    final_dir = next(d for k, _, d in runs if k == final_key)
    for name in [*PLOTS.values(), *VAL_PRED]:
        shutil.copy2(Path(final_dir) / name, plots_out / name)
    return {
        "final": final_key,
        "runs": [{"key": k, "label": label, **load_run(d)} for k, label, d in runs],
        "plots": {**{k: f"data/plots/{v}" for k, v in PLOTS.items()},
                  "val_pred": [f"data/plots/{v}" for v in VAL_PRED]},
    }


def summarize(records: list[dict], infos: list[dict], map50: float, playback: set[str]) -> dict:
    """Overall and per-video counts; rates are over all captures (0 when there are none)."""
    n = len(records)
    matched = sum(r["match"] for r in records)
    counts = [0] * (len(SCORE_EDGES) - 1)
    for r in records:
        counts[min(max(bisect_right(SCORE_EDGES, r["score"]) - 1, 0), len(counts) - 1)] += 1
    per_video = []
    for info in infos:
        v = info["video"]
        mine = [r for r in records if r["video"] == v]
        per_video.append({
            "video": v,
            "captures": len(mine),
            "matched": sum(r["match"] for r in mine),
            "duration_s": info["duration_s"],
            "playback": f"data/videos/{v}.mp4" if v in playback else None,
        })
    return {
        "videos": len(infos),
        "captures": n,
        "matched": matched,
        "match_rate": round(matched / n, 3) if n else 0,
        "avg_score": round(sum(r["score"] for r in records) / n, 3) if n else 0,
        "map50": map50,
        "per_video": per_video,
        "score_hist": {"edges": SCORE_EDGES, "counts": counts},
    }


def select_playback(per_video: list[dict], k: int = 4) -> list[str]:
    """Videos with the most verified plates (ties: more captures); never ones with none."""
    ranked = sorted((p for p in per_video if p["matched"] > 0), key=lambda p: (-p["matched"], -p["captures"]))
    return [p["video"] for p in ranked[:k]]
