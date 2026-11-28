"""Generate everything the dashboard shows, from the test videos. Needs Tesseract and ffmpeg.

    python tools/build_site_data.py --videos videos/testing_video --out web/data --playback 4 [--only test6 ...]

Env: TESSERACT_CMD, FFMPEG (default: found on PATH), OXR_MODEL (the OXR-1.0 weights folder; needs a CUDA GPU). Inputs are checked first; the build goes to <out>.new
and replaces <out> only when finished. An existing <out> is replaced only if an earlier run made it.
"""
from __future__ import annotations  # `X | None` hints on Python 3.9

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `python tools/build_site_data.py` can import tools.*

from tools.lpr_pipeline import Models, process_video  # noqa: E402
from tools.site_data import build_metrics, letter_model_scores, select_playback, summarize  # noqa: E402

TR = ROOT / "training_results"
RUNS = [
    ("yolov8s_yaml(3)", "YOLOv8s (final)", TR / "Results_yolov8s" / "yolov8s_yaml(3)"),
    ("yolov8n_yaml", "YOLOv8n", TR / "Results_yolov8n" / "yolov8n_yaml"),
    ("yolov8s_yaml(2)", "YOLOv8s (earlier run)", TR / "Results_yolov8s" / "yolov8s_yaml(2)"),
]
FINAL = "yolov8s_yaml(3)"
PUBLISHED = ROOT / "web" / "data"  # metrics.json + plots/ from those runs, in the repo: same model files
MB = 1024 * 1024


def transcode_h264(src: Path, dst: Path, max_mb: float = 25) -> None:
    """Browser-playable H.264; step quality down until the file fits max_mb."""
    ffmpeg = os.environ.get("FFMPEG", "ffmpeg")
    for height, crf in ((720, 28), (720, 32), (720, 36), (540, 36), (360, 36)):  # 360p: 5-min dashcam clips
        subprocess.run([ffmpeg, "-nostdin", "-y", "-v", "error", "-i", str(src), "-vf", f"scale=-2:{height}",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", "-an", str(dst)], check=True)
        if dst.stat().st_size <= max_mb * MB:
            return
    raise RuntimeError(f"{dst.name} is still over {max_mb} MB at 360p crf 36")


def have_runs(runs) -> bool:
    return all((Path(d) / "results.csv").is_file() for _, _, d in runs)


def preflight(out: Path, ffmpeg: str, runs=RUNS, published: Path = PUBLISHED) -> None:
    """Fail in seconds, not hours: check inputs and refuse to replace a folder this script didn't make."""
    if shutil.which(ffmpeg) is None:
        sys.exit(f"ffmpeg not found: {ffmpeg} (set FFMPEG or add it to PATH)")
    if not have_runs(runs) and not (published / "metrics.json").is_file():
        sys.exit(f"neither training_results/ (local-only) nor {published / 'metrics.json'} found")
    if out.exists() and any(out.iterdir()) and not (out / "summary.json").is_file():
        sys.exit(f"refusing to replace {out}: it is not empty and was not made by this script")
    staging = out.with_name(out.name + ".new")
    if staging.exists():
        sys.exit(f"leftover {staging} from an interrupted run; delete it and retry")


def training_metrics(plots_out: Path, runs=RUNS, published: Path = PUBLISHED) -> dict:
    """The Models page data from the training runs; without them (a fresh clone), the copy published in the repo."""
    if have_runs(runs):
        return build_metrics(runs, FINAL, plots_out)
    print(f"training_results/ not found: reusing {published / 'metrics.json'} and its plots", flush=True)
    shutil.copytree(published / "plots", plots_out, dirs_exist_ok=True)
    return json.loads((published / "metrics.json").read_text(encoding="utf-8"))


def publish(staging: Path, out: Path) -> None:
    """Swap the finished build into place, so a failed run never leaves the site half-written."""
    if out.exists():
        shutil.rmtree(out)
    staging.rename(out)


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, default=ROOT / "videos" / "testing_video")
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "data")
    ap.add_argument("--playback", type=int, default=4, help="how many annotated videos to publish")
    ap.add_argument("--only", nargs="*", help="video stems to process (default: all)")
    args = ap.parse_args(argv)

    videos = sorted(p for p in args.videos.glob("*.mp4") if not args.only or p.stem in args.only)
    if not videos:
        sys.exit(f"no .mp4 videos found in {args.videos}")
    if not os.environ.get("OXR_MODEL"):
        sys.exit("OXR_MODEL is not set: the site's cross-check needs OXR (see README)")
    final_out = args.out.resolve()
    preflight(final_out, os.environ.get("FFMPEG", "ffmpeg"))
    out = final_out.with_name(final_out.name + ".new")  # build here, publish() at the end
    (out / "videos").mkdir(parents=True)
    models = Models.load(ROOT / "models")

    # 1. Records for every video (fast mode: capture frames only).
    records, infos = [], []
    for i, v in enumerate(videos, 1):
        t0 = time.time()
        rs, info = process_video(v, models, out / "crops", progress=True)
        records += rs
        infos.append(info)
        print(f"[{i}/{len(videos)}] {v.stem}: {len(rs)} captures, {sum(r['match'] for r in rs)} matched, "
              f"{time.time() - t0:.0f} s", flush=True)

    metrics = training_metrics(out / "plots")
    metrics["letter"] = letter_model_scores(ROOT / "models" / "plate_letter.pt")
    map50 = next(r for r in metrics["runs"] if r["key"] == FINAL)["best"]["map50"]
    chosen = select_playback(summarize(records, infos, map50, set())["per_video"], args.playback)

    # 2. Annotated playback videos for the best few; must reproduce the fast-mode records exactly.
    work = Path(os.environ.get("TEMP", tempfile.gettempdir())) / "lpr_build"
    work.mkdir(parents=True, exist_ok=True)
    strip = lambda rs: [{k: v for k, v in r.items() if k != "crop"} for r in rs]
    print(f"playback videos for {len(chosen)} of {len(videos)} videos: {chosen}", flush=True)
    for i, stem in enumerate(chosen, 1):
        t0 = time.time()
        raw = work / f"{stem}.mp4"
        rs, _ = process_video(next(v for v in videos if v.stem == stem), models, None, annotate_to=raw, progress=True)
        if strip(rs) != strip([r for r in records if r["video"] == stem]):
            raise RuntimeError(f"{stem}: annotate mode produced different records than fast mode")
        print(f"[{i}/{len(chosen)}] {stem}: compressing the playback video", flush=True)
        transcode_h264(raw, out / "videos" / f"{stem}.mp4")
        raw.unlink()
        print(f"[{i}/{len(chosen)}] {stem}: playback video written, {time.time() - t0:.0f} s", flush=True)

    # 3. JSON for the site.
    for r in records:
        r["crop"] = f"data/crops/{r['crop']}"
    _write_json(out / "records.json", records)
    _write_json(out / "summary.json", {"generated": date.today().isoformat(),
                                       **summarize(records, infos, map50, set(chosen))})
    _write_json(out / "metrics.json", metrics)
    publish(out, final_out)
    print(f"done: {len(records)} captures, {sum(r['match'] for r in records)} matched, playback: {chosen}")


if __name__ == "__main__":
    main()
