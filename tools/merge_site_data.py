"""Combine folders built by tools/build_site_data.py into the site's data, with one highlight reel per folder.

    python tools/merge_site_data.py --out web/data --cut "Test videos=5,8,15"         "Test videos=videos/built/testing_video" "Dashcam=videos/built/raw_video"

Each folder's annotated playback videos are cut down to the stretches where plates are being read (by default 1 s
before to 2 s after each capture, cutting gaps of more than 3 s) and joined into one reel, data/videos/reel-N.mp4.
--cut "Label=5,8,15" keeps 5 s before and 8 s after for that folder and cuts only gaps over 15 s: on a fixed camera a
cut makes the car jump, like a skipped frame. summary.json lists each reel's segments and cut, so the site can find
any capture in its reel.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.build_site_data import _write_json, preflight, publish, transcode_h264  # noqa: E402
from tools.site_data import summarize  # noqa: E402

REEL_FPS = 25                    # reels are written at a fixed 25 fps, so frame k of a reel plays at k / 25 s
CUT = (1, 2, 3)                  # default: keep 1 s before and 2 s after each capture, cut gaps of more than 3 s
REEL_MAX_MB = 45                 # GitHub warns about files over 50 MB
FONT = Path("C:/Windows/Fonts/arial.ttf")


def reel_segments(frames: list[int], n_frames: int, cut: tuple[int, int, int] = CUT,
                  fps: int = REEL_FPS) -> list[tuple[int, int]]:
    """1-based capture frames of one video -> the 0-based, inclusive frame ranges the reel keeps.

    cut is (seconds kept before each capture, seconds kept after, longest gap kept)."""
    pre, post, gap = cut
    segs: list[list[int]] = []
    for f in sorted(frames):
        a, b = max(f - 1 - pre * fps, 0), min(f - 1 + post * fps, n_frames - 1)
        if segs and a - segs[-1][1] <= gap * fps:
            segs[-1][1] = b
        else:
            segs.append([a, b])
    return [(a, b) for a, b in segs]


def frame_count(path: Path, ffprobe: str) -> int:
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    return int(out.strip())


def cut_reel(parts: list[tuple[str, Path]], records: list[dict], dst: Path, ffmpeg: str, ffprobe: str,
             cut: tuple[int, int, int] = CUT) -> list[dict]:
    """Join the capture stretches of each (video, playback file) into dst; returns the reel's segments in order."""
    segments, chains, inputs, at = [], [], [], 0
    font = "fontfile='" + FONT.as_posix().replace(":", r"\:") + "':" if FONT.is_file() else ""  # drawtext escapes C:
    for i, (video, path) in enumerate(parts):
        segs = reel_segments([r["frame"] for r in records if r["video"] == video], frame_count(path, ffprobe), cut)
        for a, b in segs:
            segments.append({"video": video, "from": a, "to": b, "at": at})
            at += b - a + 1
        keep = "+".join(f"between(n,{a},{b})" for a, b in segs)
        inputs += ["-i", str(path)]
        chains.append(f"[{i}:v]select='{keep}',setpts=N/{REEL_FPS}/TB,"
                      "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                      f"drawtext={font}text='{video}':x=20:y=h-50:fontsize=32:fontcolor=white:"
                      f"box=1:boxcolor=black@0.5:boxborderw=8[v{i}]")
    graph = ";".join(chains) + ";" + "".join(f"[v{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[out]"
    cut = dst.with_name(dst.stem + ".cut.mp4")
    subprocess.run([ffmpeg, "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[out]", "-r", str(REEL_FPS),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(cut)], check=True)
    transcode_h264(cut, dst, REEL_MAX_MB)
    cut.unlink()
    return segments


def check_sources(sources: list[tuple[str, Path]]) -> None:
    """Record ids and crops are named after the video, so a video may only come from one folder."""
    seen: dict[str, str] = {}
    for label, folder in sources:
        for p in json.loads((folder / "summary.json").read_text(encoding="utf-8"))["per_video"]:
            if p["video"] in seen:
                sys.exit(f"{p['video']} is in both {seen[p['video']]} and {label}")
            seen[p["video"]] = label


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "data")
    ap.add_argument("--cut", action="append", default=[], metavar="LABEL=BEFORE,AFTER,GAP",
                    help="seconds kept before and after each capture and the longest gap kept, for one folder")
    ap.add_argument("sources", nargs="+", help='"Label=folder" for each folder built by build_site_data.py')
    args = ap.parse_args(argv)
    sources = [(label, Path(folder)) for label, _, folder in (s.partition("=") for s in args.sources)]
    cuts = {label: tuple(int(x) for x in v.split(",")) for label, _, v in (c.partition("=") for c in args.cut)}
    if unknown := set(cuts) - {label for label, _ in sources}:
        sys.exit(f"--cut names no source: {', '.join(sorted(unknown))}")
    check_sources(sources)
    ffmpeg, ffprobe = os.environ.get("FFMPEG", "ffmpeg"), os.environ.get("FFPROBE", "ffprobe")
    final_out = args.out.resolve()
    preflight(final_out, ffmpeg, runs=())
    out = final_out.with_name(final_out.name + ".new")
    (out / "videos").mkdir(parents=True)

    records, infos, reels, label_of, generated = [], [], [], {}, []
    for n, (label, folder) in enumerate(sources, 1):
        recs = json.loads((folder / "records.json").read_text(encoding="utf-8"))
        s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        records += recs
        infos += [{"video": p["video"], "vehicles": p["vehicles"], "duration_s": p["duration_s"]} for p in s["per_video"]]
        label_of.update({p["video"]: label for p in s["per_video"]})
        generated.append(s["generated"])
        shutil.copytree(folder / "crops", out / "crops", dirs_exist_ok=True)
        parts = [(p["video"], folder / "videos" / f"{p['video']}.mp4") for p in s["per_video"] if p["playback"]]
        if parts:
            cut = cuts.get(label, CUT)
            reels.append({"label": label, "src": f"data/videos/reel-{n}.mp4", "fps": REEL_FPS,
                          "cut": dict(zip(("before", "after", "gap"), cut)),
                          "segments": cut_reel(parts, recs, out / "videos" / f"reel-{n}.mp4", ffmpeg, ffprobe, cut)})
        print(f"{label}: {len(s['per_video'])} videos, {len(recs)} captures, reel of {len(parts)} videos", flush=True)
    first = sources[0][1]
    shutil.copy2(first / "metrics.json", out / "metrics.json")  # same models for every folder
    shutil.copytree(first / "plots", out / "plots")
    summary = {"generated": max(generated), **summarize(records, infos, s["map50"], set())}
    for p in summary["per_video"]:
        p["source"] = label_of[p["video"]]
    summary["reels"] = reels
    _write_json(out / "records.json", records)
    _write_json(out / "summary.json", summary)
    publish(out, final_out)
    print(f"done: {summary['videos']} videos, {summary['captures']} captures, {summary['matched']} verified")


if __name__ == "__main__":
    main()
