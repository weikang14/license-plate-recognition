"""Time each pipeline stage on one 1080p test frame and one plate crop, and record it in web/speed.json.

    python tools/benchmark.py --device cpu
    python tools/benchmark.py --device 0      # first CUDA GPU (needs a CUDA build of PyTorch)

Stand-alone on purpose (no lpr_pipeline import), so it also runs in an environment without pytesseract;
the Tesseract step is then skipped.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def ms(fn, n: int) -> float:
    for _ in range(3):  # warm-up (CUDA init, first-call allocations)
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return round((time.perf_counter() - t0) / n * 1000, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", help="cpu, or a CUDA device index such as 0")
    ap.add_argument("--video", type=Path, default=ROOT / "videos" / "testing_video" / "test6.mp4")
    ap.add_argument("--crop", type=Path, default=ROOT / "web" / "data" / "crops" / "test6-000427.jpg")
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "speed.json")
    ap.add_argument("--hardware", help="name to show for the device (default: what Python reports)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 3600)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read a frame from {args.video}")
    gray = cv2.cvtColor(cv2.imread(str(args.crop)), cv2.COLOR_BGR2GRAY)  # same preparation as lpr_pipeline.prepare_plate
    plate = cv2.resize(gray, (800, int(gray.shape[0] * 800 / gray.shape[1])), interpolation=cv2.INTER_CUBIC)

    d = args.device
    car, det, letter = (YOLO(str(ROOT / "models" / m)) for m in ("yolov8n.pt", "plate.pt", "plate_letter.pt"))
    stages = {
        "vehicle_ms": ms(lambda: car(frame, conf=0.5, agnostic_nms=True, device=d, verbose=False), 30),
        "plate_ms": ms(lambda: det(frame, conf=0.5, device=d, verbose=False), 30),
        "characters_ms": ms(lambda: letter(cv2.cvtColor(plate, cv2.COLOR_GRAY2RGB), conf=0.4, device=d, verbose=False), 30),
    }
    try:
        import os
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_CMD", "tesseract")
        stages["tesseract_ms"] = ms(lambda: pytesseract.image_to_data(cv2.bitwise_not(plate), config="--psm 6"), 10)
    except ImportError:
        pass

    hw = args.hardware or (torch.cuda.get_device_name(int(d)) if d != "cpu" else platform.processor() or "CPU")
    result = {"hardware": hw, "frame": f"{frame.shape[1]}x{frame.shape[0]}", "torch": torch.__version__,
              "ultralytics": ultralytics.__version__, **stages}
    print(json.dumps(result))
    data = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    data["gpu" if d != "cpu" else "cpu"] = result
    args.out.write_text(json.dumps(data, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
