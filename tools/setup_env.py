"""Set up what the pipeline needs, inside this folder. Needs Python 3.12 and an NVIDIA GPU.

    python tools/setup_env.py [--cuda cu128]

1. .venv: PyTorch with CUDA (cu128 by default; cu126 for older drivers, cu130 is what built the site) and requirements.txt
2. oxr_model/: the OXR-1.0 weights from Hugging Face (3.1 GB), the version the site was built with
3. local.env: where OXR, Tesseract and ffmpeg are; every tool reads it (tools/__init__.py)
Tesseract and ffmpeg are installed separately: it says how if it can't find them. Safe to run again.
"""
from __future__ import annotations  # so an older Python reaches the version warning

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
OXR = ROOT / "oxr_model"
OXR_REVISION = "c708407f9d2b4df658f2543d9132b62bb7221df4"
WIN_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")  # the UB-Mannheim installer's default
INSTALL = {  # what to run when a program is missing
    "TESSERACT_CMD": ("winget install -e --id UB-Mannheim.TesseractOCR", "sudo apt install tesseract-ocr"),
    "FFMPEG": ("winget install -e --id Gyan.FFmpeg", "sudo apt install ffmpeg"),
}


def find(var: str, program: str) -> str | None:
    """The path the shell variable gives, else the program on PATH, else Tesseract where its Windows installer puts it."""
    if found := os.environ.get(var) or shutil.which(program):
        return found
    if program == "tesseract" and os.name == "nt" and WIN_TESSERACT.is_file():
        return str(WIN_TESSERACT)  # the installer does not add itself to PATH
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuda", default="cu128", choices=["cu126", "cu128", "cu130"], help="PyTorch CUDA build")
    args = ap.parse_args()
    if sys.version_info[:2] != (3, 12):
        print(f"warning: the pinned packages were tested on Python 3.12, this is {sys.version.split()[0]}")
    paths = {"OXR_MODEL": str(OXR), "TESSERACT_CMD": find("TESSERACT_CMD", "tesseract"),
             "FFMPEG": find("FFMPEG", "ffmpeg"), "FFPROBE": find("FFPROBE", "ffprobe")}
    missing = [k for k in INSTALL if not paths[k]] + (["FFMPEG"] if paths["FFMPEG"] and not paths["FFPROBE"] else [])
    if missing:  # before the 6 GB of downloads
        print("not found; install it, open a new terminal and run this script again (or set the variable to its path):")
        for k in dict.fromkeys(missing):
            print(f"  {k}: " + INSTALL[k][os.name != "nt"])
        sys.exit(1)

    if not PY.is_file():
        print("1/3 creating .venv", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    pip = [str(PY), "-m", "pip", "install", "--disable-pip-version-check"]
    print(f"1/3 installing PyTorch ({args.cuda}) and requirements.txt into .venv", flush=True)
    subprocess.run([*pip, "torch==2.11.0", "torchvision==0.26.0", "--index-url",
                    f"https://download.pytorch.org/whl/{args.cuda}"], check=True)
    subprocess.run([*pip, "-r", str(ROOT / "requirements.txt")], check=True)

    if not (OXR / "model.safetensors").is_file():
        print("2/3 downloading the OXR-1.0 weights (3.1 GB) into oxr_model/", flush=True)
        subprocess.run([str(PY), "-c", "from huggingface_hub import snapshot_download; "
                        f"snapshot_download('SoMarkAI/OXR-1.0', revision={OXR_REVISION!r}, local_dir={str(OXR)!r})"],
                       check=True)

    (ROOT / "local.env").write_text("# written by tools/setup_env.py; a variable set in the shell wins\n"
                                    + "".join(f"{k}={v}\n" for k, v in paths.items()), encoding="utf-8")
    print("3/3 wrote local.env", flush=True)
    run = r".venv\Scripts\python" if os.name == "nt" else ".venv/bin/python"
    print(f"\nready. Next: {run} tools/build_site_data.py --help  (README: Download and run)")


if __name__ == "__main__":
    main()
