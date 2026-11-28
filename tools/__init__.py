"""Every tool imports this package first, so local.env (written by tools/setup_env.py) is loaded before anything runs."""
import os
from pathlib import Path


def load_env(path: Path) -> None:
    """KEY=VALUE lines; a variable already set in the shell wins."""
    if Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip())


load_env(Path(__file__).resolve().parent.parent / "local.env")
