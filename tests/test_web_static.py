"""The site is served from a sub-path (/license-plate-recognition/), so every URL in web/ must be relative."""
import re
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"
ROOT_ABSOLUTE = re.compile(r"""(?:src|href)\s*=\s*["']/|fetch\(\s*["'`]/|url\(\s*["']?/""")


def test_no_regex_lookbehind():
    """Safari < 16.4 can't parse lookbehind; one such regex stops every module from loading."""
    bad = [f"{p.relative_to(WEB)}:{n}" for p in WEB.rglob("*.js")
           for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if "(?<=" in line or "(?<!" in line]
    assert not bad, "regex lookbehind breaks older Safari:\n" + "\n".join(bad)


def test_no_root_absolute_urls():
    bad = []
    for path in WEB.rglob("*"):
        if path.suffix in {".html", ".js", ".css"}:
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if ROOT_ABSOLUTE.search(line):
                    bad.append(f"{path.relative_to(WEB)}:{n}: {line.strip()}")
    assert not bad, "root-absolute URLs break on GitHub Pages:\n" + "\n".join(bad)
