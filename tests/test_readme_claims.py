"""README claims about results must agree with the published data."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "web" / "data"


def test_readme_failure_claims_match_records():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    records = json.loads((DATA / "records.json").read_text(encoding="utf-8"))
    empty = sum(not r["plate"] for r in records)
    verified = {r["plate"] for r in records if r["match"]}
    assert f"returned no text on {empty} of {len(records)} captures" in text
    if len(verified) == 1:
        plate = next(iter(verified))
        assert f"All {sum(r['match'] for r in records)} verified captures are the same plate" in text
        assert plate[:3] in text
    assert "often reads the same plate correctly" not in text
