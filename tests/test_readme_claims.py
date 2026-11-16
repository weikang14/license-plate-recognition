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


def test_readme_results_table_matches_the_data():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    records = json.loads((DATA / "records.json").read_text(encoding="utf-8"))
    cars = lambda rs: len({(r["video"], r["car_id"]) for r in rs})
    verified = [r for r in records if r["match"]]
    assert f"| Vehicles tracked (counted per video) | {summary['vehicles']} |" in text
    assert f"| {summary['captures']}, from {cars(records)} vehicles |" in text
    assert f"| {summary['matched']} ({summary['match_rate'] * 100:.1f}%), from {cars(verified)} vehicles |" in text


def test_readme_speed_table_matches_the_benchmark():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    speed = json.loads((ROOT / "web" / "speed.json").read_text(encoding="utf-8"))
    for device in speed.values():
        for key, value in device.items():
            if key.endswith("_ms"):
                assert f"{value} ms" in text, f"{device['hardware']} {key} = {value} ms is not in the README"
