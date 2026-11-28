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


def confirmed_cars(records):
    """Cars whose same agreed reading passed the cross-check twice (the pipeline's rule for confirming a plate)."""
    passed, done = set(), set()
    for r in sorted(records, key=lambda r: (r["video"], r["frame"])):
        if r["match"]:
            key = (r["video"], r["car_id"], r["agreed"])
            (done.add(key[:2]) if key in passed else passed.add(key))
    return done


def test_readme_results_table_matches_the_data():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    records = json.loads((DATA / "records.json").read_text(encoding="utf-8"))
    sources = list(dict.fromkeys(p["source"] for p in summary["per_video"]))
    columns = [[p for p in summary["per_video"] if p["source"] == s] for s in sources] + [summary["per_video"]]
    cars = lambda rs: len({(r["video"], r["car_id"]) for r in rs})

    def row(label, cell):
        cells = []
        for per in columns:
            mine = {p["video"] for p in per}
            cells.append(cell(per, [r for r in records if r["video"] in mine]))
        return f"| {label} | " + " | ".join(cells) + " |"

    assert "| | " + " | ".join(sources) + " | All |" in text
    assert row("Videos", lambda per, rs: str(len(per))) in text
    assert row("Vehicles tracked (counted per video)", lambda per, rs: str(sum(p["vehicles"] for p in per))) in text
    assert row("Plate captures (plates on a tracked car)", lambda per, rs: f"{len(rs)}, from {cars(rs)} vehicles") in text
    verified = lambda per, rs: (lambda v: f"{len(v)} ({len(v) / len(rs) * 100:.1f}%), from {cars(v)} vehicles")(
        [r for r in rs if r["match"]])
    assert row("Verified by the cross-check (2 of 3 readers agree)", verified) in text
    assert row("Vehicles with a confirmed plate (same reading verified twice)",
               lambda per, rs: f"{len(confirmed_cars(rs))} of {cars(rs)}") in text
    assert row("Average detection confidence", lambda per, rs: f"{sum(r['score'] for r in rs) / len(rs):.3f}") in text


def test_readme_speed_table_matches_the_benchmark():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    speed = json.loads((ROOT / "web" / "speed.json").read_text(encoding="utf-8"))
    for device in speed.values():
        for key, value in device.items():
            if key.endswith("_ms"):
                assert f"{value} ms" in text, f"{device['hardware']} {key} = {value} ms is not in the README"
