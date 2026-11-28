"""local.env, written by tools/setup_env.py, fills in the variables the shell has not set."""
import os

from tools import load_env


def test_load_env_sets_only_missing_variables(tmp_path, monkeypatch):
    for k in ("LPR_TEST_A", "LPR_TEST_B"):
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)  # restored (removed) after the test
    monkeypatch.setenv("LPR_TEST_B", "from the shell")
    f = tmp_path / "local.env"
    f.write_text("# comment\nLPR_TEST_A=C:\\Program Files\\a=b.exe\nLPR_TEST_B=from the file\n\n", encoding="utf-8")
    load_env(f)
    assert os.environ["LPR_TEST_A"] == "C:\\Program Files\\a=b.exe"
    assert os.environ["LPR_TEST_B"] == "from the shell"


def test_load_env_without_the_file_changes_nothing(tmp_path):
    before = dict(os.environ)
    load_env(tmp_path / "missing.env")
    assert dict(os.environ) == before
