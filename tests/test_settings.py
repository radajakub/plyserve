from pathlib import Path

import pytest

from plyserve.settings import environment, home, read_env_file


def test_env_file_syntax(tmp_path: Path) -> None:
    path = tmp_path / "a.env"
    path.write_text("# comment\n\nA=1\nexport B = two \nC=\"quoted value\"\nD='x'\nE=\nF=a=b\n")
    assert read_env_file(path) == {"A": "1", "B": "two", "C": "quoted value", "D": "x", "F": "a=b"}


def test_missing_env_file_is_empty(tmp_path: Path) -> None:
    assert read_env_file(tmp_path / "missing") == {}


def test_malformed_line_names_its_place(tmp_path: Path) -> None:
    path = tmp_path / "a.env"
    path.write_text("A=1\njust words\n")
    with pytest.raises(ValueError, match=r"a\.env:2"):
        read_env_file(path)


def test_shell_wins_over_env_file(no_env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    no_env_file.write_text("PLYSERVE_TEST_A=file\nPLYSERVE_TEST_B=file\n")
    monkeypatch.setenv("PLYSERVE_TEST_A", "shell")
    monkeypatch.delenv("PLYSERVE_TEST_B", raising=False)
    env = environment()
    assert env["PLYSERVE_TEST_A"] == "shell"
    assert env["PLYSERVE_TEST_B"] == "file"


def test_home_from_env_file_expands_variables(no_env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PLYSERVE_HOME", raising=False)
    monkeypatch.setenv("USER", "someone")
    no_env_file.write_text("PLYSERVE_HOME=/mnt/personal/$USER/plyserve-home\n")
    assert home() == Path("/mnt/personal/someone/plyserve-home")


def test_home_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PLYSERVE_HOME", raising=False)
    with pytest.raises(RuntimeError, match="PLYSERVE_HOME is not set"):
        home()
