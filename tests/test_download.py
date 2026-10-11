import subprocess
from pathlib import Path
from typing import Any

import pytest

from plyserve import settings
from plyserve.download import download
from plyserve.manifest import ModelSpec

_SPEC = ModelSpec("m-fp8", "org/model", "abc123", 18001, ())


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "hf").touch()
    monkeypatch.setattr(settings, "RUNTIME", venv)
    return venv


def test_download_fills_the_plyserve_cache(tmp_path: Path, runtime: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLYSERVE_HOME", str(tmp_path / "home"))
    calls: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))

    download(_SPEC)

    [(command, kwargs)] = calls
    assert command == [str(runtime / "bin" / "hf"), "download", "org/model", "--revision", "abc123"]
    assert kwargs["env"]["HF_HOME"] == str(tmp_path / "home" / "hf")
    assert kwargs["check"] is True
    assert (tmp_path / "home" / "hf").is_dir()


def test_plyserve_home_is_required(runtime: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PLYSERVE_HOME", raising=False)
    with pytest.raises(RuntimeError, match="PLYSERVE_HOME is not set"):
        download(_SPEC)


def test_env_file_reaches_the_download(tmp_path: Path, runtime: Path, no_env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PLYSERVE_HOME", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    no_env_file.write_text(f"PLYSERVE_HOME={tmp_path / 'home'}\nHF_TOKEN=hf_test\nHF_HOME=/ignored\n")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append(kwargs))

    download(_SPEC)

    [kwargs] = calls
    assert kwargs["env"]["HF_TOKEN"] == "hf_test"
    # plyserve's cache, whatever HF_HOME says, so downloads and servers share it
    assert kwargs["env"]["HF_HOME"] == str(tmp_path / "home" / "hf")


def test_missing_runtime_says_how_to_build_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLYSERVE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(settings, "RUNTIME", tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="make runtime"):
        download(_SPEC)
