from pathlib import Path

import pytest

from plyserve import settings


@pytest.fixture(autouse=True)
def no_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # a .env in the clone must not leak into the tests; a test that needs one writes this path
    path = tmp_path / ".env"
    monkeypatch.setattr(settings, "ENV_FILE", path)
    return path
