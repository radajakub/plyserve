from pathlib import Path

import pytest

from plyserve.manifest import DEFAULT_MANIFEST, load_manifest

_ENTRY = 'repo = "org/model"\nrevision = "abc123"\ntunnel_port = 18001\n'


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "models.toml"
    path.write_text(text)
    return path


def test_repository_manifest_loads() -> None:
    specs = load_manifest(DEFAULT_MANIFEST)
    assert specs, "the shipped models.toml has no models"
    for name, spec in specs.items():
        assert spec.name == name
        assert len(spec.revision) == 40, f"{name} is not pinned to a full commit hash"


def test_vllm_args_pin_weights_and_tokenizer(tmp_path: Path) -> None:
    path = _write(tmp_path, f'[models."m-fp8"]\n{_ENTRY}args = ["--max-model-len", "4096"]\n')
    spec = load_manifest(path)["m-fp8"]
    assert spec.vllm_args("127.0.0.1", 8000) == [
        "serve",
        "org/model",
        "--revision",
        "abc123",
        "--tokenizer-revision",
        "abc123",
        "--served-model-name",
        "m-fp8",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-model-len",
        "4096",
    ]


def test_args_default_to_none(tmp_path: Path) -> None:
    assert load_manifest(_write(tmp_path, f'[models."m"]\n{_ENTRY}'))["m"].args == ()


@pytest.mark.parametrize("flag", ["--port", "--api-key=secret", "--served-model-name", "--tokenizer-revision"])
def test_reserved_flags_are_refused(tmp_path: Path, flag: str) -> None:
    path = _write(tmp_path, f'[models."m"]\n{_ENTRY}args = ["{flag}", "x"]\n')
    with pytest.raises(ValueError, match="plyserve sets itself"):
        load_manifest(path)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ('repo = "org/model"\ntunnel_port = 18001\n', "missing"),
        (_ENTRY + "quantization = 'fp8'\n", "unknown keys"),
        ('repo = "org/model"\nrevision = "abc"\ntunnel_port = 80\n', "unprivileged port"),
        ('repo = "org/model"\nrevision = "abc"\ntunnel_port = true\n', "unprivileged port"),
        ('repo = "org/model"\nrevision = ""\ntunnel_port = 18001\n', "commit hash"),
        (_ENTRY + 'args = "--max-model-len 4096"\n', "list of strings"),
    ],
)
def test_invalid_entries_are_refused(tmp_path: Path, entry: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_manifest(_write(tmp_path, f'[models."m"]\n{entry}'))


def test_shared_tunnel_port_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, f'[models."a"]\n{_ENTRY}\n[models."b"]\n{_ENTRY}')
    with pytest.raises(ValueError, match="share tunnel_port 18001"):
        load_manifest(path)


def test_unknown_top_level_key_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown top-level keys"):
        load_manifest(_write(tmp_path, '[model."m"]\nrepo = "x"\n'))
