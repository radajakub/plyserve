from pathlib import Path

import pytest

from plyserve.cli import main
from plyserve.manifest import DEFAULT_MANIFEST, default_manifest, load_manifest

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


def test_default_manifest_is_the_repositorys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PLYSERVE_MANIFEST", raising=False)
    assert default_manifest() == DEFAULT_MANIFEST


def test_default_manifest_from_env_file_expands_variables(no_env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PLYSERVE_MANIFEST", raising=False)
    monkeypatch.setenv("USER", "someone")
    no_env_file.write_text("PLYSERVE_MANIFEST=/data/$USER/models-a.toml\n")
    assert default_manifest() == Path("/data/someone/models-a.toml")


def test_cli_reads_plyserve_manifest_unless_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    variant = tmp_path / "variant"
    variant.mkdir()
    monkeypatch.setenv("PLYSERVE_MANIFEST", str(_write(variant, f'[models."from-env"]\n{_ENTRY}')))
    main(["models"])
    assert capsys.readouterr().out.startswith("from-env\t")
    main(["--manifest", str(_write(tmp_path, f'[models."from-flag"]\n{_ENTRY}')), "models"])
    assert capsys.readouterr().out.startswith("from-flag\t")


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


_OLLAMA_ENTRY = f'backend = "ollama"\n{_ENTRY}file = "model-Q8_0.gguf"\n'


def test_ollama_entry_renders_a_modelfile(tmp_path: Path) -> None:
    path = _write(tmp_path, f'[models."m-q8"]\n{_OLLAMA_ENTRY}parameters = {{ num_ctx = 4096, top_p = 0.95, stop = ["<|im_end|>", "a b"] }}\n')
    spec = load_manifest(path)["m-q8"]
    assert (spec.backend, spec.file, spec.args) == ("ollama", "model-Q8_0.gguf", ())
    assert spec.modelfile(Path("/hf/snap")) == 'FROM /hf/snap/model-Q8_0.gguf\nPARAMETER num_ctx 4096\nPARAMETER top_p 0.95\nPARAMETER stop "<|im_end|>"\nPARAMETER stop "a b"\n'


def test_backend_defaults_to_vllm(tmp_path: Path) -> None:
    spec = load_manifest(_write(tmp_path, f'[models."m"]\n{_ENTRY}'))["m"]
    assert spec.backend == "vllm" and spec.file is None and spec.parameters == ()


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ('backend = "llamacpp"\n' + _ENTRY, "backend must be one of"),
        ('backend = "ollama"\n' + _ENTRY, r"missing \['file'\]"),
        (_OLLAMA_ENTRY + 'args = ["--max-model-len", "4096"]\n', r"unknown keys \['args'\] for backend ollama"),
        (_ENTRY + 'file = "model.gguf"\n', r"unknown keys \['file'\] for backend vllm"),
        ('backend = "ollama"\n' + _ENTRY + 'file = "model.safetensors"\n', "must name a .gguf file"),
        (_OLLAMA_ENTRY + "parameters = { num_ctx = true }\n", "parameter num_ctx must be"),
        (_OLLAMA_ENTRY + 'parameters = "num_ctx 4096"\n', "parameters must be a table"),
    ],
)
def test_invalid_ollama_entries_are_refused(tmp_path: Path, entry: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_manifest(_write(tmp_path, f'[models."m"]\n{entry}'))
