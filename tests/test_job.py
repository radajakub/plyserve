"""`plyserve job`: the serve job's partition comes from .env per length, its time limit from the length unless --time is given, it is submitted
from this clone with the arguments of `serve` unchanged, and partitions left in local.mk stop it instead of being ignored."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from plyserve import job
from plyserve.cli import main

PARTITIONS = "PLYSERVE_PARTITION_SHORT=h200fast\nPLYSERVE_PARTITION_LONG=h200long\n"


@pytest.fixture(autouse=True)
def no_local_mk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "local.mk"
    monkeypatch.setattr(job, "LOCAL_MK", path)
    for length in job.LENGTHS:
        monkeypatch.delenv(job.partition_variable(length), raising=False)
        monkeypatch.delenv(job.time_variable(length), raising=False)
    return path


def test_each_length_has_its_partition_and_time(no_env_file: Path) -> None:
    no_env_file.write_text(PARTITIONS)
    assert job.sbatch_command(["--tunnel", "m"]) == ["sbatch", "-p", "h200fast", "-t", "4:00:00", str(job.SBATCH_SCRIPT), "--tunnel", "m"]
    assert job.sbatch_command(["m"], "long", "10:00:00")[:5] == ["sbatch", "-p", "h200long", "-t", "10:00:00"]
    # a length without a partition leaves the choice to Slurm
    assert job.sbatch_command(["m"], "day")[:3] == ["sbatch", "-t", "1-00:00:00"]
    with pytest.raises(ValueError, match="length must be one of"):
        job.sbatch_command(["m"], "week")


def test_a_site_sets_its_own_time_limits(no_env_file: Path) -> None:
    no_env_file.write_text(PARTITIONS + "PLYSERVE_TIME_LONG=2-00:00:00\n")
    assert job.sbatch_command(["m"], "long")[:5] == ["sbatch", "-p", "h200long", "-t", "2-00:00:00"]
    # --time still wins for one job, and a length without a site time keeps the default
    assert job.sbatch_command(["m"], "long", "5:00:00")[4] == "5:00:00"
    assert job.time_limit("short") == "4:00:00"


def test_partitions_left_in_local_mk_stop_the_job(no_local_mk: Path) -> None:
    no_local_mk.write_text("CPU_PARTITION = cpufast\nGPU_PARTITION = gpufast\n")
    assert job.partition("short") is None
    no_local_mk.write_text("CPU_PARTITION = cpufast\nSERVE_LONG_PARTITION = h200long\n")
    with pytest.raises(RuntimeError, match="moved to .*PLYSERVE_PARTITION_SHORT"):
        job.partition("short")


def test_the_cli_submits_from_this_clone_with_the_serve_arguments_unchanged(
    tmp_path: Path, no_env_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    no_env_file.write_text(PARTITIONS)
    record = tmp_path / "sbatch.json"
    fake = tmp_path / "bin" / "sbatch"
    fake.parent.mkdir()
    fake.write_text(
        f"#!{sys.executable}\nimport json, os, sys\njson.dump({{'argv': sys.argv[1:], 'cwd': os.getcwd()}}, open({str(record)!r}, 'w'))\nprint('Submitted batch job 7')\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ['PATH']}")

    with pytest.raises(SystemExit) as exit:
        main(["job", "--length", "long", "--", "--tunnel", "m-fp8", "--", "uv", "run", "--directory", "../client", "python", "run.py"])
    assert exit.value.code == 0
    called = json.loads(record.read_text())
    assert called["argv"] == ["-p", "h200long", "-t", "3-00:00:00", str(job.SBATCH_SCRIPT), "--tunnel", "m-fp8", "--", "uv", "run", "--directory", "../client", "python", "run.py"]
    assert Path(called["cwd"]).resolve() == job.REPOSITORY.resolve()

    main(["partition", "long"])
    main(["partition", "day"])
    assert capsys.readouterr().out.splitlines()[-2:] == ["h200long", ""]
