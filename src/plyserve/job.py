from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence

from plyserve.settings import ENV_FILE, REPOSITORY, environment

# A batch job of slurm/serve_model.sbatch (the arguments are those of `plyserve serve`), on the partition and with the time limit of a length:
#   short (tests, throughput checks), day, long (full runs), extralong.
# Both are site settings in .env: PLYSERVE_PARTITION_<LENGTH> (empty: Slurm chooses) and PLYSERVE_TIME_<LENGTH> (empty: the defaults below, the limits of
# RCI's H200 partitions: 4 hours, 1 day, 3 days, 21 days).
# Submitted from this clone, so a client command runs here (hence `uv run --directory <client>`), and the log is plyserve-<job id>.out here.
LENGTHS = {"short": "4:00:00", "day": "1-00:00:00", "long": "3-00:00:00", "extralong": "21-00:00:00"}
SBATCH_SCRIPT = REPOSITORY / "slurm" / "serve_model.sbatch"
LOCAL_MK = REPOSITORY / "local.mk"
# the variables of local.mk that named the serve partitions before they moved to .env
_OLD_PARTITIONS = re.compile(r"^\s*SERVE_(DAY_|LONG_|EXTRALONG_)?PARTITION\s*[:?+]?=", re.MULTILINE)


def partition_variable(length: str) -> str:
    if length not in LENGTHS:
        raise ValueError(f"length must be one of {', '.join(LENGTHS)}, not {length!r}")
    return f"PLYSERVE_PARTITION_{length.upper()}"


def time_variable(length: str) -> str:
    return partition_variable(length).replace("_PARTITION_", "_TIME_")


def time_limit(length: str, env: dict[str, str] | None = None) -> str:
    # the time limit of a length in Slurm's format: the site's, else the default
    partition_variable(length)
    return (env if env is not None else environment()).get(time_variable(length)) or LENGTHS[length]


def partition(length: str, env: dict[str, str] | None = None) -> str | None:
    # the partition of a length, None when it is not set (Slurm chooses)
    check_no_old_partitions()
    return (env if env is not None else environment()).get(partition_variable(length)) or None


def check_no_old_partitions() -> None:
    # partitions left in local.mk would be ignored without a word, and a job would land on Slurm's default partition
    if LOCAL_MK.exists() and _OLD_PARTITIONS.search(LOCAL_MK.read_text()):
        raise RuntimeError(
            f"{LOCAL_MK} names serve partitions (SERVE_*PARTITION), which moved to {ENV_FILE}: set PLYSERVE_PARTITION_SHORT, _DAY, _LONG and _EXTRALONG there "
            "(see .env.example) and remove them from local.mk"
        )


def sbatch_command(serve_args: Sequence[str], length: str = "short", time: str | None = None, env: dict[str, str] | None = None) -> list[str]:
    # `time` replaces the time limit of the length, up to the partition's limit (a shorter one can start sooner)
    chosen = partition(length, env)
    return ["sbatch", *(["-p", chosen] if chosen else []), "-t", time or time_limit(length, env), str(SBATCH_SCRIPT), *serve_args]


def submit_job(serve_args: Sequence[str], length: str = "short", time: str | None = None) -> int:
    # submits the job from this clone; sbatch prints "Submitted batch job <id>". Returns sbatch's exit code
    if not serve_args:
        raise ValueError("a job needs the arguments of `plyserve serve`: [--expose] [--tunnel] [--monitor] <model> [-- client command...]")
    command = sbatch_command(serve_args, length, time)
    print(f"plyserve: {' '.join(command)}", flush=True)
    # the shell's environment, as `make serve-job` passed it; serve reads .env itself
    return subprocess.run(command, cwd=REPOSITORY).returncode
