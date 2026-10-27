from __future__ import annotations

import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO, cast

# seconds between GPU samples (nvidia-smi, DCGM) and between CPU samples (top)
GPU_INTERVAL = 2
CPU_INTERVAL = 5
# seconds a sampler gets after SIGTERM before SIGKILL
STOP_GRACE = 5.0
# what nvidia-smi reports for the job's GPUs. utilization.gpu is only the share of time any kernel runs, so it reads 100% on a busy server whatever
# the kernels do; DCGM's fields below say how much of the GPU they use. pci.bus_id finds the job's GPUs in DCGM, which lists every GPU of the node
NVIDIA_SMI_QUERY = "timestamp,index,pci.bus_id,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks_event_reasons.active"
# DCGM profiling fields: graphics engine active, SM active, SM occupancy, tensor pipes active, DRAM active
DCGM_FIELDS = "1001,1002,1003,1004,1005"


@dataclass(frozen=True)
class Sampler:
    file: str  # in the run folder
    argv: tuple[str, ...]
    timestamps: bool = False  # prefix each line with the local time, for tools that print none (dcgmi)


def samplers(env: dict[str, str]) -> tuple[list[Sampler], list[str]]:
    # the samplers for `serve --monitor`, and notes on those skipped. Checked before the server starts, so a missing tool costs no model load.
    # DCGM is optional: not every cluster runs it, and without it nvidia-smi still shows memory, power and clocks
    path = env.get("PATH")
    for tool in ("nvidia-smi", "top"):
        if shutil.which(tool, path=path) is None:
            raise RuntimeError(f"--monitor needs {tool}, which is not on PATH")
    found = [
        Sampler("gpu.csv", ("nvidia-smi", f"--query-gpu={NVIDIA_SMI_QUERY}", "--format=csv", "-l", str(GPU_INTERVAL))),
        Sampler("top.txt", ("top", "-b", "-d", str(CPU_INTERVAL), "-u", str(os.getuid()))),
    ]
    if shutil.which("dcgmi", path=path) is None:
        return found, ["dcgmi: not on PATH"]
    found.append(Sampler("dcgm.txt", ("dcgmi", "dmon", "-e", DCGM_FIELDS, "-d", str(GPU_INTERVAL * 1000)), timestamps=True))
    return found, []


def _copy_with_time(source: TextIO, target: TextIO) -> None:
    with source, target:
        for line in source:
            target.write(f"{datetime.now().isoformat(timespec='milliseconds')} {line}")
            target.flush()


class Monitor:
    # runs the samplers into the run folder until stop(); with DCGM, its GPU list (dcgm-gpus.txt) maps its ids to bus ids first
    def __init__(self, run_dir: Path, env: dict[str, str], found: list[Sampler]) -> None:
        self.files: list[str] = []
        self.processes: list[tuple[Sampler, subprocess.Popen[str]]] = []
        self.threads: list[threading.Thread] = []
        self.reported: set[str] = set()
        if any(sampler.argv[0] == "dcgmi" for sampler in found):
            with open(run_dir / "dcgm-gpus.txt", "w", encoding="utf-8") as out:
                try:
                    subprocess.run(["dcgmi", "discovery", "-l"], stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, timeout=60)
                except subprocess.TimeoutExpired:
                    out.write("plyserve: dcgmi discovery -l did not answer within 60 s\n")
            self.files.append("dcgm-gpus.txt")
        for sampler in found:
            out = open(run_dir / sampler.file, "w", encoding="utf-8")
            # its own session, so a Ctrl-C reaches only plyserve, which stops the samplers itself
            if sampler.timestamps:
                process = subprocess.Popen(sampler.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, text=True, start_new_session=True)
                thread = threading.Thread(target=_copy_with_time, args=(cast(TextIO, process.stdout), out), daemon=True)
                thread.start()
                self.threads.append(thread)
            else:
                with out:
                    process = subprocess.Popen(sampler.argv, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, text=True, start_new_session=True)
            self.processes.append((sampler, process))
            self.files.append(sampler.file)

    def check(self) -> None:
        # a sampler that ends early (e.g. a field this driver does not know) leaves the server running; say so once
        for sampler, process in self.processes:
            if sampler.file not in self.reported and (code := process.poll()) is not None:
                self.reported.add(sampler.file)
                print(f"plyserve: the {sampler.argv[0]} sampler exited with code {code}; its output is in {sampler.file}", flush=True)

    def stop(self) -> None:
        for _, process in self.processes:
            if process.poll() is None:
                process.terminate()
        for _, process in self.processes:
            try:
                process.wait(STOP_GRACE)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for thread in self.threads:
            thread.join(STOP_GRACE)
