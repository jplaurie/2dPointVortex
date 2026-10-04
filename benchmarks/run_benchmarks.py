#!/usr/bin/env python3
"""Run repeatable CPU/OpenMP, MPI, and CUDA RK4 scaling benchmarks."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


BACKENDS = {
    "serial": ("CPU serial", "point_vortex_backend_benchmark_serial"),
    "cpu": ("CPU/OpenMP", "point_vortex_backend_benchmark_cpu"),
    "mpi": ("MPI", "point_vortex_backend_benchmark_mpi"),
    "cuda": ("CUDA", "point_vortex_backend_benchmark_cuda"),
    "cuda_mixed": ("CUDA mixed FP32/FP64", "point_vortex_backend_benchmark_cuda_mixed"),
}


def checked_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def physical_core_count() -> int:
    topology = Path("/sys/devices/system/cpu")
    cores: set[tuple[str, str]] = set()
    for cpu in topology.glob("cpu[0-9]*"):
        try:
            package = (cpu / "topology/physical_package_id").read_text().strip()
            core = (cpu / "topology/core_id").read_text().strip()
            cores.add((package, core))
        except OSError:
            pass
    return len(cores) or (os.cpu_count() or 1)


def parse_result(stdout: str) -> dict[str, str]:
    line = next((line for line in reversed(stdout.splitlines()) if line.startswith("backend=")), "")
    # The backend value is the only value that may contain a slash, but no spaces.
    fields = dict(re.findall(r"([a-zA-Z_]+)=([^ ]+)", line))
    required = {"backend", "N", "steps", "warmup_steps", "seconds",
                "seconds_per_step", "interactions_per_second", "checksum"}
    if not required.issubset(fields):
        raise RuntimeError(f"could not parse benchmark output:\n{stdout}")
    return fields


def command_for(backend: str, executable: Path, count: int, steps: int, warmup: int,
                mpi_ranks: int) -> list[str]:
    arguments = [str(executable), str(count), str(steps), str(warmup)]
    if backend == "mpi":
        return ["mpirun", "--bind-to", "core", "--map-by", "core", "-n",
                str(mpi_ranks), *arguments]
    return arguments


def run_once(backend: str, executable: Path, count: int, steps: int, warmup: int,
             mpi_ranks: int, cpu_threads: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": str(cpu_threads if backend == "cpu" else 1),
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    })
    command = command_for(backend, executable, count, steps, warmup, mpi_ranks)
    completed = subprocess.run(command, text=True, capture_output=True, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark failed ({' '.join(command)}):\n{completed.stdout}{completed.stderr}"
        )
    result = parse_result(completed.stdout)
    result["backend"] = BACKENDS[backend][0]
    return result


def calibrated_steps(backend: str, executable: Path, count: int, minimum_warmup: int,
                     mpi_ranks: int, cpu_threads: int, target_seconds: float,
                     warmup_seconds: float, minimum_timed_steps: int) -> tuple[int, int]:
    steps = 1
    while True:
        result = run_once(backend, executable, count, steps, minimum_warmup,
                          mpi_ranks, cpu_threads)
        seconds = float(result["seconds"])
        if seconds >= 0.1 or steps >= 1_000_000:
            break
        steps *= 4
    seconds_per_step = seconds / steps
    timed_steps = max(minimum_timed_steps,
                      min(1_000_000, math.ceil(target_seconds / seconds_per_step)))
    warmup_steps = max(minimum_warmup, math.ceil(warmup_seconds / seconds_per_step))
    return timed_steps, warmup_steps


def machine_metadata(repo: Path, cpu_threads: int, mpi_ranks: int,
                     resolutions: list[int], trials: int, target_seconds: float,
                     minimum_warmup: int, warmup_seconds: float,
                     minimum_timed_steps: int) -> dict[str, object]:
    cpu_model = "unknown"
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
        if match:
            cpu_model = match.group(1).strip()
    except OSError:
        pass
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "physical_cores_detected": physical_core_count(),
        "cpu_openmp_threads": cpu_threads,
        "mpi_ranks": mpi_ranks,
        "gpu": checked_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        "gpu_driver": checked_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "compiler": checked_output(["c++", "--version"]).splitlines()[0],
        "mpi": checked_output(["mpirun", "--version"]).splitlines()[0],
        "cuda_compiler": checked_output(["nvcc", "--version"]).splitlines()[-1],
        "git_commit": checked_output(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "git_dirty": bool(checked_output(["git", "-C", str(repo), "status", "--porcelain"])
                          not in ("", "unknown")),
        "resolutions": resolutions,
        "trials": trials,
        "target_timed_seconds_per_trial": target_seconds,
        "minimum_timed_steps": minimum_timed_steps,
        "minimum_warmup_steps": minimum_warmup,
        "target_warmup_seconds": warmup_seconds,
        "integrator": "RK4",
        "geometry": "infinite plane",
        "core_radius": 1.0e-3,
        "timing_excludes": [
            "process and MPI startup", "CUDA context creation", "backend allocation",
            "state initialization and upload", "warm-up steps", "final state download",
            "checksum", "file output",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build/benchmarks"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("benchmarks/system.json"))
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--resolutions", nargs="+", type=int,
                        default=[256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--target-seconds", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--minimum-timed-steps", type=int, default=3)
    parser.add_argument("--warmup-seconds", type=float, default=0.0,
                        help="calibrate at least this much untimed warm-up before every trial")
    parser.add_argument("--cpu-threads", type=int, default=physical_core_count())
    parser.add_argument("--mpi-ranks", type=int, default=physical_core_count())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    positive = [*args.resolutions, args.trials, args.warmup_steps, args.minimum_timed_steps,
                args.cpu_threads, args.mpi_ranks]
    if (any(value <= 0 for value in positive) or args.target_seconds <= 0 or
            args.warmup_seconds < 0):
        parser.error("resolutions, counts, and timed durations must be positive; warm-up must be nonnegative")
    if args.output.exists() and not args.overwrite:
        parser.error(f"{args.output} exists; pass --overwrite to replace it")

    repo = Path(__file__).resolve().parents[1]
    build = (repo / args.build_dir).resolve() if not args.build_dir.is_absolute() else args.build_dir
    executables: dict[str, Path] = {}
    for backend in args.backends:
        executable = build / BACKENDS[backend][1]
        if not executable.is_file():
            parser.error(f"missing {executable}; build the backend benchmark targets first")
        if backend == "mpi" and not shutil.which("mpirun"):
            parser.error("mpirun is not available")
        executables[backend] = executable

    output = (repo / args.output).resolve() if not args.output.is_absolute() else args.output
    metadata_path = ((repo / args.metadata).resolve()
                     if not args.metadata.is_absolute() else args.metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = machine_metadata(repo, args.cpu_threads, args.mpi_ranks, args.resolutions,
                                args.trials, args.target_seconds, args.warmup_steps,
                                args.warmup_seconds, args.minimum_timed_steps)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    columns = ["backend", "N", "trial", "steps", "warmup_steps", "seconds",
               "seconds_per_step", "interactions_per_second", "checksum",
               "cpu_threads", "mpi_ranks"]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for count in args.resolutions:
            for backend in args.backends:
                executable = executables[backend]
                steps, warmup = calibrated_steps(
                    backend, executable, count, args.warmup_steps, args.mpi_ranks,
                    args.cpu_threads, args.target_seconds, args.warmup_seconds,
                    args.minimum_timed_steps
                )
                print(f"{BACKENDS[backend][0]:>10} N={count:<6} timed_steps={steps} "
                      f"warmup_steps={warmup}", flush=True)
                for trial in range(1, args.trials + 1):
                    result = run_once(backend, executable, count, steps, warmup,
                                      args.mpi_ranks, args.cpu_threads)
                    writer.writerow({
                        **{name: result[name] for name in columns[:2]},
                        "trial": trial,
                        **{name: result[name] for name in columns[3:9]},
                        "cpu_threads": args.cpu_threads,
                        "mpi_ranks": args.mpi_ranks,
                    })
                    stream.flush()
                    print(f"  trial {trial}: {float(result['seconds_per_step']):.6g} s/step",
                          flush=True)
    print(f"wrote {output}")
    print(f"wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
