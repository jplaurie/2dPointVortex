#!/usr/bin/env python3
"""Plot backend timing and speedup from run_benchmarks.py output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


COLORS = {"CPU serial": "#577590", "CPU/OpenMP": "#277da1", "MPI": "#f8961e",
          "CUDA": "#43aa8b", "CUDA mixed FP32/FP64": "#d1495b"}
MARKERS = {"CPU serial": "D", "CPU/OpenMP": "o", "MPI": "s", "CUDA": "^",
           "CUDA mixed FP32/FP64": "v"}
BACKEND_ORDER = ("CPU serial", "CPU/OpenMP", "MPI", "CUDA", "CUDA mixed FP32/FP64")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path,
                        default=[Path("benchmarks/results.csv")])
    parser.add_argument("--metadata", type=Path, default=Path("benchmarks/system.json"))
    parser.add_argument("--output-prefix", type=Path,
                        default=Path("benchmarks/backend_scaling"))
    args = parser.parse_args()

    rows = []
    for input_path in args.input:
        with input_path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    if not rows:
        parser.error("the input files have no benchmark rows")
    metadata = json.loads(args.metadata.read_text())

    samples: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        samples.setdefault(row["backend"], {}).setdefault(int(row["N"]), []).append(
            float(row["seconds_per_step"])
        )

    labels = {
        "CPU serial": "CPU serial (1 core)",
        "CPU/OpenMP": f"CPU/OpenMP ({metadata['cpu_openmp_threads']} threads)",
        "MPI": f"MPI ({metadata['mpi_ranks']} ranks)",
        "CUDA": "CUDA FP64",
        "CUDA mixed FP32/FP64": "CUDA mixed FP32/FP64",
    }
    figure, (timing_axis, speedup_axis) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    medians: dict[str, dict[int, float]] = {}
    for backend in BACKEND_ORDER:
        if backend not in samples:
            continue
        counts = sorted(samples[backend])
        values = [median(samples[backend][count]) for count in counts]
        medians[backend] = dict(zip(counts, values))
        lower = [value - min(samples[backend][count])
                 for count, value in zip(counts, values)]
        upper = [max(samples[backend][count]) - value
                 for count, value in zip(counts, values)]
        timing_axis.errorbar(counts, values, yerr=[lower, upper], label=labels[backend],
                             color=COLORS[backend], marker=MARKERS[backend], linewidth=2,
                             capsize=3)

    timing_axis.set(xscale="log", yscale="log", xlabel="Number of vortices, N",
                    ylabel="Time per RK4 step (s)", title="End-to-end timestep throughput")
    timing_axis.grid(True, which="both", alpha=0.25)
    timing_axis.legend(frameon=False)

    baseline = medians.get("CPU serial", {})
    for backend in BACKEND_ORDER:
        if backend not in medians:
            continue
        counts = sorted(set(baseline) & set(medians[backend]))
        speedups = [baseline[count] / medians[backend][count] for count in counts]
        speedup_axis.plot(counts, speedups, label=labels[backend], color=COLORS[backend],
                          marker=MARKERS[backend], linewidth=2)
    speedup_axis.axhline(1.0, color="0.45", linewidth=1, linestyle="--")
    speedup_axis.set(xscale="log", yscale="log", xlabel="Number of vortices, N",
                     ylabel="Speedup over one CPU core", title="Backend speedup")
    speedup_axis.grid(True, which="both", alpha=0.25)

    all_counts = sorted({count for backend in samples.values() for count in backend})
    for axis in (timing_axis, speedup_axis):
        axis.set_xscale("log", base=2)
        axis.set_xticks(all_counts, labels=[str(count) for count in all_counts])
        axis.xaxis.set_minor_locator(NullLocator())

    gpu = metadata.get("gpu", "unknown")
    cpu = metadata.get("cpu_model", "unknown")
    figure.suptitle(f"2dPointVortex RK4 benchmark\n{cpu} · {gpu}", fontsize=11)
    figure.tight_layout()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        output = args.output_prefix.with_suffix(f".{extension}")
        figure.savefig(output, dpi=180, bbox_inches="tight")
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
