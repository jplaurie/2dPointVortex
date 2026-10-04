"""Check the mixed CUDA backend against the FP64 CPU reference.

Usage: python3 tests/mixed_precision_consistency.py CPU_EXEC MIXED_CUDA_EXEC
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile


def ring(count: int) -> list[tuple[float, float, float]]:
    return [(math.cos(2 * math.pi * i / count), math.sin(2 * math.pi * i / count),
             1.0 if i % 2 == 0 else -1.0) for i in range(count)]


def run(executable: Path, directory: Path, vortices: list[tuple[float, float, float]],
        geometry: str, integrator: str, time_step: float, end_time: float,
        extra_parameters: str = "") -> None:
    directory.mkdir()
    initial = directory / "initial.dat"
    initial.write_text("".join(f"{x:.17g} {y:.17g} {gamma:.17g}\n"
                               for x, y, gamma in vortices))
    params = directory / "run.params"
    params.write_text(
        "initialCondition file\n"
        f"initialConditionFile {initial}\n"
        f"boundaryCondition {geometry}\nintegrator {integrator}\n"
        f"coreRadius {0.001 if geometry == 'infinite' else 0}\n"
        "boxLengthX 2\nboxLengthY 2\ndiskRadius 1\nperiodicImageLayers 8\n"
        "absoluteTolerance 1e-9\nrelativeTolerance 1e-7\n"
        f"timeStep {time_step:.17g}\nendTime {end_time:.17g}\n"
        f"outputTime {end_time:.17g}\ndiagnosticsTime {end_time:.17g}\n"
        f"checkpointTime {end_time:.17g}\nnumThreads 2\n"
        f"{extra_parameters}"
        f"runDirectory {directory}\n"
    )
    completed = subprocess.run([str(executable), str(params)], text=True, capture_output=True,
                               timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def trajectory_metrics(reference: Path, candidate: Path) -> tuple[float, float, float]:
    with reference.open(newline="") as stream:
        left = list(csv.DictReader(stream))
    with candidate.open(newline="") as stream:
        right = list(csv.DictReader(stream))
    assert len(left) == len(right)
    maximum_position_error = 0.0
    velocity_error_squared = 0.0
    velocity_reference_squared = 0.0
    maximum_velocity_error = 0.0
    for expected, actual in zip(left, right):
        assert (expected["time"], expected["frame"], expected["index"],
                expected["circulation"]) == (actual["time"], actual["frame"],
                                               actual["index"], actual["circulation"])
        dx = float(actual["x"]) - float(expected["x"])
        dy = float(actual["y"]) - float(expected["y"])
        du = float(actual["u"]) - float(expected["u"])
        dv = float(actual["v"]) - float(expected["v"])
        maximum_position_error = max(maximum_position_error, math.hypot(dx, dy))
        maximum_velocity_error = max(maximum_velocity_error, math.hypot(du, dv))
        velocity_error_squared += du * du + dv * dv
        velocity_reference_squared += float(expected["u"]) ** 2 + float(expected["v"]) ** 2
    relative_velocity_l2 = math.sqrt(velocity_error_squared / velocity_reference_squared)
    return maximum_position_error, relative_velocity_l2, maximum_velocity_error


def hamiltonian_error(reference: Path, candidate: Path) -> float:
    with reference.open(newline="") as stream:
        expected = list(csv.DictReader(stream))[-1]
    with candidate.open(newline="") as stream:
        actual = list(csv.DictReader(stream))[-1]
    left, right = float(expected["hamiltonian"]), float(actual["hamiltonian"])
    return abs(right - left) / max(1.0, abs(left))


def main(cpu: Path, mixed: Path) -> None:
    generator = random.Random(1234567)
    cloud = [(generator.uniform(-1.0, 1.0), generator.uniform(-1.0, 1.0),
              1.0 if i % 2 == 0 else -1.0) for i in range(257)]
    close_pairs = [(-0.00055, 0.0, 1.0), (0.00055, 0.0, -1.0),
                   (-0.31, 0.22, 0.75), (0.27, -0.19, -0.75)]
    ordinary = [(-0.31, -0.12, 1.0), (0.23, 0.14, -1.0),
                (0.11, -0.29, 2.0), (-0.14, 0.32, -2.0)]
    cases = [("ring", ring(1024), "infinite", "rk4", 1e-6, 1e-4),
             ("cloud", cloud, "infinite", "rk4", 1e-6, 1e-4),
             ("close_pairs", close_pairs, "infinite", "rk4", 1e-8, 1e-6)]
    cases += [(f"{geometry}_{integrator}", ordinary, geometry, integrator, 1e-4, 3e-3)
              for geometry in ("infinite", "periodic_x", "periodic", "disk")
              for integrator in ("rk4", "dopri5")]
    with tempfile.TemporaryDirectory(prefix="point_vortex_mixed_test_") as temporary:
        root = Path(temporary)
        for name, vortices, geometry, integrator, time_step, end_time in cases:
            cpu_run, mixed_run = root / f"{name}_cpu", root / f"{name}_mixed"
            run(cpu, cpu_run, vortices, geometry, integrator, time_step, end_time)
            run(mixed, mixed_run, vortices, geometry, integrator, time_step, end_time)
            position, velocity_l2, velocity_max = trajectory_metrics(
                cpu_run / "trajectory.csv", mixed_run / "trajectory.csv"
            )
            hamiltonian = hamiltonian_error(cpu_run / "diagnostics.csv",
                                            mixed_run / "diagnostics.csv")
            print(f"{name}: max_position_error={position:.6e} "
                  f"relative_velocity_l2_error={velocity_l2:.6e} "
                  f"max_velocity_error={velocity_max:.6e} "
                  f"relative_hamiltonian_error={hamiltonian:.6e}")
            assert position < 1e-7
            assert velocity_l2 < 5e-4
            assert velocity_max < 5e-4
            assert hamiltonian < 1e-8

        # Exercise production plumbing beyond a fresh trajectory: a dipole-enabled run writes a
        # checkpoint, then a second mixed-CUDA process restores it and continues with DOPRI5.
        workflow_rows = [(-0.01, 0.0, 1.0), (0.01, 0.0, -1.0),
                         (-0.4, 0.2, 1.0), (0.4, -0.2, -1.0)]
        workflow = root / "workflow"
        dipole_parameters = (
            "dipoleRemoval true\ndipoleRemovalDistance 0.05\n"
            "dipoleRemovalInterval 0.0005\ndipoleReinjection paired\n"
        )
        run(mixed, workflow, workflow_rows, "periodic", "dopri5", 1e-4, 2e-3,
            dipole_parameters)
        checkpoint = workflow / "checkpoints" / "checkpoint_00000001.dat"
        assert checkpoint.exists()
        resumed = root / "workflow_resumed"
        run(mixed, resumed, workflow_rows, "periodic", "dopri5", 1e-4, 3e-3,
            dipole_parameters + f"restartFile {checkpoint}\n")
        assert (resumed / "trajectory.csv").exists()
        assert (resumed / "diagnostics.csv").exists()
        print("restart/dipole workflow: passed", flush=True)
    print("mixed-precision CUDA consistency check passed")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
