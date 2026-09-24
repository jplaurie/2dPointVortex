"""Compare geometry kernels and short runs against a CPU executable.

Usage: python3 tests/backend_consistency.py CPU_EXEC BACKEND_EXEC
       python3 tests/backend_consistency.py CPU_EXEC mpirun -n 2 MPI_EXEC
"""
import csv
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile


def run(command, directory, geometry, rows, end, integrator='rk4'):
    directory.mkdir()
    initial = directory / 'initial.dat'
    initial.write_text(''.join(f'{x:.17g} {y:.17g} {gamma:.17g}\n' for x, y, gamma in rows))
    params = directory / 'run.params'
    params.write_text(
        f'initialConditionFile {initial}\nboundaryCondition {geometry}\n'
        f'endTime {end}\nintegrator {integrator}\ntimeStep 0.001\noutputTime 0.002\n'
        'diagnosticsTime 0.001\ncheckpointTime 0.003\nnumThreads 2\n'
        f'runDirectory {directory}\n')
    result = subprocess.run([*command, str(params)], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def compare(first, second):
    with first.open() as stream:
        left = list(csv.DictReader(stream))
    with second.open() as stream:
        right = list(csv.DictReader(stream))
    assert len(left) == len(right), (first, second)
    for a, b in zip(left, right):
        assert a.keys() == b.keys()
        for key in a:
            x, y = float(a[key]), float(b[key])
            assert math.isfinite(x) and math.isfinite(y)
            assert math.isclose(x, y, rel_tol=2e-10, abs_tol=2e-11), (first, key, x, y)


def main(cpu, backend):
    ordinary = [(-.31, -.12, 1), (.23, .14, -1), (.11, -.29, 2), (-.14, .32, -2)]
    rng = random.Random(1234)
    large = [(rng.uniform(-.4, .4), rng.uniform(-.4, .4), 1 if i % 2 == 0 else -1)
             for i in range(258)]
    cases = [(geometry + '_run', geometry, ordinary, .003)
             for geometry in ('infinite', 'periodic_x', 'periodic', 'disk')]
    cases += [(geometry + '_openmp', geometry, large, 0)
              for geometry in ('infinite', 'periodic_x', 'periodic', 'disk')]
    cases += [('periodic_close', 'periodic', [(0, 0, 1), (1e-10, 0, -1)], 0),
              ('periodic_x_non_neutral', 'periodic_x', [(0, 0, 1), (.2, .1, 1)], .003),
              ('disk_center', 'disk', [(.2, 0, 1), (1e-155, 0, 1)], 0)]
    with tempfile.TemporaryDirectory(prefix='point_vortex_backend_test_') as tmp:
        root = Path(tmp)
        for name, geometry, rows, end in cases:
            left, right = root / (name + '_cpu'), root / (name + '_backend')
            run([cpu], left, geometry, rows, end)
            run(backend, right, geometry, rows, end)
            for filename in ('trajectory.csv', 'diagnostics.csv'):
                compare(left / filename, right / filename)
            print(f'{name}: matched CPU reference', flush=True)
        for geometry in ('infinite', 'periodic_x', 'periodic', 'disk'):
            left, right = root / (geometry + '_dopri_cpu'), root / (geometry + '_dopri_backend')
            run([cpu], left, geometry, ordinary, .003, integrator='dopri5')
            run(backend, right, geometry, ordinary, .003, integrator='dopri5')
            for filename in ('trajectory.csv', 'diagnostics.csv'):
                compare(left / filename, right / filename)
            print(f'{geometry}_dopri: matched CPU reference', flush=True)
    print('all backend consistency tests passed')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
