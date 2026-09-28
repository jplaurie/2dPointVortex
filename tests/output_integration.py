"""Exercise real solver output/restart behavior; accepts an executable or MPI command."""
import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def check_times(actual, expected):
    assert len(actual) == len(expected), (actual, expected)
    assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0) for a, b in zip(actual, expected)), (actual, expected)


def csv_times(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        assert all(math.isfinite(float(value)) for value in row.values()), row
        identity = (row['time'], row.get('index'))
        assert identity not in seen, ('duplicate output row', row)
        seen.add(identity)
    times = list(dict.fromkeys(float(row['time']) for row in rows))
    assert times == sorted(times)
    return times


def csv_frames(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    return list(dict.fromkeys(int(row['frame']) for row in rows))


def checkpoint_times(directory):
    return [float(path.read_text().splitlines()[1].split()[1])
            for path in sorted(directory.glob('checkpoint_*.dat'))]


def main(command):
    with tempfile.TemporaryDirectory(prefix='point_vortex_output_test_') as temporary:
        root = Path(temporary)

        def run(name, extra='', integrator='rk4', success=True):
            directory = root / name
            params = root / (name.replace('/', '_') + '.params')
            params.write_text(
                f'N 2\nboundaryCondition infinite\nintegrator {integrator}\n'
                'timeStep 0.007\nnumThreads 1\nendTime 0.13\noutputTime 0.04\n'
                f'runDirectory {directory}\n' + extra)
            result = subprocess.run([*command, str(params)], text=True, capture_output=True, timeout=30)
            assert (result.returncode == 0) == success, result.stdout + result.stderr
            return directory, result

        for integrator in ('rk4', 'dopri5'):
            # No parent directories exist: the solver creates all of them.
            common, result = run(integrator + '_common', integrator=integrator)
            expected = [0, 0.04, 0.08, 0.12, 0.13]
            check_times(csv_times(common / 'trajectory.csv'), expected)
            check_times(csv_times(common / 'diagnostics.csv'), expected)
            check_times(checkpoint_times(common / 'checkpoints'), expected)
            assert str(common / 'trajectory.csv') in result.stdout
            assert str(common / 'diagnostics.csv') in result.stdout
            assert str(common / 'checkpoints') in result.stdout

            split, _ = run(integrator + '_split', 'diagnosticsTime 0.03\ncheckpointTime 0.05\n', integrator)
            check_times(csv_times(split / 'trajectory.csv'), expected)
            check_times(csv_times(split / 'diagnostics.csv'), [0, .03, .06, .09, .12, .13])
            check_times(checkpoint_times(split / 'checkpoints'), [0, .05, .1, .13])
            source = split / 'checkpoints/checkpoint_00000001.dat'
            branch, _ = run(integrator + '_branch',
                            f'diagnosticsTime 0.03\ncheckpointTime 0.05\nrestartFile {source}\n', integrator)
            check_times(csv_times(branch / 'trajectory.csv'), [.05, .08, .12, .13])
            check_times(csv_times(branch / 'diagnostics.csv'), [.05, .06, .09, .12, .13])
            check_times(checkpoint_times(branch / 'checkpoints'), [.1, .13])
            assert (branch / 'checkpoints/checkpoint_00000002.dat').exists()
            with (split / 'trajectory.csv').open() as stream:
                full_rows = list(csv.DictReader(stream))[-2:]
            with (branch / 'trajectory.csv').open() as stream:
                branch_rows = list(csv.DictReader(stream))[-2:]
            for full, resumed in zip(full_rows, branch_rows):
                for key in ('x', 'y', 'circulation', 'u', 'v'):
                    assert math.isclose(float(full[key]), float(resumed[key]), abs_tol=1e-12)

            changed, _ = run(integrator + '_changed',
                             f'outputTime 0.02\ndiagnosticsTime 0.04\ncheckpointTime 0.03\nrestartFile {source}\n', integrator)
            check_times(csv_times(changed / 'trajectory.csv'), [.05, .07, .09, .11, .13])
            check_times(csv_times(changed / 'diagnostics.csv'), [.05, .09, .13])
            check_times(checkpoint_times(changed / 'checkpoints'), [.08, .11, .13])

            # Version 4 has only the shared next-output clock.
            old = root / (integrator + '_v4.dat')
            old_lines = []
            for line in (common / 'checkpoints/checkpoint_00000001.dat').read_text().splitlines():
                if line.startswith(('output_schedule ', 'dipole_schedule ', 'event_index ')):
                    continue
                line = line.replace('POINT_VORTEX_CHECKPOINT 8',
                                    'POINT_VORTEX_CHECKPOINT 4')
                if line.startswith('dipole_config '):
                    fields = line.split()
                    line = ' '.join([*fields[:3], fields[-1]])
                if line.startswith('dipole_counts '):
                    fields = line.split()
                    line = ' '.join([*fields[:2], fields[-1]])
                old_lines.append(line)
            old.write_text('\n'.join(old_lines) + '\n')
            legacy, _ = run(integrator + '_legacy', f'restartFile {old}\n', integrator)
            check_times(csv_times(legacy / 'trajectory.csv'), [.04, .08, .12, .13])
            check_times(csv_times(legacy / 'diagnostics.csv'), [.04, .08, .12, .13])
            check_times(checkpoint_times(legacy / 'checkpoints'), [.08, .12, .13])

        for key in ('diagnosticsTime', 'checkpointTime'):
            for value in ('0', '-1', 'nan', 'inf'):
                _, result = run(key + value, f'{key} {value}\n', success=False)
                assert 'error:' in result.stderr
        for value in ('-1', 'nan', 'inf'):
            _, result = run('dipole_interval_' + value,
                            f'dipoleRemovalInterval {value}\n', success=False)
            assert 'error:' in result.stderr

        _, result = run('legacy_output_key', 'outputFile obsolete.csv\n', success=False)
        assert 'unknown parameter: outputFile' in result.stderr

        # Existing managed output must not be replaced unless explicitly requested.
        protected = root / 'protected'
        protected.mkdir()
        sentinel = protected / 'diagnostics.csv'
        sentinel.write_text('previous results\n')
        _, result = run('protected', success=False)
        assert 'already contains solver output' in result.stderr
        assert sentinel.read_text() == 'previous results\n'
        assert not (protected / 'trajectory.csv').exists()
        _, result = run('protected', 'overwriteRun true\n')
        assert result.returncode == 0
        assert sentinel.read_text().startswith('time,frame,')

        malformed = root / 'malformed.dat'
        malformed.write_text('not a vortex\n0 0 1\n')
        _, result = run('malformed',
                        f'initialCondition file\ninitialConditionFile {malformed}\n',
                        success=False)
        assert 'invalid initial condition on line 1' in result.stderr

        outside = root / 'outside.dat'
        outside.write_text('0 0 1\n1.01 0 -1\n')
        _, result = run('outside_disk',
                        'boundaryCondition disk\ndiskRadius 1\ninitialCondition file\n'
                        f'initialConditionFile {outside}\n', success=False)
        assert 'disk vortex lies on or outside the boundary' in result.stderr

        outside.write_text('-0.5 0 1\n0.5 0 -1\n')
        _, result = run('outside_periodic',
                        'boundaryCondition periodic\nboxLengthX 1\nboxLengthY 1\n'
                        'initialCondition file\n'
                        f'initialConditionFile {outside}\n', success=False)
        assert 'periodic vortex lies outside the fundamental box' in result.stderr

        # A physical-time dipole cadence is independent of all output clocks and
        # is saved after being advanced, so restarts preserve the next event.
        scheduled, _ = run(
            'dipole_schedule',
            'dipoleRemoval true\ndipoleRemovalInterval 0.03\ncheckpointTime 0.05\n')
        schedule_line = next(
            line for line in
            (scheduled / 'checkpoints/checkpoint_00000001.dat').read_text().splitlines()
            if line.startswith('dipole_schedule '))
        _, interval, next_removal = schedule_line.split()
        assert math.isclose(float(interval), 0.03)
        assert math.isclose(float(next_removal), 0.06)
        scheduled_restart, _ = run(
            'dipole_schedule_restart',
            'dipoleRemoval true\ndipoleRemovalInterval 0.03\ncheckpointTime 0.05\n'
            f'restartFile {scheduled}/checkpoints/checkpoint_00000001.dat\n')
        schedule_line = next(
            line for line in
            (scheduled_restart / 'checkpoints/checkpoint_00000002.dat').read_text().splitlines()
            if line.startswith('dipole_schedule '))
        _, interval, next_removal = schedule_line.split()
        assert math.isclose(float(interval), 0.03)
        assert math.isclose(float(next_removal), 0.12)

        upper_initial = root / 'upper-removal.dat'
        upper_initial.write_text('0 0 1\n0.1 0 1\n10 0 -1\n10.1 0 -1\n')
        upper_removed, _ = run(
            'upper_removal',
            'initialCondition file\n'
            f'initialConditionFile {upper_initial}\n'
            'dipoleRemoval true\ndipoleRemovalDistance 0.01\n'
            'dipoleRemovalUpper true\ndipoleRemovalUpperDistance 5\nendTime 0\n')
        assert csv_times(upper_removed / 'trajectory.csv') == []
        with (upper_removed / 'diagnostics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1 and int(rows[0]['removed_pairs']) == 2
        assert int(rows[0]['removed_upper_pairs']) == 2

        # The solver's built-in random selection must honor each bounded domain.
        periodic_random, _ = run(
            'periodic_random',
            'N 20\nboundaryCondition periodic\nboxLengthX 1\nboxLengthY 1\n'
            'initialCondition random\nendTime 0\n')
        with (periodic_random / 'trajectory.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 20
        assert all(-0.5 <= float(row['x']) < 0.5 and
                   -0.5 <= float(row['y']) < 0.5 for row in rows)

        disk_random, _ = run(
            'disk_random',
            'N 20\nboundaryCondition disk\ndiskRadius 0.75\n'
            'initialCondition random\nendTime 0\n')
        with (disk_random / 'trajectory.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 20
        assert all(float(row['x']) ** 2 + float(row['y']) ** 2 < 0.75 ** 2
                   for row in rows)

        # End times differing from a scheduled output only by roundoff need one final frame.
        rounded, _ = run('rounded', 'endTime 0.1200000000000001\n')
        check_times(csv_times(rounded / 'trajectory.csv'), [0, .04, .08, .1200000000000001])
        # Tolerances must scale with time, not with an artificial unit-time floor.
        tiny, _ = run('tiny', 'N 1\ntimeStep 1e-17\nendTime 1e-15\noutputTime 2e-16\n')
        check_times(csv_times(tiny / 'trajectory.csv'), [0, 2e-16, 4e-16, 6e-16, 8e-16, 1e-15])

        for key, value in [('output_index', '-1'), ('event_index', '-1'),
                           ('vortex_count', '999999999999999999'), ('next_output_time', '0'),
                           ('accepted_steps', '-1')]:
            corrupt = root / (key + '.dat')
            lines = source.read_text().splitlines()
            corrupt.write_text('\n'.join(f'{key} {value}' if line.startswith(key + ' ') else line
                                         for line in lines) + '\n')
            run('corrupt_' + key, f'restartFile {corrupt}\n', 'dopri5', success=False)

        initial = root / 'removed.dat'
        initial.write_text('0 0 1\n0.001 0 -1\n')
        removed, _ = run('removed', f'initialCondition file\ninitialConditionFile {initial}\n'
                         'dipoleRemoval true\ndipoleRemovalDistance 0.01\n'
                         'diagnosticsTime 0.03\ncheckpointTime 0.05\n')
        assert csv_times(removed / 'trajectory.csv') == []
        check_times(csv_times(removed / 'diagnostics.csv'), [0, .03, .06, .09, .12, .13])
        empty_source = removed / 'checkpoints/checkpoint_00000001.dat'
        resumed, _ = run('removed_restart', f'restartFile {empty_source}\n'
                         'dipoleRemoval true\ndipoleRemovalDistance 0.01\n'
                         'diagnosticsTime 0.03\ncheckpointTime 0.05\n')
        assert csv_times(resumed / 'trajectory.csv') == []
        check_times(checkpoint_times(resumed / 'checkpoints'), [.1, .13])

        zero, _ = run('zero', 'endTime 0\n')
        check_times(csv_times(zero / 'trajectory.csv'), [0])
        check_times(csv_times(zero / 'diagnostics.csv'), [0])
        check_times(checkpoint_times(zero / 'checkpoints'), [0])

        # Every run owns all outputs and records exactly how it was started.
        managed = root / 'managed_run'
        managed_params = root / 'managed.params'
        managed_params.write_text(
            'N 2\nboundaryCondition infinite\nintegrator rk4\ntimeStep 0.007\n'
            'endTime 0.08\noutputTime 0.04\nnumThreads 1\n'
            f'runDirectory {managed}\n')
        result = subprocess.run([*command, str(managed_params)], text=True, capture_output=True,
                                timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        trajectory = managed / 'trajectory.csv'
        diagnostics = managed / 'diagnostics.csv'
        assert trajectory.exists() and diagnostics.exists()
        assert (managed / 'checkpoints/checkpoint_00000002.dat').exists()
        assert csv_frames(trajectory) == [0, 1, 2], csv_frames(trajectory)
        assert csv_frames(diagnostics) == [0, 1, 2], csv_frames(diagnostics)
        record = (managed / 'resolved_parameters.txt').read_text()
        assert 'POINT_VORTEX_RUN_RECORD 1' in record
        assert 'backend ' in record
        assert 'trajectory_file ' in record and 'checkpoints' in record
        segment = managed / 'segments/segment_00000001/resolved_parameters.txt'
        assert segment.exists()

        managed_branch = root / 'managed_branch'
        branch_params = root / 'managed_branch.params'
        branch_params.write_text(
            'boundaryCondition infinite\nintegrator rk4\ntimeStep 0.007\nendTime 0.12\n'
            'outputTime 0.04\nnumThreads 1\n'
            f'restartFile {managed}/checkpoints/checkpoint_00000001.dat\n'
            f'runDirectory {managed_branch}\n')
        result = subprocess.run([*command, str(branch_params)], text=True, capture_output=True,
                                timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        branch_record = (managed_branch / 'resolved_parameters.txt').read_text()
        assert 'restarting 1' in branch_record
        assert csv_frames(managed_branch / 'trajectory.csv') == [1, 2, 3], \
            csv_frames(managed_branch / 'trajectory.csv')

    print('output scheduling, automatic directories, overwrite protection, and restart tests passed')


if __name__ == '__main__':
    main(sys.argv[1:])
