"""Smoke-test the generator, plotting tools, analysis notebook, and movie tools.

Requires the optional plotting/movie dependencies documented in README.md.
Run from the repository root: python3 tests/tooling_tests.py build/release
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def checked(command, **kwargs):
    result = subprocess.run([str(arg) for arg in command], text=True, capture_output=True,
                            timeout=120, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def execute_notebook(path, replacements):
    notebook = json.loads(Path(path).read_text())
    sources = [''.join(cell['source']) for cell in notebook['cells']
               if cell['cell_type'] == 'code']
    namespace = {}
    for source in sources:
        for old, new in replacements.items():
            source = source.replace(old, new)
        exec(compile(source, str(path), 'exec'), namespace)
    namespace['plt'].close('all')


def main(build):
    repo = Path.cwd()
    build = Path(build).resolve()
    with tempfile.TemporaryDirectory(prefix='point_vortex_tooling_test_') as temporary:
        root = Path(temporary)
        os.environ['MPLCONFIGDIR'] = str(root / 'mplconfig')
        os.environ['MPLBACKEND'] = 'Agg'
        notebook = json.loads((repo / 'scripts/analysis/point_vortex_analysis.ipynb').read_text())
        code = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        for geometry in ('infinite', 'periodic', 'disk'):
            directory = root / geometry
            initial = directory / 'nested/initial.dat'
            checked([build / 'point_vortex_initial', '--geometry', geometry, '--case', 'random',
                     '--count', '6', '--min-separation', '.05', '--output', initial])
            params = root / (geometry + '.params')
            params.write_text(f'boundaryCondition {geometry}\ninitialConditionFile {initial}\n'
                              'integrator rk4\ntimeStep 0.00001\nendTime 0.00002\noutputTime 0.00001\n'
                              'numThreads 1\n'
                              f'runDirectory {directory}\n')
            checked([build / 'point_vortex_cpu', params])
            for mode in ('ordinary', 'zero_strength', 'empty'):
                trajectory = directory / 'trajectory.csv'
                if mode != 'ordinary':
                    trajectory = directory / (mode + '.csv')
                    rows = (directory / 'trajectory.csv').read_text().splitlines()
                    if mode == 'empty':
                        rows = rows[:1]
                    else:
                        rows = [rows[0]] + [','.join(row.split(',')[:4] + ['0'] + row.split(',')[5:])
                                           for row in rows[1:]]
                    trajectory.write_text('\n'.join(rows) + '\n')
                figures = directory / ('figures_' + mode)
                first = code[0].replace('geometry = "periodic"', f'geometry = {geometry!r}')
                first = first.replace('box_length = 1.0', 'box_length = 2.0')
                first = first.replace('trajectory_name = "runs/default/trajectory.csv"', f'trajectory_name = {str(trajectory)!r}')
                first = first.replace('diagnostics_name = "runs/default/diagnostics.csv"',
                                      f'diagnostics_name = {str(directory / "diagnostics.csv")!r}')
                first = first.replace('figure_directory_name = "runs/default/figures"',
                                      f'figure_directory_name = {str(figures)!r}')
                namespace = {}
                for source in (first, *code[1:]):
                    exec(compile(source, 'point_vortex_analysis.ipynb', 'exec'), namespace)
                assert (figures / 'diagnostics_evolution.pdf').stat().st_size > 100
                assert (figures / 'vortex_configurations.pdf').exists() == (mode != 'empty')
                namespace['plt'].close('all')
            movie = directory / 'movie.mp4'
            checked([sys.executable, repo / 'scripts/movie/make_vortex_movie.py',
                     directory / 'trajectory.csv', '--geometry', geometry, '--box-length', '2',
                     '--dpi', '50', '--fps', '4', '-o', movie])
            assert movie.stat().st_size > 100
            figures = directory / 'figures'
            configuration = figures / 'vortices.pdf'
            diagnostics = figures / 'diagnostics.pdf'
            modern_movie = figures / 'vortices.mp4'
            notebook_settings = {
                "RUN_DIRECTORY = ROOT / 'runs/default'":
                    f'RUN_DIRECTORY = Path({str(directory)!r})',
                'USE_TEX = True': 'USE_TEX = False',
            }
            execute_notebook(repo / 'scripts/vortices.ipynb', {
                **notebook_settings,
                'FRAMES = [-1]': 'FRAMES = [0, -1]',
            })
            execute_notebook(repo / 'scripts/diagnostics.ipynb', notebook_settings)
            checked([sys.executable, repo / 'scripts/movie_vortices.py',
                     '--run-dir', directory, '--frames', '0', '-1', '--dpi', '50', '--fps', '4',
                     '--output', modern_movie, '--no-tex'])
            assert configuration.stat().st_size > 100
            assert diagnostics.stat().st_size > 100
            assert modern_movie.stat().st_size > 100
            print(f'{geometry}: generator, notebook, plots, and movies passed', flush=True)

        # Bad data and empty selections must fail before destroying an existing movie.
        module_path = repo / 'scripts/movie/make_vortex_movie.py'
        spec = importlib.util.spec_from_file_location('vortex_movie', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        unordered = root / 'unordered.csv'
        unordered.write_text('time,x,y,circulation\n1,0,0,1\n0,0,0,1\n')
        try:
            list(module.trajectory_frames(unordered))
        except ValueError:
            pass
        else:
            raise AssertionError('decreasing movie times were accepted')
        sentinel = root / 'protected.mp4'
        sentinel.write_text('existing movie')
        result = subprocess.run([sys.executable, str(module_path), str(root / 'disk/empty.csv'),
                                 '--geometry', 'disk', '-o', str(sentinel)], capture_output=True, timeout=30)
        assert result.returncode != 0 and sentinel.read_text() == 'existing movie'
        for args in (['0'], ['-1'], ['3junk'], ['2', '0'], ['2', 'bad']):
            result = subprocess.run([str(build / 'point_vortex_benchmark'), *args],
                                    capture_output=True, timeout=30)
            assert result.returncode == 1
    print('all tooling tests passed')


if __name__ == '__main__':
    main(sys.argv[1])
