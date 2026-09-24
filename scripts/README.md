# Plotting and movies

These notebooks and scripts read the self-contained run directories produced by every
PointVortex backend. They follow the same layout and plotting conventions
as the tools in `2dGrossPitaevskii/scripts/` and `2dNavierStokes/scripts/`.

## Requirements

All tools require Python 3, NumPy, and Matplotlib; the notebooks additionally
require Jupyter. PDF figures use external LaTeX by default; set `USE_TEX=False`
in a notebook (or pass `--no-tex` to the movie) when LaTeX is unavailable. MP4 output also
requires `ffmpeg`, while GIF output uses Matplotlib's Pillow writer.

The notebook configuration cells default to `runs/default`. Change
`RUN_DIRECTORY` to select another completed run; its `trajectory.csv`,
`diagnostics.csv`, and `resolved_parameters.txt` are then found automatically:

```bash
jupyter lab scripts/vortices.ipynb scripts/diagnostics.ipynb
python3 scripts/movie_vortices.py --run-dir runs/periodic_n400 \
  --output runs/periodic_n400/figures/vortices.mp4 --no-tex
```

Run the movie script with `--help` for every command-line option.

## Geometry and box configuration

The plotting notebooks normally read `boundaryCondition`, `boxLengthX`,
`boxLengthY`, and `diskRadius` from the run's `resolved_parameters.txt`. This
keeps a figure consistent with its simulation without repeating settings.
The configuration cell values take precedence when an override is useful:

```python
# Square periodic display. Coordinates are wrapped for display only.
GEOMETRY = 'periodic'
BOX_LENGTH = 2.0
BOX_LENGTH_X = BOX_LENGTH_Y = None

# Rectangular periodic display instead:
BOX_LENGTH = None
BOX_LENGTH_X = 2.0
BOX_LENGTH_Y = 4.0

# Singly periodic display; only x is wrapped.
GEOMETRY = 'periodic_x'
BOX_LENGTH = 2.0
Y_LIMITS = (-3.0, 3.0)

# Disk boundary and an explicit viewing window instead:
GEOMETRY = 'disk'
DISK_RADIUS = 1.5
X_LIMITS = (-2.0, 2.0)
Y_LIMITS = (-2.0, 2.0)
```

`BOX_LENGTH` cannot be combined with `BOX_LENGTH_X` or `BOX_LENGTH_Y`. Box
lengths and disk radius must be positive. The solver supports square doubly
periodic dynamics and the `periodic_x` cylinder geometry; separate display
lengths make the plotting tools usable with rectangular trajectory data too.

## Configuration figures

`vortices.ipynb` writes a multi-panel physical-space figure. `FRAMES` selects exact labels, with
negative indices such as `[-1]` selecting from the end. Set `FRAMES=None` to use the inclusive
`FRAME_START`, `FRAME_STOP`, and `FRAME_STRIDE` range and choose up to `SNAPSHOT_COUNT` evenly
spaced configurations:

```python
FRAMES = [0, 50, 100, -1]
PANEL_COLUMNS = 2
CONFIGURATION_FIGURE = RUN_DIRECTORY / 'figures/selected_vortices.pdf'
```

Positive, negative, and zero circulations are shown in red, blue, and grey.
Marker area scales with circulation magnitude using one scale shared by all
panels.

## Diagnostics

`diagnostics.ipynb` writes a six-panel figure containing circulation,
Hamiltonian, the invariants appropriate to the selected geometry, drift from
the initial and current segment reference states, and cumulative dipole
events. The following options select or smooth the data:

```python
TIME_RANGE = (0.2, 0.8)
ROLLING_WINDOW = 5
DIAGNOSTICS_FIGURE = RUN_DIRECTORY / 'figures/diagnostics_excerpt.pdf'
```

`TIME_RANGE` and `FRAME_RANGE` are inclusive and either bound may be `None`. A rolling window of one
(the default) plots the saved data unchanged. Dipole counts remain unsmoothed
step plots.

The linear impulses are displayed for `infinite`, `periodic_x`, and
`periodic`; angular impulse is displayed for `infinite` and `disk`.

## Movies

`movie_vortices.py` accepts the same geometry and frame-selection controls as
the configuration plot. The output suffix chooses MP4 or GIF. MP4 defaults to
H.264; pass `--codec h265` for HEVC/H.265:

```bash
python3 scripts/movie_vortices.py --run-dir runs/periodic_n400 \
  --start 10 --stop 200 --stride 2 --fps 20 --codec h265 \
  --output runs/periodic_n400/figures/vortices.mp4
```

All movie frames use fixed axes and one circulation-to-marker-size scale, so
the display does not jump or visually rescale during the animation.

## File summary

| Script | Output |
| --- | --- |
| `vortices.ipynb` | Selected vortex configurations (PDF by default) |
| `diagnostics.ipynb` | Invariants, drift, and dipole-event figure |
| `movie_vortices.py` | Vortex-configuration MP4 or GIF |
| `point_vortex_plotting.py` | Shared readers, geometry, style, and writers |

The older notebook under `scripts/analysis/` and movie tool under
`scripts/movie/` remain available for compatibility. New workflows should use
the notebooks and movie tool above.
