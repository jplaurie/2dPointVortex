# Point-vortex movie tool

`make_vortex_movie.py` reads the `trajectory.csv` file written by the C++
solver and streams it into an MP4. It groups rows by `time`, so the entire
trajectory does not need to fit in memory. The extra `index`, `u`, and `v`
columns may be present and are ignored. The number of vortices may vary between
saved frames when dipole removal is enabled.

## Requirements

- Python 3.10 or newer
- Matplotlib
- FFmpeg available as `ffmpeg`
- A LaTeX installation available as `latex`

For example, install Matplotlib with your Python environment's package manager:

```bash
python3 -m pip install matplotlib
```

## Examples

Run these commands from the repository root:

```bash
# Infinite plane; the script determines fixed limits from the selected frames.
python3 scripts/movie/make_vortex_movie.py runs/default/trajectory.csv \
    --geometry infinite --output runs/default/infinite.mp4

# Square periodic box. Coordinates are wrapped for display only.
python3 scripts/movie/make_vortex_movie.py runs/default/trajectory.csv \
    --geometry periodic --box-length 2.0 --output runs/default/periodic.mp4

# Periodic in x and unbounded in y. Only x coordinates are wrapped.
python3 scripts/movie/make_vortex_movie.py runs/default/trajectory.csv \
    --geometry periodic_x --box-length 2.0 --output runs/default/periodic_x.mp4

# Circular disk.
python3 scripts/movie/make_vortex_movie.py runs/default/trajectory.csv \
    --geometry disk --radius 1.0 --output runs/default/disk.mp4
```

`--start`, `--stop`, and `--stride` select output-frame indices. For example,
this renders frames 100 through 499, taking every second frame:

```bash
python3 scripts/movie/make_vortex_movie.py runs/default/trajectory.csv \
    --geometry disk --start 100 --stop 500 --stride 2 --output runs/default/excerpt.mp4
```

For an infinite-plane movie, `--xlim MIN MAX` and `--ylim MIN MAX` override the
automatic limits. For `periodic_x`, `--ylim` overrides the automatic unbounded-direction limits.
Other useful controls include `--fps`, `--dpi`,
`--marker-size`, and `--title`. Run the script with `--help` for the complete
command-line reference.

The script validates numeric CSV data and axis ranges before rendering, and
reports missing LaTeX or FFmpeg executables explicitly.

The trajectory must be ordered by output time, as produced by PointVortex.
Positive and negative vortices are drawn in red and blue respectively. Rows
with zero circulation are not displayed. Plot text is rendered with LaTeX and
Computer Modern serif fonts.

Only nonempty trajectory frames can be rendered. Once the last vortex is removed there are no
further trajectory rows to animate. An empty selection is rejected before opening the movie
output. The movie output path must differ from the input CSV path.

`script_movie_disc.py` is a historical example for legacy numbered snapshots, with hard-coded
paths. Use `make_vortex_movie.py` for current solver CSV output.
