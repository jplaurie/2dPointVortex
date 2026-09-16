"""Shared readers and plotting helpers for the PointVortex solver."""

from __future__ import annotations

import csv
import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


GEOMETRIES = ("infinite", "periodic", "disk")


@dataclass(frozen=True)
class VortexFrame:
    """One saved point-vortex configuration."""

    frame: int
    time: float
    x: np.ndarray
    y: np.ndarray
    circulation: np.ndarray


@dataclass(frozen=True)
class Domain:
    """Geometry and plotting limits resolved from parameters and overrides."""

    geometry: str
    length_x: float
    length_y: float
    radius: float
    x_limits: tuple[float, float] | None = None
    y_limits: tuple[float, float] | None = None


def repository_root() -> Path:
    """Return the repository containing this module."""
    return Path(__file__).resolve().parent.parent


def use_plot_style(use_tex: bool = True, font_size: float = 14.0) -> None:
    """Apply the publication-oriented style shared with the PDE solvers."""
    mpl.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "legend.fontsize": 0.78 * font_size,
            "xtick.labelsize": 0.85 * font_size,
            "ytick.labelsize": 0.85 * font_size,
            "lines.linewidth": 2.0,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "savefig.bbox": "tight",
            "savefig.format": "pdf",
            "figure.constrained_layout.use": True,
        }
    )


def read_parameters(path: str | Path) -> dict[str, str]:
    """Read a user parameter file or solver ``resolved_parameters.txt``."""
    parameters: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line or line.startswith("POINT_VORTEX_RUN_RECORD"):
                continue
            try:
                fields = shlex.split(line.replace("=", " ", 1))
            except ValueError as error:
                raise ValueError(
                    f"invalid parameter syntax on line {line_number}: {path}"
                ) from error
            if len(fields) >= 2:
                parameters[fields[0]] = " ".join(fields[1:])
    return parameters


def parameter_float(parameters: dict[str, str], key: str, default: float) -> float:
    """Return one finite floating-point parameter or its default."""
    value = float(parameters[key]) if key in parameters else default
    if not math.isfinite(value):
        raise ValueError(f"parameter {key} must be finite")
    return value


def read_numeric_csv(
    path: str | Path, required_columns: Iterable[str], *, allow_empty: bool = False
) -> dict[str, np.ndarray]:
    """Load a numeric CSV into named one-dimensional arrays with validation."""
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        missing = set(required_columns).difference(fields)
        if missing:
            raise ValueError(f"{source} is missing columns: {', '.join(sorted(missing))}")
        columns: dict[str, list[float]] = {name: [] for name in fields}
        for line_number, row in enumerate(reader, start=2):
            try:
                values = {name: float(row[name]) for name in fields}
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid numeric value in {source} on row {line_number}"
                ) from error
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"non-finite numeric value in {source} on row {line_number}")
            for name, value in values.items():
                columns[name].append(value)
    if not allow_empty and not next(iter(columns.values()), []):
        raise ValueError(f"CSV file has no data rows: {source}")
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def read_trajectory(path: str | Path, *, allow_empty: bool = False) -> dict[int, VortexFrame]:
    """Load and group ``trajectory.csv`` rows by their saved frame number."""
    table = read_numeric_csv(
        path,
        {"time", "frame", "x", "y", "circulation"},
        allow_empty=allow_empty,
    )
    if not table["frame"].size:
        return {}
    integer_frames = table["frame"].astype(np.int64)
    if not np.array_equal(table["frame"], integer_frames):
        raise ValueError(f"trajectory frame labels must be integers: {path}")
    if np.any(np.diff(integer_frames) < 0) or np.any(np.diff(table["time"]) < 0.0):
        raise ValueError(f"trajectory rows are not ordered by frame and time: {path}")

    frames: dict[int, VortexFrame] = {}
    for frame in np.unique(integer_frames):
        mask = integer_frames == frame
        times = np.unique(table["time"][mask])
        if times.size != 1:
            raise ValueError(f"trajectory frame {frame} contains multiple output times")
        frames[int(frame)] = VortexFrame(
            frame=int(frame),
            time=float(times[0]),
            x=table["x"][mask],
            y=table["y"][mask],
            circulation=table["circulation"][mask],
        )
    return frames


def read_diagnostics(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate the solver diagnostics table."""
    required = {
        "time",
        "frame",
        "circulation",
        "linear_impulse_x",
        "linear_impulse_y",
        "angular_impulse",
        "hamiltonian",
        "delta_circulation",
        "delta_linear_impulse_x",
        "delta_linear_impulse_y",
        "delta_angular_impulse",
        "delta_hamiltonian",
        "segment_delta_circulation",
        "segment_delta_linear_impulse_x",
        "segment_delta_linear_impulse_y",
        "segment_delta_angular_impulse",
        "segment_delta_hamiltonian",
        "removed_pairs",
        "reinjected_pairs",
    }
    table = read_numeric_csv(path, required)
    if np.any(np.diff(table["time"]) <= 0.0):
        raise ValueError(f"diagnostics times must be strictly increasing: {path}")
    return table


def select_frames(
    available: Iterable[int],
    requested: Sequence[int] | None = None,
    *,
    start: int | None = None,
    stop: int | None = None,
    stride: int = 1,
) -> list[int]:
    """Select labels, accepting negative indices such as ``-1`` for the last frame."""
    choices = sorted(set(int(frame) for frame in available))
    if not choices:
        raise ValueError("there are no saved vortex configurations")
    if requested is not None:
        selected: list[int] = []
        for frame in requested:
            if frame < 0:
                try:
                    resolved = choices[frame]
                except IndexError as error:
                    raise ValueError(f"negative frame index {frame} is out of range") from error
            else:
                resolved = int(frame)
            if resolved not in choices:
                raise ValueError(f"frame {resolved} is not available")
            if resolved not in selected:
                selected.append(resolved)
        if not selected:
            raise ValueError("the frame selection is empty")
        return selected
    if stride < 1:
        raise ValueError("stride must be at least one")
    selected = [
        frame
        for frame in choices
        if (start is None or frame >= start) and (stop is None or frame <= stop)
    ][::stride]
    if not selected:
        raise ValueError("the frame selection is empty")
    return selected


def evenly_spaced_frames(frames: Sequence[int], count: int) -> list[int]:
    """Choose at most ``count`` approximately evenly spaced entries."""
    if count < 1:
        raise ValueError("snapshot count must be at least one")
    if len(frames) <= count:
        return list(frames)
    indices = np.rint(np.linspace(0, len(frames) - 1, count)).astype(int)
    return [frames[index] for index in np.unique(indices)]


def resolve_domain(
    parameters: dict[str, str],
    *,
    geometry: str | None = None,
    box_length: float | None = None,
    box_length_x: float | None = None,
    box_length_y: float | None = None,
    disk_radius: float | None = None,
    x_limits: Sequence[float] | None = None,
    y_limits: Sequence[float] | None = None,
) -> Domain:
    """Resolve geometry and dimensions, giving command-line values precedence."""
    resolved_geometry = geometry or parameters.get("boundaryCondition", "infinite")
    if resolved_geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {', '.join(GEOMETRIES)}")
    if box_length is not None and (box_length_x is not None or box_length_y is not None):
        raise ValueError("--box-length cannot be combined with --box-length-x or --box-length-y")
    if box_length is not None:
        box_length_x = box_length_y = box_length
    length_x = (
        box_length_x
        if box_length_x is not None
        else parameter_float(parameters, "boxLengthX", 1.0)
    )
    length_y = (
        box_length_y
        if box_length_y is not None
        else parameter_float(parameters, "boxLengthY", 1.0)
    )
    radius = (
        disk_radius
        if disk_radius is not None
        else parameter_float(parameters, "diskRadius", 1.0)
    )
    if not all(math.isfinite(value) and value > 0.0 for value in (length_x, length_y, radius)):
        raise ValueError("box lengths and disk radius must be positive and finite")

    def limits(values: Sequence[float] | None, name: str) -> tuple[float, float] | None:
        if values is None:
            return None
        lower, upper = float(values[0]), float(values[1])
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError(f"{name} requires finite MIN < MAX")
        return lower, upper

    return Domain(
        resolved_geometry,
        float(length_x),
        float(length_y),
        float(radius),
        limits(x_limits, "x limits"),
        limits(y_limits, "y limits"),
    )


def wrap_periodic(values: np.ndarray, length: float) -> np.ndarray:
    """Wrap coordinates into a centered periodic interval."""
    return (np.asarray(values) + 0.5 * length) % length - 0.5 * length


def display_coordinates(frame: VortexFrame, domain: Domain) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinates as displayed for the selected geometry."""
    if domain.geometry == "periodic":
        return (
            wrap_periodic(frame.x, domain.length_x),
            wrap_periodic(frame.y, domain.length_y),
        )
    return frame.x, frame.y


def axis_limits(
    frames: Sequence[VortexFrame], domain: Domain
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return fixed limits for configuration plots and movies."""
    if domain.geometry == "periodic":
        defaults = (
            (-0.5 * domain.length_x, 0.5 * domain.length_x),
            (-0.5 * domain.length_y, 0.5 * domain.length_y),
        )
    elif domain.geometry == "disk":
        defaults = ((-domain.radius, domain.radius), (-domain.radius, domain.radius))
    else:
        x = np.concatenate([frame.x for frame in frames])
        y = np.concatenate([frame.y for frame in frames])
        if not x.size:
            raise ValueError("the selected configurations contain no vortices")
        span = max(float(np.ptp(x)), float(np.ptp(y)))
        if span < 1.0e-12:
            span = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1.0)
        padding = 0.05 * span
        defaults = (
            (float(np.min(x) - padding), float(np.max(x) + padding)),
            (float(np.min(y) - padding), float(np.max(y) + padding)),
        )
    return domain.x_limits or defaults[0], domain.y_limits or defaults[1]


def draw_boundary(axis: plt.Axes, domain: Domain) -> None:
    """Draw the physical boundary for periodic and disk geometries."""
    if domain.geometry == "periodic":
        axis.add_patch(
            Rectangle(
                (-0.5 * domain.length_x, -0.5 * domain.length_y),
                domain.length_x,
                domain.length_y,
                fill=False,
                color="black",
                linewidth=1.2,
            )
        )
    elif domain.geometry == "disk":
        axis.add_patch(
            Circle((0.0, 0.0), domain.radius, fill=False, color="black", linewidth=1.4)
        )


def marker_areas(
    circulation: np.ndarray,
    minimum: float,
    maximum: float,
    *,
    scale: float | None = None,
) -> np.ndarray:
    """Scale marker area by absolute circulation using a stable reference."""
    if minimum <= 0.0 or maximum < minimum:
        raise ValueError("marker sizes require 0 < minimum <= maximum")
    strength = np.abs(np.asarray(circulation, dtype=float))
    denominator = (
        float(np.max(strength))
        if scale is None and strength.size
        else float(scale or 0.0)
    )
    if denominator <= 0.0:
        return np.full(strength.shape, minimum)
    return minimum + (maximum - minimum) * np.minimum(strength / denominator, 1.0)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Return a centered moving average with NaNs at incomplete edges."""
    array = np.asarray(values, dtype=float)
    if window <= 1:
        return array.copy()
    if window > array.size:
        raise ValueError("rolling-average window exceeds the selected diagnostics length")
    result = np.full(array.shape, np.nan)
    valid = np.convolve(array, np.ones(window) / window, mode="valid")
    left = (window - 1) // 2
    result[left : left + valid.size] = valid
    return result


def filter_diagnostics(
    table: dict[str, np.ndarray],
    *,
    time_range: Sequence[float | None] | None = None,
    frame_range: Sequence[int | None] | None = None,
) -> dict[str, np.ndarray]:
    """Select inclusive time and frame intervals from diagnostics."""
    mask = np.ones(table["time"].size, dtype=bool)
    if time_range is not None:
        lower, upper = time_range
        if lower is not None:
            mask &= table["time"] >= lower
        if upper is not None:
            mask &= table["time"] <= upper
    if frame_range is not None:
        lower, upper = frame_range
        if lower is not None:
            mask &= table["frame"] >= lower
        if upper is not None:
            mask &= table["frame"] <= upper
    if not np.any(mask):
        raise ValueError("the requested diagnostics interval is empty")
    return {name: values[mask] for name, values in table.items()}


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 200) -> Path:
    """Create the destination directory and save a tightly cropped figure."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi)
    return destination


def write_animation(
    movie_animation,
    path: str | Path,
    fps: float,
    dpi: int,
    codec: str = "h264",
) -> Path:
    """Save a Matplotlib animation with FFmpeg (MP4) or Pillow (GIF)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".gif":
        movie_animation.save(destination, writer="pillow", fps=fps, dpi=dpi)
    elif destination.suffix.lower() == ".mp4":
        codecs = {"h264": "libx264", "h265": "libx265"}
        writer = mpl_animation.FFMpegWriter(
            fps=fps,
            codec=codecs[codec],
            extra_args=["-pix_fmt", "yuv420p"],
        )
        movie_animation.save(destination, writer=writer, dpi=dpi)
    else:
        raise ValueError("movie output must end in .mp4 or .gif")
    return destination
