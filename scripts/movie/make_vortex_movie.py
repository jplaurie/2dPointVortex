#!/usr/bin/env python3
"""Render a PointVortex trajectory CSV as an MP4 movie."""

from __future__ import annotations

import argparse
import csv
import math
from itertools import chain
import shutil
from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "text.usetex": True,
    }
)
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Rectangle


REQUIRED_COLUMNS = {"time", "x", "y", "circulation"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an MP4 from the trajectory CSV written by PointVortex."
    )
    parser.add_argument("input", type=Path, help="trajectory CSV, normally runDirectory/trajectory.csv")
    parser.add_argument("-o", "--output", type=Path, default=Path("vortices.mp4"))
    parser.add_argument(
        "--geometry", choices=("infinite", "periodic_x", "periodic", "disk"), default="infinite"
    )
    parser.add_argument("--box-length", type=float, default=2.0, help="periodic length or square side")
    parser.add_argument("--radius", type=float, default=1.0, help="disk radius")
    parser.add_argument("--xlim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--start", type=int, default=0, help="first frame index")
    parser.add_argument("--stop", type=int, help="exclusive final frame index")
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--marker-size", type=float, default=24.0)
    parser.add_argument("--title", default="Point-vortex dynamics")
    args = parser.parse_args()

    positive_options = (args.box_length, args.radius, args.fps, args.dpi, args.marker_size)
    if not all(math.isfinite(value) and value > 0 for value in positive_options):
        parser.error("box length, radius, fps, dpi, and marker size must be positive and finite")
    if args.start < 0 or (args.stop is not None and args.stop < args.start):
        parser.error("invalid frame range")
    if args.stride <= 0:
        parser.error("--stride must be positive")
    for option, limits in (("--xlim", args.xlim), ("--ylim", args.ylim)):
        if limits and (not all(map(math.isfinite, limits)) or limits[0] >= limits[1]):
            parser.error(f"{option} requires finite MIN < MAX")
    return args


def trajectory_frames(path: Path) -> Iterator[tuple[float, list[float], list[float], list[float]]]:
    """Yield consecutive time groups while retaining only one frame in memory."""
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
            raise ValueError(f"trajectory is missing columns: {', '.join(sorted(missing))}")

        current_time: float | None = None
        x: list[float] = []
        y: list[float] = []
        circulation: list[float] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                time = float(row["time"])
                row_x = float(row["x"])
                row_y = float(row["y"])
                gamma = float(row["circulation"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid numeric value on CSV row {row_number}") from error
            if not all(math.isfinite(value) for value in (time, row_x, row_y, gamma)):
                raise ValueError(f"non-finite numeric value on CSV row {row_number}")

            if current_time is not None and time < current_time:
                raise ValueError(f"output times decrease on CSV row {row_number}")
            if current_time is not None and time != current_time:
                yield current_time, x, y, circulation
                x, y, circulation = [], [], []
            current_time = time
            x.append(row_x)
            y.append(row_y)
            circulation.append(gamma)

        if current_time is not None:
            yield current_time, x, y, circulation


def selected_frames(path: Path, start: int, stop: int | None, stride: int):
    for index, frame in enumerate(trajectory_frames(path)):
        if index < start:
            continue
        if stop is not None and index >= stop:
            break
        if (index - start) % stride == 0:
            yield index, frame


def infinite_limits(path: Path, args: argparse.Namespace) -> tuple[tuple[float, float], tuple[float, float]]:
    """Find fixed limits in a low-memory first pass so the movie axes do not jump."""
    minimum_x = minimum_y = math.inf
    maximum_x = maximum_y = -math.inf
    for _, (_, x, y, circulation) in selected_frames(path, args.start, args.stop, args.stride):
        visible = [(px, py) for px, py, gamma in zip(x, y, circulation) if gamma != 0.0]
        if visible:
            frame_x, frame_y = zip(*visible)
            minimum_x, maximum_x = min(minimum_x, min(frame_x)), max(maximum_x, max(frame_x))
            minimum_y, maximum_y = min(minimum_y, min(frame_y)), max(maximum_y, max(frame_y))
    if not math.isfinite(minimum_x):
        raise ValueError("no non-zero-circulation vortices in the selected frames")

    span = max(maximum_x - minimum_x, maximum_y - minimum_y, 1.0e-12)
    padding = 0.05 * span
    return (minimum_x - padding, maximum_x + padding), (minimum_y - padding, maximum_y + padding)


def periodic_x_y_limits(path: Path, args: argparse.Namespace) -> tuple[float, float]:
    """Find fixed limits in the unbounded direction without using unwrapped x."""
    minimum_y, maximum_y = math.inf, -math.inf
    for _, (_, _, y, circulation) in selected_frames(path, args.start, args.stop, args.stride):
        visible_y = [py for py, gamma in zip(y, circulation) if gamma != 0.0]
        if visible_y:
            minimum_y = min(minimum_y, min(visible_y))
            maximum_y = max(maximum_y, max(visible_y))
    if not math.isfinite(minimum_y):
        raise ValueError("no non-zero-circulation vortices in the selected frames")
    if maximum_y - minimum_y < 1.0e-12:
        center = 0.5 * (minimum_y + maximum_y)
        return center - 0.5 * args.box_length, center + 0.5 * args.box_length
    padding = 0.05 * max(maximum_y - minimum_y, args.box_length)
    return minimum_y - padding, maximum_y + padding


def wrap_periodic(values: list[float], length: float) -> list[float]:
    half_length = 0.5 * length
    return [((value + half_length) % length) - half_length for value in values]


def offsets(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Matplotlib requires a two-column shape, even when no points are present."""
    return points if points else [(math.nan, math.nan)]


def main() -> None:
    args = parse_arguments()
    if not args.input.is_file():
        raise FileNotFoundError(f"trajectory file not found: {args.input}")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("movie output must differ from the trajectory input")
    frames = iter(selected_frames(args.input, args.start, args.stop, args.stride))
    first_frame = next(frames, None)
    if first_frame is None:
        raise ValueError("no trajectory frames selected; the file may contain no surviving vortices")
    if shutil.which("latex") is None:
        raise RuntimeError("LaTeX executable not found; install LaTeX to render plot text")
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("FFmpeg executable not found; install FFmpeg to encode MP4 output")

    if args.geometry == "periodic":
        half_length = 0.5 * args.box_length
        x_limits = y_limits = (-half_length, half_length)
    elif args.geometry == "periodic_x":
        half_length = 0.5 * args.box_length
        automatic_y = periodic_x_y_limits(args.input, args)
        x_limits = (-half_length, half_length)
        y_limits = tuple(args.ylim) if args.ylim else automatic_y
    elif args.geometry == "disk":
        x_limits = y_limits = (-args.radius, args.radius)
    else:
        automatic_x, automatic_y = infinite_limits(args.input, args)
        x_limits = tuple(args.xlim) if args.xlim else automatic_x
        y_limits = tuple(args.ylim) if args.ylim else automatic_y

    figure, axes = plt.subplots(figsize=(7, 7))
    axes.set(xlim=x_limits, ylim=y_limits, xlabel=r"$x$", ylabel=r"$y$", title=args.title)
    axes.set_aspect("equal", adjustable="box")
    axes.grid(alpha=0.2)

    if args.geometry == "periodic":
        axes.add_patch(
            Rectangle(
                (-0.5 * args.box_length, -0.5 * args.box_length),
                args.box_length,
                args.box_length,
                fill=False,
                color="black",
                linewidth=1.2,
            )
        )
    elif args.geometry == "periodic_x":
        axes.axvline(-0.5 * args.box_length, color="black", linewidth=1.0)
        axes.axvline(0.5 * args.box_length, color="black", linewidth=1.0)
    elif args.geometry == "disk":
        axes.add_patch(Circle((0.0, 0.0), args.radius, fill=False, color="black", linewidth=1.5))

    positive = axes.scatter([], [], s=args.marker_size, c="tab:red", label=r"$\Gamma>0$")
    negative = axes.scatter([], [], s=args.marker_size, c="tab:blue", label=r"$\Gamma<0$")
    time_label = axes.text(0.02, 0.97, "", transform=axes.transAxes, va="top")
    axes.legend(loc="upper right", frameon=False)

    writer = FFMpegWriter(fps=args.fps, metadata={"title": args.title, "artist": "PointVortex"})
    frame_count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(figure, str(args.output), args.dpi):
        for frame_index, (time, x, y, circulation) in chain((first_frame,), frames):
            if args.geometry == "periodic":
                x = wrap_periodic(x, args.box_length)
                y = wrap_periodic(y, args.box_length)
            elif args.geometry == "periodic_x":
                x = wrap_periodic(x, args.box_length)

            positive.set_offsets(
                offsets(
                    [(px, py) for px, py, gamma in zip(x, y, circulation) if gamma > 0]
                )
            )
            negative.set_offsets(
                offsets(
                    [(px, py) for px, py, gamma in zip(x, y, circulation) if gamma < 0]
                )
            )
            time_label.set_text(rf"$t = {time:.6g}$")
            writer.grab_frame()
            frame_count += 1
            print(f"frame {frame_index}: time={time:.9g}")

    plt.close(figure)
    print(f"wrote {frame_count} frames to {args.output}")


if __name__ == "__main__":
    main()
