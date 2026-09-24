#!/usr/bin/env python3
"""Create an MP4 or GIF from saved point-vortex configurations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from point_vortex_plotting import (
    GEOMETRIES,
    axis_limits,
    display_coordinates,
    draw_boundary,
    marker_areas,
    read_parameters,
    read_trajectory,
    repository_root,
    resolve_domain,
    select_frames,
    use_plot_style,
    write_animation,
)


def parse_arguments() -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)

    # Input/output configuration.
    parser.add_argument("--run-dir", type=Path, default=root / "runs/default")
    parser.add_argument("--input", type=Path, help="trajectory CSV; defaults below --run-dir")
    parser.add_argument(
        "--parameters", type=Path, help="resolved parameters; defaults below --run-dir"
    )
    parser.add_argument("--output", type=Path, help="default: RUN_DIR/figures/vortices.mp4")

    # Frame configuration. The range endpoints are inclusive, matching the
    # movie scripts in 2dGrossPitaevskii and 2dNavierStokes.
    parser.add_argument("--frames", nargs="*", type=int, help="exact labels; -1 means last")
    parser.add_argument("--start", type=int, help="first frame label")
    parser.add_argument("--stop", type=int, help="last frame label (inclusive)")
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth available frame")

    # Geometry configuration. Normally these values are inferred from the run
    # record. They are exposed so a trajectory can be displayed in a different
    # box without editing the script.
    parser.add_argument("--geometry", choices=GEOMETRIES)
    parser.add_argument("--box-length", type=float, help="periodic length or square side")
    parser.add_argument("--box-length-x", type=float, help="x-periodic length or box width")
    parser.add_argument("--box-length-y", type=float, help="periodic box height")
    parser.add_argument("--disk-radius", type=float, help="disk boundary radius")
    parser.add_argument("--xlim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"))

    # Movie and appearance configuration.
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--codec", choices=["h264", "h265"], default="h264")
    parser.add_argument("--marker-min", type=float, default=18.0)
    parser.add_argument("--marker-max", type=float, default=40.0)
    parser.add_argument("--title", default="Point-vortex dynamics")
    parser.add_argument("--no-tex", action="store_true", help="disable external LaTeX text")
    args = parser.parse_args()
    if args.stride < 1 or args.fps <= 0.0 or args.dpi < 1:
        parser.error("stride, fps, and dpi must be positive")
    if not (0.0 < args.marker_min <= args.marker_max):
        parser.error("marker sizes require 0 < --marker-min <= --marker-max")
    return args


def empty_offsets() -> np.ndarray:
    """Return the two-column empty shape expected by Matplotlib scatter."""
    return np.empty((0, 2), dtype=float)


def main() -> None:
    args = parse_arguments()
    trajectory_path = args.input or args.run_dir / "trajectory.csv"
    parameter_path = args.parameters or args.run_dir / "resolved_parameters.txt"
    output_path = args.output or args.run_dir / "figures/vortices.mp4"
    if trajectory_path.resolve() == output_path.resolve():
        raise ValueError("movie output must differ from the trajectory input")

    use_plot_style(not args.no_tex)
    parameters = read_parameters(parameter_path) if parameter_path.exists() else {}
    domain = resolve_domain(
        parameters,
        geometry=args.geometry,
        box_length=args.box_length,
        box_length_x=args.box_length_x,
        box_length_y=args.box_length_y,
        disk_radius=args.disk_radius,
        x_limits=args.xlim,
        y_limits=args.ylim,
    )
    trajectory = read_trajectory(trajectory_path)
    selected = select_frames(
        trajectory,
        args.frames if args.frames else None,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
    )
    configurations = [trajectory[frame] for frame in selected]
    x_limits, y_limits = axis_limits(configurations, domain)
    global_strength = max(
        (float(np.max(np.abs(frame.circulation))) for frame in configurations),
        default=0.0,
    )

    fig, axis = plt.subplots(figsize=(7, 7))
    axis.set(xlim=x_limits, ylim=y_limits, xlabel=r"$x$", ylabel=r"$y$")
    axis.set_aspect("equal", adjustable="box")
    draw_boundary(axis, domain)
    positive = axis.scatter([], [], color="tab:red", label=r"$\Gamma>0$", alpha=0.85)
    negative = axis.scatter([], [], color="tab:blue", label=r"$\Gamma<0$", alpha=0.85)
    zero = axis.scatter([], [], color="0.5", label=r"$\Gamma=0$", alpha=0.85)
    axis.legend(loc="upper right", frameon=False)
    title = axis.set_title("")

    def update(index: int):
        frame = configurations[index]
        x, y = display_coordinates(frame, domain)
        gamma = frame.circulation
        sizes = marker_areas(
            gamma, args.marker_min, args.marker_max, scale=global_strength
        )
        for artist, mask in (
            (positive, gamma > 0.0),
            (negative, gamma < 0.0),
            (zero, gamma == 0.0),
        ):
            offsets = (
                np.column_stack((x[mask], y[mask]))
                if np.any(mask)
                else empty_offsets()
            )
            artist.set_offsets(offsets)
            artist.set_sizes(sizes[mask])
        title.set_text(
            rf"{args.title}: frame {frame.frame}, $t={frame.time:.6g}$, $N={frame.x.size}$"
        )
        return positive, negative, zero, title

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(configurations),
        interval=1000.0 / args.fps,
        blit=False,
    )
    destination = write_animation(movie, output_path, args.fps, args.dpi, args.codec)
    plt.close(fig)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
