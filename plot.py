"""Draw the learning curve and PPO diagnostics of one run, or compare several runs.

Usage:
    python plot.py runs/<run_name>
        -> runs/<run_name>/plots/learning_curve.png and diagnostics.png
    python plot.py runs/<a> runs/<b> --out compare
        -> compare_learning_curve.png and compare_diagnostics.png
"""

import argparse
import sys
from pathlib import Path

from hopper_ppo.config import RunPaths, display_path, load_config
from hopper_ppo.media import plot_diagnostics, plot_learning_curves


def plot_runs(
    runs: dict[str, RunPaths],
    curve_file: Path,
    diagnostics_file: Path,
    title: str | None = None,
) -> None:
    """Draw both charts for the runs, keyed by their legend label."""
    first = next(iter(runs.values()))
    name = title or f"PPO on {load_config(first.config_file).run.env_id}"
    plot_learning_curves(
        {label: run.evals_file for label, run in runs.items()}, curve_file, f"{name}: eval return during training"
    )
    plot_diagnostics(
        {label: run.progress_file for label, run in runs.items()}, diagnostics_file, f"{name}: training diagnostics"
    )
    print(f"[plot] {display_path(curve_file)}")
    print(f"[plot] {display_path(diagnostics_file)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot learning curves and PPO diagnostics.")
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, help="output prefix (required when comparing several runs)")
    parser.add_argument("--title")
    args = parser.parse_args()

    runs = {}
    for run_dir in args.run_dirs:
        run = RunPaths(run_dir.resolve())
        if not run.evals_file.exists() or not run.progress_file.exists():
            sys.exit(f"No logs found in {display_path(run.logs_dir)}")
        runs[run.root.name] = run

    if args.out:
        prefix = args.out.resolve()
        curve = prefix.with_name(f"{prefix.name}_learning_curve.png")
        diagnostics = prefix.with_name(f"{prefix.name}_diagnostics.png")
    elif len(runs) == 1:
        plots_dir = next(iter(runs.values())).plots_dir
        curve, diagnostics = plots_dir / "learning_curve.png", plots_dir / "diagnostics.png"
    else:
        sys.exit("--out is required when comparing several runs")
    plot_runs(runs, curve, diagnostics, args.title)


if __name__ == "__main__":
    main()
