"""Train several config variants (optionally with several seeds) and compare them.

A sweep file lists variants (a tag plus config overrides) and, optionally,
seeds. Each (variant, seed) pair becomes one run under `runs/<sweep name>/`.
After a run finishes, its final model is scored over 10 episodes.
Running the same command again skips finished work and resumes stopped runs,
so a sweep can be interrupted with Ctrl+C and continued later.

Outputs in the sweep folder:
    summary.csv                   one row per run
    compare_learning_curve.png    one line per variant (mean over seeds, min-max band)
    compare_diagnostics.png       PPO metrics, mean over seeds

Usage:
    python sweep.py configs/sweeps/<sweep>.toml
    python sweep.py configs/sweeps/<sweep>.toml --jobs 3   # 3 runs at a time
    python sweep.py configs/sweeps/<sweep>.toml --summary-only
"""

import argparse
import csv
import json
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from hopper_ppo.checkpoint import has_snapshot, load_state
from hopper_ppo.config import PROJECT_ROOT, RUNS_DIR, RunPaths, display_path
from hopper_ppo.media import plot_diagnostics, plot_learning_curves
from hopper_ppo.metrics import COLLAPSE_RETURN, GOOD_RETURN, summarize_evals

SCORE_EPISODES = 10


def run_tag(variant: dict, seed: int | None) -> str:
    return variant["tag"] if seed is None else f"{variant['tag']}_s{seed}"


def find_run(sweep_dir: Path, tag: str) -> RunPaths | None:
    """Return the most recent run folder with this tag, if any."""
    matches = sorted(p for p in sweep_dir.glob(f"*_{tag}") if (p / "state.json").exists())
    return RunPaths(matches[-1]) if matches else None


def score_path(run: RunPaths) -> Path:
    return run.logs_dir / "score_final.json"


def run_one(sweep_dir: Path, base: Path, common: list[str], variant: dict, seed: int | None) -> str:
    """Train (or resume) one run, then score its final model. Returns what happened."""
    tag = run_tag(variant, seed)
    existing = find_run(sweep_dir, tag)
    log_file = sweep_dir / f"{tag}.log"

    def call(command: list[str]) -> bool:
        with open(log_file, "a") as log:
            return subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT).returncode == 0

    if existing and load_state(existing).status == "completed":
        action = "already trained"
    else:
        if existing and has_snapshot(existing):
            command = [sys.executable, "train.py", "--no-progress-bar", "--resume", str(existing.root)]
            action = "resumed"
        else:
            command = [
                sys.executable, "train.py", "--no-progress-bar",
                "--config", str(base), "--runs-dir", str(sweep_dir), "--tag", tag,
            ]  # fmt: skip
            overrides = [*common, *variant.get("set", [])]
            if seed is not None:
                overrides.append(f"run.seed={seed}")
            for item in overrides:
                command += ["--set", item]
            action = "trained"
        if not call(command):
            return f"FAILED (see {display_path(log_file)})"

    run = find_run(sweep_dir, tag)
    if not run or load_state(run).status != "completed":
        return "stopped before finishing"
    if not score_path(run).exists():
        scored = call([
            sys.executable, "evaluate.py", str(run.root),
            "--model", "final", "--episodes", str(SCORE_EPISODES), "--save-score",
        ])  # fmt: skip
        if not scored:
            return f"{action}, scoring FAILED (see {display_path(log_file)})"
        action += ", scored"
    return action


def run_row(variant: dict, seed: int | None, run: RunPaths) -> dict | None:
    summary = summarize_evals(run.evals_file)
    if summary is None:
        return None
    row = {"variant": variant["tag"], "seed": seed, "run": run.root.name, **summary}
    if score_path(run).exists():
        score = json.loads(score_path(run).read_text())
        row["final_mean"] = score["mean_reward"]
        row["final_std"] = score["std_reward"]
        row["final_full"] = score["full_length_episodes"]
    return row


def _fmt(value, spec: str = ".0f") -> str:
    return "-" if value is None else format(value, spec)


def _spread(values: list) -> str:
    """mean (min-max) of the values that exist, plus how many are missing."""
    present = [v for v in values if v is not None]
    if not present:
        return "-"
    text = f"{np.mean(present):.0f}"
    if len(present) > 1:
        text += f" ({min(present):.0f}-{max(present):.0f})"
    if len(present) < len(values):
        text += f" [{len(values) - len(present)} none]"
    return text


def summarize(sweep_dir: Path, variants: list[dict], seeds: list[int | None], name: str) -> None:
    rows, groups = [], {}
    for variant in variants:
        for seed in seeds:
            run = find_run(sweep_dir, run_tag(variant, seed))
            row = run_row(variant, seed, run) if run else None
            if row is None:
                continue
            rows.append(row)
            groups.setdefault(variant["tag"], []).append(run)

    if not rows:
        print("[sweep] no results yet")
        return

    print(
        f"\nPer run. good = first eval >= {GOOD_RETURN:.0f}; "
        f"collapses = later evals < {COLLAPSE_RETURN:.0f}; final = final model over {SCORE_EPISODES} episodes"
    )
    print(f"{'run':18s} {'best':>6s} {'last10':>7s} {'good at':>8s} {'collapses':>9s} {'final':>16s}")
    for row in rows:
        tag = row["variant"] + ("" if row["seed"] is None else f"_s{row['seed']}")
        final = "-" if "final_mean" not in row else f"{row['final_mean']:.0f} +/- {row['final_std']:.0f} ({row['final_full']}/{SCORE_EPISODES})"
        print(
            f"{tag:18s} {row['best']:6.0f} {row['last10']:7.0f} {_fmt(row['first_good']):>8s} "
            f"{_fmt(row['collapses'], 'd'):>9s} {final:>16s}"
        )

    if len(seeds) > 1:
        print("\nPer variant: mean (min-max) over seeds")
        print(f"{'variant':12s} {'best':>16s} {'last10':>16s} {'good at':>22s} {'collapses':>12s} {'final':>16s}")
        for variant in variants:
            mine = [row for row in rows if row["variant"] == variant["tag"]]
            if not mine:
                continue
            print(
                f"{variant['tag']:12s} {_spread([r['best'] for r in mine]):>16s} "
                f"{_spread([r['last10'] for r in mine]):>16s} {_spread([r['first_good'] for r in mine]):>22s} "
                f"{_spread([r['collapses'] for r in mine]):>12s} {_spread([r.get('final_mean') for r in mine]):>16s}"
            )

    fields = ["variant", "seed", "run", "steps", "best", "last10", "first_good", "collapses",
              "final_mean", "final_std", "final_full"]  # fmt: skip
    with open(sweep_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[sweep] {display_path(sweep_dir / 'summary.csv')}")

    if len(groups) > 8:
        print("[plot] more than 8 variants: draw subsets with plot.py instead")
        return
    curve = sweep_dir / "compare_learning_curve.png"
    diagnostics = sweep_dir / "compare_diagnostics.png"
    plot_learning_curves(
        {tag: [run.evals_file for run in runs] for tag, runs in groups.items()},
        curve,
        f"Sweep {name}: eval return during training",
    )
    plot_diagnostics(
        {tag: [run.progress_file for run in runs] for tag, runs in groups.items()},
        diagnostics,
        f"Sweep {name}: training diagnostics",
    )
    print(f"[plot] {display_path(curve)}")
    print(f"[plot] {display_path(diagnostics)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train several config variants and compare them.")
    parser.add_argument("sweep_file", type=Path)
    parser.add_argument("--jobs", type=int, default=1, help="number of runs to train at the same time")
    parser.add_argument("--summary-only", action="store_true", help="only print the tables and redraw the charts")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="parent folder of the sweep folder")
    args = parser.parse_args()

    with open(args.sweep_file, "rb") as f:
        spec = tomllib.load(f)
    sweep, variants = spec["sweep"], spec["variants"]
    seeds = sweep.get("seeds") or [None]  # None: use the seed from the config
    tags = [run_tag(v, s) for v in variants for s in seeds]
    if len(set(tags)) != len(tags):
        sys.exit("Variant tags must be unique")
    if any(a != b and a.endswith(f"_{b}") for a in tags for b in tags):
        sys.exit("A run tag must not end with '_<another tag>'")

    sweep_dir = args.runs_dir.resolve() / sweep["name"]
    sweep_dir.mkdir(parents=True, exist_ok=True)
    base = (PROJECT_ROOT / sweep["base"]).resolve()
    common = sweep.get("common", [])

    if not args.summary_only:
        # Seed-major order: every variant gets its first seed before any gets a second,
        # so a sweep stopped early still covers all variants.
        jobs = [(v, s) for s in seeds for v in variants]
        print(f"[sweep] {sweep['name']}: {len(jobs)} runs, {args.jobs} at a time -> {display_path(sweep_dir)}")
        try:
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futures = {run_tag(v, s): pool.submit(run_one, sweep_dir, base, common, v, s) for v, s in jobs}
                for tag, future in futures.items():
                    print(f"[sweep] {tag}: {future.result()}", flush=True)
        except KeyboardInterrupt:
            # Ctrl+C also reaches the training processes, which save their progress.
            print("\n[sweep] interrupted; run the same command again to continue.")
            return

    summarize(sweep_dir, variants, seeds, sweep["name"])


if __name__ == "__main__":
    main()
