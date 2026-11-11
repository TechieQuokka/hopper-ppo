"""Train several config variants and compare them.

A sweep file lists variants (a tag plus config overrides) and, optionally,
seeds. Each (variant, seed) pair becomes one run under `runs/<sweep name>/`.
Running the same command again skips finished runs and resumes stopped ones,
so a sweep can be interrupted with Ctrl+C and continued later.

Usage:
    python sweep.py configs/sweeps/<sweep>.toml
    python sweep.py configs/sweeps/<sweep>.toml --jobs 2   # 2 runs at a time
    python sweep.py configs/sweeps/<sweep>.toml --summary-only

Note: each run already uses `run.n_envs` worker processes, so running several
at once only helps while there are idle CPU cores.
"""

import argparse
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from hopper_ppo.checkpoint import has_snapshot, load_state
from hopper_ppo.config import PROJECT_ROOT, RUNS_DIR, RunPaths, display_path
from hopper_ppo.metrics import read_rows
from plot import plot_runs


def run_tag(variant: dict, seed: int | None) -> str:
    return variant["tag"] if seed is None else f"{variant['tag']}_s{seed}"


def find_run(sweep_dir: Path, tag: str) -> RunPaths | None:
    """Return the most recent run folder with this tag, if any."""
    matches = sorted(p for p in sweep_dir.glob(f"*_{tag}") if (p / "state.json").exists())
    return RunPaths(matches[-1]) if matches else None


def run_one(sweep_dir: Path, base: Path, common: list[str], variant: dict, seed: int | None) -> str:
    """Start, resume, or skip one run. Returns what happened."""
    tag = run_tag(variant, seed)
    existing = find_run(sweep_dir, tag)

    if existing and load_state(existing).status == "completed":
        return "skipped (already completed)"

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

    log_file = sweep_dir / f"{tag}.log"
    with open(log_file, "a") as log:
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        return f"FAILED (see {display_path(log_file)})"

    run = find_run(sweep_dir, tag)
    return action if run and load_state(run).status == "completed" else "stopped before finishing"


def eval_summary(run: RunPaths) -> dict[str, float] | None:
    rows = read_rows(run.evals_file)
    if not rows:
        return None
    rewards = [float(row["mean_reward"]) for row in rows]
    return {
        "steps": int(rows[-1]["timesteps"]),
        "best": max(rewards),
        "last": rewards[-1],
        "last3": float(np.mean(rewards[-3:])),
    }


def summarize(sweep_dir: Path, variants: list[dict], seeds: list[int | None], name: str) -> None:
    print(f"\n{'run':22s} {'steps':>8s} {'best eval':>10s} {'last eval':>10s} {'mean last 3':>12s}")
    finished = {}
    for variant in variants:
        per_seed = []
        for seed in seeds:
            tag = run_tag(variant, seed)
            run = find_run(sweep_dir, tag)
            summary = eval_summary(run) if run else None
            if summary is None:
                print(f"{tag:22s} (no results)")
                continue
            per_seed.append(summary)
            finished[tag] = run
            print(
                f"{tag:22s} {summary['steps']:>8d} {summary['best']:10.0f} "
                f"{summary['last']:10.0f} {summary['last3']:12.0f}"
            )
        if len(per_seed) > 1:
            last3 = [s["last3"] for s in per_seed]
            print(f"{'  ' + variant['tag'] + ' (seeds)':22s} {'':>8s} {'':>10s} {'':>10s} "
                  f"{np.mean(last3):7.0f} +/- {np.std(last3):.0f}")  # fmt: skip

    if not finished:
        return
    if len(finished) > 8:
        print("\n[plot] more than 8 runs: draw subsets with plot.py instead")
        return
    plot_runs(finished, sweep_dir / "compare_learning_curve.png", sweep_dir / "compare_diagnostics.png", f"Sweep {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train several config variants and compare them.")
    parser.add_argument("sweep_file", type=Path)
    parser.add_argument("--jobs", type=int, default=1, help="number of runs to train at the same time")
    parser.add_argument("--summary-only", action="store_true", help="only print the table and redraw the charts")
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
        jobs = [(v, s) for v in variants for s in seeds]
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
