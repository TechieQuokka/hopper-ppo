"""Test whether sweep variants really differ from a reference variant.

For each metric, compares each variant with the reference (default: the first
variant in the sweep file) using:
    - the difference of means, with a 95% bootstrap confidence interval
    - a two-sided permutation test: how often a random relabeling of the runs
      gives a difference at least as large as the observed one (exact when the
      number of relabelings is small, otherwise 200,000 random ones)

Each run (one seed) counts as one sample. Episodes within a run are not
independent samples, so they are never pooled across runs.
With very few runs per variant (e.g. 3) the bootstrap interval is too narrow
to trust; read the permutation p-value instead.

Usage:
    python compare.py configs/sweeps/<sweep>.toml
    python compare.py configs/sweeps/<sweep>.toml --reference clip0.1
"""

import argparse
import itertools
import json
import math
import sys
import tomllib
from pathlib import Path

import numpy as np

from hopper_ppo.config import RUNS_DIR
from hopper_ppo.metrics import COLLAPSE_RETURN, GOOD_RETURN, read_rows
from sweep import find_run, run_tag, score_path

EXACT_LIMIT = 200_000
RANDOM_RELABELINGS = 200_000
BOOTSTRAP = 20_000

# (key, description, higher is better, format)
METRICS = [
    ("last10", "mean of the last 10 evals", True, ".0f"),
    ("final", "final model, mean over its scoring episodes", True, ".0f"),
    ("first_good", f"first eval >= {GOOD_RETURN:.0f} (steps; never = 1M+)", False, ",.0f"),
    ("collapse_rate", f"share of evals < {COLLAPSE_RETURN:.0f} after first good", False, ".1%"),
]


def run_metrics(evals_file: Path, score_file: Path) -> dict[str, float]:
    rows = read_rows(evals_file)
    steps = [int(row["timesteps"]) for row in rows]
    rewards = [float(row["mean_reward"]) for row in rows]
    good = next((i for i, reward in enumerate(rewards) if reward >= GOOD_RETURN), None)
    after = rewards[good + 1 :] if good is not None else []
    metrics = {
        "last10": float(np.mean(rewards[-10:])),
        # A run that never got there is counted as one eval past the end, which
        # is conservative: the true value is at least that late.
        "first_good": float(steps[good]) if good is not None else float(steps[-1] + (steps[1] - steps[0])),
        "collapse_rate": float(np.mean([r < COLLAPSE_RETURN for r in after])) if after else 1.0,
    }
    if score_file.exists():
        metrics["final"] = float(json.loads(score_file.read_text())["mean_reward"])
    return metrics


def permutation_p(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[float, str]:
    """Two-sided p-value for the difference of means. Returns (p, 'exact' or 'random')."""
    pooled = np.concatenate([a, b])
    observed = abs(a.mean() - b.mean())
    total = pooled.sum()
    n_a, n = len(a), len(pooled)
    tolerance = 1e-9 * max(1.0, abs(observed))
    if math.comb(n, n_a) <= EXACT_LIMIT:
        sums = np.array([pooled[list(idx)].sum() for idx in itertools.combinations(range(n), n_a)])
        kind = "exact"
    else:
        idx = np.argsort(rng.random((RANDOM_RELABELINGS, n)), axis=1)[:, :n_a]
        sums = pooled[idx].sum(axis=1)
        kind = "random"
    diffs = np.abs(sums / n_a - (total - sums) / (n - n_a))
    count = int((diffs >= observed - tolerance).sum())
    if kind == "random":
        return (count + 1) / (len(diffs) + 1), kind
    return count / len(diffs), kind


def bootstrap_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """95% interval for mean(a) - mean(b), resampling runs within each group."""
    da = a[rng.integers(0, len(a), (BOOTSTRAP, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), (BOOTSTRAP, len(b)))].mean(axis=1)
    low, high = np.percentile(da - db, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether sweep variants differ from a reference variant.")
    parser.add_argument("sweep_file", type=Path)
    parser.add_argument("--reference", help="variant tag to compare against (default: the first one)")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--seed", type=int, default=0, help="seed for the random parts of the tests")
    args = parser.parse_args()

    with open(args.sweep_file, "rb") as f:
        spec = tomllib.load(f)
    sweep, variants = spec["sweep"], spec["variants"]
    seeds = sweep.get("seeds") or [None]
    sweep_dir = args.runs_dir.resolve() / sweep["name"]
    reference = args.reference or variants[0]["tag"]

    data: dict[str, list[dict[str, float]]] = {}
    for variant in variants:
        for seed in seeds:
            run = find_run(sweep_dir, run_tag(variant, seed))
            if run is None or not run.evals_file.exists():
                continue
            data.setdefault(variant["tag"], []).append(run_metrics(run.evals_file, score_path(run)))
    if reference not in data:
        sys.exit(f"No finished runs for the reference variant {reference!r}")

    rng = np.random.default_rng(args.seed)
    print(f"Sweep {sweep['name']}: each run is one sample; reference = {reference}")
    print("Counts: " + ", ".join(f"{tag} {len(runs)}" for tag, runs in data.items()))
    for key, description, higher_better, spec_ in METRICS:
        print(f"\n{key}: {description} ({'higher' if higher_better else 'lower'} is better)")
        print(f"  {'variant':16s} {'mean':>10s} {'median':>10s} {'min-max':>22s} "
              f"{'diff vs ref':>12s} {'95% CI of diff':>24s} {'p':>7s}")  # fmt: skip
        ref_values = np.array([run[key] for run in data[reference] if key in run])
        for tag, runs in data.items():
            values = np.array([run[key] for run in runs if key in run])
            if len(values) == 0:
                continue
            spread = f"{format(values.min(), spec_)}-{format(values.max(), spec_)}"
            line = f"  {tag:16s} {format(values.mean(), spec_):>10s} {format(float(np.median(values)), spec_):>10s} {spread:>22s}"
            if tag != reference and len(values) > 1 and len(ref_values) > 1:
                p, kind = permutation_p(values, ref_values, rng)
                low, high = bootstrap_ci(values, ref_values, rng)
                diff = values.mean() - ref_values.mean()
                ci = f"[{format(low, spec_)}, {format(high, spec_)}]"
                line += f" {format(diff, '+' + spec_):>12s} {ci:>24s} {p:7.4f}"
                if kind == "random":
                    line += " (random)"
            print(line)


if __name__ == "__main__":
    main()
