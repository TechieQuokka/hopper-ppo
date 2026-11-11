"""CSV logs of a run: evaluation results and PPO training metrics.

Both files grow across resumes. When a run resumes from a snapshot, rows
recorded after that snapshot are dropped first, because those steps will be
trained again and would otherwise appear twice.
"""

import csv
from pathlib import Path
from typing import Any

from stable_baselines3.common.logger import KVWriter

EVAL_FIELDS = ["timesteps", "mean_reward", "std_reward", "min_reward", "max_reward", "mean_length"]

# SB3 logger keys kept in progress.csv. One row is written per PPO iteration.
PROGRESS_KEYS = [
    "rollout/ep_rew_mean",  # mean raw return of recent training episodes
    "rollout/ep_len_mean",
    "train/approx_kl",  # how far one update moved the policy
    "train/clip_fraction",  # share of samples whose ratio was clipped
    "train/clip_range",
    "train/entropy_loss",  # minus the policy entropy
    "train/explained_variance",  # how well the value net predicts returns (1 = perfect)
    "train/learning_rate",
    "train/loss",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/std",  # mean action standard deviation (exploration noise)
    "train/n_updates",
    "time/fps",
]
PROGRESS_FIELDS = ["timesteps", *PROGRESS_KEYS]


def _append_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def trim_rows(path: Path, max_timesteps: int) -> int:
    """Drop rows recorded after `max_timesteps`. Returns how many were dropped."""
    if not path.exists():
        return 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    kept = [row for row in rows if int(row["timesteps"]) <= max_timesteps]
    if len(kept) == len(rows):
        return 0
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    tmp.replace(path)
    return len(rows) - len(kept)


def append_eval_row(path: Path, row: dict[str, Any]) -> None:
    _append_row(path, EVAL_FIELDS, row)


class ProgressCsv(KVWriter):
    """SB3 logger output that appends selected metrics to a CSV file.

    SB3's own CSV writer starts a new file on every run, which would erase the
    history of a resumed run, and its columns change as new keys appear. This
    one appends, and always writes the same columns.
    """

    def __init__(self, path: Path):
        self.path = path

    def write(self, key_values: dict[str, Any], key_excluded: dict[str, tuple[str, ...]], step: int = 0) -> None:
        row = {"timesteps": step}
        for key in PROGRESS_KEYS:
            value = key_values.get(key)
            if value is not None:
                row[key] = value
        _append_row(self.path, PROGRESS_FIELDS, row)

    def close(self) -> None:
        pass


def column(rows: list[dict[str, str]], key: str) -> tuple[list[int], list[float]]:
    """Return (timesteps, values) for the rows where `key` has a value."""
    steps, values = [], []
    for row in rows:
        raw = row.get(key, "")
        if raw not in ("", None):
            steps.append(int(row["timesteps"]))
            values.append(float(raw))
    return steps, values
