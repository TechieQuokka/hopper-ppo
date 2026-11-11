"""Measure PPO training speed on Hopper-v5 for different hardware settings.

Compared settings:
    - device: cpu vs cuda
    - number of parallel environments (n_envs), with one process each
      (SubprocVecEnv) or all in the main process (DummyVecEnv)
    - PyTorch CPU threads (default vs 1)

The rollout size (n_envs * n_steps = 2048) and every PPO setting stay the
same, so each setting does the same learning work per rollout. Each setting
trains for a fixed number of steps, repeated, and the best repeat is reported
(other programs on the machine can only slow a run down, not speed it up).
A share of the time is spent in the gradient update; the rest is collecting
experience.

Usage:
    python tests/bench_speed.py
    python tests/bench_speed.py --steps 20480 --repeats 2
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from hopper_ppo.builders import add_normalization, make_model, make_train_env  # noqa: E402
from hopper_ppo.config import DEFAULT_CONFIG, load_config  # noqa: E402

ROLLOUT_SIZE = 2048

# (label, device, n_envs, vec_env, torch threads or None for the default)
SETTINGS = [
    ("cpu  dummy   x1", "cpu", 1, "dummy", None),
    ("cpu  dummy   x8", "cpu", 8, "dummy", None),
    ("cpu  subproc x2", "cpu", 2, "subproc", None),
    ("cpu  subproc x4", "cpu", 4, "subproc", None),
    ("cpu  subproc x8", "cpu", 8, "subproc", None),
    ("cpu  subproc x16", "cpu", 16, "subproc", None),
    ("cpu  subproc x8, 1 thread", "cpu", 8, "subproc", 1),
    ("cpu  subproc x4, 1 thread", "cpu", 4, "subproc", 1),
    ("cpu  dummy   x8, 1 thread", "cpu", 8, "dummy", 1),
    ("cpu  dummy   x16, 1 thread", "cpu", 16, "dummy", 1),
    ("cuda dummy   x1", "cuda", 1, "dummy", None),
    ("cuda subproc x8", "cuda", 8, "subproc", None),
]


class PhaseTimer(BaseCallback):
    """Split wall time into experience collection and gradient updates."""

    def __init__(self):
        super().__init__()
        self.collect = 0.0
        self._start = 0.0

    def _on_rollout_start(self) -> None:
        self._start = time.perf_counter()

    def _on_rollout_end(self) -> None:
        self.collect += time.perf_counter() - self._start

    def _on_step(self) -> bool:
        return True


def measure(device: str, n_envs: int, vec_env: str, threads: int | None, steps: int) -> tuple[float, float]:
    """Return (steps per second, share of time spent collecting)."""
    cfg = load_config(
        DEFAULT_CONFIG,
        [
            f'run.device="{device}"',
            f"run.n_envs={n_envs}",
            f'run.vec_env="{vec_env}"',
            f"ppo.n_steps={ROLLOUT_SIZE // n_envs}",
        ],
    )
    default_threads = torch.get_num_threads()
    if threads is not None:
        torch.set_num_threads(threads)
    env = add_normalization(cfg, make_train_env(cfg))
    model = make_model(cfg, env)
    timer = PhaseTimer()
    start = time.perf_counter()
    model.learn(total_timesteps=steps, callback=timer)
    elapsed = time.perf_counter() - start
    env.close()
    torch.set_num_threads(default_threads)
    return model.num_timesteps / elapsed, timer.collect / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure PPO training speed for hardware settings.")
    parser.add_argument("--steps", type=int, default=10 * ROLLOUT_SIZE)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    print(f"PyTorch {torch.__version__}, default CPU threads: {torch.get_num_threads()}, "
          f"CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")  # fmt: skip
    print(f"{args.steps} steps per measurement, best of {args.repeats}\n")
    print(f"{'setting':28s} {'steps/s':>8s} {'all repeats':>22s} {'collecting':>11s} {'1M steps':>9s}")

    for label, device, n_envs, vec_env, threads in SETTINGS:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        results = [measure(device, n_envs, vec_env, threads, args.steps) for _ in range(args.repeats)]
        best_fps, collect_share = max(results)
        repeats = ", ".join(f"{fps:.0f}" for fps, _ in results)
        minutes = 1_000_000 / best_fps / 60
        print(f"{label:28s} {best_fps:8.0f} {repeats:>22s} {collect_share:10.0%} {minutes:7.1f} min", flush=True)


if __name__ == "__main__":
    main()
