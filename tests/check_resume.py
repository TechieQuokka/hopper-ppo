"""Check that a stopped training run can be resumed without gaps.

Two scenarios, each run through the real `train.py` command line. Signals go
to the whole process group, like a Ctrl+C in a terminal, so the parallel
environment workers receive them too:
    1. Ctrl+C (SIGINT): progress is saved at the moment of the interrupt.
    2. Hard kill (SIGKILL): the run restarts from the latest snapshot.

After resuming, the run must reach the target step count with continuous logs,
and the normalization statistics must have kept counting from where they were.

Usage:
    python tests/check_resume.py
"""

import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hopper_ppo.checkpoint import model_files  # noqa: E402
from hopper_ppo.config import RunPaths, load_config  # noqa: E402
from hopper_ppo.metrics import read_rows  # noqa: E402

WORK_DIR = PROJECT_ROOT / "tests" / "output" / "resume_check"
TOTAL_STEPS = 40_000
EVAL_FREQ = 2_000
SNAPSHOT_FREQ = 4_000
CHECKPOINT_FREQ = 10_000
OVERRIDES = [
    f"run.total_timesteps={TOTAL_STEPS}",
    f"eval.freq={EVAL_FREQ}",
    "eval.episodes=2",
    f"save.snapshot_freq={SNAPSHOT_FREQ}",
    f"save.checkpoint_freq={CHECKPOINT_FREQ}",
]


def train_command(extra: list[str]) -> list[str]:
    return [sys.executable, "train.py", "--no-progress-bar", *extra]


def start_new_run(tag: str, log_file: Path) -> tuple[subprocess.Popen, RunPaths]:
    runs_dir = WORK_DIR / tag
    args = ["--runs-dir", str(runs_dir), "--tag", tag]
    for item in OVERRIDES:
        args += ["--set", item]
    process = subprocess.Popen(
        train_command(args),
        cwd=PROJECT_ROOT,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group, so signals reach the workers too
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        found = list(runs_dir.glob("*/state.json"))
        if found:
            return process, RunPaths(found[0].parent)
        time.sleep(0.1)
    os.killpg(process.pid, signal.SIGKILL)
    raise RuntimeError("run folder was not created in time")


def read_state(paths: RunPaths) -> dict:
    return json.loads(paths.state_file.read_text())


def eval_steps(paths: RunPaths) -> list[int]:
    return [int(row["timesteps"]) for row in read_rows(paths.evals_file)]


def wait_until(condition, process: subprocess.Popen, timeout: float = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("training finished before it could be stopped")
        if condition():
            return
        time.sleep(0.02)
    os.killpg(process.pid, signal.SIGKILL)
    raise RuntimeError("timed out waiting for training progress")


def resume(paths: RunPaths, log_file: Path) -> None:
    subprocess.run(
        train_command(["--resume", str(paths.root)]),
        cwd=PROJECT_ROOT,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        check=True,
    )


def check(condition: bool, message: str) -> None:
    print(f"  [{'ok' if condition else 'FAIL'}] {message}")
    if not condition:
        raise SystemExit(f"check failed: {message}")


def stats_count(stats_file: Path) -> float:
    """How many observations the running mean/variance has seen."""
    with open(stats_file, "rb") as f:
        return float(pickle.load(f).obs_rms.count)


def verify_completed(paths: RunPaths) -> None:
    n_envs = load_config(paths.config_file).run.n_envs
    state = read_state(paths)
    steps = state["num_timesteps"]
    check(state["status"] == "completed", f"status is completed (got {state['status']})")
    check(steps >= TOTAL_STEPS, f"reached {TOTAL_STEPS} steps (got {steps})")

    # Each eval interval appears exactly once, in order.
    evals = eval_steps(paths)
    intervals = [s // EVAL_FREQ for s in evals]
    expected = list(range(1, TOTAL_STEPS // EVAL_FREQ + 1))
    check(intervals == expected, f"eval log is continuous, no gaps or duplicates ({len(evals)} rows)")

    progress = [int(row["timesteps"]) for row in read_rows(paths.progress_file)]
    check(progress == sorted(set(progress)), f"progress log steps strictly increase ({len(progress)} rows)")

    checkpoints = sorted(p.stem for p in paths.checkpoints_dir.glob("step_*.zip"))
    check(
        [int(name[5:]) // CHECKPOINT_FREQ for name in checkpoints] == list(range(1, TOTAL_STEPS // CHECKPOINT_FREQ + 1)),
        f"one checkpoint per interval: {checkpoints}",
    )
    check(all(model_files(paths, name).exists() for name in checkpoints), "every checkpoint has its normalization stats")
    for name in ("best", "final", "resume"):
        check(model_files(paths, name).exists(), f"{name} model and its normalization stats exist")

    monitors = sorted(p.name for p in paths.monitor_dir.glob("*.monitor.csv"))
    check(len(monitors) == n_envs, f"one training monitor file per environment ({len(monitors)})")

    # The stats see n_envs observations per step of each environment, plus one
    # batch per reset at the start of each training session.
    count = stats_count(model_files(paths, "final").stats)
    check(
        abs(count - steps) <= n_envs * 3,
        f"normalization stats kept counting across the resume ({count:.0f} observations for {steps} steps)",
    )


def scenario_sigint() -> None:
    print("\n=== Scenario 1: Ctrl+C (SIGINT to the process group), then resume ===")
    process, paths = start_new_run("sigint", WORK_DIR / "sigint_first.log")
    wait_until(lambda: read_state(paths)["num_timesteps"] >= 16_000, process)
    snapshot_before = read_state(paths)["num_timesteps"]
    time.sleep(0.3)  # stop between snapshots, not exactly on one
    os.killpg(process.pid, signal.SIGINT)
    returncode = process.wait(timeout=120)

    state = read_state(paths)
    stopped_at = state["num_timesteps"]
    check(returncode == 0, f"training exited cleanly after Ctrl+C (exit code {returncode})")
    check(state["status"] == "interrupted", f"status is interrupted (got {state['status']})")
    check(snapshot_before < stopped_at < TOTAL_STEPS, f"progress saved at the interrupt: step {stopped_at}")

    resume(paths, WORK_DIR / "sigint_resume.log")
    verify_completed(paths)


def scenario_kill() -> None:
    print("\n=== Scenario 2: hard kill (SIGKILL to the process group), then resume ===")
    process, paths = start_new_run("kill", WORK_DIR / "kill_first.log")

    # Kill once the eval log and a checkpoint are ahead of the last snapshot,
    # so that resuming has to drop what it is about to produce again.
    def log_ahead_of_snapshot() -> bool:
        steps = eval_steps(paths)
        snapshot_step = read_state(paths)["num_timesteps"]
        checkpoint_ahead = any(
            int(p.stem.split("_")[1]) > snapshot_step for p in paths.checkpoints_dir.glob("step_*_vecnormalize.pkl")
        )
        return snapshot_step >= 8_000 and bool(steps) and steps[-1] > snapshot_step and checkpoint_ahead

    wait_until(log_ahead_of_snapshot, process)
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=60)

    state = read_state(paths)
    snapshot_step = state["num_timesteps"]
    check(state["status"] == "running", "state still says running (no chance to save)")
    check(eval_steps(paths)[-1] > snapshot_step, f"eval log is ahead of snapshot step {snapshot_step}")

    resume(paths, WORK_DIR / "kill_resume.log")
    log = (WORK_DIR / "kill_resume.log").read_text()
    check("1 newer checkpoints" in log, "the checkpoint written after the snapshot was dropped on resume")
    verify_completed(paths)


def main() -> None:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    start = time.perf_counter()
    scenario_sigint()
    scenario_kill()
    print(f"\n[done] resume check passed in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
