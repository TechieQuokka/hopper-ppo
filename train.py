"""Train PPO on Hopper-v5, or continue a run that was stopped.

Usage:
    python train.py
    python train.py --set ppo.n_epochs=5 --tag epochs5
    python train.py --resume runs/<run_name>

Press Ctrl+C to stop: progress is saved and the run can be resumed later.
"""

import argparse
import sys
import time
from pathlib import Path

from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat

from hopper_ppo.builders import add_normalization, make_eval_env, make_model, make_train_env
from hopper_ppo.callbacks import RunCallback
from hopper_ppo.checkpoint import (
    RunState,
    drop_checkpoints_after,
    has_snapshot,
    load_snapshot,
    load_state,
    model_files,
    save_model,
    save_snapshot,
    save_state,
)
from hopper_ppo.config import (
    DEFAULT_CONFIG,
    RUNS_DIR,
    RunPaths,
    display_path,
    load_config,
    new_run_paths,
    save_config,
)
from hopper_ppo.metrics import ProgressCsv, trim_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Hopper-v5.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", type=Path, metavar="RUN_DIR", help="continue a stopped run")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one config value (can be repeated)",
    )
    parser.add_argument("--tag", help="short label appended to the run folder name")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="where new runs are created")
    parser.add_argument("--no-progress-bar", action="store_true", help="for logs written to a file")
    return parser.parse_args()


def make_logger(paths: RunPaths) -> Logger:
    """TensorBoard plus an appending CSV of PPO metrics, both kept across resumes."""
    return Logger(
        folder=None,
        output_formats=[TensorBoardOutputFormat(str(paths.tensorboard_dir)), ProgressCsv(paths.progress_file)],
    )


def main() -> None:
    args = parse_args()
    resuming = args.resume is not None

    if resuming:
        paths = RunPaths(args.resume.resolve())
        if not paths.state_file.exists():
            sys.exit(f"Not a run folder (no state.json): {display_path(paths.root)}")
        # The run's own config is used; --set can still adjust it,
        # e.g. run.total_timesteps to train for longer.
        cfg = load_config(paths.config_file, args.overrides)
        state = load_state(paths)
        if state.status == "completed" and state.num_timesteps >= cfg.run.total_timesteps:
            print(f"[run] already completed at {state.num_timesteps} steps: {display_path(paths.root)}")
            return
        if not has_snapshot(paths):
            sys.exit(f"No snapshot to resume from in {display_path(paths.resume_dir)}; start a new run instead.")
    else:
        cfg = load_config(args.config, args.overrides)
        paths = new_run_paths(cfg, args.tag, args.runs_dir)
        paths.create()
        state = RunState()
        save_state(paths, state)
    save_config(cfg, paths.config_file)
    print(f"[run] {display_path(paths.root)}")

    venv = make_train_env(cfg, monitor_dir=paths.monitor_dir, append=resuming)
    eval_env = make_eval_env(cfg)

    if resuming:
        env = add_normalization(cfg, venv, model_files(paths, "resume").stats)
        model = load_snapshot(cfg, env, paths)
        dropped = trim_rows(paths.evals_file, model.num_timesteps)
        trim_rows(paths.progress_file, model.num_timesteps)
        removed = drop_checkpoints_after(paths, model.num_timesteps)
        print(
            f"[resume] continuing from step {model.num_timesteps} "
            f"(dropped {dropped} newer eval rows, {len(removed)} newer checkpoints)"
        )
    else:
        env = add_normalization(cfg, venv)
        model = make_model(cfg, env)
    model.set_logger(make_logger(paths))

    start_step = model.num_timesteps
    remaining = cfg.run.total_timesteps - start_step
    print(
        f"[run] device={model.device} n_envs={cfg.run.n_envs} rollout={cfg.rollout_size} "
        f"steps {start_step} -> {cfg.run.total_timesteps}"
    )

    start = time.perf_counter()
    try:
        model.learn(
            total_timesteps=remaining,
            callback=RunCallback(cfg, paths, eval_env, state),
            reset_num_timesteps=not resuming,
            progress_bar=not args.no_progress_bar,
        )
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
    elapsed = time.perf_counter() - start

    save_snapshot(model, paths, state, status)
    done = model.num_timesteps - start_step
    print(f"[train] {done} steps in {elapsed:.1f}s ({done / max(elapsed, 1e-9):.0f} steps/s)")

    if status == "completed":
        save_model(model, env, model_files(paths, "final"))
        print(f"[save] {display_path(model_files(paths, 'final').model)}")
        print(f"[done] best eval mean reward: {state.best_mean_reward:.1f} at step {state.best_timesteps}")
    else:
        print(f"[stop] interrupted at step {model.num_timesteps}; progress saved.")
        print(f"[stop] continue with: python train.py --resume {display_path(paths.root)}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
