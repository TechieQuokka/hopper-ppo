"""Score a saved model, and optionally record videos.

Usage:
    python evaluate.py runs/<run_name>                      # best model, 10 episodes
    python evaluate.py runs/<run_name> --model final
    python evaluate.py runs/<run_name> --model step_0100000
    python evaluate.py runs/<run_name> --video              # also save videos/<model>.mp4
    python evaluate.py runs/<run_name> --video --gif        # plus a small GIF for documents
    python evaluate.py runs/<run_name> --video-stages       # one video per checkpoint + best
    python evaluate.py runs/<run_name> --model final --save-score   # -> logs/score_final.json

Every video comes with a contact sheet (videos/<model>_sheet.png: frames at
0/25/50/75/100%) and a camera check that warns if the robot leaves the view.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Render off-screen on the GPU. Must be set before MuJoCo is imported.
# (The default windowed backend logs OpenGL warnings on this machine.)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.evaluation import evaluate_policy  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402

from hopper_ppo.builders import EVAL_SEED_OFFSET, make_eval_env, make_render_env  # noqa: E402
from hopper_ppo.checkpoint import list_checkpoints, model_files  # noqa: E402
from hopper_ppo.config import Config, RunPaths, display_path, load_config  # noqa: E402
from hopper_ppo.media import record_episodes, save_contact_sheet, save_gif, save_video  # noqa: E402


def load(cfg: Config, paths: RunPaths, name: str, device: str) -> tuple[PPO, VecNormalize]:
    """Load a model and an evaluation environment that uses its saved statistics."""
    files = model_files(paths, name)
    if not files.exists():
        sys.exit(f"Model or its normalization stats not found: {display_path(files.model)}, {display_path(files.stats)}")
    model = PPO.load(files.model, device=device)
    return model, make_eval_env(cfg, stats_file=files.stats)


def score(model: PPO, env: VecNormalize, episodes: int) -> dict[str, float]:
    rewards, lengths = evaluate_policy(
        model, env, n_eval_episodes=episodes, deterministic=True, return_episode_rewards=True
    )
    result = {
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "full_length_episodes": int(sum(length >= 1000 for length in lengths)),
    }
    print(
        f"[eval] {episodes} episodes: return {result['mean_reward']:.1f} +/- {result['std_reward']:.1f} "
        f"(min {result['min_reward']:.1f}, max {result['max_reward']:.1f}), mean length {result['mean_length']:.0f}"
    )
    return result


def score_file(paths: RunPaths, name: str) -> Path:
    return paths.logs_dir / f"score_{name}.json"


def video(cfg: Config, model: PPO, stats: VecNormalize, out_dir: Path, name: str, episodes: int, gif: bool) -> bool:
    """Record, save the video and its contact sheet, and report the camera check."""
    env = make_render_env(cfg)
    fps = env.metadata.get("render_fps", 30)
    recording = record_episodes(
        model, stats, env, cfg.run.render_camera, episodes=episodes, seed=cfg.run.seed + EVAL_SEED_OFFSET
    )
    env.close()

    path = out_dir / f"{name}.mp4"
    save_video(recording.frames, path, fps)
    save_contact_sheet(recording.frames, out_dir / f"{name}_sheet.png")
    if gif:
        save_gif(recording.frames, out_dir / f"{name}.gif", fps)
    shown = ", ".join(f"{r:.0f}" for r in recording.returns)
    print(f"[video] {display_path(path)} (episode returns: {shown})")

    if recording.camera is None:
        print("[camera] WARNING: no named camera set (run.render_camera); the view was not checked")
        return False
    print(f"[camera] {recording.camera.summary()}")
    return recording.camera.ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a saved model, optionally record videos.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default="best", help="best, final, or a checkpoint name like step_0100000")
    parser.add_argument("--episodes", type=int, default=10, help="episodes used for scoring")
    parser.add_argument("--video", action="store_true", help="record the chosen model")
    parser.add_argument("--gif", action="store_true", help="also save a small GIF of each recording")
    parser.add_argument("--video-stages", action="store_true", help="record every checkpoint and the best model")
    parser.add_argument("--video-episodes", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-score", action="store_true", help="write logs/score_<model>.json")
    args = parser.parse_args()

    paths = RunPaths(args.run_dir.resolve())
    if not paths.config_file.exists():
        sys.exit(f"Not a run folder (no config.toml): {display_path(paths.root)}")
    cfg = load_config(paths.config_file)

    model, env = load(cfg, paths, args.model, args.device)
    print(f"[load] {display_path(model_files(paths, args.model).model)}")
    result = score(model, env, args.episodes)
    if args.save_score:
        score_file(paths, args.model).write_text(json.dumps(result, indent=2) + "\n")

    camera_ok = True
    if args.video:
        camera_ok &= video(cfg, model, env, paths.videos_dir, args.model, args.video_episodes, args.gif)
    env.close()

    if args.video_stages:
        for name in [*list_checkpoints(paths), "best"]:
            if args.video and name == args.model:
                continue  # already recorded above
            stage_model, stage_env = load(cfg, paths, name, args.device)
            camera_ok &= video(cfg, stage_model, stage_env, paths.videos_dir, name, args.video_episodes, args.gif)
            stage_env.close()

    if not camera_ok:
        sys.exit("[camera] at least one video failed the camera check")


if __name__ == "__main__":
    main()
