"""Quick end-to-end check of the whole pipeline (about a minute).

Runs the real commands on a short budget: train -> evaluate with a video,
contact sheet and camera check -> plots. The agent will NOT learn to hop in
this few steps; the goal is only to confirm that everything works together.

Usage:
    python tests/quick_check.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = PROJECT_ROOT / "tests" / "output" / "quick_check"
OVERRIDES = [
    "run.total_timesteps=20000",
    "eval.freq=5000",
    "eval.episodes=3",
    "save.snapshot_freq=5000",
    "save.checkpoint_freq=10000",
]


def run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run([sys.executable, *command], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    train = ["train.py", "--no-progress-bar", "--runs-dir", str(WORK_DIR.relative_to(PROJECT_ROOT)), "--tag", "quick"]
    for item in OVERRIDES:
        train += ["--set", item]
    run(train)

    run_dir = next(WORK_DIR.glob("*_quick"))
    relative = str(run_dir.relative_to(PROJECT_ROOT))
    run(["evaluate.py", relative, "--episodes", "3", "--video", "--video-stages"])
    run(["plot.py", relative])

    expected = [
        "models/best_model.zip",
        "models/best_vecnormalize.pkl",
        "models/final_model.zip",
        "models/final_vecnormalize.pkl",
        "videos/best.mp4",
        "videos/best_sheet.png",
        "plots/learning_curve.png",
        "plots/diagnostics.png",
    ]
    missing = [name for name in expected if not (run_dir / name).exists()]
    if missing:
        raise SystemExit(f"missing outputs: {missing}")
    print(f"\n[done] quick check passed: {relative}")


if __name__ == "__main__":
    main()
