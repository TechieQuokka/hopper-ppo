"""Check that videos keep the robot in view, however far it travels.

An untrained policy falls over in place, so a short training run cannot show
whether the camera follows the robot. This script moves the robot directly
instead: it places the standing robot further and further along the track,
renders a frame through the same builder the videos use, and checks where the
torso appears in the image.

Checks:
    1. With the configured camera, the torso stays near the image center at
       every distance (0 to 100 m; a good Hopper policy covers about 3-4 m
       per 1000-step episode).
    2. The detector itself works: a camera that does NOT follow the robot
       must be reported as out of view.
    3. The floor is drawn far enough for a full episode (Hopper draws it to
       20 m only; a good policy travels about 20 m), and drawing it wider does
       not change the physics: the same actions give the same trajectory.
    4. A marked image is saved so the projection can be checked by eye:
       tests/output/camera_check.png (a red cross should sit on the torso,
       with the floor visible even at 100 m).

Usage:
    python tests/check_camera.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from hopper_ppo.builders import RENDER_FLOOR_HALF_SIZE, _make_gym_env, make_render_env  # noqa: E402
from hopper_ppo.config import DEFAULT_CONFIG, display_path, load_config  # noqa: E402
from hopper_ppo.media import TRACKED_BODY, is_in_view, torso_on_screen  # noqa: E402

OUTPUT = PROJECT_ROOT / "tests" / "output" / "camera_check.png"
DISTANCES = [0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]


def check(condition: bool, message: str) -> None:
    print(f"  [{'ok' if condition else 'FAIL'}] {message}")
    if not condition:
        raise SystemExit(f"check failed: {message}")


def place_robot(env, x: float) -> None:
    """Put the robot at forward position `x` in its initial standing pose."""
    sim = env.unwrapped
    qpos = sim.init_qpos.copy()
    qpos[0] = x  # rootx: forward position along the track
    sim.set_state(qpos, np.zeros_like(sim.init_qvel))  # also recomputes body and camera poses


def draw_cross(frame: np.ndarray, x: float, y: float, size: int = 10) -> np.ndarray:
    frame = frame.copy()
    height, width = frame.shape[:2]
    px, py = int(round(x * width)), int(round(y * height))
    for d in range(-size, size + 1):
        for cx, cy in ((px + d, py), (px, py + d)):
            if 0 <= cx < width and 0 <= cy < height:
                frame[max(cy - 1, 0) : cy + 2, max(cx - 1, 0) : cx + 2] = (230, 30, 30)
    return frame


def main() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    camera = cfg.run.render_camera
    print(f"=== Camera '{camera}' follows the robot ===")
    check(bool(camera), "a named camera is configured (run.render_camera)")

    env = make_render_env(cfg)
    env.reset(seed=0)
    marked = []
    for distance in DISTANCES:
        place_robot(env, distance)
        frame = env.render()
        x, y = torso_on_screen(env, camera)
        check(is_in_view(x, y), f"robot at {distance:5.1f} m appears at ({x:.2f}, {y:.2f}) of the frame")
        if distance in (0.0, 10.0, 100.0):
            marked.append(draw_cross(frame, x, y))

    print("\n=== The detector reports a camera that stays still ===")
    # Freeze the camera where it was for the robot at 0 m, then move only the
    # robot's body. This is what a non-following camera would see.
    place_robot(env, 0.0)
    sim = env.unwrapped
    for distance in (3.0, 10.0):
        place_robot(env, 0.0)
        torso = sim.data.body(TRACKED_BODY)
        torso.xpos[0] += distance
        x, y = torso_on_screen(env, camera)
        check(not is_in_view(x, y), f"robot {distance:.0f} m away from a still camera is out of view (x={x:.2f})")
    env.close()

    print("\n=== The floor is drawn far enough, and the physics is unchanged ===")
    env = make_render_env(cfg)
    half_size = float(env.unwrapped.model.geom("floor").size[0])
    check(half_size >= RENDER_FLOOR_HALF_SIZE, f"floor drawn out to {half_size:.0f} m")
    plain = _make_gym_env(cfg.run.env_id)
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(300, plain.action_space.shape[0])).astype(np.float32)
    trajectories = []
    for sim in (plain, env):
        sim.reset(seed=0)
        positions = []
        for action in actions:
            _, _, terminated, truncated, _ = sim.step(action)
            positions.append(sim.unwrapped.data.qpos.copy())
            if terminated or truncated:
                sim.reset(seed=1)
        trajectories.append(np.array(positions))
    check(np.array_equal(*trajectories), "same actions give an identical trajectory with the wider floor")
    plain.close()
    env.close()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    gap = np.full((marked[0].shape[0], 6, 3), 255, dtype=np.uint8)
    imageio.imwrite(OUTPUT, np.concatenate([marked[0], gap, marked[1], gap, marked[2]], axis=1))
    print(f"\n[image] {display_path(OUTPUT)} (robot at 0, 10, 100 m; the red cross marks the computed torso position)")
    print(f"[done] camera check passed (MuJoCo {mujoco.__version__})")


if __name__ == "__main__":
    main()
