"""Outputs for people: videos with a camera check, contact sheets, and charts."""

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import imageio
import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402

from hopper_ppo.metrics import column, read_rows  # noqa: E402

# Chart colors: fixed categorical order, neutral inks for all text.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# The robot counts as "in view" while its torso stays in the middle part of the
# frame: within this fraction of the frame size from the center, on both axes.
IN_VIEW_MARGIN = 0.35
TRACKED_BODY = "torso"
CONTACT_SHEET_POINTS = (0.0, 0.25, 0.5, 0.75, 1.0)


# --------------------------------------------------------------------------- camera check


def torso_on_screen(env: gym.Env, camera_name: str) -> tuple[float, float]:
    """Where the torso appears in the camera image, as (x, y) in [0, 1].

    (0.5, 0.5) is the image center; values outside [0, 1] are off screen.
    Computed from the camera pose MuJoCo used for the last rendered frame.
    """
    sim = env.unwrapped
    model, data = sim.model, sim.data
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Camera {camera_name!r} does not exist in {sim.spec.id}")

    # Torso position in the camera frame. MuJoCo cameras look along -z, with +y up.
    rotation = data.cam_xmat[cam_id].reshape(3, 3)
    local = rotation.T @ (data.body(TRACKED_BODY).xpos - data.cam_xpos[cam_id])
    depth = -local[2]
    if depth <= 0:
        return float("nan"), float("nan")  # behind the camera
    focal = 0.5 / np.tan(np.deg2rad(model.cam_fovy[cam_id]) / 2)  # in image heights
    aspect = sim.width / sim.height
    x = 0.5 + focal * local[0] / depth / aspect
    y = 0.5 - focal * local[1] / depth
    return float(x), float(y)


def is_in_view(x: float, y: float, margin: float = IN_VIEW_MARGIN) -> bool:
    return abs(x - 0.5) <= margin and abs(y - 0.5) <= margin


@dataclass
class CameraReport:
    """How well the camera kept the robot in view during a recording."""

    camera: str
    frames: int = 0
    out_of_view: int = 0
    worst_offset: float = 0.0  # largest distance of the torso from the center, in frame sizes
    positions: list[tuple[float, float]] = field(default_factory=list)

    def add(self, x: float, y: float) -> None:
        self.frames += 1
        self.positions.append((x, y))
        if not is_in_view(x, y):
            self.out_of_view += 1
        offset = max(abs(x - 0.5), abs(y - 0.5))
        self.worst_offset = max(self.worst_offset, offset if np.isfinite(offset) else float("inf"))

    @property
    def ok(self) -> bool:
        return self.frames > 0 and self.out_of_view == 0

    def summary(self) -> str:
        status = "ok" if self.ok else "WARNING: robot left the view"
        return (
            f"camera '{self.camera}': {status} "
            f"({self.out_of_view}/{self.frames} frames out of view, "
            f"worst offset {self.worst_offset:.2f} of frame from center)"
        )


# --------------------------------------------------------------------------- videos


@dataclass
class Recording:
    returns: list[float]
    frames: list[np.ndarray]
    camera: CameraReport | None


def record_episodes(
    model: PPO,
    stats: VecNormalize,
    env: gym.Env,
    camera_name: str,
    episodes: int = 1,
    seed: int = 0,
) -> Recording:
    """Run the agent in a rendering environment (see builders.make_render_env).

    Observations are normalized with the model's saved statistics. Every frame
    is checked for the robot staying in view, unless no named camera is used.
    """
    frames, returns = [], []
    report = CameraReport(camera_name) if camera_name else None
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_return, done = 0.0, False
        while not done:
            action, _ = model.predict(stats.normalize_obs(obs), deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            if report is not None:
                report.add(*torso_on_screen(env, camera_name))
            episode_return += float(reward)
            done = terminated or truncated
        returns.append(episode_return)
    return Recording(returns, frames, report)


def save_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def save_gif(frames: list[np.ndarray], path: Path, fps: int, every: int = 2, scale: int = 2) -> None:
    """Save a smaller GIF: every `every`-th frame, each side divided by `scale`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    small = [frame[::scale, ::scale] for frame in frames[::every]]
    imageio.mimsave(path, small, duration=1000 * every / fps, loop=0)


def save_contact_sheet(frames: list[np.ndarray], path: Path, gap: int = 6) -> None:
    """Save five frames (start, 25%, 50%, 75%, end) side by side in one image.

    A quick way to see whether the camera followed the robot without watching
    the whole video.
    """
    last = len(frames) - 1
    picks = [frames[round(point * last)] for point in CONTACT_SHEET_POINTS]
    height = picks[0].shape[0]
    spacer = np.full((height, gap, 3), 255, dtype=np.uint8)
    row = []
    for i, frame in enumerate(picks):
        if i:
            row.append(spacer)
        row.append(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.concatenate(row, axis=1))


# --------------------------------------------------------------------------- charts


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:,.0f}k")
    ax.margins(x=0.02)
    ax.set_xlim(left=0)


def _check_series_count(count: int) -> None:
    if count > len(SERIES_COLORS):
        raise ValueError(f"At most {len(SERIES_COLORS)} runs can share one chart, got {count}")


def _aligned(series: list[tuple[list[int], list[float]]]) -> tuple[list[int], np.ndarray]:
    """Keep the steps every series has, and stack the values: shape (series, steps)."""
    common = sorted(set.intersection(*(set(steps) for steps, _ in series)))
    stacked = [[dict(zip(steps, values))[step] for step in common] for steps, values in series]
    return common, np.array(stacked, dtype=float)


def _label_line_ends(ax: plt.Axes, ends: list[tuple[int, float]]) -> None:
    """Write each line's last value next to its end, nudged apart so labels never overlap.

    The value is in ink, not in the series color.
    """
    low, high = ax.get_ylim()
    gap = 0.045 * (high - low)
    placed: list[float] = []
    for _, value in sorted(ends, key=lambda end: end[1]):
        placed.append(max(value, placed[-1] + gap) if placed else value)
    for (step, value), y in zip(sorted(ends, key=lambda end: end[1]), placed):
        ax.annotate(
            f"{value:,.0f}",
            (step, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
            annotation_clip=False,
        )


def plot_learning_curves(groups: dict[str, list[Path]], path: Path, title: str) -> None:
    """Draw eval mean return against env steps, one line per group.

    `groups` maps a label to the `evals.csv` files of its runs (usually one
    per seed). A group of one run shows +/- 1 std over its eval episodes; a
    group of several runs shows the mean over runs and a min-max band.
    Up to 8 groups, one fixed color each.
    """
    _check_series_count(len(groups))
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    band_note = "band: +/- 1 std over eval episodes"
    ends: list[tuple[int, float]] = []

    for (label, files), color in zip(groups.items(), SERIES_COLORS):
        if len(files) == 1:
            rows = read_rows(files[0])
            steps = [int(row["timesteps"]) for row in rows]
            mean = np.array([float(row["mean_reward"]) for row in rows])
            std = np.array([float(row["std_reward"]) for row in rows])
            low, high = mean - std, mean + std
        else:
            steps, values = _aligned([column(read_rows(f), "mean_reward") for f in files])
            mean, low, high = values.mean(axis=0), values.min(axis=0), values.max(axis=0)
            band_note = "line: mean over seeds; band: min-max over seeds"
        ax.fill_between(steps, low, high, color=color, alpha=0.15, linewidth=0)
        ax.plot(steps, mean, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=label)
        if len(steps):
            ends.append((steps[-1], float(mean[-1])))

    _style_axes(ax)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=20)
    ax.text(0, 1.02, band_note, transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")
    ax.set_xlabel("Environment steps", color=INK_SECONDARY)
    ax.set_ylabel("Eval return (mean of episodes)", color=INK_SECONDARY)
    ax.tick_params(labelsize=9)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:,.0f}")
    if ax.get_ylim()[0] >= 0:
        ax.set_ylim(bottom=0)
    if len(groups) <= 4:
        _label_line_ends(ax, ends)
    if len(groups) > 1:
        legend = ax.legend(frameon=False, fontsize=9, loc="lower right")
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# (key in progress.csv, panel title, what a reader should look for)
DIAGNOSTIC_PANELS = [
    ("rollout/ep_rew_mean", "Training episode return", "raw return, stochastic actions"),
    ("train/explained_variance", "Explained variance", "value net fit; 1 is perfect"),
    ("train/approx_kl", "Approx. KL per update", "how far each update moves the policy"),
    ("train/clip_fraction", "Clip fraction", "share of samples hitting the clip"),
    ("train/std", "Action std", "exploration noise of the policy"),
    ("train/value_loss", "Value loss", "on normalized rewards"),
]


def plot_diagnostics(groups: dict[str, list[Path]], path: Path, title: str) -> None:
    """Small multiples of PPO training metrics, one panel per metric.

    `groups` maps a label to the `progress.csv` files of its runs. A group of
    several runs is drawn as its mean over runs. Up to 8 groups, one fixed color each.
    """
    _check_series_count(len(groups))
    data = {label: [read_rows(f) for f in files] for label, files in groups.items()}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    for ax, (key, panel_title, hint) in zip(axes.flat, DIAGNOSTIC_PANELS):
        for (label, runs), color in zip(data.items(), SERIES_COLORS):
            series = [column(rows, key) for rows in runs]
            series = [s for s in series if s[0]]
            if not series:
                continue
            steps, values = _aligned(series)
            ax.plot(steps, values.mean(axis=0), color=color, linewidth=1.5, solid_joinstyle="round", label=label)
        _style_axes(ax)
        ax.set_title(f"{panel_title}\n", loc="left", fontsize=10, color=INK, pad=2)
        ax.text(0, 1.02, hint, transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")
        if key == "train/value_loss":
            ax.set_yscale("log")

    several = any(len(files) > 1 for files in groups.values())
    fig.suptitle(title + ("  (mean over seeds)" if several else ""), x=0.01, ha="left", fontsize=12, color=INK)
    if len(groups) > 1:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=min(len(groups), 5))
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
