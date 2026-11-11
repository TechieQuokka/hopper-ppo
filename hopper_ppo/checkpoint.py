"""Saving and loading models together with their normalization statistics.

A PPO model trained behind VecNormalize only works when its inputs are
normalized with the same running mean and variance it saw in training. So
every saved model has a partner file with those statistics:

    best   -> models/best_model.zip         + models/best_vecnormalize.pkl
    final  -> models/final_model.zip        + models/final_vecnormalize.pkl
    step_N -> checkpoints/step_N.zip        + checkpoints/step_N_vecnormalize.pkl
    resume -> resume/model.zip              + resume/vecnormalize.pkl

Every file is first written under a temporary name and then renamed, so a
crash in the middle of saving leaves the previous file intact.
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from hopper_ppo.config import Config, RunPaths


@dataclass
class RunState:
    """Progress of a run, stored as `<run>/state.json`."""

    num_timesteps: int = 0
    best_mean_reward: float | None = None
    best_timesteps: int | None = None
    status: str = "running"  # running | interrupted | completed
    updated_at: str = ""


@dataclass(frozen=True)
class SavedModel:
    """A model file and its normalization statistics file."""

    model: Path
    stats: Path

    def exists(self) -> bool:
        return self.model.exists() and self.stats.exists()


def model_files(paths: RunPaths, name: str) -> SavedModel:
    """Resolve `best`, `final`, `resume`, or a checkpoint name such as `step_0100000`."""
    if name in ("best", "final"):
        return SavedModel(paths.models_dir / f"{name}_model.zip", paths.models_dir / f"{name}_vecnormalize.pkl")
    if name == "resume":
        return SavedModel(paths.resume_dir / "model.zip", paths.resume_dir / "vecnormalize.pkl")
    return SavedModel(paths.checkpoints_dir / f"{name}.zip", paths.checkpoints_dir / f"{name}_vecnormalize.pkl")


def checkpoint_name(step: int) -> str:
    return f"step_{step:07d}"


def list_checkpoints(paths: RunPaths) -> list[str]:
    """Checkpoint names in step order (only complete model + stats pairs)."""
    names = sorted(p.stem for p in paths.checkpoints_dir.glob("step_*.zip"))
    return [name for name in names if model_files(paths, name).exists()]


def drop_checkpoints_after(paths: RunPaths, step: int) -> list[str]:
    """Delete checkpoints newer than `step` (left behind by a hard kill).

    Those steps are trained again after resuming, so the old files belong to a
    history that no longer exists. Returns the names that were removed.
    """
    removed = []
    for name in list_checkpoints(paths):
        if int(name.removeprefix("step_")) > step:
            files = model_files(paths, name)
            files.model.unlink()
            files.stats.unlink()
            removed.append(name)
    return removed


def _commit(tmp: Path, final: Path) -> None:
    """Flush a finished temporary file to disk, then move it into place."""
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, final)


def save_model(model: PPO, env: VecNormalize, target: SavedModel) -> None:
    """Save the statistics first, then the model.

    If saving is cut short, a model file is never left without statistics.
    """
    target.model.parent.mkdir(parents=True, exist_ok=True)
    tmp_stats = target.stats.with_name(target.stats.stem + ".tmp.pkl")
    env.save(str(tmp_stats))
    _commit(tmp_stats, target.stats)

    tmp_model = target.model.with_name(target.model.stem + ".tmp.zip")
    model.save(tmp_model)
    _commit(tmp_model, target.model)


def save_state(paths: RunPaths, state: RunState) -> None:
    state.updated_at = datetime.now().isoformat(timespec="seconds")
    tmp = paths.state_file.with_name(paths.state_file.name + ".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2) + "\n")
    _commit(tmp, paths.state_file)


def load_state(paths: RunPaths) -> RunState:
    return RunState(**json.loads(paths.state_file.read_text()))


def has_snapshot(paths: RunPaths) -> bool:
    return model_files(paths, "resume").exists()


def save_snapshot(model: PPO, paths: RunPaths, state: RunState, status: str = "running") -> None:
    """Save everything needed to continue training later, then the progress state.

    The model file stores its own step count, so it stays the source of truth
    for where to resume from even if the state file is not updated.
    """
    env = model.get_vec_normalize_env()
    if env is None:
        raise RuntimeError("Training environment is not wrapped in VecNormalize")
    save_model(model, env, model_files(paths, "resume"))
    state.num_timesteps = model.num_timesteps
    state.status = status
    save_state(paths, state)


def load_snapshot(cfg: Config, env: VecNormalize, paths: RunPaths) -> PPO:
    """Load the model of the latest snapshot onto an environment that already has its statistics."""
    model = PPO.load(
        model_files(paths, "resume").model,
        env=env,
        device=cfg.run.device,
        tensorboard_log=str(paths.tensorboard_dir),
    )
    # Loading re-applies the original seed. Re-seed with the step count so the
    # resumed part does not replay the same random sequence as the beginning.
    model.set_random_seed(cfg.run.seed + model.num_timesteps)
    return model
