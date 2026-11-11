"""Build environments and the PPO model from a Config.

Three kinds of environment exist, and each has exactly one builder here:
    - training:   n_envs copies in parallel, each in a Monitor, all behind VecNormalize
    - evaluation: one copy behind VecNormalize that never updates its statistics
    - rendering:  one plain copy that draws frames with the configured camera
"""

import multiprocessing
import signal
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import ConstantSchedule, LinearSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize
from torch import nn

from hopper_ppo.config import Config

EVAL_SEED_OFFSET = 10_000
RENDER_FLOOR_HALF_SIZE = 200.0  # meters; Hopper's own floor is drawn to 20 m
ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}


def _make_gym_env(env_id: str, render_mode: str | None = None, camera_name: str | None = None) -> gym.Env:
    """Create one environment. Runs inside a worker process for SubprocVecEnv."""
    if multiprocessing.current_process().name != "MainProcess":
        # Ctrl+C is sent to the whole process group. Workers ignore it so that the
        # main process can still talk to them while it saves progress.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    kwargs = {"camera_name": camera_name} if camera_name else {}
    return gym.make(env_id, render_mode=render_mode, **kwargs)


def make_train_env(cfg: Config, monitor_dir: Path | None = None, append: bool = False, seed: int | None = None) -> VecEnv:
    """Create the parallel training environments, without normalization.

    Each copy is wrapped in a Monitor that tracks raw episode returns. If
    `monitor_dir` is given, each copy writes `<monitor_dir>/<rank>.monitor.csv`;
    use `append=True` when resuming so the existing files are kept.
    """
    use_subproc = cfg.run.vec_env == "subproc" and cfg.run.n_envs > 1
    return make_vec_env(
        _make_gym_env,
        n_envs=cfg.run.n_envs,
        seed=cfg.run.seed if seed is None else seed,
        monitor_dir=str(monitor_dir) if monitor_dir else None,
        env_kwargs={"env_id": cfg.run.env_id},
        vec_env_cls=SubprocVecEnv if use_subproc else DummyVecEnv,
        monitor_kwargs={"override_existing": not append},
    )


def add_normalization(cfg: Config, venv: VecEnv, stats_file: Path | None = None) -> VecNormalize:
    """Wrap the training environments in VecNormalize, fresh or from saved statistics."""
    if stats_file is not None:
        env = VecNormalize.load(str(stats_file), venv)
        env.training = True
        return env
    return VecNormalize(
        venv,
        norm_obs=cfg.normalize.obs,
        norm_reward=cfg.normalize.reward,
        clip_obs=cfg.normalize.clip_obs,
        gamma=cfg.ppo.gamma,
    )


def make_eval_env(cfg: Config, stats_file: Path | None = None) -> VecNormalize:
    """Create one environment for evaluation, with its own seed.

    Observations are normalized with the training statistics (copied in before
    each evaluation, or loaded from `stats_file`), which are never updated here.
    Rewards are left raw so the scores are real returns.
    """
    venv = DummyVecEnv([lambda: Monitor(_make_gym_env(cfg.run.env_id))])
    venv.seed(cfg.run.seed + EVAL_SEED_OFFSET)
    if stats_file is not None:
        env = VecNormalize.load(str(stats_file), venv)
    else:
        env = VecNormalize(venv, norm_obs=cfg.normalize.obs, norm_reward=False, clip_obs=cfg.normalize.clip_obs)
    env.training = False
    env.norm_reward = False
    return env


def widen_floor(env: gym.Env, half_size: float = RENDER_FLOOR_HALF_SIZE) -> None:
    """Draw the floor out to `half_size` meters instead of Hopper's 20.

    A good policy travels about 20 m in one episode and would hop past the
    edge of the drawn floor. A MuJoCo plane collides as an infinite plane
    whatever its size, so this changes the picture only, not the physics.
    The checker texture is repeated more often so the squares keep their size.
    """
    model = env.unwrapped.model
    floor = model.geom("floor")
    scale = half_size / floor.size[0]
    if scale <= 1:
        return
    floor.size[:2] *= scale
    material = model.mat(int(floor.matid[0]))
    material.texrepeat[:] *= scale


def make_render_env(cfg: Config) -> gym.Env:
    """Create the one environment used for videos. All video code must use this."""
    env = _make_gym_env(cfg.run.env_id, render_mode="rgb_array", camera_name=cfg.run.render_camera or None)
    widen_floor(env)
    return env


def make_model(cfg: Config, env: VecEnv, tensorboard_dir: Path | None = None) -> PPO:
    """Create a fresh PPO model with the hyperparameters from the config."""
    ppo = cfg.ppo
    if ppo.lr_schedule == "linear":
        learning_rate = LinearSchedule(ppo.learning_rate, 0.0, 1.0)
    else:
        learning_rate = ConstantSchedule(ppo.learning_rate)
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=ppo.n_steps,
        batch_size=ppo.batch_size,
        n_epochs=ppo.n_epochs,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
        clip_range=ppo.clip_range,
        target_kl=ppo.target_kl or None,
        ent_coef=ppo.ent_coef,
        vf_coef=ppo.vf_coef,
        max_grad_norm=ppo.max_grad_norm,
        policy_kwargs={
            "net_arch": {"pi": list(ppo.net_arch), "vf": list(ppo.net_arch)},
            "activation_fn": ACTIVATIONS[ppo.activation],
            "log_std_init": ppo.log_std_init,
            "ortho_init": ppo.ortho_init,
        },
        tensorboard_log=str(tensorboard_dir) if tensorboard_dir else None,
        seed=cfg.run.seed,
        device=cfg.run.device,
        verbose=0,
    )
