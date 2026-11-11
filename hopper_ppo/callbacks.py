"""Periodic work during training: evaluation, checkpoints, resumable snapshots."""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

from hopper_ppo.checkpoint import RunState, checkpoint_name, model_files, save_model, save_snapshot, save_state
from hopper_ppo.config import Config, RunPaths
from hopper_ppo.metrics import append_eval_row


class RunCallback(BaseCallback):
    """Evaluate, checkpoint, and snapshot at the intervals set in the config.

    With n_envs parallel environments the step counter grows by n_envs at a
    time, so a task is due as soon as the counter passes the next multiple of
    its interval.

    Evaluations and checkpoints happen during a rollout (the policy does not
    change within a rollout). Snapshots happen only at the start of a rollout,
    right after a policy update, so a resumed run continues from a state where
    every counted step has been trained on.
    """

    def __init__(self, cfg: Config, paths: RunPaths, eval_env: VecNormalize, state: RunState):
        super().__init__()
        self.cfg = cfg
        self.paths = paths
        self.eval_env = eval_env
        self.state = state
        self._last_index: dict[str, int] = {}

    def _intervals(self) -> dict[str, int]:
        return {
            "eval": self.cfg.eval.freq,
            "checkpoint": self.cfg.save.checkpoint_freq,
            "snapshot": self.cfg.save.snapshot_freq,
        }

    def _on_training_start(self) -> None:
        # Count intervals from the current step, so a resumed run does not
        # repeat work that was already done before it stopped.
        for name, freq in self._intervals().items():
            self._last_index[name] = self.num_timesteps // freq

    def _is_due(self, name: str) -> bool:
        index = self.num_timesteps // self._intervals()[name]
        if index > self._last_index[name]:
            self._last_index[name] = index
            return True
        return False

    def _train_env(self) -> VecNormalize:
        env = self.model.get_vec_normalize_env()
        if env is None:
            raise RuntimeError("Training environment is not wrapped in VecNormalize")
        return env

    def _on_step(self) -> bool:
        step = self.num_timesteps
        # Evaluate first so a checkpoint at the same step sees the latest best score.
        if self._is_due("eval"):
            self._evaluate(step)
        if self._is_due("checkpoint"):
            save_model(self.model, self._train_env(), model_files(self.paths, checkpoint_name(step)))
        return True

    def _on_rollout_start(self) -> None:
        if self._is_due("snapshot"):
            save_snapshot(self.model, self.paths, self.state)

    def _evaluate(self, step: int) -> None:
        # Use the current training statistics, without letting evaluation change them.
        sync_envs_normalization(self._train_env(), self.eval_env)
        rewards, lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.cfg.eval.episodes,
            deterministic=True,
            return_episode_rewards=True,
        )
        mean_reward = float(np.mean(rewards))
        append_eval_row(
            self.paths.evals_file,
            {
                "timesteps": step,
                "mean_reward": round(mean_reward, 3),
                "std_reward": round(float(np.std(rewards)), 3),
                "min_reward": round(float(np.min(rewards)), 3),
                "max_reward": round(float(np.max(rewards)), 3),
                "mean_length": round(float(np.mean(lengths)), 1),
            },
        )
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_length", float(np.mean(lengths)))

        best = self.state.best_mean_reward
        if best is None or mean_reward > best:
            self.state.best_mean_reward = mean_reward
            self.state.best_timesteps = step
            save_model(self.model, self._train_env(), model_files(self.paths, "best"))
            save_state(self.paths, self.state)
