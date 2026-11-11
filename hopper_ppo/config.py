"""Experiment settings (loaded from TOML) and the folder layout of a run."""

import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ppo_hopper.toml"


@dataclass
class RunConfig:
    name: str = "ppo_hopper"
    env_id: str = "Hopper-v5"
    seed: int = 0
    device: str = "cpu"
    total_timesteps: int = 1_000_000
    # Parallel environments. Each PPO rollout collects n_envs * ppo.n_steps steps.
    n_envs: int = 8
    # "dummy": all environments in this process; "subproc": one process each.
    # Hopper's physics step is so cheap that both run at the same speed here.
    vec_env: str = "dummy"
    # PyTorch CPU threads; 0 keeps PyTorch's default. One thread was slightly
    # faster for this small network and leaves cores free for parallel runs.
    torch_threads: int = 1
    # Camera used for videos. Hopper's "track" camera follows the robot;
    # the default free camera stays still, so the robot hops out of view.
    render_camera: str = "track"


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    lr_schedule: str = "constant"  # "constant" or "linear" (decays to 0)
    n_steps: int = 256  # steps per environment per rollout
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    # Stop a round of updates early once the approximate KL to the policy that
    # collected the rollout exceeds 1.5 x this value. 0 turns the limit off.
    target_kl: float = 0.0
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: list[int] = field(default_factory=lambda: [64, 64])  # policy and value nets each
    activation: str = "tanh"  # "tanh" or "relu"
    log_std_init: float = 0.0
    ortho_init: bool = True


@dataclass
class NormalizeConfig:
    obs: bool = True  # running mean/std normalization of observations
    reward: bool = True  # scale rewards by a running estimate of the return's std
    clip_obs: float = 10.0


@dataclass
class EvalConfig:
    freq: int = 10_000
    episodes: int = 5


@dataclass
class SaveConfig:
    snapshot_freq: int = 10_000  # resumable snapshot (model + normalization stats)
    checkpoint_freq: int = 100_000  # kept for later comparison and videos


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    save: SaveConfig = field(default_factory=SaveConfig)

    @property
    def rollout_size(self) -> int:
        return self.run.n_envs * self.ppo.n_steps


SECTIONS = {f.name: f.default_factory for f in fields(Config)}


def _section_keys(section: str) -> set[str]:
    return {f.name for f in fields(SECTIONS[section])}


def _parse_value(raw: str) -> Any:
    """Parse an override value with TOML rules; fall back to a plain string."""
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def _apply_override(data: dict[str, dict[str, Any]], item: str) -> None:
    """Apply one `section.key=value` (or unambiguous `key=value`) override."""
    if "=" not in item:
        raise ValueError(f"Override must look like key=value, got: {item!r}")
    key, raw = item.split("=", 1)
    key = key.strip()

    if "." in key:
        section, name = key.split(".", 1)
    else:
        matches = [s for s in SECTIONS if key in _section_keys(s)]
        if len(matches) != 1:
            raise ValueError(f"Override key {key!r} matches sections {matches}; use section.key=value")
        section, name = matches[0], key

    if section not in SECTIONS or name not in _section_keys(section):
        raise ValueError(f"Unknown setting: {section}.{name}")
    data.setdefault(section, {})[name] = _parse_value(raw.strip())


def validate(cfg: Config) -> None:
    """Reject settings that would fail late or silently misbehave."""
    if cfg.run.n_envs < 1:
        raise ValueError("run.n_envs must be at least 1")
    if cfg.run.torch_threads < 0:
        raise ValueError("run.torch_threads must be 0 (PyTorch default) or more")
    if cfg.run.vec_env not in ("subproc", "dummy"):
        raise ValueError(f"run.vec_env must be 'subproc' or 'dummy', got {cfg.run.vec_env!r}")
    if cfg.ppo.lr_schedule not in ("constant", "linear"):
        raise ValueError(f"ppo.lr_schedule must be 'constant' or 'linear', got {cfg.ppo.lr_schedule!r}")
    if cfg.ppo.activation not in ("tanh", "relu"):
        raise ValueError(f"ppo.activation must be 'tanh' or 'relu', got {cfg.ppo.activation!r}")
    if cfg.ppo.target_kl < 0:
        raise ValueError("ppo.target_kl must be 0 (off) or positive")
    if cfg.rollout_size % cfg.ppo.batch_size != 0:
        raise ValueError(
            f"Rollout size n_envs * n_steps = {cfg.rollout_size} must be a multiple of "
            f"ppo.batch_size = {cfg.ppo.batch_size}, so no minibatch is cut short"
        )
    for name in ("freq", "episodes"):
        if getattr(cfg.eval, name) < 1:
            raise ValueError(f"eval.{name} must be at least 1")
    for name in ("snapshot_freq", "checkpoint_freq"):
        if getattr(cfg.save, name) < 1:
            raise ValueError(f"save.{name} must be at least 1")


def load_config(path: Path = DEFAULT_CONFIG, overrides: list[str] | None = None) -> Config:
    """Read a TOML file, apply command-line overrides, and validate the keys."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    for item in overrides or []:
        _apply_override(data, item)

    unknown_sections = set(data) - set(SECTIONS)
    if unknown_sections:
        raise ValueError(f"Unknown config sections: {sorted(unknown_sections)}")

    sections = {}
    for section, cls in SECTIONS.items():
        values = data.get(section, {})
        unknown = set(values) - _section_keys(section)
        if unknown:
            raise ValueError(f"Unknown settings in [{section}]: {sorted(unknown)}")
        sections[section] = cls(**values)
    cfg = Config(**sections)
    validate(cfg)
    return cfg


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return repr(value)


def save_config(cfg: Config, path: Path) -> None:
    """Write the resolved settings as TOML, so a run records what it used."""
    lines = []
    for section, values in asdict(cfg).items():
        lines.append(f"[{section}]")
        lines.extend(f"{key} = {_toml_value(value)}" for key, value in values.items())
        lines.append("")
    path.write_text("\n".join(lines))


@dataclass(frozen=True)
class RunPaths:
    """Locations of everything one run produces."""

    root: Path

    @property
    def config_file(self) -> Path:
        return self.root / "config.toml"

    @property
    def state_file(self) -> Path:
        return self.root / "state.json"

    @property
    def resume_dir(self) -> Path:
        return self.root / "resume"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def tensorboard_dir(self) -> Path:
        return self.root / "tensorboard"

    @property
    def evals_file(self) -> Path:
        return self.logs_dir / "evals.csv"

    @property
    def progress_file(self) -> Path:
        return self.logs_dir / "progress.csv"

    @property
    def monitor_dir(self) -> Path:
        return self.logs_dir / "train_monitor"

    @property
    def videos_dir(self) -> Path:
        return self.root / "videos"

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots"

    def create(self) -> None:
        for directory in (
            self.resume_dir,
            self.models_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.monitor_dir,
            self.tensorboard_dir,
            self.videos_dir,
            self.plots_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def display_path(path: Path) -> str:
    """Path relative to the project when possible, so logs carry no home-folder names."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def new_run_paths(cfg: Config, tag: str | None = None, runs_dir: Path = RUNS_DIR) -> RunPaths:
    """Build a unique run folder name: <name>_seed<seed>_<timestamp>[_<tag>]."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{cfg.run.name}_seed{cfg.run.seed}_{stamp}"
    if tag:
        name += f"_{tag}"
    return RunPaths(Path(runs_dir).resolve() / name)
