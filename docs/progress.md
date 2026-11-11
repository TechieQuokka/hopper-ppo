# Progress and handoff

Last updated: 2026-11-11 17:10

This document records where the work stands so it can be continued later.

## Goal

Learn how PPO behaves in practice by training Stable-Baselines3 PPO on Gymnasium's
`Hopper-v5` (the same robot as the earlier SAC project), then changing one setting at a
time and reading the PPO diagnostics. The score matters less than understanding what
each setting does. Implementing PPO from scratch is out of scope.

## Status in one paragraph

Stages 0-4 are done. The pipeline trains, evaluates, records videos with a camera check,
draws charts, and can be stopped and resumed. The baseline (standard MuJoCo PPO settings
with observation and reward normalization) trains 1M steps in about 6 minutes on the CPU
and reaches an eval return of 3445 +/- 4 over 10 episodes, all 1000 steps long.
Nothing is running now. Next: design the sweep (stage 5) together with the user.

## Stages

| Stage | Result |
|---|---|
| 0. Repo setup | Done. `.gitignore`, `requirements.txt`, local git (not pushed) |
| 1. Config, environments, model, training | Done. `train.py`, `hopper_ppo/`, `configs/ppo_hopper.toml` |
| 2. Snapshots and resume | Done. Verified by `tests/check_resume.py` (Ctrl+C and hard kill) |
| 3. Speed benchmark | Done. `tests/bench_speed.py`; CPU, 8 envs in one process, 1 thread |
| 4. Baseline run, videos, charts | Done. Results below; copies in `assets/` |
| 5. Sweep and multi-seed check | Not started. To be designed with the user |
| 6. README report | Not started |

## How to use

```bash
python train.py                                   # baseline, about 6 min
python train.py --set ppo.n_epochs=5 --tag epochs5
python train.py --resume runs/<run>               # continue after Ctrl+C or a crash
python evaluate.py runs/<run> --video --gif       # 10-episode score + video, sheet, GIF
python evaluate.py runs/<run> --video-stages      # one video per checkpoint
python plot.py runs/<run>                         # learning curve + PPO diagnostics
python plot.py runs/<a> runs/<b> --out compare    # several runs in one chart
python sweep.py configs/sweeps/<sweep>.toml       # variants x seeds, resumable

python tests/quick_check.py                       # whole pipeline in about a minute
python tests/check_resume.py                      # stop/resume correctness
python tests/check_camera.py                      # videos keep the robot in view
python tests/bench_speed.py                       # hardware settings
```

## Baseline results (`runs/ppo_hopper_seed0_20261111-165338_baseline`)

Settings: `configs/ppo_hopper.toml` (lr 3e-4, rollout 8 envs x 256 steps = 2048,
batch 64, 10 epochs, gamma 0.99, GAE lambda 0.95, clip 0.2, no entropy bonus,
64x64 tanh networks, observation and reward normalization). Seed 0.

| | |
|---|---|
| Training time | 374 s for 1,001,472 steps (2680 steps/s including evaluations) |
| Best model, 10 eval episodes | 3445.5 +/- 4.0, all episodes 1000 steps (best at step 970k) |
| Final model, 10 eval episodes | 3281.5 +/- 472.1, one episode fell early (1865) |
| First eval above 1000 / 3000 | 110k / 180k steps |
| Last 10 evals (910k-1M) | mean 3427, all 1000 steps long |
| Reward threshold (3800) | not reached |

Observations:

- **Learning is fast, then unstable.** After the first eval above 3000 (180k), 10 of the
  next 82 evaluations dropped below 2000 (lowest 1042 at 440k). It settled only in the
  last ~100k steps. The best model is saved separately for this reason.
- **Updates get bigger as the policy gets sharper.** The action std fell from about 0.8
  to 0.17. Over the same time `approx_kl` rose from about 0.011 to 0.046 and
  `clip_fraction` from 0.13 to 0.36 (means of the first and last 50 updates). With a
  fixed learning rate, 10 epochs over the same data move a narrow policy further.
  This is the main thing to probe in the sweep.
- **The value function fits well.** `explained_variance` stays near 0.99 after the first
  ~20k steps, with brief dips that line up with return collapses.
- The best policy travels about 19.5 m per episode.

Charts: `assets/baseline_learning_curve.png`, `assets/baseline_diagnostics.png`.
Video frames: `assets/best_policy_sheet.png`; animation: `assets/best_policy.gif`.

## Speed benchmark (2026-11-11, i3-14100F 4C/8T, RTX 3060)

20,480 steps per measurement, best of 3, same rollout size (2048) everywhere.

| Setting | Steps/s | 1M steps |
|---|---|---|
| cpu, 1 env | 1854 | 9.0 min |
| cpu, 8 envs in one process (dummy) | 3403 | 4.9 min |
| cpu, 8 envs in one process, 1 thread | 3551 | 4.7 min |
| cpu, 8 envs one process each (subproc) | 3381 | 4.9 min |
| cpu, 8 envs subproc, 1 thread | 3681 | 4.5 min |
| cpu, 16 envs subproc | 3504 | 4.8 min |
| cuda, 1 env | 1148 | 14.5 min |
| cuda, 8 envs subproc | 2598 | 6.4 min |

- The GPU is slower: the 64x64 networks are too small to benefit.
- One process per environment gives no gain: a Hopper physics step is very cheap, and
  most of the time is the batched network forward pass and the update.
- Decision: CPU, 8 envs in one process (`dummy`), 1 PyTorch thread. Speed is within
  about 5% of the fastest setting, and each run uses about one core, which leaves room
  to train several sweep runs at once.

## Design notes

- **Normalization statistics are part of the model.** Every saved model has a partner
  `*_vecnormalize.pkl`; `hopper_ppo/checkpoint.py` saves and resolves them as pairs.
  Evaluation copies the training statistics in and never updates them, and scores use
  raw rewards.
- **Snapshots** are taken only at the start of a rollout (right after an update), so a
  hard kill resumes from a state where every counted step was trained on. Ctrl+C saves
  at the moment of the interrupt. On resume, eval/progress rows and checkpoints newer
  than the snapshot are dropped; the best model is kept even if it came from the
  dropped part (it was really evaluated).
- **Ctrl+C with worker processes** (`vec_env = "subproc"`): workers ignore SIGINT so the
  main process can still save. Tested with signals sent to the whole process group.
- **Metrics** go to TensorBoard and to `logs/progress.csv` (fixed columns, appended
  across resumes). SB3's own CSV writer would overwrite the file on resume.

## Camera (lesson from the SAC project)

In the SAC project the default camera stayed still and the robot hopped out of view.
Here:

- Videos are made only through `builders.make_render_env`, which uses the `track`
  camera from the config (default and baseline config both set it).
- Every recorded frame is checked: the torso position is projected into the image, and
  a warning (and a non-zero exit from `evaluate.py`) follows if it leaves the middle
  of the frame. All baseline videos passed (worst offset 0.21 of the frame).
- `tests/check_camera.py` moves the robot to 0-100 m and confirms it stays centered,
  and confirms the detector does flag a camera that stays still.
- Each video also gets a contact sheet (`<name>_sheet.png`, frames at 0/25/50/75/100%).
- Hopper draws its floor only to 20 m, and the best policy reaches 19.5 m, so the end of
  the first baseline videos showed the edge of the floor. The rendering environment now
  draws the floor to 200 m. A MuJoCo plane collides as an infinite plane, so physics is
  unchanged: `check_camera.py` confirms identical trajectories, and the best model
  scored exactly the same before and after (3445.5).

## Environment

- Conda env `rl`, Python 3.13.12; versions in `requirements.txt` (same as the SAC
  project, nothing upgraded).
- Rendering uses `MUJOCO_GL=egl` (set in `evaluate.py` and the camera test).

## Known limitations

- Single seed so far; the instability above means one run says little about a setting.
- After a hard kill, up to one snapshot interval (10k steps) is trained again.
- Ctrl+C during an update saves the parameters as they are at that moment; the
  interrupted rollout's data is discarded.
- Resuming re-seeds with the step count, so a resumed run does not reproduce an
  uninterrupted one exactly.

## Possible next steps (stage 5, to be agreed)

Candidates, one change at a time against the baseline, with several seeds since a run
takes about 6 minutes:

1. No normalization (obs and/or reward off), to see why it matters.
2. Smaller or decaying updates late in training: linear learning-rate decay,
   `n_epochs` 5, `clip_range` 0.1, or a KL limit (`target_kl`, would need a config field).
3. `n_steps` (rollout length) and `gae_lambda`.
4. `ent_coef` > 0 (exploration), larger networks.
5. The RL Zoo's tuned Hopper settings as a reference point.
