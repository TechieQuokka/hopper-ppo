# Progress and handoff

Last updated: 2026-11-11 19:05

This document records where the work stands so it can be continued later.

## Goal

Learn how PPO behaves in practice by training Stable-Baselines3 PPO on Gymnasium's
`Hopper-v5` (the same robot as the earlier SAC project), then changing one setting at a
time and reading the PPO diagnostics. The score matters less than understanding what
each setting does. Implementing PPO from scratch is out of scope.

## Status in one paragraph

Stages 0-5 are done. The pipeline trains, evaluates, records videos with a camera check,
draws charts, and can be stopped and resumed. The baseline (standard MuJoCo PPO settings
with observation and reward normalization) trains 1M steps in about 6 minutes on the CPU
and reaches an eval return of 3445 +/- 4 over 10 episodes (seed 0). Sweep 1 (5 variants
x 3 seeds) suggested that `clip_range = 0.1` ends better; sweep 2 (3 variants x 10 seeds)
did **not** confirm it (p = 0.20), but did confirm that it learns about 2x slower
(p = 0.001). Nothing is running now. Next: decide with the user.

## Stages

| Stage | Result |
|---|---|
| 0. Repo setup | Done. `.gitignore`, `requirements.txt`, local git (not pushed) |
| 1. Config, environments, model, training | Done. `train.py`, `hopper_ppo/`, `configs/ppo_hopper.toml` |
| 2. Snapshots and resume | Done. Verified by `tests/check_resume.py` (Ctrl+C and hard kill) |
| 3. Speed benchmark | Done. `tests/bench_speed.py`; CPU, 8 envs in one process, 1 thread |
| 4. Baseline run, videos, charts | Done. Results below; copies in `assets/` |
| 5a. Sweep 1: update size (5 variants x 3 seeds) | Done. Results below |
| 5b. Sweep 2: confirm clip 0.1 (3 variants x 10 seeds) | Done. Results below |
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

- The baseline section uses one seed. Seed-to-seed variation is large (sweep 2).
- After a hard kill, up to one snapshot interval (10k steps) is trained again.
- Ctrl+C during an update saves the parameters as they are at that moment; the
  interrupted rollout's data is discarded.
- Resuming re-seeds with the step count, so a resumed run does not reproduce an
  uninterrupted one exactly.

## Sweep 1: update size (`configs/sweeps/update_size.toml`, `runs/update_size/`)

Question: in the baseline, `approx_kl` and `clip_fraction` grew as the policy got sharper,
and the eval return kept collapsing. Does limiting the update size help?
Each variant changes one setting; seeds 0, 1, 2; 1M steps each. 15 runs took about
35 minutes, 3 at a time (each run ran at full speed in parallel).

| Variant | Change |
|---|---|
| baseline | - |
| lr_linear | learning rate decays linearly from 3e-4 to 0 |
| epochs5 | `n_epochs` 10 -> 5 |
| clip0.1 | `clip_range` 0.2 -> 0.1 |
| kl0.02 | `target_kl` 0.02 (a round of updates stops once approx_kl > 0.03) |

Mean over seeds (min-max in brackets). "Good at": first eval >= 3000. "Collapse rate":
share of evals below 2000 after that point. "Final": final model over 10 episodes.
Per-run numbers: `docs/update_size_summary.csv`.

| Variant | Best eval | Mean of last 10 evals | Good at | Collapse rate | Final model | Late approx_kl | Late epochs per rollout |
|---|---|---|---|---|---|---|---|
| baseline | 3604 (3445-3714) | 3205 (2877-3427) | 167k (130-190k) | 9% (7-12%) | 3342 (3265-3480) | 0.042 | 10.0 |
| lr_linear | 3463 (3308-3585) | 3143 (2644-3489) | 270k (180-320k) | 3% (0-7%) | 3100 (2517-3496) | 0.006 | 10.0 |
| epochs5 | 3512 (3312-3627) | 3025 (1992-3566) | 373k (190-550k) | 15% (4-31%) | 2651 (1268-3544) | 0.019 | 5.0 |
| clip0.1 | 3585 (3376-3772) | **3490 (3325-3734)** | 337k (270-390k) | 6% (2-15%) | **3478 (3353-3706)** | 0.010 | 10.0 |
| kl0.02 | 3400 (3240-3549) | 2477 (1903-2823) | 307k (140-550k) | 26% (9-55%) | 2802 (1692-3511) | 0.018 | 2.7 |

"Late" = mean of the last 50 updates. Charts: `assets/update_size_learning_curve.png`,
`assets/update_size_diagnostics.png`.

**Significance check** (`python compare.py configs/sweeps/update_size.toml`): exact
two-sided permutation tests on per-run values, vs the baseline. With 3 runs against 3
there are only 20 ways to split the 6 runs, so **no difference can reach p < 0.10**.

| Variant | Last 10 evals, diff | p | Final model, diff | p | First eval >= 3000, diff | p |
|---|---|---|---|---|---|---|
| lr_linear | -63 | 0.80 | -242 | 0.90 | +103k | 0.30 |
| epochs5 | -181 | 1.00 | -691 | 0.50 | +207k | 0.20 |
| clip0.1 | +285 | 0.40 | +136 | 0.50 | +170k | 0.10 |
| kl0.02 | -728 | 0.10 | -540 | 0.50 | +140k | 0.40 |

None of the differences is statistically significant. The strongest signals (p = 0.10,
the smallest possible here, meaning the groups did not overlap at all) are: `kl0.02`
ends lower than the baseline, and `clip0.1` reaches 3000 later. The findings below are
therefore **hypotheses**; sweep 2 tested the `clip0.1` one with 10 seeds (not confirmed).

Findings (hypotheses; 3 seeds):

1. **Every variant lowered the late approx_kl** (0.006-0.019 vs 0.042), so each knob did
   what it says. A smaller KL alone did not make training more stable.
2. **Cutting the number of updates hurt.** `kl0.02` stopped each round after about 2.7 of
   10 epochs late in training: the worst variant on every measure, with the training
   return stuck near 1800-2000. `epochs5` was also worse than the baseline and varied
   most between seeds. Fewer passes over the data means less learning per rollout.
3. **Limiting the step per sample may help at the end** (not significant, p = 0.40).
   `clip0.1` had the best and most consistent end result: all three seeds above 3325 over the last 10 evals and above
   3350 for the final model; seed 2 reached the best single score (3772, final 3706).
   Its clip fraction stays as high as the baseline's (a narrower clip catches more
   samples), but its KL stays low.
4. **Learning-rate decay calmed the end of training** (lowest collapse rate, 0% for seed 2),
   but it also freezes whatever the policy is when the rate reaches 0: seed 1 ended at
   2517 with only 2 of 10 full-length episodes.
5. **The baseline learns fastest** (first 3000 at 167k on average, about 2x earlier than
   the others), then oscillates. Every stabilizing change also slowed early learning.
6. The action std shrinks more slowly in all variants (0.26-0.35 at the end vs 0.19),
   because smaller updates also shrink the exploration noise more slowly.

Reproducibility: `baseline_s0` in the sweep produced exactly the same eval log as the
standalone baseline run (same seed, same settings).

## Sweep 2: confirm clip 0.1 (`configs/sweeps/confirm_clip.toml`, `runs/confirm_clip/`)

Question: is the `clip0.1` advantage from sweep 1 real? The analysis was written into the
sweep file before running: primary metric = mean of the last 10 evals, exact two-sided
permutation test against the baseline (10 runs vs 10: 184,756 relabelings).
Seeds 0-9; seeds 0-2 reproduce sweep 1 exactly. 30 runs took 74 minutes, 3 at a time.

`python compare.py configs/sweeps/confirm_clip.toml`. Means over 10 seeds; difference vs
baseline with a 95% bootstrap interval; p from the permutation test.

| Metric | baseline | clip0.1 | clip0.1_lrlin |
|---|---|---|---|
| **Mean of last 10 evals** (primary) | 2937 (2020-3639) | 3217, +280 [-115, 666], **p = 0.20** | 3118, +181 [-290, 618], p = 0.46 |
| Final model (10 episodes) | 3102 (1664-3547) | 3245, +143, p = 0.57 | 2989, -113, p = 0.72 |
| First eval >= 3000 | 217k (130-330k) | 444k, +227k, **p = 0.001** | 486k, +269k, **p = 0.0002** |
| Collapse rate after that | 10.7% | 9.4%, p = 0.75 | 16.2%, p = 0.82 (see note) |

Note: one `clip0.1_lrlin` run (seed 9) never reached 3000 (best 2983) and counts as a
100% collapse rate by the script's conservative rule; its median is 0.7%.

Conclusions:

1. **The end-performance advantage of `clip0.1` is not confirmed.** The mean difference
   is about the same as in sweep 1 (+280 vs +285), but with 10 seeds the baseline's
   spread turned out much wider (last-10 mean 2020-3639; the first 3 seeds happened to be
   among its better ones), and the interval still includes zero and harmful values.
   A real effect smaller than about +650 cannot be ruled out either.
2. **`clip0.1` learns about 2x slower: confirmed** (p = 0.001). Halving the clip range
   halves how far each update may move the policy, and it shows early in training.
3. **Adding learning-rate decay did not help** on any metric.
4. Run-to-run variation is the dominant effect in this setup: the same settings give
   last-10 means from about 2000 to 3600 depending on the seed alone.

Lessons on method:

- 3 seeds per variant could not have shown significance (minimum two-sided p = 0.10),
  and the 3 baseline seeds understated its spread. A ranking from 3 seeds was misleading.
- Deciding the metric and test before running avoided picking whichever metric looked best.

## Possible next steps (to be agreed)

1. Normalization ablation (obs and/or reward off), to see why it matters.
2. Other settings: `n_steps` (rollout length), `gae_lambda`, `ent_coef`, network size,
   the RL Zoo's tuned Hopper settings.
3. README report (stage 6).
