# Resume here — 2026-08-25

## Where things stand

**HIL-SERL runs end-to-end on the real SO-101.** 2026-08-24, 10 minutes, no incident:

```
episodes 34 · rewarded 29 · EE-jump clamps 0 · errors 0
learner 7.8 optimization steps/s · wandb run ri71n3m9
```

Not autonomous yet — that 85% includes interventions. 34 episodes vs the ~1,900 the
sim needed. Today proved stability, not competence.

## What actually fixed the shaking / folding / crashes

**The workspace box was too big.** Derived from the leader sweep, it reached x=0.429,
about 14 cm beyond anywhere the 24 successful demos ever went — into the region where a
nearly-extended 5-DoF arm is kinematically singular and the IK swings between solutions.

FK over all 3472 demo frames showed the task lives in:

```
x 0.113..0.287   y -0.053..0.175   z -0.014..0.114
```

Box is now that, plus ~3 cm: `x 0.09..0.32  y -0.09..0.21  z -0.015..0.16`
(23 x 30 x 17.5 cm, was 33 x 37 x 26). After this, zero shaking and zero clamps.

**Lesson worth keeping:** the demo data, not reasoning about IK internals, found this.
Several plausible-sounding code fixes (orientation_weight=1.0, seeding IK from measured
joints, max_relative_target) all made it worse and were reverted. Change one thing at a
time and test against the known-good baseline.

Also fixed today: a pinched servo loom in the base was causing grinding on shoulder_pan
plus an intermittent bus (partial pings, failing reads). Reseated. Route it with slack.

## Learner throughput — change APPLIED, needs measuring

The learner is the bottleneck and it costs robot time directly.

```
actor collects 10 env steps/s; utd_ratio 2 wants 20 optimization steps/s
run of 2026-08-24 delivered ~7.8  ->  effective UTD 0.78, not 2
```

So each minute at the bench bought ~1/3 of the learning the config intends. Suspected
cause: `storage_device: "cpu"` meant every batch (256 x 2 cameras x 128x128 x float32 x
(obs+next) ~= 200 MB) was copied CPU->GPU ~8 times a second, while the 4090 sat at 1.1 GB
of 24 GB used.

**Already changed in configs/real/train_config.json:**

```json
"storage_device": "cuda", "online_buffer_capacity": 12000, "offline_buffer_capacity": 4000
```

Verified it allocates: learner starts clean, GPU 3.8 GB with the offline buffer resident,
projecting ~13 GB once the online buffer fills. No OOM.

**Not yet measured:** the Hz gain needs an actor feeding the learner. First thing on the
next run, watch `Optimization frequency loop [Hz]`:
- was ~7.8. If it climbs toward 15-20, the transfer was the bottleneck and robot time is
  now worth ~2-3x more.
- if it stays ~8, the bottleneck is compute or buffer sampling, not transfer — revert
  storage_device to "cpu" (it costs GPU memory for nothing) and look at batch_size.
- if it OOMs mid-run, drop online_buffer_capacity to 10000.

## Then: the long run

1–2 hours continuous. That is the paper's budget per task and what autonomy needs.

```bash
cd /mnt/Storage/projects/hil-serl
set -a && . ./.env && set +a && source .venv/bin/activate
rm -rf outputs/hilserl_cube_run1 outputs/hilserl_cube_run1_actor

python -m lerobot.rl.learner --config_path configs/real/train_config.json
# second terminal:
python -m lerobot.rl.actor   --config_path configs/real/train_config.json \
    --output_dir=outputs/hilserl_cube_run1_actor
```

`--output_dir` on the actor is required (both validate the same config; the actor refuses
the directory the learner just made). If the actor dies, restart the learner too —
MAX_WORKERS=3 in learner_service.py equals the RPCs one actor opens, and a half-open
connection blocks the next actor silently.

Watch: intervention rate falling, and autonomous (non-intervened) success rising. Those
two are the real progress signal, not raw episode reward.

## Preflight, every time

```bash
ls /dev/so101_follower && lsusb | grep -q 045e && echo pad ok
```
Follower needs USB **and** 12 V. Symptom decoder:
- no motors found / no status packet on all 6 -> servos unpowered
- partial pings, failing reads -> loose connector (check the base loom)
- `Incorrect status packet` -> transient, retry once

Controls: hold **RB** to take over · left stick X/Y · right stick Z · **LT** close ·
**RT** open · **Y** success · **A** rerecord. On-screen help is Logitech-ordered; ignore.
Reward is the Y button — no classifier yet, so you are the reward function.

## Open threads

- **IK walkthrough** — promised, not done. Now worth more: there is a concrete failure
  (5-DoF arm + a 6-DoF action space + a singular workspace edge) to explain.
- Dataset is named `..._ball_...` but contains **white cube** demos (the ball was too
  slippery). Rename before it reaches the thesis.
- TPU gripper pads for the ball task will change what the wrist camera sees; any policy
  trained now must run without them.
- Upstream PR candidates queued: #4297 (open), crop_dataset_roi finalize + boundary fixes
  (committed on local/hilserl), the docs claiming leader-arm support that does not exist.

---

# Session 2026-08-26 — first checkpointed run

```
121 episodes · 94 rewarded · 14,000 optimization steps · 9.0 Hz
7 checkpoints, latest outputs/hilserl_cube_run1/checkpoints/0014000
wandb run jd6m1z4y
```

Still needs interventions throughout; releasing RB gives wild behaviour. Expected at
121 episodes — the sim needed ~1,900.

## Two bugs found and handled

1. **Checkpointing killed the learner.** `save_training_checkpoint` writes the policy
   weights fine, then dumps the replay buffers back out as PNGs; the buffer holds float
   images in [0,255] while the dataset writer wants [0,1] or uint8. The exception
   propagated and took down the training loop *after* the weights were already written.
   Wrapped the dump in try/except in `rl/learner.py` — warns and continues.
   **Upstream impact: any HIL-SERL run reaching its first checkpoint dies.** Hidden
   because the shipped `save_freq` is 2,000,000, so it effectively never fires.
   Good PR: either fix the dtype in `buffer.to_lerobot_dataset` or make the dump non-fatal.

2. GPU-resident replay buffer gave only ~18% (7.8 -> 9.0 Hz), not the 2-3x hoped.
   The PCIe transfer was not the main bottleneck; the learner is compute/sampling bound.
   Keeping it (GPU has headroom, ~12 GB used of 24) but do not expect more from it.

## Resuming — read this first

`resume: true` + `--config_path` restores **policy weights and optimizer state**, NOT the
replay buffer: the buffer dump is exactly what fails above, so the online buffer restarts
empty. The demo dataset reloads normally. So a resumed run keeps everything the policy
has *learned* but loses everything it has *experienced*.

Practical consequence: prefer long single sessions over many short ones.

## The intervention balance — important

Holding RB the whole time collects an excellent demonstration set but teaches the critic
nothing about which states are bad: only the human's (good) transitions enter the online
buffer, so nothing pushes the actor away from flailing. Some autonomous-and-bad data is
required for learning.

Next session: release RB during **transit** (gripper high, away from the board) for 1-3 s
at a time, keep it held during **approach and insertion** where a bad action costs
hardware. The workspace box bounds the damage; the board is the remaining hazard.

## Session 2 produced a much better demo set — use it

You guided 99.1% of frames, so essentially all of session 2 is demonstration-quality.
Filtered to successes only:

```
lucarp/so101_cube_session2_clean   92 episodes · 10,347 frames · 92/92 success
```

That is ~4x the original 24-demo set, from the identical scene, with faster episodes
(10.4 s mean vs 14.5 s). Verified it loads via LeRobotDataset with correct shapes.

**Next run: point `dataset.repo_id` at it.**

```json
"dataset": { "repo_id": "lucarp/so101_cube_session2_clean", "use_imagenet_stats": false }
```

Then update `policy.dataset_stats.observation.state` min/max from that dataset's
meta/stats.json (the current values came from the old 24-demo set).

**Not merged with the original 24.** The session-2 dataset comes from a replay-buffer
dump and lacks metadata columns `lerobot-edit-dataset --operation.type merge` requires
(`meta/episodes/chunk_index`). Aligning the feature schema fixed two rounds of errors and
a third appeared. Not worth more conversions for +26% data.

Derivation chain, if it ever needs redoing:
```
outputs/hilserl_cube_run1/dataset            (115 eps, raw buffer dump)
  -> delete_episodes [23 without success]    -> so101_cube_session2_success (92)
  -> convert_image_to_video                  -> so101_cube_session2_vid
  -> remove_feature is_intervention          -> so101_cube_session2_clean
```

Also resume the policy rather than starting cold:
`checkpoints/0014000` holds 14,000 optimization steps.

---

# Session 2026-08-26 (afternoon) — 230 episodes, no improvement yet

```
230 episodes · 24,000 optimization steps · 12 checkpoints · latest 0024000
offline data: 92-episode demo set · GPU buffers · ~10 Hz (GPU free)
intervention 65.9% (was 99.1%) · 4,090 autonomous frames (was ~109)
wandb opp47yi6
```

**Success fell 74% -> 30% across the session. That is the intervention rate, not the
policy.** Per-block, success tracks how much RB was held; comparing to the 82% of the
previous session compares human flying to policy flying. No upward trend visible yet.

## The structural problem: every session starts from scratch

Session 1: 14,000 steps discarded. Session 2: 24,000 steps discarded (this one, unless
resumed). The early-training phase keeps being repeated instead of accumulated.

**Next session: RESUME.** It was correct to start fresh today (the dataset and its
normalization changed). Nothing changes now, so:

```json
"resume": true
```

and launch against the same output_dir so it picks up `checkpoints/last` (0024000).
Do NOT change `dataset.repo_id` or `dataset_stats` while resuming — a resumed policy
carries a normalizer fitted to the current data; changing the data underneath it is
exactly the inconsistency that made a fresh start right today.

## Throughput, finally measured cleanly

```
CPU buffers                       7.8 Hz
GPU buffers, sharing with the LLM 9.0 Hz
GPU buffers, GPU free            10.0 Hz
```
The PCIe transfer was never the main bottleneck; the learner is compute-bound. ~28% total.

**llama-server holds 20.9 GB of the 4090 and must be stopped before training** — with it
running, even CPU-resident buffers OOM because the model and batch activations need a few
GB. Restart it after training:

```
llama-server --model ~/local-llm/models/Qwen3.8-27B-UD-Q4_K_XL.gguf --alias qwen3.8-27b \
  --host 127.0.0.1 --port 8080 --n-gpu-layers all --flash-attn on --ctx-size 131072 \
  --cache-type-k q4_0 --cache-type-v q4_0 --spec-type draft-mtp ...
```

## Expectation setting

The sim reached 100% after ~1,900 episodes / 32,000 steps. You are at 230 episodes with a
policy that has seen 24,000 steps. Episodes, not gradient steps, are the scarce resource:
~200/hour on hardware. Budget several accumulated hours, and resume every time.
