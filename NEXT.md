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

## Do this first tomorrow: learner throughput

The learner is the bottleneck and it costs robot time directly.

```
actor collects 10 env steps/s; utd_ratio 2 wants 20 optimization steps/s
learner delivers ~7.8  ->  effective UTD 0.78, not 2
```

So each minute at the bench buys ~1/3 of the learning the config intends. Suspected cause
is `policy.storage_device: "cpu"` — a batch is 256 x 2 cameras x 128x128 x float32 x
(obs+next) ~= 200 MB, copied CPU->GPU ~8 times a second.

Change to try (configs/real/train_config.json), then watch the Hz line:

```json
"policy": { "storage_device": "cuda", "online_buffer_capacity": 12000 }
```

GPU is a 24 GB 4090 with 22.9 GB free; 12000+4000 frames ~= 12.6 GB buffer + ~2 GB model.
Do NOT leave online_buffer_capacity at 15000 with cuda storage — 14.9 GB is too tight.
If it OOMs, drop to 10000. If Hz does not improve, revert; it is the transfer or not.

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
