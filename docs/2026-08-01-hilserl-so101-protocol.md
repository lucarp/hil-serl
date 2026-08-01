# HIL-SERL on SO-101 — Setup & Training Protocol

**Author:** Lucas · **Date:** 2026-08-01
**Goal:** Reproduce HIL-SERL on a real SO-101 arm as a faithful baseline, then extend.
**Status:** Plan approved, not yet executed.

> This is a protocol, not a tutorial transcript. Where it disagrees with
> <https://huggingface.co/docs/lerobot/hilserl>, the disagreement is deliberate and the
> reason is stated. Every claim marked ✅ was verified against source in a local checkout
> on 2026-08-01; claims marked ⚠️ are unverified and must be settled by testing.

---

## 0. What you are actually building

HIL-SERL is not "record demos, then train." It is an **online, off-policy RL loop running on
physical hardware, with a human in it**. Three processes matter:

1. **Learner** — holds the replay buffers, runs SAC gradient steps, serves weights over gRPC.
2. **Actor** — owns the robot and cameras, executes the current policy at 10 Hz, streams
   transitions back to the learner.
3. **You** — watching, and taking over the instant the policy is about to fail.

Your corrections enter the same replay buffer as the policy's own experience. This is the
whole trick: the human supplies the on-policy corrective data that random exploration would
take millions of steps to find. The paper's ablation is the clearest statement of why you
cannot skip any leg of it:

| Configuration | Success rate |
|---|---|
| No demos, no interventions | **0%** |
| Demos only | 49% |
| Demos + interventions | **100%** |

*(Luo et al. 2024, averaged over three ablated tasks.)*

**Therefore the metric that matters is not reward. It is your intervention rate falling over
time.** A run where reward is high because you kept rescuing the policy is a failed run.

### The one counter-intuitive rule

From the paper (§3.4), verbatim:

> we should avoid persistently providing long sparse interventions that lead to task
> successes. Such an intervention strategy will cause the overestimation of the value
> function, particularly in the early stages of the training process; which can result in
> unstable training dynamics.

**Intervening all the way to success actively damages the Q-function.** Short corrections
that put the policy back on track, then hands off. This is the single most common way people
waste a week.

---

## 1. Verified starting state

Checked on this machine, 2026-08-01:

| Item | State |
|---|---|
| OS | Ubuntu 24.04.4, kernel 6.8.0-136 |
| GPU | RTX 4090 24 GB, driver 580.173.02 — clears both cu128 (570.86) and cu130 (580.65) floors |
| CPU / RAM | 12 cores / 61 GB |
| Disk | 358 GB free on `/mnt/Storage` |
| Python tooling | `uv` 0.11.16, `pyenv` (3.12.7 available), **no conda**, system python 3.13 |
| ffmpeg | 6.1.1 system-wide |
| Session | **X11** (not Wayland) — matters for keyboard teleop |
| Serial perms | user in `dialout` ✅ — no `chmod` needed |
| Cameras | 2 USB (ARC International, Lenovo Performance) — **both on Bus 003, one 480 Mbit/s hub** ⚠️ |
| Arms | Assembled + calibrated, teleop working. **Not currently plugged in.** |
| Calibration | `~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower_arm.json`, `.../teleoperators/so_leader/leader_arm.json` |
| Gamepad | Xbox controller(s) on hand. (`/dev/input/js0` is an ASRock RGB controller — ignore it.) |
| Existing LeRobot | `/mnt/Storage/projects/robot-arm/lerobot` @ `8194897`, v0.5.2, 2026-05-22. Working venv, torch 2.11.0+cu128, CUDA ✅. **`placo` and `grpcio` absent** → cannot run HIL-SERL. |
| Existing dataset | `lucarp/so101_duck_20260602_221736` — 30 fps, joint space, cameras `wrist` + `scene`, 480×640, AV1 |

**Calibration ids to reuse in every command and config: `follower_arm` and `leader_arm`.**
Robot/teleop type strings remain `so101_follower` / `so101_leader` (registered as aliases on
the unified `SOFollower`/`SOLeader` classes) ✅.

### What does not carry over

The duck dataset is **not reusable as HIL-SERL demonstrations**. It is 30 Hz joint-space; HIL-SERL
is 10 Hz end-effector-delta space. Keep it as a reference for camera framing and nothing else.

---

## 2. Decisions taken, and why

| Decision | Choice | Reasoning |
|---|---|---|
| Repo layout | `hil-serl/` is *your* repo; LeRobot is a **pinned submodule** | The submodule SHA *is* your baseline. Reproducible by construction; swap the remote to your fork when extending. |
| Pin target | A **recent `main`**, not the local May commit | The May checkout's `placo-dep` lacks the `cmeel-urdfdom`/`cmeel-tinyxml2` pins added upstream for issue #3755. Without them `import placo` can fail *after a clean install*, which kills IK, which kills HIL-SERL. |
| Python | **3.12** (pyenv 3.12.7) | `requires-python = ">=3.12"`. 3.13 is nominally supported but `placo`, `feetech-servo-sdk`, `torchcodec`, `mujoco` are exactly the packages most likely to lack 3.13 wheels. Zero community precedent on 3.13. |
| Env manager | **uv**, not conda | Already installed; cleaner CUDA selector. Conda's sole advantage is `conda install ffmpeg -c conda-forge` — keep as fallback, don't lead with it. |
| Install extras | **`.[all]`**, not `.[hilserl]` | `hilserl = [transformers-dep, dataset, gym-hil, grpcio-dep, placo-dep]` — **no motor SDK, no gamepad, no pynput**. Following the docs literally guarantees failure at first hardware contact. |
| Torch backend | `--torch-backend cu128` | What upstream CI exercises; 10 driver versions of headroom vs cu130's 0.5. |
| Task | **Pick and place into a container** | Chosen. Slightly beyond the docs' validated "pick and lift", still forgiving. See §7 for the risk this adds. |
| Action space | **EE delta (Δx, Δy, Δz) + discrete gripper** | Not a choice — it is what the pipeline implements (`hil_processor.py:499-504`). No wrist reorientation available. |
| Intervention device | **Xbox gamepad** | The only device that satisfies both teleop contracts on stock code. |
| Reward, run #1 | **Manual annotation** | Docs explicitly bless it. Removes ~1.5 h and an entire class of silent failure. Classifier on run #2. |
| Leader arm | **Deferred to extension phase** | See §3. |

### Why the leader arm is not the intervention device ✅

Verified in the local checkout. A HIL-SERL teleoperator must satisfy two contracts; `SOLeader`
satisfies neither:

1. **`get_teleop_events()`** → `{is_intervention, terminate_episode, success, rerecord_episode}`.
   Only `GamepadTeleop` and `KeyboardEndEffectorTeleop` implement it. `SOLeader` defines 13
   methods, none of them this one. `hil_processor.py:89` raises
   `TypeError: Teleoperator SOLeader must implement get_teleop_events() method`.
2. **`get_action()`** → must return `{delta_x, delta_y, delta_z, gripper}`. `SOLeader.get_action()`
   returns **joint positions**. Bridging needs FK on the leader's joints → EE pose → delta vs a
   reference, plus scaling.

So the tutorial's "Setting up the SO101 leader" section documents a path that does not run.
Upstream issue [#2952](https://github.com/huggingface/lerobot/issues/2952) is open; PR #3086 is
unmerged.

**This is your first extension, not your starting point.** Building it means writing an
`SOLeaderEndEffector` teleoperator (placo gives you the FK) composed with a keyboard event
source. Tractable in days, and a real upstream contribution. But doing it before you have ever
run the pipeline means debugging your own new code against a stack you have never seen work.

---

## 3. Phases and gates

Do not pass a gate without the stated evidence. Each gate exists because skipping it produces a
silent failure hours later.

### Phase 0 — Repository and environment

**Target layout:**

```
hil-serl/
├── vendor/lerobot/          # submodule, pinned SHA
├── .venv/                   # uv, python 3.12
├── configs/                 # your JSON configs (NOT inside the submodule)
├── assets/SO-ARM100/        # URDF source
├── docs/                    # this file + lab notebook
├── runs/                    # gitignored: checkpoints, wandb, datasets
└── README.md
```

Keep configs **outside** `vendor/lerobot` so `git pull` on the submodule never touches them.

**Gate — all four must pass:**
```
python -c "import placo; print('placo ok')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from torchcodec.decoders import VideoDecoder; print('decoder ok')"
python -c "import grpc, pygame, feetech_servo_sdk; print('hw ok')"
```

`import placo` is the one that bites on Ubuntu 24.04 (issue #3755). **Result: passed** — the
`cmeel-urdfdom`/`cmeel-tinyxml2` pins present in v0.6.1 do their job.

**Do not use `.[all]`.** It pulls `hf-libero` → `robomimic` → `egl-probe`, which builds from
source and requires CMake. LIBERO is an imitation-learning simulation benchmark irrelevant to
HIL-SERL. Use `.[hilserl,feetech,gamepad,core_scripts,training]` — same capability, no CMake.

**torchcodec / NPP — Q2 resolved, and the answer was not what the question assumed.** ffmpeg
6.1.1 is fine; `libtorchcodec_core6.so` resolves every FFmpeg 6 library correctly. The actual
failure is that torchcodec's CUDA build links **NVIDIA NPP (`libnppicc.so.12`)** while
(a) declaring no dependency on `nvidia-npp-cu12`, and (b) shipping **no RPATH** — compare
`libtorch_cuda.so`, which carries `$ORIGIN/../../nvidia/*/lib` for exactly this reason. With no
system CUDA toolkit present, the loader cannot find it. Fix applied:

```
uv pip install nvidia-npp-cu12          # provides libnppicc.so.12
# + LD_LIBRARY_PATH export appended to .venv/bin/activate
```

**Consequence: always `source .venv/bin/activate`.** Calling `.venv/bin/python` directly skips
the export and torchcodec will fail again. Fallbacks if this ever regresses: CPU torchcodec build
(GPU decode is irrelevant here — the offline buffer is 15–30 episodes loaded once), or
`--dataset.video_backend=pyav`.

**Note:** the example configs the docs reference (`src/lerobot/configs/env_config_so100.json`
etc.) **do not exist in the repo** — verified. Get them from
`https://huggingface.co/datasets/lerobot/config_examples/`. Avoid the older
`aractingi/lerobot-example-config-files`, still linked from the docs but predating several refactors.

---

### Phase 1 — Simulation smoke test **(do not skip)**

Run the full actor/learner stack against `gym_hil`'s Panda pick-cube task. No hardware at risk.

Configs: `lerobot/config_examples` → `rl/gym_hil/env_config.json`, `rl/gym_hil/train_config.json`.

```bash
# terminal 1
python -m lerobot.rl.learner --config_path configs/gym_hil_train.json
# terminal 2
python -m lerobot.rl.actor   --config_path configs/gym_hil_train.json
```

**This phase exists to answer four questions cheaply:**
- Does gRPC actor↔learner work on this box?
- **Does the Xbox mapping match?** (See §4 — it partly does not.)
- Does WandB log what you expect?
- What does a healthy reward curve look like, so you recognise an unhealthy one later?

**Gate:** reward curve rises; you have personally felt the intervene-and-release rhythm.

---

### Phase 2 — Hardware bring-up

1. **Move one camera to a USB-3 port.** Verified: both are on Bus 003 (480 Mbit/s); Buses 002
   and 004 (20 Gbit/s) are empty. Two 640×480 streams saturate one USB-2 hub → `VIDIOC_STREAMON:
   No space left on device`. Free, permanent fix.
2. **Write udev rules.** `/dev/ttyACM*` order and `/dev/videoN` indices both reshuffle across
   reboots and replugs. LeRobot's docs contain no udev guidance at all — their only advice is
   `sudo chmod 666`, which does not survive a replug. Key on VID:PID (+serial) to get stable
   names like `/dev/so101_follower`, `/dev/cam_scene`. **Do this before recording demos**, or
   every crop you compute becomes wrong after a reboot.
   For cameras note the `ATTR{index}=="0"` idiom — each UVC device exposes several video nodes
   and only one is real.
3. **Verify calibration still loads** against ids `follower_arm` / `leader_arm`.
4. Force MJPG **and** the V4L2 backend in camera config. With OpenCV's default `backend=ANY`
   on Linux, FFMPEG is chosen and **silently ignores `fourcc="MJPG"`**, falling back to
   uncompressed YUYV (issue #3198). The shipped config sets `fourcc` but not `backend`.

**Gate:** both cameras stream simultaneously at 640×480 MJPG 30 fps for 5 unbroken minutes.
Confirm with `v4l2-ctl --list-formats-ext` and `lsusb -t`.

---

### Phase 3 — URDF and workspace bounds

```bash
git clone https://github.com/TheRobotStudio/SO-ARM100.git assets/SO-ARM100
# → assets/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
```

No URDF ships with LeRobot ✅ — `kinematics.py:62` calls `placo.RobotWrapper(urdf_path)` on a path
you supply. The SO-ARM100 URDF is the one recommended in `lerobot_find_joint_limits.py`'s own
source comment.

⚠️ **Resolve before running:** is `target_frame_name` `"gripper"` or `"gripper_frame_link"`? The
script default, the shipped config, and the script's docstring disagree. Grep the link names in
the URDF and settle it.

```bash
lerobot-find-joint-limits \
  --robot.type=so101_follower --robot.port=/dev/so101_follower --robot.id=follower_arm \
  --teleop.type=so101_leader  --teleop.port=/dev/so101_leader  --teleop.id=leader_arm \
  --urdf_path=assets/SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
  --target_frame_name=<resolved> \
  --teleop_time_s=30
```

**The docs' version of this command omits `--urdf_path` and it has no default — as printed, it
fails.** This is also the one place your leader arm is genuinely useful right now: you physically
move it through the task space while the script records EE extrema.

**Gate:** a **deliberately tight** EE box. The shipped example is ~14×29×11 cm and that is
correct, not a bug. Bounds that are too large are the #1 reported cause of "the agent never sees
a reward." Accept up front that tight bounds limit generalization — that is a known, published
limitation of the method, and belongs in your thesis framing rather than being engineered away.

---

### Phase 4 — Record demonstrations

15–30 episodes, `mode: "record"`, at **`fps: 10`** (every shipped HIL-SERL example overrides the
base `EnvConfig` default of 30 — set it explicitly).

```bash
python -m lerobot.rl.gym_manipulator --config_path configs/env_record.json
```

**Gate — and this one is silent if you get it wrong:**
`env.processor.reset.fixed_reset_joint_positions` must produce **exactly** the pose your
demonstrations start from. A field report: *"I wasted hours debugging because my reset height was
z=0.03 but my demos were recorded at z=0.07."* Every episode then begins out of distribution,
with no error message.

Reset randomization that is known to work: **±1 cm** in x/y/z. Larger caused failures.

Budget realistically: you will reset this workspace **150–750 times**. If the reset is not a
scripted motion or a ≤1 s hand movement, redesign the task now.

---

### Phase 5 — Crop ROI

```bash
python -m lerobot.rl.crop_dataset_roi --repo-id <you>/<dataset>
```

Draw the box, press `c`. Outputs crop params and a new cropped dataset. Then set
`resize_size: [128, 128]` (validated; 64×64 if you need VRAM).

**Why this matters more than it looks:** a randomly-initialised critic bootstrapping on 128×128
pixels will happily fit spurious background correlates of reward. Because the reward is *itself*
a function of the same pixels once you add a classifier, lighting drift corrupts reward and value
simultaneously. One field report logged an entire failure mode as *"Robot either follows shadows
or shadows follow robot."*

**Gate:** nothing outside the workspace is in frame. Windows excluded. A dominant desk lamp
strong enough to overpower daylight. Cameras rigidly mounted **outside the arm's swing envelope**
— they get hit, and a bumped camera invalidates every crop.

---

### Phase 6 — Train

```bash
# terminal 1
python -m lerobot.rl.learner --config_path configs/train_hilserl.json
# terminal 2
python -m lerobot.rl.actor   --config_path configs/train_hilserl.json
```

**Mandatory overrides to the shipped `train_config.json`:**

| Field | Shipped | Set to | Why |
|---|---|---|---|
| `policy.actor_learner_config.policy_parameters_push_frequency` | `50` | `2` | The field is in **seconds** (`learner.py:420`). Class default is 4; docs recommend 1–2. At 50 s and 10 Hz the actor runs ~500 steps on stale weights between pushes. Learner and actor are on loopback here — network cost is nil. |
| `online_buffer_capacity` / `offline_buffer_capacity` | 30000 / 20000 | 10000 / 10000 | `ReplayBuffer._initialize_storage` allocates **float32**, not uint8. Two 128×128×3 cameras on `storage_device: "cuda"` ⇒ ~19.7 GB **before** weights, optimizer state, or activations. Does not fit 24 GB. |
| `policy.storage_device` | `"cpu"` | `"cuda"` | Only *after* the capacity fix. Removes CPU→GPU transfer per update. Fallback to `"cpu"` is safe — you have 61 GB RAM — at a throughput cost. |

**Starting hyperparameters** (class defaults are *not* usable — they differ sharply from every
working config):

| Field | Class default | Use |
|---|---|---|
| `temperature_init` | 1.0 | **0.01** — too high makes interventions ineffective |
| `utd_ratio` | 1 | **2** — matches the reference implementation |
| `discount` | 0.99 | **0.97** |
| `grad_clip_norm` | 40.0 | **10.0** |
| `gripper_penalty` | 0.0 | **-0.02** |
| `end_effector_step_sizes` | — | ~`{x:0.009, y:0.009, z:0.01}` (≈9–10 mm/step at 10 Hz) |

Also: `policy.type` must be **`"gaussian_actor"`**, not `"sac"` — renamed by PR #3075. The
*algorithm* key stays `"sac"`. A config saying `policy.type: "sac"` fails to parse.

**Operational rule:** when the actor dies (and it will — camera drops are common), **restart the
learner too, every time.** `learner_service.py` sets `MAX_WORKERS = 3`, exactly the number of
long-lived RPCs one actor opens, and `StreamParameters` never checks `context.is_active()`. A
half-open connection from a crashed actor holds all three threads forever, and the new actor
blocks silently at `Send ready message to Learner` with no error and no timeout (issue #3979).

**Gate / success criterion:** **intervention rate decays** in WandB. Not reward.

---

### Phase 7 — Evaluate

Do **not** build the evaluation protocol on `lerobot.rl.eval_policy` — strong static evidence it
is broken (`eval_policy.py:58` calls `make_robot_env(env_cfg)`, which returns a 2-tuple since the
refactor, and reads an attribute absent from both config classes). Use `lerobot-rollout`, and
define N trials, a fixed initial-state distribution, and a success criterion **before** running.

---

## 4. Xbox gamepad mapping ⚠️

`gamepad_utils.py` hardcodes pygame indices for a Logitech F710. Against standard Xbox/`xpad`/SDL:

| Function | LeRobot index | Xbox actual | Status |
|---|---|---|---|
| Intervene | button 5 | RB | ✅ |
| Success | button 3 | Y | ✅ |
| Failure | button 1 | B | ⚠️ works, label differs |
| Rerecord | button 0 | A | ⚠️ works, label differs |
| Close gripper | button 6 | **Back** | ❌ |
| Open gripper | button 7 | **Start** | ❌ |
| Z axis | axis 3 | **right stick X** (Z is axis 4) | ❌ |

`GamepadTeleopConfig` exposes only `use_gripper` — no mapping fields, so this means editing
`gamepad_utils.py`. **Verify empirically in Phase 1**; the table above is inference from the
standard mapping, not a measurement.

---

## 5. Pitfall register (the ones that apply here)

Ordered by expected cost.

1. **`.[hilserl]` alone** — no motor SDK, no gamepad. Certain failure if you follow the docs. → `.[all]`
2. **`import placo` fails on Noble** — issue #3755. Never hand-install placo; use the pinned extra.
3. **VRAM OOM** — §Phase 6 table. Watch `nvidia-smi` for the first 100 learner steps. ⚠️ unconfirmed empirically.
4. **Reset pose ≠ demo pose** — silent, expensive, trivially preventable.
5. **Both cameras on one USB-2 hub** — confirmed on this machine. Move one.
6. **Long interventions to success** — corrupts the Q-function. Short corrections only.
7. **`policy_parameters_push_frequency: 50`** — seconds, not steps.
8. **Actor restart hangs** — restart the learner too, always.
9. **`/dev` name churn** — udev rules before demos, not after.
10. **Lighting drift** — dominant lamp, aggressive crop.
11. **`wrist_roll` calibration crash** — `ValueError: Magnitude 2073 exceeds 2047` (issue #3193). Rotate `wrist_roll` to raw ≈2048 *before* `lerobot-calibrate`. (Not needed unless you recalibrate.)
12. **STS3215 thermal cutout** — torque disables around 70 °C; HIL-SERL runs 500–750 back-to-back episodes, exactly that duty cycle. Looks like a policy failure, is a hardware failure. ⚠️ figures are vendor-sourced; monitor temperature directly.
13. **Stale-tutorial schema errors** — `lerobot.scripts.rl.*` moved to `lerobot.rl.*`; flat env config → nested `env.processor.*`; reward classifier under `reward_model`, not `policy`.
14. **SO-101 is not the platform HIL-SERL was developed on** — it was built on SO-100 and Koch. Issue #1387 (open) reports SO-101 state-dim mismatch and an IK state-caching bug. One reproducer's summary: *"basically nothing worked out of the box."* **Budget three weeks, not three days.**

---

## 6. Open questions — resolve by testing

| # | Question | How to settle |
|---|---|---|
| Q1 | Is `target_frame_name` `"gripper"` or `"gripper_frame_link"`? | Grep link names in `so101_new_calib.urdf` |
| ~~Q2~~ | ~~Is ffmpeg 6.1.1 ABI-compatible with the shipped torchcodec?~~ | **RESOLVED 2026-08-01.** ffmpeg was never the problem — missing NVIDIA NPP was. See Phase 0. |
| Q3 | Does the shipped config actually OOM on this 4090? | `nvidia-smi` during first 100 learner steps |
| Q4 | Exact Xbox button/axis indices? | Phase 1, in sim |
| Q5 | Does the SO-101 state-dim mismatch (#1387) affect this version? | First `gym_manipulator` run with real hardware |

---

## 7. Task-choice note

You chose **pick and place into a container**. The docs' validated first task — and what both
published SO-101 reproductions did, each reaching ~70% — is **pick and lift**. Pick-and-place adds:

- a longer horizon (keep it ≤10 s at 10 Hz ⇒ ≤100 steps)
- **two-stage success**, which a reward classifier must distinguish ("grasped" vs "placed")
- a container that must be inside the tight EE bounds *and* inside the crop

This is a reasonable target and not reckless. But since Phase 6 is where weeks disappear, consider
running the pipeline end-to-end on **pick-and-lift first** — same hardware, same cameras, same
crop, same configs, one less failure mode — and switching the task once the machinery is proven.
That converts pick-and-place from "did the pipeline work?" into "did the task work?", which is a
far cheaper question to debug.

---

## 8. Sources

- Tutorial (local, matches code): `vendor/lerobot/docs/source/hilserl.mdx`
- Rendered: <https://huggingface.co/docs/lerobot/hilserl> · <https://huggingface.co/docs/lerobot/hilserl_sim>
- Config examples: <https://huggingface.co/datasets/lerobot/config_examples/>
- URDF: <https://github.com/TheRobotStudio/SO-ARM100>
- Paper: Luo, Xu, Wu, Levine (2024), *Precise and Dexterous Robotic Manipulation via
  Human-in-the-Loop Reinforcement Learning*, arXiv:2410.21845 · <https://hil-serl.github.io>
- Lineage: SERL; RLPD (Ball et al., symmetric sampling, LayerNorm, high UTD)
- Issues referenced: [#2952](https://github.com/huggingface/lerobot/issues/2952) (leader teleop),
  [#3755](https://github.com/huggingface/lerobot/issues/3755) (placo/Noble),
  [#3198](https://github.com/huggingface/lerobot/issues/3198) (MJPG/backend),
  [#3979](https://github.com/huggingface/lerobot/issues/3979) (actor restart hang),
  [#1387](https://github.com/huggingface/lerobot/issues/1387) (SO-101 support),
  [#3193](https://github.com/huggingface/lerobot/issues/3193) (wrist_roll calibration),
  [#2431](https://github.com/huggingface/lerobot/issues/2431) (find-joint-limits urdf_path)

**Note on LeRobot's SAC:** it is a deliberately de-tuned RLPD — 2 critics not 10, UTD 1–2 not 20,
γ=0.97 not 0.99, α₀=1e-2 not 1.0. LayerNorm is the one RLPD ingredient kept at full strength.
High UTD is unaffordable at 10 Hz on real hardware; human interventions substitute for the sample
efficiency it would buy. Two undocumented divergences from the reference implementation worth an
ablation if training is unstable: `use_backup_entropy=True` (reference: `False`) and `nn.SiLU()`
(reference: `tanh`). **That is a legitimate first paper contribution.**
