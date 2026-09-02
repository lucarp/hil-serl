# HIL-SERL on a real SO-101

Reproducing [HIL-SERL](https://hil-serl.github.io/) (human-in-the-loop sample-efficient
reinforcement learning) on a physical [SO-101](https://github.com/TheRobotStudio/SO-ARM100)
arm, on top of [LeRobot](https://github.com/huggingface/lerobot).

The robot learns a manipulation task directly on hardware, from a handful of
demonstrations plus corrections made with a gamepad while it is training. There is no
simulator anywhere in this loop.

Four bugs found along the way are now open pull requests against LeRobot
([#4558](https://github.com/huggingface/lerobot/pull/4558),
[#4560](https://github.com/huggingface/lerobot/pull/4560),
[#4561](https://github.com/huggingface/lerobot/pull/4561),
[#4562](https://github.com/huggingface/lerobot/pull/4562)).

## Result

Task: pick a 3 cm cube off the table and lift it clear. Binary reward, given by the
operator on success.

Starting point was 30 teleoperated demonstrations (1913 frames, about 3 minutes of
robot time). After that, everything the policy learned came from its own attempts and
from corrections issued mid-episode.

Measured from the learner's own replay buffer at 48,000 gradient steps, over the 240
episodes it still holds:

| | |
|---|---|
| episodes run with under 5% human input | 63 |
| of those, reached the reward | 39 (62%) |
| best 20-episode window | 7 of 8 autonomous episodes succeeded |
| demonstrations used | 30 |

Two caveats worth stating. These numbers come from the training buffer, not from a held-out
evaluation, so they mix in whatever the policy happened to be doing at the time rather than
measuring it under fixed conditions. And the cube always started in roughly the same place;
generalisation to new positions was never tested. Getting even this far took considerably
more debugging of the pipeline than tuning of the algorithm.

## How it runs

Two processes talk over gRPC:

- **learner** holds the replay buffers and does SAC updates on the GPU
- **actor** drives the robot at 10 Hz, sends transitions to the learner, and pulls fresh
  weights back every 2 gradient steps

Actions are end-effector deltas (dx, dy, dz) plus a discrete gripper command. Observations
are two 128x128 camera crops (wrist and scene) and the six servo angles. The vision encoder
is a frozen `lerobot/resnet10`, so only about 891K parameters are actually learned.

Following RLPD, every batch is half demonstrations and half online experience
(`online_ratio: 0.5`). Corrections are written to *both* buffers, which is the whole point
of the method and also, as it turns out, the source of its most interesting failure mode.

Holding RB on the gamepad takes over; releasing it hands control back. The operator holds
it whenever the robot is about to do something stupid, and lets go otherwise.

## What is in here

```
configs/real/          training, recording and workspace configs (the real inputs)
docs/                  protocol, study plan, and the write-ups below
tools/inspect_dataset.py   read any LeRobot dataset: episodes, actions, joints, frames
system/                udev rules giving the arms and cameras stable device names
vendor/lerobot         pinned LeRobot submodule
NEXT.md                open landmines, read this before restarting anything
RESUME_COMMANDS.md     the runbook for resuming a training session
```

`configs/real/workspace_bounds.json` is worth a look: the end-effector box is derived from
the demonstrations by forward kinematics rather than guessed. An earlier hand-picked box
included the fully-extended pose, where a 5-DoF arm is near-singular, and the arm shook
itself around trying to reach commanded positions it could not hold.

## Two findings worth reading

### The intervention feedback loop

[`docs/2026-09-02-intervention-feedback-loop.md`](docs/2026-09-02-intervention-feedback-loop.md)

A policy improves, plateaus, then degrades, and the degradation is caused by the operator
helping. Intervention frames go into a fixed-size FIFO buffer, so as the operator rescues
more episodes the policy's own *autonomous successes* are the oldest entries and get
evicted first. Half of every batch then comes from a buffer that no longer contains any
record of the policy succeeding on its own, so it depends on the operator more, so the
operator intervenes more.

Nothing errors. Episode reward *rises* while this happens, because the operator is
rescuing episodes. In the worst window measured, overall success read 90% while autonomous
success was 0%.

Observed twice, on two different tasks. The metric that catches it is autonomous success
in windows of about 15 episodes, computed from the buffer dump, since no logged scalar
shows it.

### A 255x scaling bug that made half of every batch noise

`make_dataset` loads frames with `return_uint8=True`, and `ReplayBuffer._initialize_storage`
pre-allocates with `torch.empty`, which is float32. Assigning uint8 `[0, 255]` into a
float32 tensor gives float `[0.0, 255.0]`, and nothing on that path divides by 255. Those
values then reach a normalizer expecting `[0, 1]` with ImageNet statistics.

So on a fresh run, the demonstration half of every batch, which is the anchor RLPD depends
on, was noise. This explains a completely flat learning curve I could not otherwise account
for. Fixed on both the dump and the load path
([#4558](https://github.com/huggingface/lerobot/pull/4558)).

## Upstream pull requests

| PR | Problem |
|---|---|
| [#4558](https://github.com/huggingface/lerobot/pull/4558) | Replay-buffer dumps write float images in `[0, 255]`, which kills the learner at the first checkpoint |
| [#4560](https://github.com/huggingface/lerobot/pull/4560) | Resuming a run silently re-initialises the policy at random, keeping only the critic and stale optimizer state |
| [#4561](https://github.com/huggingface/lerobot/pull/4561) | `crop_dataset_roi` never calls `finalize()` and drops an episode boundary, producing a dataset that raises `IndexError` on load |
| [#4562](https://github.com/huggingface/lerobot/pull/4562) | Checkpointing deletes the previous replay-buffer dataset before writing the new one, so an interrupted checkpoint is unrecoverable |

[#4560](https://github.com/huggingface/lerobot/pull/4560) is the one that cost the most.
`make_policy` branches on `cfg.pretrained_path`, and the resume path returned a config
parsed from the checkpoint JSON where that field is `null`, so the `else` branch built a
fresh policy. The actor weights live only in `pretrained_model/model.safetensors`; the
`algorithm/` checkpoint holds critic, targets and `log_alpha` and zero actor tensors. A
resume therefore gave a random actor with a trained critic, which is worse than starting
over. Verified after the fix by cosine similarity of flattened policy weights across a
resume boundary: 0.968, against 0.975 for two consecutive checkpoints inside one session.

## Hardware

- SO-101 leader and follower arms, Feetech STS3215 servos
- two USB cameras, wrist and scene, 640x480 at 30 fps, MJPG
- Xbox gamepad for demonstrations and interventions
- one consumer GPU, which the learner needs to itself

`system/install_udev.sh` installs rules giving `/dev/so101_leader`,
`/dev/so101_follower`, `/dev/cam_wrist` and `/dev/cam_scene`, because USB enumeration
order is not stable and plugging things in in a different order otherwise silently swaps
the arms.

## Running it

```bash
git clone git@github.com:lucarp/hil-serl.git && cd hil-serl
git submodule update --init vendor/lerobot     # see the note below
uv venv && source .venv/bin/activate
uv pip install -e "vendor/lerobot[hilserl,feetech,gamepad,core_scripts,training]"
sudo bash system/install_udev.sh
```

The submodule is pinned to a commit on a local branch that is not published anywhere, so
the `git submodule update` above will not resolve it as-is. Point it at upstream LeRobot
`main` and apply the four pull requests listed below, or wait for them to land. The extras
list matters: `[all]` pulls LIBERO and a CMake build, and `[hilserl]` alone gives you
neither the motor SDK nor gamepad support.

Then, in order: calibrate both arms, verify teleoperation (see `CLAUDE.md`), derive the
workspace box, record demonstrations, crop the camera views, and start the two training
processes. `RESUME_COMMANDS.md` has the exact commands, and
`docs/2026-08-01-hilserl-so101-protocol.md` has the full protocol including the failure
register.

Some things that are easy to get wrong and expensive to discover:

- HIL-SERL configs run at 10 Hz in end-effector delta space. `fps: 10` must be set
  explicitly, because the base `EnvConfig` default of 30 is wrong for every shipped example.
- `fixed_reset_joint_positions` must match the pose the demonstrations start from. A
  mismatch is silent.
- The OpenCV camera backend must be set to V4L2 explicitly. With the default, OpenCV picks
  FFMPEG on Linux and silently ignores `fourcc: MJPG`, falling back to uncompressed YUYV
  ([lerobot#3198](https://github.com/huggingface/lerobot/issues/3198)).
- RB must be held for every step of a demonstration. Release it and the pipeline records
  the neutral action `[0, 0, 0, 1]`, giving a dataset that looks perfectly valid and is worthless.

## Status and next steps

The lift task works. Open items, in `NEXT.md`:

- a proper evaluation, 30 to 50 hands-off episodes with a confidence interval, rather than
  counting autonomous successes out of a training buffer
- promoting the policy's own autonomous successes into the offline buffer, where nothing
  evicts them, as a direct fix for the feedback loop above
- an A/B test of that loop: identical runs from one checkpoint, one where the operator
  rescues a failing policy and one where failures are allowed to stand

## Notes

LeRobot is pinned as a submodule on a local branch carrying the four fixes above plus two
local-only changes (gamepad axis mapping and trigger handling for an Xbox pad, and a
relaxed safety check). See `CLAUDE.md` for what is local and why.

Parts of this work were done with AI assistance, disclosed in each pull request per
LeRobot's AI policy. The diagnoses and fixes were verified against real hardware runs.
