# hil-serl — working notes

Reproducing HIL-SERL on a real SO-101. LeRobot is a pinned submodule at `vendor/lerobot`.
Full protocol: `docs/2026-08-01-hilserl-so101-protocol.md`. Guide: `docs/hilserl-so101-guide.md`.

## Environment

**Always `source .venv/bin/activate`** — never call `.venv/bin/python` directly. The activate
script exports the `LD_LIBRARY_PATH` that lets torchcodec find `libnppicc.so.12`
(`nvidia-npp-cu12`); skipping it makes video decode fail.

The venv installs lerobot **editable** from the submodule, which must stay on branch
**`local/hilserl`** (= PR #4297 fix + Xbox Z-axis one-liner; **never push it**). PR branches
in the submodule: `fix/reset-config-joint-positions-type` (#4297, open),
`feat/gamepad-controller-profiles` (parked, design in protocol §4.1). Xbox pad controls:
left stick X/Y, right stick vertical Z, **LT/RT = close/open gripper** (View/Menu also work),
hold RB = intervene (**required every step** - without it the pipeline records the neutral
action `[0,0,0,1]` and the dataset is silently worthless),
Y/B/A = success/failure/rerecord (on-screen help text is Logitech-ordered — ignore it).

## Stable device names

Installed by `system/install_udev.sh` (rules in `system/99-so101-hilserl.rules`):

| Symlink | Device |
|---|---|
| `/dev/so101_leader` | leader arm, CH343 serial `5B61033417` |
| `/dev/so101_follower` | follower arm, CH343 serial `5B3E090551` |
| `/dev/cam_wrist` | Sonix module |
| `/dev/cam_scene` | Lenovo Performance Camera, **RGB interface only** (interface 02 is IR) |

Calibration ids are `follower_arm` / `leader_arm` (files under
`~/.cache/huggingface/lerobot/calibration/{robots/so_follower,teleoperators/so_leader}/`).
Type strings stay `so101_follower` / `so101_leader` — aliases on the unified `SOFollower`/`SOLeader`.

## Teleoperate with both camera views

```bash
cd /mnt/Storage/projects/hil-serl
source .venv/bin/activate
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/so101_follower \
  --robot.id=follower_arm \
  --robot.cameras='{ wrist: {type: opencv, index_or_path: /dev/cam_wrist, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: 200}, scene: {type: opencv, index_or_path: /dev/cam_scene, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: 200} }' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/so101_leader \
  --teleop.id=leader_arm \
  --fps=30 \
  --display_data=true
```

Opens a rerun viewer with both camera streams plus joint traces. Ctrl-C to stop; the follower
torque is disabled on disconnect. Add `--teleop_time_s=300` to bound the run (Phase 2 gate is
5 unbroken minutes with no `ENOSPC` and no frame timeouts).

Notes on that command:

- `backend: 200` is `Cv2Backends.V4L2`. **Required.** With the default `ANY`, OpenCV picks FFMPEG
  on Linux and *silently ignores* `fourcc: MJPG`, falling back to uncompressed YUYV
  (lerobot issue #3198). Both `fourcc` and `backend` must be set together.
- Camera keys `wrist` / `scene` match the existing dataset convention — keep them.
- `--fps=30` matches the cameras; the script's default is 60, which just repeats frames.
- Align the leader roughly to the follower's current pose before starting, or the follower snaps
  to it on the first `send_action`. `--robot.max_relative_target=<deg>` caps per-step motion if
  a slower, safer first move is wanted.
- Needs X11 (`XDG_SESSION_TYPE=x11`), not Wayland.

## Pitfalls that cost the most (see protocol §5 for the full register)

- Install extras are `.[hilserl,feetech,gamepad,core_scripts,training]`, **not** `.[all]`
  (pulls LIBERO → CMake build) and **not** `.[hilserl]` alone (no motor SDK, no gamepad).
- HIL-SERL configs are 10 Hz EE-delta space, not 30 Hz joint space. `fps: 10` must be set
  explicitly — the base `EnvConfig` default of 30 is wrong for every shipped example.
- `env.processor.reset.fixed_reset_joint_positions` must exactly match the pose demos start
  from. Mismatch is silent and expensive.
- Module paths are `lerobot.rl.*` (tutorials still say `lerobot.scripts.rl.*`).
