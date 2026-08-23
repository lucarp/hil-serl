# Resume here — 2026-08-23

## State: everything is ready to train. One physical step, then two commands.

Both arms were unplugged at 18:09 (kernel confirmed two USB disconnects), which is
the only reason training did not start. Nothing is broken.

### 1. Plug the follower back in — USB **and** its 12 V supply. Then check:

```bash
ls -l /dev/so101_follower        # must exist (udev symlink by adapter serial)
lsusb | grep 1a86                # must list the CH343 adapter
```

The leader can stay unplugged — HIL-SERL never uses it. Also connect the Xbox pad.

Symptom decoder, both seen today:
- `There is no status packet!` on all 6 motors → servos unpowered (12 V), USB is fine
- `Incorrect status packet!` → a real bus glitch; retry once before investigating

### 2. Start the learner

```bash
cd /mnt/Storage/projects/hil-serl
set -a && . ./.env && set +a && source .venv/bin/activate
rm -rf outputs/hilserl_cube_run1 outputs/hilserl_cube_run1_actor   # or bump job_name
python -m lerobot.rl.learner --config_path configs/real/train_config.json
```

### 3. Start the actor, in a second terminal

```bash
cd /mnt/Storage/projects/hil-serl
set -a && . ./.env && set +a && source .venv/bin/activate
python -m lerobot.rl.actor --config_path configs/real/train_config.json \
  --output_dir=outputs/hilserl_cube_run1_actor
```

The `--output_dir` override is required: both processes validate the same config
and the actor refuses to start on the directory the learner just created.

**If the actor dies, restart the learner too.** `learner_service.py` sets
MAX_WORKERS = 3, exactly the number of long-lived RPCs one actor opens; a
half-open connection from a dead actor holds all three forever and the next
actor silently fails to connect.

## Controls during training

Hold **RB** to take over · left stick X/Y · right stick vertical Z ·
**LT** close gripper · **RT** open · **Y** success · **B** failure · **A** rerecord.
The on-screen help is Logitech-ordered — ignore it.

Reward for this run is the **Y button**: no classifier yet, so you are the reward
function for the whole session.

## What is trained

Task is the **white cube**, not the ball — the ball was too slippery to grasp.
Dataset `lucarp/so101_ball_circular_hole_cropped` (24 episodes, 3472 frames,
24/24 successes) is misnamed for that reason; rename before it becomes a habit.

## Known follow-ups

- Green ball sits in-frame in every demo as a bystander. Leave it exactly where it
  is during training, or the scene stops matching the demos.
- TPU gripper pads for the ball task will change what the wrist camera sees, so any
  policy trained now must run without them. Decide before recording the ball task.
- Demos average 14.5 s with ~60% zero-motion frames. Record the next task brisker.
- Images are 128x128 because `resize_size` was set at record time; the full 640x480
  was discarded. For the next task, record unresized and crop from full resolution.
