# HIL-SERL — copy-paste commands for one training session

Resumes from `outputs/hilserl_cube_run1/checkpoints/last` (currently step 36,000,
wandb run `opp47yi6`) — and since 2026-08-28 also restores the online (12,000 frames)
and offline (10,347 frames) replay buffers, so startup takes a couple of extra minutes.
Run the blocks in order, each in a normal terminal
(NOT the opencode/Claude Code terminal — that one is what you are killing in step 0).

## 0. Stop the LLM (frees the 21.7 GB of GPU) — do this FIRST

llama-server is a systemd user service (`llama-server.service`, Restart=on-failure).
A plain `pkill` makes systemd restart it, and it grabs the GPU back — stop the unit:

```bash
systemctl --user stop llama-server
systemctl --user reset-failed llama-server 2>/dev/null
nvidia-smi
```

`nvidia-smi` must show no processes (or ~0 MiB used) before training.
If the GPU is still busy, wait a few seconds and check again (the process needs a
moment to release its context). Stopping the LLM also kills the opencode/Claude
Code session — expected.

## 1. Preflight — robot + pad

```bash
ls /dev/so101_follower && lsusb | grep -q 045e && echo pad ok
```

Must print `pad ok`. Follower needs USB **and** 12 V. No `pad ok` → check power
and the base loom before starting anything else.

## 2. Environment — paste this in EVERY new terminal

```bash
cd /mnt/Storage/projects/hil-serl
set -a && . ./.env && set +a && source .venv/bin/activate
```

## 3. Terminal A — the LEARNER (start this one first)

```bash
python -m lerobot.rl.learner --config_path configs/real/train_config.json
```

Wait for the startup lines, and check them:

- `Valid checkpoint found: resume=True detected, resuming previous run`
- `Resuming from step 36000, interaction step ...`  (step number grows each session)
- `Loading weights from local directory`
- you must **NOT** see `instantiating a policy from scratch` — if you do, stop and check

The learner logs to `outputs/hilserl_cube_run1/logs/learner_cube_run1.log`.
It writes a checkpoint every 2,000 steps (~3.3 min at ~10 Hz) and at shutdown.

## 4. Terminal B — the ACTOR (second terminal, step 2 first, then:)

```bash
python -m lerobot.rl.actor --config_path configs/real/train_config.json \
    --output_dir=outputs/hilserl_cube_run1_actor
```

Check for: `[ACTOR] Loaded initial parameters from Learner before first action.`
Then train. Controls: hold **RB** to intervene · left stick X/Y · right stick Z ·
**LT** close · **RT** open · **Y** success · **A** re-record.
Watch: intervention rate falling + autonomous success rising (the real progress signal).

## 5. Ending the session

```bash
# Ctrl-C in the actor terminal, then Ctrl-C in the learner terminal
```

The learner writes a final checkpoint on the way out.
If the actor died for another reason, restart the **learner** too before a new
actor (MAX_WORKERS=3: a half-open actor connection silently blocks the next actor).

## 6. Bring the LLM back online — only AFTER training is done

```bash
systemctl --user start llama-server
```

Model load takes ~30–60 s. Verify:

```bash
systemctl --user status llama-server --no-pager
journalctl --user -u llama-server -n 20 --no-pager
curl -s http://127.0.0.1:8080/health
```

If `start` refuses (start-limit hit), run
`systemctl --user reset-failed llama-server && systemctl --user start llama-server`.
Once port 8080 serves, opencode / Claude Code reconnect normally.
