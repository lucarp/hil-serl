# Ball task — demo dataset

Recorded 2026-08-23 with `configs/real/env_record.json` (gamepad, 10 Hz, EE deltas).

| Dataset | Episodes | Frames | Successes |
|---|---|---|---|
| `lucarp/so101_ball_circular_hole` | 25 | 3671 | 24/25 |
| **`lucarp/so101_ball_circular_hole_clean`** | **24** | **3472** | **24/24** |

Use the `_clean` one for training. Episode 11 of the original ran past
`control_time_s` (199 steps = 19.9 s), was truncated, and got saved as a
reward-0 failure — the timeout trap: only `A` (rerecord) discards, everything
else saves. Removed with:

```bash
lerobot-edit-dataset --repo_id lucarp/so101_ball_circular_hole \
  --new_repo_id lucarp/so101_ball_circular_hole_clean \
  --operation.type delete_episodes --operation.episode_indices "[11]"
```

Verified: every episode carries exactly one reward-1 frame, on its final step.
Action stats show full range on all three EE axes plus gripper 0 and 2, which
confirms RB was held throughout (a released RB records the neutral `[0,0,0,1]`
and yields a valid-looking, worthless dataset).

Episode length mean 145 steps (14.5 s), min 76, max 190 — on the long side;
about 60% of frames carry zero EE delta (operator pauses). Worth shortening on
the cube task.

Datasets live under `/mnt/ai/lucas/huggingface/lerobot/` (`~/.cache/huggingface`
is a symlink to `/mnt/ai/lucas/huggingface`). Not pushed to the Hub.
