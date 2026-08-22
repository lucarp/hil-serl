# HIL-SERL study plan and implementation audit

**Status:** working document, 2026-08-22. Written from a code audit of the vendored
lerobot at `vendor/lerobot` (branch `local/hilserl`, HEAD `1667afc2`) plus the HIL-SERL,
RLPD, SERL and SAC papers. Produced with AI assistance (parallel readers over each
subsystem, then a verification pass that re-opened every cited line range); it is an
internal note, not prose to paste into a PR or a thesis.

Line numbers are the verifier-corrected ones. Independently re-checked by hand before
filing, all confirmed:

- `raise_on_jump: bool = True` (`robots/so_follower/robot_kinematic_processor.py:212`),
  raise at `:245`, and `rl/gym_manipulator.py:510-512` constructs `EEBoundsAndSafety`
  without overriding it — an over-limit EE step aborts the run rather than clamping.
- The reward classifier's per-camera `Sequential` is built with learnable
  `SpatialLearnedEmbeddings` + `Linear` + `LayerNorm` (`modeling_classifier.py:164-180`)
  but is invoked entirely inside `torch.no_grad()` (`:203-211`), while `_freeze_encoder`
  (`:158-161`) only freezes the ResNet trunk. Those parameters never receive gradient
  and stay at random initialisation; only `classifier_head` trains.
- `EnvConfig.fps` defaults to 30 (`envs/configs.py:58`).
- `DONE` is set by `terminate_episode or (terminate_on_success and success)`, while
  `REWARD = float(success)` unconditionally (`processor/hil_processor.py:513-517`).
- The data mixer computes `n_online` before the offline-`None` early exit, so with no
  demo dataset the effective batch is half the configured size
  (`rl/data_sources/data_mixer.py:77` vs `:85-87`).
- Sim config values cited at `configs/sim/train_config.json` lines 26/29/30/43/49/168 are
  accurate: `discount 0.97`, `num_critics 2`, `num_subsample_critics null`, `utd_ratio 2`,
  `online_ratio 0.5`, `policy_parameters_push_frequency 50`.

---


All code anchors are relative to `/mnt/Storage/projects/hil-serl`; library paths are `vendor/lerobot/src/lerobot/…` on submodule branch `local/hilserl` (HEAD `1667afc2`). Line numbers below are the **verifier-corrected** ones where the original maps were wrong.

---

## 0. Read this before the maps: where the maps contradict each other

Four disagreements matter operationally. In each case the verifier's ruling is authoritative and I use it below.

1. **The EE jump limit raises, it does not clamp.** The actor map calls it "the 5 cm/step jump limiter" and treats it as a silent reshaping invisible to the buffer. The env-pipeline map says it raises. The env-pipeline map is right: `raise_on_jump` defaults `True` (`vendor/lerobot/src/lerobot/robots/so_follower/robot_kinematic_processor.py:212`), `gym_manipulator.py:510-512` never overrides it, and the raise is at `robot_kinematic_processor.py:245`. **This will kill a training run mid-episode.** Only the `end_effector_bounds` `np.clip` at `robot_kinematic_processor.py:233` is silent-and-invisible-to-the-buffer.

2. **What pollutes the actor's gradient clip.** The sac-core map blamed the continuous critic's gradient from `sac_algorithm.py:225`. Wrong for the shipped config: `optimizers["discrete_critic"].zero_grad()` at `vendor/lerobot/src/lerobot/rl/algorithms/sac/sac_algorithm.py:236` clears it first, and the gradient actually sitting in the shared encoder at clip time is the **discrete critic's**, from line 237. The line-225 story holds only when `num_discrete_actions is None`. Also drop the map's "3.2× inflation" number — not reproducible (the verifier measured ~42× on the same config; it is seed- and batch-dependent). Keep the structural fact: the clip at `sac_algorithm.py:250-252` walks 62 tensors, only 12 of which the actor's Adam will ever step (`policies/gaussian_actor/modeling_gaussian_actor.py:55-63`).

3. **The shared encoder is stepped by two optimizers, not one.** The sac-core map says "the critic loss alone." False. `sac_algorithm.py:487-495` builds `critic` over `critic_ensemble.parameters()` **and** `discrete_critic` over `policy.discrete_critic.parameters()`; both register the same `policy.encoder_critic` object (`sac_algorithm.py:647`, `modeling_gaussian_actor.py:373`). Measured: critic = 70 tensors of which 50 are encoder; discrete_critic = 60 of which 50 are encoder. Two Adams with independent moment estimates step the same weights every iteration.

4. **The offline truncation bug is real but small.** The sac-core map says "systematic pessimism injected at 50% of every batch." Only the terminal frame of each demo episode carries `done=1` (`vendor/lerobot/src/lerobot/rl/gym_manipulator.py:727`, `rl/buffer.py:686-687`), so it is ~1/episode_length of the offline half — under 1% of a batch at 150 steps. Do not spend a day on it.

Smaller ones, corrected and used below without further comment: the gripper-penalty step's install condition is `cfg.processor.gripper is not None` (default `None`, `envs/configs.py:297`) plus `use_gripper`, **not** "max_gripper_pos explicitly set" (it defaults to 100.0, `envs/configs.py:301`); `crop_dataset_roi.py:225` **does** clamp, so the claimed train/inference asymmetry there does not exist; `AddBatchDimensionProcessorStep` is a no-op for the observation but **does** add a batch dim to the 1-D action (`processor/batch_processor.py:60-62, 249-250`); `eval_policy.py`'s first exception is at line 68, not line 1; DrQ's per-image independent shifts match DrQ-v2 and are not a deviation; and `policy_fps` wraps preprocess+forward+postprocess (`rl/actor.py:312-323`), not inference alone.

---

## 1. Answer the strategic question

### Recommendation

**Run 1 is a reproduction of *this implementation*, on ball-into-circular-hole, with the algorithm untouched. Only two categories of change are permitted before it, and neither is research.**

The reason is attribution, and it is quantitative. Between the four maps there are at least eight independently verified mechanisms that produce the exact symptom "reward stays 0 for an hour":

- the success button tap can be swallowed — `JOYBUTTONUP` clears `episode_end_status` for buttons 0/2/3, so a Y press shorter than one 100 ms control period is set and cleared before the pipeline reads it (`vendor/lerobot/src/lerobot/teleoperators/gamepad/gamepad_utils.py:264-285`, read-and-clear at `:89-91`);
- `terminate_on_success` gates the *only* path by which a human success ends an episode, because SUCCESS never sets `terminate_episode` (`processor/hil_processor.py:513-517`, `teleoperators/gamepad/teleop_gamepad.py:151-156`);
- the classifier is a linear probe on random projections, because the whole learnable per-camera `Sequential` sits under `torch.no_grad()` (`rewards/classifier/modeling_classifier.py:203-235`);
- the classifier sees a different input distribution at RL time than it was fitted on — the MEAN_STD normalizer exists only in the training pipeline (`rewards/classifier/processor_classifier.py:58-67`) while the RL step loads a bare `Classifier` (`processor/hil_processor.py:579-584`);
- a `lerobot-record` dataset has no `next.reward` column at all, so classifier training needs `gym_manipulator --mode=record` (`rl/gym_manipulator.py:653-657`);
- images must be exactly 128×128 or the spatial-embedding kernel shape-errors (`rewards/classifier/modeling_classifier.py:167-168`);
- `cfg.env.fps` defaults to 30 (`envs/configs.py:58`) and silently rescales both episode length and the joint-velocity finite difference;
- the whole demo/offline half of every batch disappears if `cfg.dataset` is unset, and the batch silently halves too (`rl/data_sources/data_mixer.py:77` vs `:85-87`).

If he also changes SAC, a failed run 1 has three or more candidate causes with no way to separate them, and the debugging currency is robot-hours and his own attention. Once he has a curve with a real number on it, every subsequent change is a controlled experiment against a baseline he owns.

### The honest caveat: "pure reproduction" of the *paper* is not on the menu

The shipped defaults are not the paper's, in ways that are not stylistic:

| Knob | Shipped | Paper / RLPD | Anchor |
|---|---|---|---|
| `discount` | 0.99 default, 0.97 in this repo's sim config | 0.96–0.985 per task | `configs/sim/train_config.json:26` |
| `num_critics` | 2 | E=10 | `configs/sim/train_config.json:29` |
| `num_subsample_critics` | `null` (subsampling OFF) | Z∈{1,2} of E | `configs/sim/train_config.json:30` |
| `utd_ratio` | 2 here, 1 library default | G≈20 | `configs/sim/train_config.json:43` |
| `fps` | 30 | 10 | `envs/configs.py:58` |
| `target_entropy` | `-(|A_cont| + 1)/2` = −2.0 | RLPD −dim(A)/2; SAC-v2 −dim(A) | `sac_algorithm.py:120-125` |

So "don't change things" means **don't change the algorithm**. Config corrections that make the code do what the paper describes are prerequisites, not experiments.

### Category A — config corrections, all before run 1

1. **`fps: 10`, explicitly.** Base `EnvConfig` default is 30 (`envs/configs.py:58`). The same value drives the trailing `precise_sleep` (`rl/actor.py:425-427`), `int(control_time_s * fps)` for the time limit (`processor/hil_processor.py:288-289`), and `dt = 1/fps` for the velocity half of `observation.state` (`rl/joint_observations_processor.py:78-83`).
2. **`discount`.** At 10 Hz, value half-life is `ln2/ln(1/γ)` steps: γ=0.99 → 69 steps (6.9 s); 0.97 → 22.8 (2.28 s); 0.96 → 17.0 (1.70 s). For a 15 s episode, 0.99 is far too flat to create cycle-time pressure. Start at 0.97 (`configs/sim/train_config.json:26`).
3. **Buffer capacity.** `_initialize_storage` passes no `dtype` (`rl/buffer.py:147`) so images are float32; both capacities default to 100_000 (`policies/gaussian_actor/configuration_gaussian_actor.py:155/157). Two 128×128 RGB cameras = 196,608 B/frame → 18.31 GiB per camera key per buffer → **~73 GiB across both buffers**. Set both capacities to what he will actually collect (~30k), and note `policy.storage_device` sets the *buffer's* device, not the policy's (`rl/learner.py:804`, `:860`), and the resume path drops it (`rl/learner.py:819-825`).
4. **Action stats must stay identity.** The learner normalizes observations only (`rl/trainer.py:77-84`); the buffer stores the post-processed executed action (`rl/actor.py:317-323`) while `_compute_loss_actor` feeds the critic raw tanh output in [−1,1]. That is only consistent because action min/max are [−1,−1,−1]/[1,1,1] (`configs/sim/train_config.json:131-134`). If he sets real SO-101 action stats, the critic silently trains on two scales.
5. **`end_effector_step_sizes`.** The per-frame EE norm limit is 0.05 m. Three axes at step size `s` give a worst-case norm `s√3`, so `s < 0.0289 m` is guaranteed safe. Pick 0.02 m and the diagonal is 0.0346 m.
6. **`terminate_on_success: true`** (`envs/configs.py:287`), or the Y press books +1 and the episode keeps running.
7. **`save_freq` ≠ 0** — `rl/learner.py:459` inlines the modulo and raises `ZeroDivisionError`, bypassing the guard that exists for exactly this (`common/train_utils.py:55-63`).

### Category B — bugs to patch or explicitly accept, before run 1

1. **`use_torch_compile: false`.** The repo's own sim config sets `true` (`configs/sim/train_config.json:46`) while the source comment right above the compile call says the policy does not converge when enabled (`sac_algorithm.py:93-97`). Compiling also renames state-dict keys to `_orig_mod.*`, defeating `_strip_encoder_keys` (`sac_algorithm.py:587-589`), and `load_state_dict(..., strict=False)` (`sac_algorithm.py:550-551`) means toggling it between checkpoint and resume loads **nothing** into the critics with no error.
2. **`raise_on_jump`.** Either keep step sizes under the bound (above) or patch `gym_manipulator.py:510-512` to pass `raise_on_jump=False`. Do not discover this at hour 2.
3. **Actor grad-clip.** One line: clip over `get_optim_params()["actor"]` instead of `policy.actor.parameters()` at `sac_algorithm.py:250-252`. Do it now, because otherwise `actor_grad_norm` in WandB is uninterpretable and he will tune `grad_clip_norm: 10.0` (`configs/sim/train_config.json:45`) against a number dominated by encoder gradients he never steps.

### Category C — defer everything else

Rotation channel, contact sensing, intervention weighting, classifier architecture changes, ensemble size. All of these are §6 material. None of them before a baseline number.

### What "test it first" means concretely

- **One task** (ball into circular hole), **one operator**, **one config file under version control**, and a written success criterion fixed *before* the run: e.g. ≥60% over 20 consecutive fully-autonomous episodes with zero interventions.
- **The eval harness exists before the training run.** `rl/eval_policy.py` cannot produce a number: first raise at `rl/eval_policy.py:68` (`env_cfg.pretrained_policy_name_or_path` exists nowhere in the tree), a tuple-unpack bug at `:41`, and no processor pipelines at all — so even fixed, `env.step()` returns the hardcoded 0.0 from `rl/gym_manipulator.py:255-280` and "success rate" would be 0 by construction. Write a real one (§6.2).
- **A 20-minute hardware dry run with an untrained policy** at 10 Hz, torque-limited, purely to confirm loop period, camera streaming at construction time (`RobotEnv.__init__` calls `_get_observation()`, `rl/gym_manipulator.py:167, 181` — a lazily-opening camera fails at construction, not at first step), reset dead time, and that no EE-jump `ValueError` fires.
- **Classifier precision/recall on held-out data reported before any RL.** Not accuracy — see §4.E8.

### What "test it first" must NOT mean

- **Not** "read the whole codebase first." The read in §2 is ~3 hours; do it while demos and classifier frames are collecting.
- **Not** "run the sim config unchanged on hardware." It will mis-time the loop and exhaust RAM.
- **Not** "if run 1 fails, HIL-SERL doesn't work on SO-101." A failed run 1 is a systems result until reward composition, loop period, and the classifier have each been independently verified. That is a diagnosis, not a finding.
- **Not** "tune hyperparameters." Nothing on the SAC side moves except `discount` and capacities.
- **Not** "skip ahead to the screw." His curriculum order is correct; keep it.

### On the "library user, not a researcher" fear

The four maps he now has are the refutation. They enumerate roughly fifteen places where this implementation diverges from what the papers describe, each with a file:line — a shared target network that isn't one (`sac_algorithm.py:82, :90`), a reward model whose learnable head never receives gradients (`modeling_classifier.py:203-235`), an action space with the rotation channel hardwired to zero (`processor/delta_action_processor.py:114-116`), an intervention up-weighting rule with no decay (`rl/learner.py:974-980`). Nobody who is "just using the library" has that list. But the list is only worth something attached to a baseline number, which is why the ordering above is not negotiable. §6 ranks the list.

---

## 2. Reading order

~2 h 50 min of essential reading in three blocks. Block A follows one control step from stick to servo; Block B follows one gradient step from buffer to loss; Block C is the reward. Each block is comprehensible on its own given the previous one. Read it in one or two sittings while data collects.

### Block A — one control step (~70 min, all essential)

| # | File:lines | Look for | Time |
|---|---|---|---|
| A1 | `vendor/lerobot/src/lerobot/rl/gym_manipulator.py:255-280` | `RobotEnv.step` returns `reward=0.0, terminated=False, truncated=False` unconditionally. Internalize: **on hardware, every reward byte comes from a processor step.** | 5 min |
| A2 | `rl/gym_manipulator.py:422-493` | The exact ordered ENV pipeline (Vanilla → [JointVel] → [MotorCurrent] → [FK-to-EE] → CropResize → TimeLimit → GripperPenalty → RewardClassifier → AddBatchDim → Device) and the ACTION pipeline. This order determines `observation.state` width and which config flags are live. Note `terminate_on_success` read once at `:374-376` and handed to two steps in two different pipelines. | 15 min |
| A3 | `rl/gym_manipulator.py:486-523` | The five hardcoded kwargs that decide every non-default kinematic behaviour and are unreachable from JSON: `use_latched_reference=False`, `use_ik_solution=True` (`:508`), `speed_factor=1.0` (`:515`), `discrete_gripper=True` (`:516`), `initial_guess_current_joints=False`. | 10 min |
| A4 | `rl/gym_manipulator.py:553-592` | `step_env_and_process_transition`. The in-place `OBSERVATION` overwrite with raw joints at `:555-557`; reward composition `r = r_env + r_action_pipeline` at `:561-565`; the env pipeline runs **last** at `:590` so the classifier can overwrite. | 10 min |
| A5 | `processor/hil_processor.py:468-531` | `InterventionActionProcessorStep`. `DONE`/`REWARD` assignment at `:513-517` (assignment, not accumulation), and the executed-action write-back into `complementary_data["teleop_action"]` at `:526-529`. This is the whole of HIL-SERL's off-policy correctness. | 10 min |
| A6 | `rl/actor.py:301-379` | `:301-327` select_action + the split unnormalize (continuous 3 dims only, discrete index passed through). `:344-379` where `executed_action` is pulled back out of complementary data and the `Transition` is assembled. Note there is no warmup and no exploration noise. | 15 min |
| A7 | `processor/delta_action_processor.py:94-130` and `robots/so_follower/robot_kinematic_processor.py:79-159, 215-262` | `target_wx/wy/wz = 0.0` at `delta_action_processor.py:114-116` (the single line where rotation is thrown away); the enable gate `||delta|| > 1e-3` on the **raw** [−1,1] action at `:104-105` (the "1 mm" comment at `:92` is wrong); the reference pose = FK of the previous IK solution; the `ValueError` at `robot_kinematic_processor.py:245`. | 15 min |

*Optional here (10 min): `model/kinematics.py:98-133` — one QP solve, not iterated IK; `position_weight=1.0` against `orientation_weight=0.01`.*

### Block B — one gradient step (~70 min, all essential)

| # | File:lines | Look for | Time |
|---|---|---|---|
| B1 | `rl/buffer.py:192-232` and `:234-300` | `add()` discards `next_state` under `optimize_memory` (`:212-214`); complementary_info keys frozen by the **first** transition (`:222-229`); sampling `high` guard (`:241-243`); `s' = states[(i+1) % capacity]` (`:256-257`); DrQ block (`:272-290`); float32 storage (`:147`). Note `self.episode_ends` (`:122`) is allocated and never read anywhere in the tree — that is the missing episode-boundary guard. | 15 min |
| B2 | `rl/data_sources/data_mixer.py:58-97` | RLPD symmetric sampling in twenty lines. `n_online = max(1, int(B·ratio))` at `:62-63`, and the bug: `n_online` is computed *before* the offline-is-None early exit (`:77` vs `:85-87`), so pure-online runs silently halve the batch — while `sample()` at `:58-60` does not have this bug. | 5 min |
| B3 | `rl/learner.py:387-422`, `:458-474`, `:944-980` | The main loop: exactly one `training_step()` per iteration (`:417`), busy-spin before warmup with no sleep (`:411-412`), wall-clock weight push (`:420`), checkpoint trigger (`:459`), and the intervention dual-write (`:974` then `:977-980`, gated on `dataset_repo_id is not None`). | 20 min |
| B4 | `rl/algorithms/sac/sac_algorithm.py:199-268`, then `:282-333`, `:356-390`, `:393-420`, `:422-440`, `:115-125` | In that order: the UTD/actor/target cadence; the Bellman target (`:307`, mask is `done` only — `truncated` never appears); the double-DQN gripper critic; the actor and temperature losses; Polyak; target entropy. Note `_prepare_forward_batch(..., include_complementary_info=True)` only inside the UTD inner loop (`:203` vs `:221`), so the gripper penalty reaches 1 of 2 discrete updates at `utd_ratio=2` and **never** at `utd_ratio=1`. | 30 min |
| B5 | `policies/gaussian_actor/modeling_gaussian_actor.py:55-63`, `:225-295`, `:446-476` | `get_optim_params` — the only 12 tensors the actor's Adam sees. The encoder: `detach` guards the image branch only (`:225-236`), so `state_encoder`/`env_encoder` gradients still flow. The tanh-Gaussian head, with the std clamp applied **after** `exp` (`:456-474`); the live bounds are `std_min=1e-5, std_max=5` from `configs/sim/train_config.json:156-157`, not the constructor's log-space defaults at `:407-408`. `use_tanh_squash` is stored and never read — a dead flag. | 15 min |

### Block C — reward and recording (~35 min, all essential)

| # | File:lines | Look for | Time |
|---|---|---|---|
| C1 | `rewards/classifier/modeling_classifier.py:164-201`, `:203-235`, `:237-289` | The architecture (`height=4, width=4` hardcoded at `:167-168` → 128×128 inputs only); `_get_encoder_output` wrapping the **entire** learnable Sequential in `no_grad` at `:205-209`; three output conventions on one model — `compute_reward` hardcodes 0.5 (`:243`), `forward` trains BCE-with-logits on `next.reward`, `predict_reward` is what RL calls. | 20 min |
| C2 | `processor/hil_processor.py:596-633` | Classifier **overwrites** reward (`:619-622`), no hysteresis, one frame over threshold ends the episode. | 5 min |
| C3 | `rewards/classifier/processor_classifier.py:58-67` | The MEAN_STD normalizer that exists only at training time. Compare with `hil_processor.py:579-584`. | 5 min |
| C4 | `rl/gym_manipulator.py:699-759` | The record loop: observation snapshotted at `:703-707` *before* the step, reward written at `:726` from the post-step transition, `clear_episode_buffer()` on rerecord at `:753-756`. Read this before recording anything. | 5 min |

### Optional appendix (~25 min, read when the corresponding thing breaks)

- `rl/learner_service.py:42-135` — `MAX_WORKERS = 3` and three long-lived streams; explains the hung-actor-on-restart failure with `stub.Ready()` having no timeout (`rl/actor.py:455`).
- `rl/eval_policy.py:38-71` — read once, only to confirm it is dead and stop reaching for it.
- `rl/learner.py:587-645` — checkpointing re-encodes the entire replay buffer to video on the training thread after `shutil.rmtree`ing the previous copy (`:624-643`).

---

## 3. Understanding checkpoints

Answers on the right; the list doubles as its own key. Nothing here is answerable from the papers alone.

### A. Off-policy correctness and the intervention pathway

| Q | A |
|---|---|
| Which tensor is stored as the transition's action during an intervention, and where is it stamped? | The human's teleop tensor, written into `complementary_data["teleop_action"]` at `processor/hil_processor.py:526-529` and read back by the actor at `rl/actor.py:346`. |
| Why does off-policy TD need no importance ratio for human actions? | `Q(s,a)` conditions on `a` as an input and resamples `a'~π` for the bootstrap; only `r` and `s'` must be genuine samples of `P(·|s,a)`. |
| What exactly breaks if you log `a_RL` instead of `a_itv`? | `s'` is no longer a sample from `P(·|s,a_RL)` — you inject a fabricated dynamics model on precisely the highest-reward-density transitions, which also carry half the batch weight. |
| Where do intervention transitions go? | **Both** buffers: `rl/learner.py:974` unconditionally, then `:977-980` again into the offline buffer, gated on `dataset_repo_id is not None` (the guard is on the wrong object — it should be `offline_replay_buffer is not None`). |
| Does SAC ever read `is_intervention`? | No. Tree-wide reads are `rl/learner.py:978`, `rl/actor.py:357/366`, `rl/gym_manipulator.py:253/279`. |
| Why must interventions be short? | Value 1 propagates back along the human chain so `Q(s,a_itv)→γ^k` while `π(a_itv|s)≈0`; a smooth critic bleeds that onto the policy's own action region where the transition never occurs → inflated `V(s)` no achievable action realizes, amplified by symmetric sampling giving these transitions half the batch. |
| What ends an episode when you press A (rerecord) during RL? | The episode ends and every transition is still shipped (`rl/actor.py:384-394` has no rerecord check) — unlike record mode, which calls `clear_episode_buffer()` (`rl/gym_manipulator.py:753-756`). In RL, "rerecord" means "end episode." |

### B. RLPD / sampling

| Q | A |
|---|---|
| What fraction of each batch is offline? | `n_online = max(1, int(B·online_ratio))`, rest offline (`rl/data_sources/data_mixer.py:62-63`); `online_ratio` at `configs/sim/train_config.json:49`, mixer selected at `:48`. |
| What happens with `cfg.dataset = None`? | `get_iterator` computes `n_online` before the early exit, so batches are `n_online` wide, not `batch_size` — at ratio 0.5 the batch silently halves (`data_mixer.py:77` vs `:85-87`). `sample()` does not have this bug, so a unit test will not catch it. |
| Is the shipped setup an RLPD ensemble? | No. `num_critics: 2` (`configs/sim/train_config.json:29`) and `num_subsample_critics: null` (`:30`) → plain clipped double-Q. The RLPD ingredient that *is* on is symmetric sampling. |
| Does 50/50 change the TD fixed point? | Tabular with full support: no, only the rate. Under function approximation: it changes the projection metric in `Π_μ T^π`, i.e. *which* states' approximation error you pay for, and transfer picks up a `‖d^π/μ‖_∞` concentrability factor. It is a reweighting, not a bias in the backup. |
| Where does `truncated` enter the backup? | Nowhere. `td_target = r + (1-done)·γ·min_q` at `sac_algorithm.py:307`; `truncated` is stored (`rl/buffer.py:298`) and never used. |
| So what goes wrong offline? | The demo path writes `DONE = terminated or truncated` (`rl/gym_manipulator.py:727`) and the converter sets `truncated = done` (`rl/buffer.py:686-687`, with a `TODO`), so a time-limited demo's last transition regresses to bare `r`. One frame per episode — under 1% of a batch, not 50%. |

### C. SAC as coded here

| Q | A |
|---|---|
| `target_entropy` for action shape [3] with `num_discrete_actions: 3`? | −(3+1)/2 = **−2.0** (`sac_algorithm.py:120-125`); SAC's heuristic is −|A|, so this is half the entropy pressure. |
| Is `critic_target` a real target network? | Only the MLP heads. Lines `:82` and `:90` pass the *same* encoder object, so Polyak on encoder params computes `p ← p·τ + p·(1−τ) = p`, a no-op (`sac_algorithm.py:422-440`), and the target Q is computed on features updated by the previous gradient step. |
| Which optimizers step the shared encoder? | `critic` **and** `discrete_critic` (`sac_algorithm.py:487-495`; `CriticEnsemble` at `:647`, `DiscreteCritic` at `modeling_gaussian_actor.py:373`). The actor's does not (`modeling_gaussian_actor.py:55-63`). |
| What is in the actor's grad-clip that its optimizer never steps? | The shared encoder's 50 tensors, carrying the discrete-critic loss gradient from `sac_algorithm.py:237` — `zero_grad()` at `:248` clears only the actor's 12 (`:250-252` clips all 62). |
| Write the temperature loss as coded. | `−exp(log α)·(logπ + H̄)` averaged over the batch, `logπ` recomputed under `no_grad` (`sac_algorithm.py:410-419`). The gradient w.r.t. `log_alpha` is scaled by α itself, so `temperature_init: 0.01` (`configs/sim/train_config.json:41`) also shrinks alpha's own learning signal. |
| Actor:critic step ratio at `utd_ratio: 2`? | 1:2. The critic takes `utd_ratio` steps per `update()`; the actor takes `policy_update_freq` steps on every `policy_update_freq`-th call — 1 on average. `policy_update_freq` batches actor updates, it does not delay them TD3-style, and all repeats reuse the same minibatch. |
| When does the gripper penalty actually reach a loss? | Only when `complementary_info` is present, i.e. the UTD inner loop (`sac_algorithm.py:203` with `include_complementary_info=True`, vs `:221` False). It enters the target at `:376-379`. At `utd_ratio: 1` it is applied **never**. |
| What is the true learnable parameter count for the sim config? | ~1,126,348. `log_training_info` prints 891,849 because it counts only `policy.parameters()` (`rl/learner.py:775-776`), missing the 234,498 critic-head params and `log_alpha`. |
| Why is `num_critics` nearly free? | One shared encoder pass, then a Python loop over head MLPs stacked to `[N, B]` (`sac_algorithm.py:628-672`). |
| What does the `.sum()` at `sac_algorithm.py:332` change? | Nothing for the independent heads; it multiplies the gradient reaching the **shared encoder** from the critic loss by N. |

### D. Env / action pipeline

| Q | A |
|---|---|
| Is rotation commandable? | No. `target_wx/wy/wz` are hardwired to 0.0 at `processor/delta_action_processor.py:114-116`, so the desired orientation is always the reference orientation. |
| What is the IK reference pose computed from? | FK of the **previous IK solution** (`use_ik_solution=True` at `rl/gym_manipulator.py:508`; written at `robot_kinematic_processor.py:629`, read at `:79-159`). The Cartesian loop closes in command space; encoder tracking error never feeds back. |
| Commanded EE step > 0.05 m? | `ValueError`, control loop dies (`robot_kinematic_processor.py:239-252`, `raise_on_jump` default True at `:212`, never overridden at `rl/gym_manipulator.py:510-512`). |
| Is the IK iterated? | No — one placo QP solve (`model/kinematics.py:98-133`). A 0.05 m request lands 0.0417 m; repeated calls converge (0.042 → 0.0072 → 0.0002). |
| Gripper command semantics? | Bang-bang, not an integrator: `delta = ±max_gripper_pos` with `speed_factor=1.0`, so CLOSE saturates to max and OPEN to 0.0 (`robot_kinematic_processor.py:414-423`). |
| Does `add_ee_pose_to_observation` grow `observation.state`? | No. It writes plain Python floats under `ee.*` keys and pops the `<motor>.pos` keys (`robot_kinematic_processor.py:439-458`), which are then dropped by the tensor guard and by the `input_features` filter at `rl/actor.py:307-309`. |
| Episode length at `control_time_s=15, fps=10`? | 149 real env steps, not 150 — `env_processor.reset()` runs before the reset transition is pushed through the same pipeline (`rl/gym_manipulator.py:601-610`), so it consumes step 1 (`processor/hil_processor.py:288-289`). |
| Which config fields are dead? | `reset_time_s` (never passed, `rl/gym_manipulator.py:348-353`; hardcoded 5.0 at `:132`), `use_tanh_squash` (`modeling_gaussian_actor.py:411/421`, never read), `async_prefetch` (`configuration_gaussian_actor.py:159`, never read — `base.py` hard-defaults True), `episode_ends` (`rl/buffer.py:122`, allocated, never read). |
| What is the minimum dead time per reset? | 5.0 s. 50 waypoints × 0.015 s ≈ 0.75 s of motion (`rl/gym_manipulator.py:108-120`), then `precise_sleep` pads to `reset_time_s` (`:224-253`). `max_relative_target` does **not** apply — the reset writes the bus directly. |

### E. Reward and evaluation

| Q | A |
|---|---|
| Human presses SUCCESS on the same step the classifier fires. Reward? | **1.0**, not 2.0. `r = r_env + r_action` at `rl/gym_manipulator.py:561-565`, then the classifier **overwrites** at `processor/hil_processor.py:619-622`. |
| Which classifier params receive gradients? | Only `classifier_head`. The whole per-camera Sequential — SpatialLearnedEmbeddings, Linear, LayerNorm — is under `no_grad` (`rewards/classifier/modeling_classifier.py:203-235`) and stays at random init. Dropout inside it is still active in train mode. |
| Why must images be 128×128? | `SpatialLearnedEmbeddings(height=4, width=4, …)` is hardcoded at `modeling_classifier.py:167-168`; only a /32-stride ResNet on 128×128 gives a 4×4 map. |
| Train vs inference normalization? | MEAN_STD normalizer only in the training pipeline (`processor_classifier.py:58-67`); RL loads a bare `Classifier` (`hil_processor.py:579-584`) and feeds raw [0,1] images from `VanillaObservationProcessorStep` (`processor/observation_processor.py:71-90`). |
| Which threshold does RL use? | `predict_reward(threshold=success_threshold)`. `compute_reward` hardcodes 0.5 (`modeling_classifier.py:237-245`) and is what the generated model card tells you to call. |
| Can a `lerobot-record` dataset train the classifier? | No — no `next.reward` column. Use `gym_manipulator --mode=record`, which adds REWARD and DONE (`rl/gym_manipulator.py:653-657`). |
| Is there a success-rate metric? | No. Only `train/Episodic reward`, `train/Intervention rate`, `train/Episode intervention` on x-axis `train/Interaction step` (`rl/actor.py:405-415` → `rl/learner.py:929-941` → `common/wandb_utils.py:195`). `pc_success` needs `info['is_success']` from a vectorized env; `RobotEnv` emits neither. |
| Why is `rl/eval_policy.py` unusable? | First raise at `:68` (`pretrained_policy_name_or_path` exists nowhere in the tree); tuple-unpack bug at `:41`; no processor pipelines so reward would be 0 by construction; and it labels mean return as "success rate" at `:52`. |

### F. Systems and throughput

| Q | A |
|---|---|
| `policy_parameters_push_frequency = 4` — four what? | Seconds, wall-clock, compared at `rl/learner.py:420`, mirrored at `rl/learner_service.py:79-80`; default 4 at `configuration_gaussian_actor.py:59`. |
| How stale is the acting policy? | One full episode. Weights are pulled only at episode boundaries (`rl/actor.py:387`), non-blocking, drain-to-latest — every intermediate push is discarded (`rl/queue.py:25-52`). |
| What crosses the wire? | `policy.actor.state_dict()` only (`sac_algorithm.py:501-510`). With `shared_encoder: true` that ships the actor's copy of the encoder, which is the stale one — in-repo TODO at `rl/actor.py:705-713`. |
| Does the FPS warning cover the control loop? | No. `policy_timer` wraps preprocess + forward + postprocess (`rl/actor.py:312-323`); the IK chain, serial writes and camera reads are outside it, and the overrun is silently absorbed by `precise_sleep(max(…, 0))` at `:425-427`. Watch the 90th-percentile figure, which is `1/(90th-percentile time)` — the near-worst-case rate (`utils/utils.py:422-435`). |
| A restarted actor hangs on "Send ready message to Learner". Why? | `MAX_WORKERS = 3` (`rl/learner_service.py:42`) and the actor holds exactly three long-lived streams; `stub.Ready()` is called with no timeout (`rl/actor.py:455`), so it queues forever instead of raising. Restart the **learner**. |
| What happens at every checkpoint? | `shutil.rmtree` then a full re-encode of both replay buffers to video LeRobotDatasets, on the training thread (`rl/learner.py:624-643`). There is a window where no saved buffer exists, and resume reloads images through lossy video compression. |

---

## 4. Experiments that build understanding

Ordered by insight-per-minute. Everything through E9 runs in sim or offline. Note that in sim the classifier step is never installed (`rl/gym_manipulator.py:403-407` returns early on the `gym_hil` branch), so classifier work is offline on recorded frames.

**E1 — Instrument the algorithm at init (5 min, no training).** Build the sim policy + `SACAlgorithm` on CPU from `configs/sim/train_config.json` and assert three things: `algo.critic_target.encoder is algo.critic_ensemble.encoder` → True; per-optimizer tensor counts (actor 12 with 0 encoder tensors, critic 70 with 50, discrete_critic 60 with 50); and the clip norm over `policy.actor.parameters()` versus over the 12 actor-optimizer tensors. *Expect:* shared target encoder, large clip inflation. *If wrong:* the pinned submodule is not what the maps describe — stop and re-verify before anything else.

**E2 — Buffer sizing (2 min, no training).** Compute 3·128·128·4 = 196,608 B/frame → 18.31 GiB per camera key per 100k-slot buffer, then actually try constructing a `ReplayBuffer` at capacity 100_000 with two camera keys on `storage_device='cuda'` and watch it die at the first transition (`rl/buffer.py:147`). *Teaches:* the number to put in the config, discovered in two minutes instead of at hour two of a real run.

**E3 — IK undershoot and the safe step size (10 min, no robot).** Call `RobotKinematics.inverse_kinematics` (`model/kinematics.py:98-133`) repeatedly on a fixed target with `assets/SO-ARM100/.../so101_new_calib.urdf`. *Expect:* a 0.05 m request lands ~0.0417 m; repeated calls converge (0.042 → 0.0072 → 0.0002). Then check what a 10-step lateral sweep does to the EE rotvec — orientation drifts because `orientation_weight` is 0.01 against `position_weight` 1.0 on a 5-DoF arm. *Teaches:* pick `end_effector_step_sizes` from the 0.05 m raise limit (`s√3 < 0.05` → `s < 0.0289`), and see the rotation problem in §6.3 before committing to a task.

**E4 — Discount sweep (30 min of sim runs).** γ ∈ {0.99, 0.97, 0.96} at fixed everything else (`configs/sim/train_config.json:26`); compare steps-to-success at matched success rate. *Expect:* lower γ shortens successful episodes, because with +1 terminal reward `Q*(s) = γ^{T(s)}` and the half-life is 6.9 s / 2.28 s / 1.70 s respectively. *If the curves are indistinguishable:* the sim task is too short or too easy for the discount to bind — which itself tells you the sim is a weak proxy for cycle-time claims on hardware.

**E5 — RLPD sampling probe (30 min).** Two runs at identical interaction counts: `online_ratio: 0.5` versus `1.0` (`configs/sim/train_config.json:49`). Then a third with `cfg.dataset` unset, with a print of `batch[ACTION].shape` inserted at `rl/trainer.py:81`. *Expect:* (a) 1.0 is dramatically slower or fails outright, because the demo half is the only reward-dense source early on — with binary sparse reward and 20 demos of ~100 steps, ~1% of demo transitions carry r=1 against ~0% of the early online buffer; (b) the unset-dataset run reports batches of 128, not 256, confirming `data_mixer.py:77` vs `:85-87`. *If (a) shows nothing:* his sim task's reward is not sparse enough to be a proxy, and no RLPD conclusion from sim transfers to the real task.

**E6 — Break the executed-action write-back (30 min). The single highest-value experiment here.** Patch `InterventionActionProcessorStep` so `complementary_data["teleop_action"]` retains the *policy* action during interventions (`processor/hil_processor.py:526-529`), then run sim with your normal intervention style. *Expect:* the critic learns "the policy's action produces the human's outcome," the actor climbs that gradient, and the policy becomes confidently wrong — degrading in a way that looks like instability rather than a bug. *Teaches:* the one quiet bug that kills reproductions, felt rather than read. Revert immediately.

**E7 — Intervention pathway and length (25 min).** First the plumbing: script a fake intervention that stamps `IS_INTERVENTION` for N steps, confirm `len(offline_replay_buffer)` grows by exactly N (`rl/learner.py:974-980`), and confirm the flag arrives as a 0-dim tensor (`utils/transition.py:66-67`), not a bool. Then the science: two matched sim runs, one with 3-step corrective interventions and one with 30-step carry-to-success interventions at the same total intervention budget. Log the critic loss and the `-min_i Q_i` term of the actor loss. *Expect:* long interventions inflate Q early and produce visible critic-loss growth. *If not:* either your task's γ^30 is already negligible or your intervention states are already in-distribution — both worth knowing before doing this on hardware.

**E8 — Classifier autopsy (40 min, offline).** On his recorded frames: (i) after `loss.backward()`, enumerate which parameters have non-None `.grad` — expect only `classifier_head` (`rewards/classifier/modeling_classifier.py:203-235`); (ii) report precision/recall/AUC on held-out data, not the accuracy the code logs (`:259-268`), and note that with ~1 success frame per 150-step episode a constant-zero predictor scores >99%; (iii) sweep `success_threshold` and plot the false-positive rate on held-out negatives; (iv) run the same images through the MEAN_STD normalizer (`processor_classifier.py:58-67`) and raw [0,1] and compare logits — that gap is the train/inference mismatch you will be flying with. *Expect:* high accuracy, poor recall, and a non-trivial normalization gap. *Teaches:* which of the two known classifier defects actually costs him anything, ranked, before §6.1.

**E9 — Truncation, quantified (15 min).** Count what fraction of stored offline transitions have `done=1` after the demo conversion (`rl/buffer.py:686-687`). *Expect:* ~1/episode_length of the offline half. *Teaches:* the discipline that separates the map from the verifier — the original map claimed 50% of every batch and was wrong by two orders of magnitude. Measure before fixing.

**E10 — Hardware timing dry run (20 min, robot powered, no learning).** Actor loop only, untrained policy, tight `end_effector_bounds`, `control_time_s` bounded. Log the *actual* loop period (add a timer around the whole iteration in `rl/actor.py`, since nothing measures it — `:425-427` absorbs the overrun silently). *Expect:* 10 Hz sustained with two 128×128 MJPG streams, IK, and serial writes. *If not:* drop a camera, drop resolution, or move the classifier off the critical path — before you spend an hour collecting demos at a rate you cannot replay.

---

## 5. Instrument the real run

### The metrics that actually exist

Produced by the actor at episode boundaries (`rl/actor.py:399-415`), shifted and logged by the learner (`rl/learner.py:929-941`), prefixed with the mode (`common/wandb_utils.py:195`):

- `train/Episodic reward` — **this is the success curve.** There is no other.
- `train/Intervention rate` — intervention steps / total steps for the episode.
- `train/Episode intervention` — 0/1 flag, any intervention this episode.
- Policy frequency stats, including a 90th-percentile figure that is `1/(90th-percentile time)`, i.e. near-worst-case (`utils/utils.py:422-435`).
- x-axis: `train/Interaction step`.

Learner-side: optimization-step Hz (`rl/learner.py:414-440` — note the timer spans the weight push and the logging block, so it over-reports relative to the true loop rate) plus SAC's loss/grad-norm dict.

### What is missing and must be added before run 1

| Add | Why | Anchor |
|---|---|---|
| True actor loop period | Nothing measures it; `policy_fps` covers preprocess+forward+postprocess only, and the trailing sleep silently absorbs overrun | `rl/actor.py:312-323`, `:425-427` |
| A clean `actor_grad_norm` over `get_optim_params()["actor"]` | The logged one walks 62 tensors carrying the discrete critic's gradient | `sac_algorithm.py:248-252`, `modeling_gaussian_actor.py:55-63` |
| Policy σ (mean over the 3 continuous dims, per episode) | The paper's free diagnostic: σ→0 = open-loop reflex; σ starting ~0.6 and decaying within an episode = closed-loop servoing | `modeling_gaussian_actor.py:456-474`, bounds `configs/sim/train_config.json:156-157` |
| `reward_classifier_frequency` | Written into info and then dropped from the actor's payload; at 10 Hz a slow classifier blows the control period invisibly | written `hil_processor.py:613/630`, dropped at `rl/actor.py:407-414` |
| Human-seconds | Sum of intervention steps ÷ 10. The metric a reproduction should report, and nobody does | derive from `rl/actor.py:355-360` |
| Corrected learnable-param count | The logged 891,849 omits the critic heads (234,498) and `log_alpha`; true ≈1,126,348 | `rl/learner.py:775-776` |

### Diagnostic table

| Curve | Healthy | What a deviation means | Stop threshold |
|---|---|---|---|
| `train/Episodic reward` | 0 or exactly 1 per episode; rising fraction of 1s | Values >1 mean `terminate_on_success` is False or the classifier is re-firing (`hil_processor.py:513-517`, `:619-622`). Flat 0 while you *are* successfully intervening means the reward path is broken, not the policy | Still exactly 0 after 1000 interaction steps *with* successful human interventions → stop; debug reward composition, not hyperparameters |
| `train/Intervention rate` | Starts high, decays monotonically toward 0 | Not decaying = the policy is not absorbing corrections. High reward **and** high intervention rate together = the human is producing the successes — the HG-DAgger signature the paper's baseline exhibits | Not below its hour-1 value by hour 3 → stop; re-examine task difficulty and reward, not SAC |
| Critic loss + actor loss `−min Q` term | Bounded, slowly decreasing | Both climbing while intervention rate is high = the long-intervention overestimation from §3.A. Symmetric sampling gives those transitions half the batch | Critic loss growing an order of magnitude over 30 min → stop, shorten interventions |
| `temperature` / α | Moves slowly upward from 0.01 | Collapse toward 0 in the first hour = premature determinism; remember `target_entropy = −2.0` is already half the SAC heuristic (`sac_algorithm.py:120-125`) and the exp-form loss scales α's own gradient by α (`:410-419`) | α < 1e-4 within the first hour with reward still 0 |
| Policy frequency, 90th pct | ≥ 10 Hz | Below 10 = inference alone misses the budget. **Silence here is not proof** the loop is keeping up — add the loop-period metric | 90th-pct < 10 Hz, or measured loop period > 120 ms |
| Learner optimization Hz | Steady, with periodic spikes at `save_freq` | Steady decline = starved learner or buffer on the wrong device. Big spikes = the checkpoint stall, which re-encodes both buffers to video on the training thread (`rl/learner.py:624-643`) | Below ~5 Hz sustained → your UTD is fictional; fix throughput before interpreting any learning curve |
| Host RSS | Flat after the buffer fills | Unbounded growth = capacity misconfigured; float32 storage at `rl/buffer.py:147` | Any approach to physical RAM → stop, resize |

### Hard-stop conditions

- **`ValueError: EE jump … > 0.05m`** — config error, not a robot problem (`robot_kinematic_processor.py:245`). Restart with smaller `end_effector_step_sizes` or `raise_on_jump=False`.
- **Actor hangs on "Send ready message to Learner"** — restart the **learner**, not the actor (`rl/learner_service.py:42`, `rl/actor.py:455`).
- **You changed `use_torch_compile` between checkpoint and resume** — the critics loaded nothing, with no error (`sac_algorithm.py:548-551`). Discard the resumed run.

### Report the run in human-minutes

Log, per session: demos collected, classifier frames collected, wall-clock, and human-seconds on the stick. The paper's headline (1–2.5 h) excludes classifier data collection, demo collection, and continuous operator attention. His own number will be more useful to a thesis committee than a comparison to theirs.

---

## 6. Where the research contribution lives

Ranked by thesis value × tractability. The constraint that PRs must not block training is satisfied by batching item 7 into one week that overlaps with data collection, and starting items 3 and 4 only after a baseline number exists.

**1. Repair the reward model, then study reward quality as the binding constraint. — Both (3-line PR + a chapter).**
Missing: the paper trains a spatial-embedding head on frozen ResNet-10 features. Here the entire learnable `Sequential` sits under `torch.no_grad()` (`rewards/classifier/modeling_classifier.py:203-235`), so the model is a linear probe on a *randomly initialized* projection — with Dropout still active on those random features. Second defect: the MEAN_STD normalizer exists only in the training pipeline (`rewards/classifier/processor_classifier.py:58-67`) and not at RL time (`hil_processor.py:579-584`). Where the fix goes: narrow the `no_grad` to the trunk; add a normalizer step to the env pipeline (`rl/gym_manipulator.py:441-493`). Difficulty: the PR is trivial; the study is not. The chapter writes itself from the SO-101 constraint table — no force/torque means the classifier carries contact inference from a single scene view, and the dominant reported failure on this hardware is the classifier tracking shadows. Deliverables: precision/recall at fixed operating point before and after, plus downstream success rate. Connects directly to RoHIL (arXiv:2605.19924). **This is the highest value-per-hour item and it is also the thing most likely to make run 2 succeed.**

**2. A real evaluation harness and a reporting protocol. — Both (PR + methods section).**
Missing: `rl/eval_policy.py` is dead four ways (§3.E), and there is no success-rate metric anywhere in the HIL path (`rl/actor.py:405-415`); `pc_success` requires `info['is_success']` from a vectorized env that `RobotEnv` is not. Where: a new script that builds env + processors via `make_processors` (`rl/gym_manipulator.py:441-493`), runs N episodes with the intervention path disabled, and reports success with a Wilson interval. Difficulty: low — an afternoon. Thesis value comes from the protocol, not the code: the paper's 100-trial-per-task number estimates *policy* variance under one seed and one operator, not method variance, and specifies no stopping rule. Pre-register n, seeds, operator, and stopping criterion, and report human-minutes. **Do this before run 1**; you cannot report a number you have no instrument for.

**3. A rotation channel in the action space. — Both; the biggest chapter.**
Missing: `target_wx/wy/wz` are hardwired to 0.0 with an inline comment "we don't have rotation input" (`processor/delta_action_processor.py:114-116`), so a 6-DoF frame task is driven by a 3-DoF command with `orientation_weight=0.01` against `position_weight=1.0` (`model/kinematics.py:98-133`). Measured consequence: over a 10-step lateral sweep the EE rotvec drifts from [0.04, 1.74, 0.04] to [−0.34, 1.74, 0.51]. This is precisely why his star-shape task is deferred. Where the change goes: extend `MapTensorToDeltaActionDictStep` → `MapDeltaActionToRobotActionStep` (`delta_action_processor.py:94-130`) → `EEReferenceAndDelta` (`robot_kinematic_processor.py:118-137`), retune the frame-task weights, and add the gamepad mapping — the parked `feat/gamepad-controller-profiles` branch is its natural home. Difficulty: medium-high, because SO-101 has 5 arm DoF and EE yaw is kinematically coupled to shoulder pan, so the honest deliverable is a characterization of *which* orientation subspace is actually commandable and at what tracking error, not a naive 6-D action space. Chapter: action-space design for underactuated low-cost arms, with a controlled comparison on the star task that 3-DoF cannot do at all. Upstreamable as a PR once the design is settled.

**4. Contact information from STS3215 `present_current`. — Both (small PR, larger study).**
Missing: `tcp_f/t` appears in 9 of the paper's 12 task configs and has no SO-101 analogue. But the plumbing already exists — `MotorCurrentProcessorStep` is in the env pipeline behind `add_current_to_observation` (`rl/gym_manipulator.py:424-439`). What is missing is calibration and evidence that the signal carries usable contact information through plastic-gear stiction. Where: a calibration processor step plus a study. Difficulty: medium-low engineering, medium-high experimental. Chapter: what contact state is recoverable from a hobby servo's current, and does adding it change insertion success on the cube-in-square-hole task — a clean ablation against a baseline he will already own. This is the most defensible "we did what the paper could not do on this hardware" result available to him.

**5. Intervention weighting. — Thesis chapter, unlikely as an upstream PR.**
Missing: `rl/learner.py:974-980` writes every intervention step to both buffers, permanently, with no decay and no confidence weighting — so a correction from minute 5 is sampled at demo rate at hour 3. The implementation treats interventions as optimal everywhere, which is exactly SiLRI's published critique (arXiv:2512.24288, claiming ≥50% reduction in time-to-90%-success). Where: the dual-write at `learner.py:974-980` and the mixer at `data_sources/data_mixer.py:58-97`. Difficulty: the code change is small; the experiment is expensive in robot time, which is why it comes after items 1–3. Note the honest baseline problem: SiLRI's comparison is against the reference HIL-SERL, not this port, so he would need his own matched runs.

**6. The half-target network. — PR + a cheap sim ablation.**
Missing: `critic_target` is constructed with the *same encoder object* as `critic_ensemble` (`sac_algorithm.py:82` and `:90`), so Polyak on the encoder parameters is `p ← p` (`:422-440`) and the target Q is computed on features updated by the previous gradient step. With `freeze_vision_encoder: true` the ResNet is frozen but 657,088 params of spatial embeddings / post-encoders / state encoder are not. Where: give `critic_target` its own encoder, or exclude encoder params from the Polyak zip. Difficulty: low code, and the ablation runs in sim overnight (measure critic-loss stability and time-to-success at matched seeds). Thin as a chapter on its own; strong as a subsection of a "what this port gets wrong and what it costs" appendix, and a good upstream credential.

**7. The one-week PR batch (do these while data collects).** Each is an hour to a day, all pure PRs, none requiring robot time:
- Actor grad-clip over `get_optim_params()["actor"]` instead of `policy.actor.parameters()` (`sac_algorithm.py:250-252`).
- Expose `raise_on_jump` and `max_ee_step_m` in the processor config (`robot_kinematic_processor.py:212`, `rl/gym_manipulator.py:510-512`) — a workspace excursion currently kills a training run instead of clamping.
- `n_online` computed before the offline-None early exit, silently halving the batch (`data_sources/data_mixer.py:77` vs `:85-87`).
- Gripper penalty reaches the loss only when `complementary_info` is included (`sac_algorithm.py:203` vs `:221`), i.e. never at `utd_ratio: 1`; and the step default −0.02 (`hil_processor.py:368`) is always overridden by `GripperConfig.gripper_penalty = 0.0` (`envs/configs.py:277`).
- `save_freq=0` → `ZeroDivisionError` at `rl/learner.py:459` instead of using the existing guard (`common/train_utils.py:55-63`).
- Resume path drops `storage_device` (`rl/learner.py:819-825`).
- Truncation in the offline converter (`rl/buffer.py:686-687`, existing `TODO`) — correct it, and report honestly that the measured effect is <1% of a batch.
- Dead config that should either be wired up or deleted: `reset_time_s` (`rl/gym_manipulator.py:348-353`), `use_tanh_squash` (`modeling_gaussian_actor.py:411/421`), `async_prefetch` (`configuration_gaussian_actor.py:159`), `episode_ends` (`rl/buffer.py:122`).

Fix `#4297` remains open; these stack behind it and give him a coherent contribution record without a single hour of robot time.

**8. Human-minutes accounting. — Methods section + a small logging PR.**
Nobody reports it, and the brief's criticism (f) is correct that it is the metric a reproduction should carry. `rl/actor.py:355-360` already counts intervention steps; adding human-seconds to the payload at `:407-414` is a five-line change. Cheap, honest, and it reframes his own results as a contribution rather than a shortfall against a number computed differently.