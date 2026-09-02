#!/usr/bin/env python
"""Inspect a LeRobot dataset: summary, per-episode detail, frames, joint angles.

Everything a HIL-SERL dataset holds is reachable from here — the recorded images,
the six servo angles, the actions the policy or the human issued, and the reward.

Usage (always `source .venv/bin/activate` first):

    # what is in the dataset
    python tools/inspect_dataset.py lucarp/so101_cube_lift

    # one episode in detail (per-step table)
    python tools/inspect_dataset.py lucarp/so101_cube_lift --episode 3

    # write a contact sheet of frames + a joint-angle plot for that episode
    python tools/inspect_dataset.py lucarp/so101_cube_lift --episode 3 --export /tmp/ep3

    # a dataset living inside a training run rather than the hub cache
    python tools/inspect_dataset.py lucarp/x --root outputs/hilserl_lift_run1/dataset

The raw video files are plain mp4 and can be opened in any player:
    <dataset root>/videos/observation.images.{scene,wrist}/chunk-*/file-*.mp4
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

HUB = Path("/mnt/ai/lucas/huggingface/lerobot")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# action[3] is the discrete gripper. NOTE the labels are inverted relative to the
# hardware: GripperVelocityToJoint maps 0 -> +velocity (raises the joint = OPENS)
# and 2 -> -velocity (CLOSES). See robot_kinematic_processor.GripperVelocityToJoint.
GRIPPER = {0: "open", 1: "stay", 2: "CLOSE"}


def resolve(repo_id: str, root: str | None) -> Path:
    if root:
        return Path(root)
    for cand in (HUB / repo_id, Path.home() / ".cache/huggingface/lerobot" / repo_id):
        if (cand / "meta/info.json").exists():
            return cand
    raise SystemExit(f"could not find {repo_id}; pass --root")


def load(path: Path) -> tuple[dict, pd.DataFrame]:
    info = json.loads((path / "meta/info.json").read_text())
    files = sorted(glob.glob(str(path / "data/**/*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet under {path}/data")
    return info, pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def summarise(path: Path, info: dict, df: pd.DataFrame) -> None:
    print(f"{path}\n  {info['total_episodes']} episodes · {info['total_frames']} frames · {info['fps']} fps")
    print("  features:")
    for key, feat in info["features"].items():
        print(f"    {key:42} {str(feat.get('dtype')):8} {feat.get('shape')}")

    ep = np.asarray(df["episode_index"]).astype(int)
    reward = np.asarray(df["next.reward"]).astype(float).ravel()
    episodes = sorted(set(ep.tolist()))
    succeeded = [e for e in episodes if (reward[ep == e] > 0).any()]
    lengths = np.array([(ep == e).sum() for e in episodes])
    print(f"\n  episodes with a reward: {len(succeeded)}/{len(episodes)}")
    print(f"  length: mean {lengths.mean():.0f} steps ({lengths.mean() / info['fps']:.1f}s)"
          f"  min {lengths.min()}  max {lengths.max()}")

    actions = np.stack(df["action"].values)
    print(f"  action dim {actions.shape[1]}: "
          f"min {np.round(actions.min(0), 2).tolist()} max {np.round(actions.max(0), 2).tolist()}")
    grip = actions[:, 3]
    print("  gripper commands: "
          + "  ".join(f"{name}={int((grip == code).sum())}" for code, name in GRIPPER.items()))

    if "complementary_info.is_intervention" in df:
        iv = np.asarray(df["complementary_info.is_intervention"]).astype(float).ravel()
        print(f"  human intervention: {100 * iv.mean():.0f}% of frames")

    videos = sorted(glob.glob(str(path / "videos/**/*.mp4"), recursive=True))
    if videos:
        print("\n  video files (open these in any player):")
        for v in videos[:4]:
            print(f"    {v}")


def show_episode(path: Path, info: dict, df: pd.DataFrame, index: int) -> pd.DataFrame:
    ep = np.asarray(df["episode_index"]).astype(int)
    rows = df[ep == index].reset_index(drop=True)
    if rows.empty:
        raise SystemExit(f"episode {index} not in dataset")
    state = np.stack(rows["observation.state"].values)
    actions = np.stack(rows["action"].values)
    reward = np.asarray(rows["next.reward"]).astype(float).ravel()
    has_iv = "complementary_info.is_intervention" in rows
    iv = (np.asarray(rows["complementary_info.is_intervention"]).astype(float).ravel()
          if has_iv else np.full(len(rows), np.nan))

    print(f"\nepisode {index}: {len(rows)} steps ({len(rows) / info['fps']:.1f}s)"
          f" · reward on {int((reward > 0).sum())} frame(s)"
          + (f" · {100 * iv.mean():.0f}% intervened" if has_iv else " · no intervention column (demo recording)"))
    print(f"\n{'step':>4} {'dx':>6} {'dy':>6} {'dz':>6} {'grip':>6} "
          + " ".join(f"{j[:9]:>9}" for j in JOINTS) + f" {'rew':>4} {'who':>6}")
    step = max(1, len(rows) // 25)  # ~25 rows regardless of episode length
    for i in range(0, len(rows), step):
        a, q = actions[i], state[i]
        print(f"{i:>4} {a[0]:6.2f} {a[1]:6.2f} {a[2]:6.2f} {GRIPPER.get(int(a[3]), '?'):>6} "
              + " ".join(f"{v:9.2f}" for v in q)
              + f" {reward[i]:4.0f} "
              + f"{'-' if not has_iv else ('human' if iv[i] > 0.5 else 'policy'):>6}")
    return rows


def export(rows: pd.DataFrame, info: dict, out: Path, index: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    state = np.stack(rows["observation.state"].values)
    reward = np.asarray(rows["next.reward"]).astype(float).ravel()

    # joint angles over the episode
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(rows)) / info["fps"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, name in enumerate(JOINTS):
        ax.plot(t, state[:, j], label=name, linewidth=1.4)
    for hit in np.where(reward > 0)[0]:
        ax.axvline(t[hit], color="k", linestyle="--", linewidth=1, label="reward")
    ax.set_xlabel("seconds"), ax.set_ylabel("joint position (deg)")
    ax.set_title(f"episode {index} — servo angles")
    ax.legend(loc="upper left", fontsize=8, ncol=4), ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"ep{index}_joints.png", dpi=110)
    print(f"  wrote {out / f'ep{index}_joints.png'}")

    # a contact sheet of frames, decoded through the dataset (handles video or image dtype)
    try:
        import torch
        from PIL import Image
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(repo_id=info.get("repo_id", "x"), root=str(ROOT))
        gi = np.where(np.asarray(ds.hf_dataset["episode_index"]).astype(int) == index)[0]
        picks = gi[np.linspace(0, len(gi) - 1, 8).astype(int)]
        for cam in ("scene", "wrist"):
            key = f"observation.images.{cam}"
            if key not in ds[int(picks[0])]:
                continue
            tiles = []
            for g in picks:
                im = ds[int(g)][key]
                arr = im.permute(1, 2, 0).numpy()
                arr = (arr * 255).astype("uint8") if im.dtype == torch.float32 else arr.astype("uint8")
                tiles.append(arr)
            sheet = np.hstack(tiles)
            Image.fromarray(sheet).save(out / f"ep{index}_{cam}.png")
            print(f"  wrote {out / f'ep{index}_{cam}.png'}  (8 frames, start -> end)")
    except Exception as exc:  # decoding needs torchcodec + the venv's LD_LIBRARY_PATH
        print(f"  (frame export skipped: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo_id")
    p.add_argument("--root", default=None, help="dataset dir, if not in the hub cache")
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--export", default=None, metavar="DIR")
    args = p.parse_args()

    ROOT = resolve(args.repo_id, args.root)
    info, df = load(ROOT)
    info.setdefault("repo_id", args.repo_id)
    summarise(ROOT, info, df)
    if args.episode is not None:
        rows = show_episode(ROOT, info, df, args.episode)
        if args.export:
            export(rows, info, Path(args.export), args.episode)
