"""Feasibility spike: LeRobotDataset v3 with video -> finalize -> reload -> ACT train steps.

Stages, each reported PASS or FAIL; the first failure stops the run:
  create    5 episodes x 40 frames; observation.images.{front,wrist} as 128x128x3 uint8 video,
            observation.state (6), observation.environment_state (8), action (6); pyav backend;
            finalize() is called at the end
  finalize  meta/info.json and one mp4 per camera exist on disk
  reload    reopen from disk; episode/frame counts and decoded sample shapes match
  train     lerobot-train ACT for 10 steps on --device

--device cpu is a pre-check. The GPU spike is --device cuda and counts only when that run passes.

Usage:
  uv run --no-sync python scripts/spike_dataset.py --device cpu
  uv run --no-sync python scripts/spike_dataset.py --device cuda
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "local/simmanip-spike"
CAMERAS = ("observation.images.front", "observation.images.wrist")
IMG = 128
STATE_DIM = 6
ENV_DIM = 8
EPISODES = 5
FRAMES = 40
FPS = 30
TRAIN_STEPS = 10
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _check(ok: bool, msg: str) -> None:
    if not ok:
        raise RuntimeError(msg)


def stage_create(root: Path) -> None:
    features: dict[str, dict] = {
        cam: {"dtype": "video", "shape": (IMG, IMG, 3), "names": ["height", "width", "channels"]}
        for cam in CAMERAS
    }
    features["observation.state"] = {"dtype": "float32", "shape": (STATE_DIM,), "names": JOINTS}
    features["observation.environment_state"] = {"dtype": "float32", "shape": (ENV_DIM,), "names": None}
    features["action"] = {"dtype": "float32", "shape": (STATE_DIM,), "names": JOINTS}
    ds = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
        root=root,
        robot_type="so101",
        use_videos=True,
        video_backend="pyav",
    )
    rng = np.random.default_rng(0)
    for _ in range(EPISODES):
        for _ in range(FRAMES):
            state = rng.standard_normal(STATE_DIM).astype(np.float32)
            frame: dict[str, object] = {
                cam: rng.integers(0, 256, size=(IMG, IMG, 3), dtype=np.uint8) for cam in CAMERAS
            }
            frame["observation.state"] = state
            frame["observation.environment_state"] = rng.standard_normal(ENV_DIM).astype(np.float32)
            frame["action"] = state.copy()
            frame["task"] = "spike reach"
            ds.add_frame(frame)
        ds.save_episode()
    ds.finalize()


def stage_finalize(root: Path) -> None:
    _check((root / "meta" / "info.json").is_file(), "meta/info.json missing")
    for cam in CAMERAS:
        mp4s = [p for p in (root / "videos").rglob("*.mp4") if cam in p.relative_to(root).parts]
        _check(bool(mp4s), f"no mp4 for {cam}")


def stage_reload(root: Path) -> None:
    ds = LeRobotDataset(REPO_ID, root=root, video_backend="pyav")
    _check(ds.num_episodes == EPISODES, f"num_episodes={ds.num_episodes}, expected {EPISODES}")
    _check(ds.num_frames == EPISODES * FRAMES, f"num_frames={ds.num_frames}, expected {EPISODES * FRAMES}")
    sample = ds[0]
    for cam in CAMERAS:
        _check(tuple(sample[cam].shape) == (3, IMG, IMG), f"{cam} shape {tuple(sample[cam].shape)}")
    _check(tuple(sample["observation.state"].shape) == (STATE_DIM,), "observation.state shape mismatch")


def stage_train(root: Path, device: str, out: Path) -> None:
    exe = shutil.which("lerobot-train")
    _check(exe is not None, "lerobot-train not on PATH; run through uv run")
    shutil.rmtree(out, ignore_errors=True)
    log = Path(f"{out}.log")
    cmd = [
        exe,
        f"--dataset.repo_id={REPO_ID}",
        f"--dataset.root={root}",
        "--dataset.video_backend=pyav",
        "--policy.type=act",
        f"--policy.device={device}",
        "--policy.push_to_hub=false",
        f"--output_dir={out}",
        "--job_name=spike_act",
        f"--steps={TRAIN_STEPS}",
        f"--save_freq={TRAIN_STEPS}",
        "--batch_size=8",
        "--num_workers=2",
        "--log_freq=1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
    log.write_text(proc.stdout + proc.stderr)
    _check(proc.returncode == 0, f"lerobot-train exit {proc.returncode}; log: {log}")
    _check((out / "checkpoints" / "last").exists(), f"no checkpoints/last under {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda", help="policy device for the train stage: cuda or cpu")
    parser.add_argument("--root", type=Path, default=Path("artifacts/spike_dataset"))
    args = parser.parse_args()
    root: Path = args.root
    out = root.parent / "spike_train"
    shutil.rmtree(root, ignore_errors=True)
    stages = [
        ("create", lambda: stage_create(root)),
        ("finalize", lambda: stage_finalize(root)),
        ("reload", lambda: stage_reload(root)),
        ("train", lambda: stage_train(root, args.device, out)),
    ]
    print(f"spike_dataset device={args.device} root={root}", flush=True)
    for name, run in stages:
        try:
            run()
        except Exception as exc:  # later stages depend on this one; stop here
            print(f"FAIL {name}: {exc}", flush=True)
            raise
        print(f"PASS {name}", flush=True)
    print("ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
