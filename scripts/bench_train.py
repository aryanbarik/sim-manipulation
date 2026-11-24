"""Feasibility spike: training-step throughput and peak memory for ACT and Diffusion Policy.

Batch 32 by default, two 128x128 cameras (observation.images.front, observation.images.wrist),
each policy's default architecture. A measured step is forward, backward and an optimizer step
on random tensors with the dataset shapes. Peak memory is torch.cuda.max_memory_allocated over
the measured window; reserved memory is reported too.

These are GPU measurements. Run them under the machine lock:
  flock /tmp/ml-portfolio-bench.lock uv run --no-sync python scripts/bench_train.py --policy act
  flock /tmp/ml-portfolio-bench.lock uv run --no-sync python scripts/bench_train.py --policy diffusion

--device cpu is a code-path smoke test; its output is not a benchmark result.
Prints one JSON object per run.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

CAMERAS = ("observation.images.front", "observation.images.wrist")
IMG = 128
STATE_DIM = 6
ACTION_DIM = 6
IS_PAD = "action_is_pad"


def build(policy: str, device: torch.device):
    inputs = {cam: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMG, IMG)) for cam in CAMERAS}
    inputs[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))
    outputs = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    if policy == "act":
        cfg = ACTConfig(
            input_features=inputs,
            output_features=outputs,
            pretrained_backbone_weights=None,
            device=device.type,
        )
        return ACTPolicy(cfg).to(device), cfg
    cfg = DiffusionConfig(
        input_features=inputs,
        output_features=outputs,
        pretrained_backbone_weights=None,
        device=device.type,
    )
    return DiffusionPolicy(cfg).to(device), cfg


def make_batch(policy: str, cfg, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    if policy == "act":
        obs, horizon = (), cfg.chunk_size
    else:
        obs, horizon = (cfg.n_obs_steps,), cfg.horizon
    batch = {cam: torch.rand(batch_size, *obs, 3, IMG, IMG, device=device) for cam in CAMERAS}
    batch[OBS_STATE] = torch.randn(batch_size, *obs, STATE_DIM, device=device)
    batch[ACTION] = torch.randn(batch_size, horizon, ACTION_DIM, device=device)
    batch[IS_PAD] = torch.zeros(batch_size, horizon, dtype=torch.bool, device=device)
    return batch


def train_step(policy, opt, scaler, batch, amp: bool, device_type: str) -> None:
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp):
        loss, _ = policy.forward(batch)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=["act", "diffusion"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    amp = args.precision == "fp16"
    if amp and device.type != "cuda":
        parser.error("--precision fp16 requires --device cuda")

    torch.manual_seed(0)
    policy, cfg = build(args.policy, device)
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    batch = make_batch(args.policy, cfg, args.batch_size, device)

    for _ in range(args.warmup):
        train_step(policy, opt, scaler, batch, amp, device.type)
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(args.steps):
        train_step(policy, opt, scaler, batch, amp, device.type)
    sync(device)
    elapsed = time.perf_counter() - t0

    result = {
        "policy": args.policy,
        "device": args.device,
        "precision": args.precision,
        "batch_size": args.batch_size,
        "image_size": IMG,
        "cameras": len(CAMERAS),
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "seconds_per_step": round(elapsed / args.steps, 4),
        "steps_per_s": round(args.steps / elapsed, 4),
        "torch": torch.__version__,
        "hip": torch.version.hip,
    }
    if device.type == "cuda":
        result.update(
            gpu=torch.cuda.get_device_name(device),
            peak_allocated_gib=round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
            peak_reserved_gib=round(torch.cuda.max_memory_reserved(device) / 2**30, 3),
        )
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
