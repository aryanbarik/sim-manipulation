"""Run independent GPU and headless-rendering feasibility probes."""

import os
import subprocess
import sys

GPU = r'''
import torch
from torchvision.models import resnet18
assert torch.cuda.is_available(), "torch reports no accessible GPU"
device = torch.device("cuda")
x = torch.randn((512, 512), device=device, dtype=torch.float16)
y = x @ x
assert torch.isfinite(y).all().item(), "non-finite fp16 matmul"
model = resnet18().to(device).train()
opt = torch.optim.SGD(model.parameters(), lr=0.001)
opt.zero_grad()
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = model(torch.randn(2, 3, 128, 128, device=device)).square().mean()
loss.backward()
opt.step()
torch.cuda.synchronize()
print("device:", torch.cuda.get_device_name(0))
print("fp16 matmul and ResNet-18 training step passed")
'''

RENDER = r'''
import time
import mujoco
model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 2"/><geom type="sphere" size=".1"/></worldbody></mujoco>')
data = mujoco.MjData(model)
with mujoco.Renderer(model, height=128, width=128) as renderer:
    renderer.update_scene(data)
    renderer.render()
    start = time.perf_counter()
    for _ in range(100):
        renderer.update_scene(data)
        renderer.render()
    print("render fps:", 100 / (time.perf_counter() - start))
'''


def probe(name: str, code: str, env: dict[str, str]) -> bool:
    print(f"--- {name} ---", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], env=env, timeout=120, check=False
        )
        passed = result.returncode == 0
    except subprocess.TimeoutExpired:
        print("timed out", flush=True)
        passed = False
    print(f"{'PASS' if passed else 'FAIL'}: {name}", flush=True)
    return passed


def main() -> int:
    base = os.environ.copy()
    base.pop("HSA_OVERRIDE_GFX_VERSION", None)
    native = probe("GPU native", GPU, base)
    override = dict(base, HSA_OVERRIDE_GFX_VERSION="10.3.0")
    retried = probe("GPU HSA override", GPU, override)
    egl = probe("render EGL", RENDER, dict(base, MUJOCO_GL="egl"))
    osmesa = probe("render OSMesa", RENDER, dict(base, MUJOCO_GL="osmesa"))
    return 0 if (native or retried) and (egl or osmesa) else 1


if __name__ == "__main__":
    raise SystemExit(main())
