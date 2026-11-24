# Feasibility gate: pending

The current session does not have `render` or `docker` group membership; `/dev/kfd` is restricted to `root:render`. TheRock ROCm wheels were installed through the documented fallback, but `torch.cuda.is_available()` returned false both with and without `HSA_OVERRIDE_GFX_VERSION=10.3.0`. This does not establish that gfx1031 training fails once device access is granted. Re-run the GPU, dataset and ACT/Diffusion spikes after the group change and a new login before choosing outcome A or B.

On this session's simple 128×128 MuJoCo rendering probe, EGL and OSMesa both worked; EGL rendered approximately 2181 frames/s and OSMesa 169 frames/s. These are simple-scene probe results, not SO-101 environment throughput.

## Spike results

CPU runs used `--device cpu` and check code paths only. Bench numbers come only from GPU runs under `flock /tmp/ml-portfolio-bench.lock`.

| Spike | Status | Result |
| --- | --- | --- |
| 1. `scripts/check_env.py` | GPU pending | `torch.cuda.is_available()` false with and without `HSA_OVERRIDE_GFX_VERSION=10.3.0` (device access, above). |
| 2. `scripts/spike_dataset.py` | CPU PASS; GPU pending | `--device cpu`: create, finalize, reload and train all PASS (ACT, 10 steps, `lerobot-train`, PyAV video); exit 0. |
| 3. `scripts/bench_train.py` | CPU smoke; GPU pending | ACT and Diffusion step paths run on CPU (batch 2, 1 step, fp32). Not a measurement. |
| 4. Menagerie SO-101 model | PASS | Loads and compiles under `mujoco` 3.17.0; details below. |

CPU checks for the Phase 0 done-check: `uv run --no-sync ruff check .` clean; `uv run --no-sync pytest -q -m "not network and not slow and not gpu"` 1 passed.

### Spike 4: SO-101 model

`mujoco_menagerie.load("robotstudio_so101", entry="so101")` (asset `robotstudio_so101-8b739c94709abfe1.tar.xz`, Apache-2.0):

- Compiled model: `nq`=6, `nu`=6, `njnt`=6, `ncam`=1, timestep 0.005 s.
- Joint ranges (rad): shoulder_pan [-1.9199, 1.9199]; shoulder_lift [-1.7453, 1.7453]; elbow_flex [-1.69, 1.69]; wrist_flex [-1.6581, 1.6581]; wrist_roll [-2.7438, 2.8412]; gripper [-0.1745, 1.7453].
- Six position actuators, one per joint, named after the joint: `kp` 998.22, `gaintype` 0, `biastype` 1, `ctrlrange` equal to the joint range.
- One camera, `wrist_cam`. The robot model has no front camera; the front view must come from the scene. Phase 1 decides and records it here.
- The wrist camera exists in the model. Rendering it is not yet tested; if it is unusable, attach a camera through MjSpec (Appendix C).
- Loads under `mujoco` 3.x, so the `mujoco==4.0.1` fallback is not needed. `trs_so_arm100` (SO-100) is not used.

### Video backend

`import torchcodec` fails in this venv (`libtorchcodec_image.so` cannot load; `ldd` reports unresolved torch and CUDA runtime libraries; cause not isolated). The spikes use `video_backend="pyav"`; see `0004-video-backend-pyav.md`.

### Outstanding (needs GPU access)

- Spike 1: `uv run --no-sync python scripts/check_env.py` after the group change and a new login.
- Spike 2: `uv run --no-sync python scripts/spike_dataset.py --device cuda`.
- Spike 3: `flock /tmp/ml-portfolio-bench.lock uv run --no-sync python scripts/bench_train.py --policy act --precision fp16`, then `--policy diffusion --precision fp16`; batch 32, 128×128 cameras. Record steps/s and peak VRAM here.

GPU results decide Outcome A or B (Step 2). A permission or device-access failure is not Outcome B; Outcome B applies only after the `HSA_OVERRIDE_GFX_VERSION=10.3.0` retry fails with device access granted. Measured values supersede the planning numbers in Appendix C §5.
