# Video encoding and decoding through PyAV

`import torchcodec` fails in this venv with `OSError: Could not load this library: .../torchcodec/libtorchcodec_image.so`. `ldd` on that library reports `not found` for `libtorch.so`, `libc10.so`, `libtorch_cpu.so`, `libtorch_cuda.so`, `libc10_cuda.so`, `libnvrtc.so.13`, and `libcudart.so.13`. The venv's torch is the ROCm build `2.14.0+rocm10.3.0a20261123`. The cause is not isolated.

Decision: use lerobot's PyAV backend, `video_backend="pyav"` in code and `--dataset.video_backend=pyav` on `lerobot-train`. This is the torchcodec fallback in the plan's Appendix C §11. In this venv PyAV 15.1.0 has the `libsvtav1` encoder (lerobot's default codec, `configs/video.py`), `libx264`, and the `libdav1d` decoder; `h264_vaapi` is unavailable, so there is no hardware encoding path. Phase 1 and later scripts pass the backend explicitly, so a torchcodec-enabled environment does not change results silently.

Consequences: video encoding and decoding run on the CPU in the training data path. Training speed with PyAV is measured by the GPU run of `scripts/spike_dataset.py`, and the result is recorded in `0003-feasibility.md`. `scripts/bench_train.py` uses random tensors and does not exercise video decoding.
