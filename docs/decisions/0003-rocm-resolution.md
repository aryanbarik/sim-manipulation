# ROCm installation fallback

`uv lock` could not resolve the explicit ROCm extra: the PyTorch wheel requires `rocm`, and after routing that package to TheRock the resolver could not find its transitive `rocm-sdk-core` package. Use the planned `make setup-rocm` contingency and run GPU commands with `uv run --no-sync` to prevent the CPU lockfile from replacing the installed ROCm wheels. GPU feasibility remains untested pending device access.
