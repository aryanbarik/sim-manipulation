.PHONY: setup test lint
setup:
	uv sync --extra cpu --all-groups
setup-rocm:
	uv sync --extra cpu
	uv pip install --reinstall --index-url https://nightly.repo.amd.com/rocm/whl-next/ 'torch[device-gfx1031]==2.14.*' 'torchvision[device-gfx1031]==0.29.*' 'numpy>=2,<2.3' 'filelock>=4.2.1,<5'
test:
	uv run --extra cpu pytest -m 'not network and not slow and not gpu'
lint:
	uv run --extra cpu ruff check
