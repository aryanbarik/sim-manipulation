"""Record reproducible experiment configuration and source revision."""

import subprocess
from pathlib import Path

import yaml


def write_manifest(run_dir: Path, config: dict, *, allow_dirty: bool = False) -> None:
    """Write configuration and revision; reject uncommitted source by default."""
    root = Path(__file__).resolve().parents[3]
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    )
    if status and not allow_dirty:
        raise RuntimeError("refusing to record a run from a dirty working tree")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    (run_dir / "git_sha.txt").write_text(sha + "\n")
