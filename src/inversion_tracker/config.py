"""Project configuration: paths from configs/paths.yaml, resolved against the repo root."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text()) or {}


def load_paths() -> dict[str, Path]:
    """Absolute paths are used as-is; relative ones are resolved from the repo root."""
    paths = {}
    for key, value in load_yaml("paths.yaml").items():
        p = Path(value)
        paths[key] = p if p.is_absolute() else REPO_ROOT / p
    return paths
