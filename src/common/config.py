"""Config loading. All thresholds live in configs/*.yaml (plan §10.3)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml (name without extension)."""
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_all() -> dict[str, dict[str, Any]]:
    return {
        "detection": load_config("detection"),
        "tracking": load_config("tracking"),
        "tagging": load_config("tagging"),
    }
