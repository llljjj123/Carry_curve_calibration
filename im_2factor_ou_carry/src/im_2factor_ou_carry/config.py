"""Configuration loading and path resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve project paths relative to the configuration file."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config = deepcopy(config)
    base = config_path.parent
    project = config.setdefault("project", {})
    for key in ("raw_dir", "output_dir", "chart_dir"):
        value = Path(project[key])
        project[key] = value if value.is_absolute() else (base / value).resolve()
    override = config.get("data", {}).get("expiry_override_path")
    if override:
        override_path = Path(override)
        config["data"]["expiry_override_path"] = (
            override_path if override_path.is_absolute() else (base / override_path).resolve()
        )
    config["config_path"] = config_path
    config["project_root"] = base
    return config


def ensure_directories(config: dict[str, Any]) -> None:
    """Create configured data and output directories."""
    for key in ("raw_dir", "output_dir", "chart_dir"):
        Path(config["project"][key]).mkdir(parents=True, exist_ok=True)
