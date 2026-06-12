"""Configuration loading with cascading defaults."""

import os
from pathlib import Path
from typing import Optional

import yaml

from forest3d.config.schema import Forest3DConfig


CONFIG_SEARCH_PATHS = [
    Path.cwd() / "forest3d.yaml",
    Path.cwd() / ".forest3d.yaml",
    Path.home() / ".config" / "forest3d" / "config.yaml",
    Path.home() / ".forest3d.yaml",
]


def find_config_file() -> Optional[Path]:
    """Search for config file in standard locations."""
    for path in CONFIG_SEARCH_PATHS:
        if path.exists():
            return path
    return None


def load_config(config_path: Optional[Path] = None) -> Forest3DConfig:
    """Load config with cascading defaults (env > explicit > auto > built-in)."""
    config_dict: dict = {}

    if config_path is None:
        config_path = find_config_file()
    elif not isinstance(config_path, Path):
        config_path = Path(config_path)

    if config_path and config_path.exists():
        with open(config_path) as f:
            config_dict = yaml.safe_load(f) or {}

    if env_blender := os.environ.get("FOREST3D_BLENDER_PATH"):
        config_dict.setdefault("blender", {})["path"] = env_blender

    if env_base := os.environ.get("FOREST3D_BASE_PATH"):
        config_dict.setdefault("paths", {})["base_path"] = env_base

    if env_models := os.environ.get("FOREST3D_MODELS_PATH"):
        config_dict.setdefault("paths", {})["models_path"] = env_models

    return Forest3DConfig(**config_dict)


def config_to_yaml(config: Forest3DConfig, exclude_none: bool = False) -> str:
    """Serialize config to YAML string."""
    config_dict = config.model_dump(exclude_none=exclude_none)

    def convert_paths(obj):
        if isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    config_dict = convert_paths(config_dict)
    return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)


def save_config(
    config: Forest3DConfig, path: Path, exclude_none: bool = False
) -> None:
    """Save configuration to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config_to_yaml(config, exclude_none=exclude_none))
