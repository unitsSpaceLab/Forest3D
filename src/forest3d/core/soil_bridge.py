"""Bridge between Forest3D terrain and gz_terramechanics soil parameters.

Reads soil types from soil_params.yaml and writes terrain_mapping.yaml
so the terramechanics plugin applies the chosen soil to generated terrain.
"""

from pathlib import Path

import yaml


def _find_plugins_dir() -> Path:
    """Find the plugins/gz_terramechanics directory relative to package root."""
    # Walk up from this file: core/ -> forest3d/ -> src/ -> Forest3D/
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent
    plugins_dir = pkg_root / "plugins" / "gz_terramechanics"
    if not plugins_dir.exists():
        raise FileNotFoundError(
            f"gz_terramechanics plugin not found at {plugins_dir}\n"
            "Ensure the plugin is installed in Forest3D/plugins/gz_terramechanics/"
        )
    return plugins_dir


def get_plugin_config_dir(config_dir: Path = None) -> Path:
    """Resolve path to plugins/gz_terramechanics/config/."""
    if config_dir is not None:
        return Path(config_dir)
    return _find_plugins_dir() / "config"


def get_plugin_build_dir() -> Path:
    """Resolve path to plugins/gz_terramechanics/build/."""
    return _find_plugins_dir() / "build"


def get_plugin_source_dir() -> Path:
    """Resolve path to plugins/gz_terramechanics/."""
    return _find_plugins_dir()


def get_available_soils(config_dir: Path = None) -> dict:
    """Read soil types from soil_params.yaml.

    Returns:
        Dict mapping soil name to its parameter dict.
    """
    config_dir = get_plugin_config_dir(config_dir)
    soil_file = config_dir / "soil_params.yaml"
    if not soil_file.exists():
        raise FileNotFoundError(f"soil_params.yaml not found at {soil_file}")
    with open(soil_file) as f:
        return yaml.safe_load(f)


def write_terrain_mapping(
    soil_name: str, pattern: str = "terrain", config_dir: Path = None
):
    """Write terrain_mapping.yaml mapping a collision pattern to a soil type.

    Args:
        soil_name: Name of soil type (must exist in soil_params.yaml).
        pattern: Collision name pattern to match (default: "terrain").
        config_dir: Override config directory path.

    Raises:
        ValueError: If soil_name not found in soil_params.yaml.
    """
    config_dir = get_plugin_config_dir(config_dir)
    soils = get_available_soils(config_dir)

    if soil_name not in soils:
        available = ", ".join(soils.keys())
        raise ValueError(
            f"Unknown soil type '{soil_name}'. Available: {available}"
        )

    mapping = {
        "default_soil": soil_name,
        "terrain_mapping": [
            {"pattern": pattern, "soil": soil_name},
        ],
    }

    mapping_file = config_dir / "terrain_mapping.yaml"
    with open(mapping_file, "w") as f:
        yaml.dump(mapping, f, default_flow_style=False, sort_keys=False)

    return mapping_file
