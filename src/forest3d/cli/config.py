"""Configuration management CLI."""

from pathlib import Path

import click
import yaml

from forest3d.config.loader import (
    config_to_yaml,
    find_config_file,
    load_config,
)
from forest3d.config.schema import DensityConfig, Forest3DConfig
from forest3d.core.terrain_base import get_terrain_class, list_terrain_types


def _tailored_template(terrain_type: str) -> str:
    """Build a YAML template scoped to a single terrain type.

    Keeps only the terrain sub-block and densities relevant to
    *terrain_type*, so a crop user isn't handed forest options (and vice
    versa). Densities are limited to fields the schema can carry, so the
    template re-loads cleanly.
    """
    cls = get_terrain_class(terrain_type)
    cfg = Forest3DConfig()
    cfg.terrain.type = terrain_type
    data = cfg.model_dump(exclude_none=True)

    # Drop the terrain sub-blocks for the other terrain types.
    terrain = data["terrain"]
    for other in list_terrain_types():
        if other != terrain_type:
            terrain.pop(other, None)

    # Densities: this terrain's class defaults, limited to schema fields.
    valid = set(DensityConfig.model_fields)
    data["density"] = {
        k: v for k, v in cls.default_densities().items() if k in valid
    }

    def convert_paths(obj):
        if isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        if isinstance(obj, Path):
            return str(obj)
        return obj

    return yaml.dump(
        convert_paths(data), default_flow_style=False, sort_keys=False
    )


@click.group()
def config():
    """Manage Forest3D configuration files."""


@config.command("init")
@click.argument("terrain_type", required=False)
@click.option(
    "--output", "-o", "output_path", type=click.Path(),
    default="./forest3d.yaml", show_default=True,
    help="Where to write the config file",
)
@click.option(
    "--force", "-f", is_flag=True,
    help="Overwrite an existing config file",
)
@click.pass_context
def init(ctx, terrain_type, output_path, force):
    """Write a forest3d.yaml template (TERRAIN_TYPE: dem or crop_rows).

    Example: forest3d config init crop_rows
    """
    console = ctx.obj["console"]
    types = list_terrain_types()
    if terrain_type is None:
        terrain_type = "dem"
    if terrain_type not in types:
        raise click.ClickException(
            f"Unknown terrain type '{terrain_type}'. "
            f"Choices: {', '.join(types)}"
        )

    path = Path(output_path)
    if path.exists() and not force:
        raise click.ClickException(
            f"{path} already exists. Use --force to overwrite."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_tailored_template(terrain_type))
    console.print(
        f"[green]Wrote {terrain_type} config template to:[/green] {path}"
    )


@config.command("show")
@click.pass_context
def show(ctx):
    """Print the effective configuration (after file + env merge)."""
    console = ctx.obj["console"]
    config_path = ctx.obj.get("config_path")
    cfg = load_config(config_path)
    source = config_path or find_config_file()
    console.print(f"[dim]Source: {source or 'built-in defaults'}[/dim]")
    console.print(config_to_yaml(cfg, exclude_none=True).rstrip())
