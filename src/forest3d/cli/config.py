"""Configuration management CLI."""

from pathlib import Path

import click

from forest3d.config.loader import (
    config_to_yaml,
    find_config_file,
    load_config,
    save_config,
)
from forest3d.config.schema import Forest3DConfig


@click.group()
def config():
    """Manage Forest3D configuration files."""


@config.command("init")
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
def init(ctx, output_path, force):
    """Write a forest3d.yaml template with all default values.

    The template is generated from the configuration schema, so it always
    reflects every available option (e.g. terrain.crop_rows.plant_spacing).
    Edit it and Forest3D will pick it up automatically from the project root.
    """
    console = ctx.obj["console"]
    path = Path(output_path)
    if path.exists() and not force:
        raise click.ClickException(
            f"{path} already exists. Use --force to overwrite."
        )
    save_config(Forest3DConfig(), path, exclude_none=True)
    console.print(f"[green]Wrote config template to:[/green] {path}")


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