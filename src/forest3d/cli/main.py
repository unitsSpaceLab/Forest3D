"""Unified CLI for Forest3D."""

import click
from rich.console import Console

from forest3d import __version__
from forest3d.utils.logging import setup_logging

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="forest3d")
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v, -vv, -vvv)")
@click.option("-q", "--quiet", is_flag=True, help="Suppress all output except errors")
@click.option(
    "-c", "--config", "config_path", type=click.Path(exists=True), help="Path to configuration file"
)
@click.pass_context
def main(ctx, verbose, quiet, config_path):
    """Forest3D - Terrain and forest generation for Gazebo simulation.

    Generate realistic outdoor environments for robotics simulation from
    DEM data and Blender assets.

    \b
    Examples:
        forest3d terrain --dem terrain.tif
        forest3d convert --input ./blender-assets --output ./models
        forest3d generate --density '{"tree": 50, "rock": 10}'

    \b
    Configuration:
        Forest3D looks for config files in these locations:
        - ./forest3d.yaml
        - ~/.config/forest3d/config.yaml
        - ~/.forest3d.yaml

    \b
    Environment Variables:
        FOREST3D_BLENDER_PATH  - Path to Blender executable
        FOREST3D_BASE_PATH     - Project base directory
        FOREST3D_MODELS_PATH   - Models output directory
    """
    ctx.ensure_object(dict)

    # Setup logging based on verbosity
    if quiet:
        log_level = "ERROR"
    else:
        log_level = ["INFO", "DEBUG", "DEBUG"][min(verbose, 2)]

    ctx.obj["logger"] = setup_logging(log_level, console=console)
    ctx.obj["config_path"] = config_path
    ctx.obj["console"] = console
    ctx.obj["verbose"] = verbose


# Import and register subcommands
from forest3d.cli.terrain import terrain
from forest3d.cli.convert import convert
from forest3d.cli.generate import generate
from forest3d.cli.launch import launch

main.add_command(terrain)
main.add_command(convert)
main.add_command(generate)
main.add_command(launch)


if __name__ == "__main__":
    main()
