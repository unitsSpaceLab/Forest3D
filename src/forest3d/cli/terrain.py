"""Terrain generation CLI — one subcommand per terrain type."""

import click
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn

from forest3d.config.loader import load_config
from forest3d.core.terrain_base import list_terrain_types, get_terrain_class

DEFAULT_OUTPUT = "./models/ground"


# ---------------------------------------------------------------------------
# Build a subcommand dynamically for each registered terrain type
# ---------------------------------------------------------------------------

def _build_terrain_cmd(terrain_type: str, cls: type) -> click.Command:
    """Create a click.Command that generates *terrain_type* terrain."""

    def handler(ctx, output, **kwargs):
        console = ctx.obj["console"]
        logger = ctx.obj["logger"]
        config = load_config(ctx.obj.get("config_path"))

        cls.cli_apply_overrides(config, kwargs)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(
                f"Generating {terrain_type} terrain...", total=None
            )

            try:
                instance = cls.cli_create(
                    output_path=Path(output),
                    config=config,
                    **kwargs,
                )
                result_path = instance.process_terrain()
                cls.cli_post_process(instance, config)

            except ImportError as e:
                raise click.ClickException(str(e))
            except Exception as e:
                logger.error(f"Terrain generation failed: {e}")
                raise click.ClickException(str(e))

        console.print(
            f"\n[green]Success![/green] Terrain created at: {result_path}"
        )

    handler.__name__ = terrain_type
    handler = click.pass_context(handler)

    params = [
        click.Option(
            ["--output", "-o"],
            type=click.Path(),
            default=DEFAULT_OUTPUT,
            help=f"Output directory (default: {DEFAULT_OUTPUT})",
        ),
    ]
    params.extend(cls.cli_options())

    return click.Command(
        name=terrain_type,
        callback=handler,
        params=params,
        help=f"Generate {terrain_type} terrain.",
    )


# ---------------------------------------------------------------------------
# Parent group
# ---------------------------------------------------------------------------

@click.group()
def terrain():
    """Generate terrain mesh.

    \b
    Examples:

        forest3d terrain dem --dem ./DEM/terrain.tif

        forest3d terrain crop_rows --field-length 50 --num-rows 12
    """


# Register one subcommand per terrain type
for _t in list_terrain_types():
    terrain.add_command(_build_terrain_cmd(_t, get_terrain_class(_t)))
