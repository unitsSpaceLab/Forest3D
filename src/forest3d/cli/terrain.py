"""Terrain generation CLI subcommand."""

import click
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn

from forest3d.config.loader import load_config


@click.command()
@click.option(
    "--dem", "-d", "dem_path", type=click.Path(exists=True), required=True,
    help="Path to DEM file (GeoTIFF)"
)
@click.option(
    "--output", "-o", "output_path", type=click.Path(),
    help="Output directory for generated terrain"
)
@click.option(
    "--scale", "-s", type=float, default=None,
    help="Scale factor for terrain (default: 1.0)"
)
@click.option(
    "--smooth", type=float, default=None,
    help="Gaussian smoothing sigma (default: 1.0)"
)
@click.option(
    "--enhance/--no-enhance", default=None,
    help="Enable DEM resolution enhancement"
)
@click.option(
    "--texture", "-t", "texture_path", type=click.Path(exists=True),
    help="Path to Blender file (.blend) for terrain texture extraction"
)
@click.option(
    "--blender", "blender_path", type=click.Path(exists=True),
    help="Path to Blender executable (auto-detected if not specified)"
)
@click.pass_context
def terrain(ctx, dem_path, output_path, scale, smooth, enhance, texture_path, blender_path):
    """Generate terrain mesh from DEM data.

    Processes a Digital Elevation Model (GeoTIFF) file and creates:

    \b
    - STL mesh file for Gazebo
    - SDF model definition
    - model.config file
    - Test world file

    \b
    Examples:
        forest3d terrain --dem ./dem/terrain.tif
        forest3d terrain -d terrain.tif --scale 2.0 --smooth 1.5
        forest3d terrain -d terrain.tif --enhance -o ./output

    \b
    Note: This command requires GDAL to be installed. Use Docker for
    easiest setup, or install GDAL manually.
    """
    console = ctx.obj["console"]
    logger = ctx.obj["logger"]
    config = load_config(ctx.obj.get("config_path"))

    # Override config with CLI options
    if scale is not None:
        config.terrain.scale_factor = scale
    if smooth is not None:
        config.terrain.smooth_sigma = smooth
    if enhance is not None:
        config.terrain.enhance = enhance
    if texture_path is not None:
        config.terrain.texture_blend = Path(texture_path)
    if blender_path is not None:
        config.blender.path = Path(blender_path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing terrain generator...", total=None)

        try:
            # Import here to give helpful error if GDAL missing
            from forest3d.core.terrain import TerrainGenerator, GDAL_AVAILABLE

            if not GDAL_AVAILABLE:
                raise click.ClickException(
                    "GDAL is required for terrain generation.\n\n"
                    "Install options:\n"
                    "  1. Use Docker: docker run -v $(pwd):/workspace forest3d terrain ...\n"
                    "  2. Ubuntu/Debian: sudo apt install python3-gdal gdal-bin\n"
                    "  3. See documentation for other platforms"
                )

            generator = TerrainGenerator(
                tif_path=Path(dem_path),
                output_path=Path(output_path) if output_path else None,
                config=config.terrain,
                blender_path=config.blender.path,
            )

            progress.update(task, description="Processing DEM data...")
            result_path = generator.process_terrain()

            # Extract textures from Blender file if provided
            if config.terrain.texture_blend:
                progress.update(task, description="Extracting textures from Blender file...")
                generator.extract_terrain_texture(config.terrain.texture_blend)

            progress.update(task, description="Complete!")

        except ImportError as e:
            raise click.ClickException(str(e))
        except Exception as e:
            logger.error(f"Terrain generation failed: {e}")
            raise click.ClickException(str(e))

    console.print(f"\n[green]Success![/green] Terrain created at: {result_path}")
    console.print(f"\n[dim]To view in Gazebo:[/dim]")
    console.print(f"  export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:{result_path.parent}")
    console.print(f"  gazebo {result_path}/test.world")
