"""Asset conversion CLI subcommand."""

import click
from pathlib import Path
from rich.progress import Progress, BarColumn, TaskProgressColumn, TextColumn

from forest3d.config.loader import load_config
from forest3d.core.converter import AssetExporter

# Known asset categories
KNOWN_CATEGORIES = ["tree", "bush", "rock", "grass", "sand"]


@click.command()
@click.option(
    "--input", "-i", "input_dir", type=click.Path(exists=True), required=True,
    help="Directory containing Blender files (or parent with category subfolders)"
)
@click.option(
    "--output", "-o", "output_dir", type=click.Path(), required=True,
    help="Output directory for Gazebo models"
)
@click.option(
    "--blender", "-b", "blender_path", type=click.Path(exists=True),
    help="Path to Blender executable (auto-detected if not specified)"
)
@click.option(
    "--category", "-c", type=click.Choice(KNOWN_CATEGORIES),
    default=None, help="Model category (auto-detected from folder name if not specified)"
)
@click.pass_context
def convert(ctx, input_dir, output_dir, blender_path, category):
    """Convert Blender assets to Gazebo models.

    Processes .blend files and generates optimized meshes for simulation:

    \b
    - Visual mesh (10% of original detail by default)
    - Collision mesh (1% of original detail by default)
    - glTF binary format with embedded textures
    - SDF and config files

    \b
    Examples:
        # Auto-detect categories from subfolders (tree/, rock/, bush/, etc.)
        forest3d convert -i ./Blender-Assets -o ./models

        # Convert specific category
        forest3d convert -i ./Blender-Assets/tree -o ./models -c tree

        # Custom Blender path
        forest3d convert -i ./assets -o ./models --blender /opt/blender/blender

    \b
    Note: Requires Blender 4.2+ to be installed.
    """
    console = ctx.obj["console"]
    logger = ctx.obj["logger"]
    config = load_config(ctx.obj.get("config_path"))

    if blender_path:
        config.blender.path = Path(blender_path)

    input_path = Path(input_dir)

    # Check for category subfolders (tree/, rock/, bush/, etc.)
    category_folders = []
    for folder in input_path.iterdir():
        if folder.is_dir() and folder.name.lower() in KNOWN_CATEGORIES:
            blend_files = list(folder.glob("*.blend"))
            if blend_files:
                category_folders.append((folder, folder.name.lower(), blend_files))

    # If category subfolders found, process each
    if category_folders:
        console.print(f"[bold]Auto-detected category folders:[/bold]")
        for folder, cat, files in category_folders:
            console.print(f"  • {cat}: {len(files)} file(s)")
        console.print(f"Output: [cyan]{output_dir}[/cyan]")
        console.print()

        try:
            exporter = AssetExporter(
                blender_path=config.blender.path,
                output_path=Path(output_dir),
                config=config.blender,
            )
        except RuntimeError as e:
            raise click.ClickException(str(e))
        except FileNotFoundError as e:
            raise click.ClickException(str(e))

        total_files = sum(len(files) for _, _, files in category_folders)
        successful = 0
        failed = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Converting assets...", total=total_files)

            for folder, cat, blend_files in category_folders:
                for blend_file in blend_files:
                    progress.update(task, description=f"[{cat}] {blend_file.name}")
                    try:
                        exporter.process_asset(blend_file, cat)
                        successful += 1
                    except Exception as e:
                        logger.warning(f"Failed to process {blend_file.name}: {e}")
                        failed += 1
                    progress.advance(task)

    else:
        # No category subfolders - look for .blend files directly
        blend_files = list(input_path.glob("*.blend"))

        if not blend_files:
            raise click.ClickException(
                f"No .blend files found in {input_dir}\n"
                f"Expected either:\n"
                f"  • .blend files directly in the folder\n"
                f"  • Subfolders named: {', '.join(KNOWN_CATEGORIES)}"
            )

        # Use provided category or default to "tree"
        cat = category or "tree"

        console.print(f"Found [bold]{len(blend_files)}[/bold] Blender files to process")
        console.print(f"Category: [cyan]{cat}[/cyan]")
        console.print(f"Output: [cyan]{output_dir}[/cyan]")
        console.print()

        try:
            exporter = AssetExporter(
                blender_path=config.blender.path,
                output_path=Path(output_dir),
                config=config.blender,
            )
        except RuntimeError as e:
            raise click.ClickException(str(e))
        except FileNotFoundError as e:
            raise click.ClickException(str(e))

        successful = 0
        failed = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Converting assets...", total=len(blend_files))

            for blend_file in blend_files:
                progress.update(task, description=f"Processing {blend_file.name}...")
                try:
                    exporter.process_asset(blend_file, cat)
                    successful += 1
                except Exception as e:
                    logger.warning(f"Failed to process {blend_file.name}: {e}")
                    failed += 1
                progress.advance(task)

    console.print()
    if successful > 0:
        console.print(f"[green]Success![/green] Converted {successful} models to: {output_dir}")
    if failed > 0:
        console.print(f"[yellow]Warning:[/yellow] {failed} files failed to convert")

    if successful > 0:
        console.print(f"\n[dim]To use in Gazebo:[/dim]")
        console.print(f"  export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:{Path(output_dir).resolve()}")
