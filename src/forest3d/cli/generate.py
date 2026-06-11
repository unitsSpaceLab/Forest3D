"""Forest generation CLI subcommand."""

import json
from typing import Dict, Optional

import click
from pathlib import Path
from rich.progress import Progress, BarColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from forest3d.config.loader import load_config
from forest3d.core.forest import WorldPopulator
from forest3d.core.terrain_base import get_terrain_class, list_terrain_types


@click.command()
@click.option(
    "--base-path", "-b", type=click.Path(exists=True),
    help="Project base path containing models/ directory"
)
@click.option(
    "--terrain-type", "-t", type=click.Choice(list_terrain_types()),
    default=None,
    help="Terrain environment (determines available categories). "
         "Defaults to forest (backward compat)."
)
@click.option(
    "--density", "-d", type=str,
    help="JSON density config: '{\"tree\": 50, \"crop\": 200}'"
)
@click.option(
    "--output", "-o", type=click.Path(),
    help="Output world file path"
)
@click.option(
    "--verbose", "-v", is_flag=True,
    help="Show detailed statistics including scale info"
)
@click.option(
    "--terrain-meta", type=str, default=None,
    help="JSON terrain metadata for structured placement: "
         "'{\"num_rows\": 8, \"row_width\": 1.0}'"
)
@click.option(
    "--seed", type=int, default=None,
    help="Random seed for reproducible placement"
)
@click.pass_context
def generate(ctx, base_path, terrain_type, density, output, verbose, terrain_meta, seed):
    """Generate a forest/crop world from existing models.

    Procedurally places models on terrain using intelligent positioning
    algorithms that consider terrain slope, model spacing, and natural
    distribution patterns.

    \b
    Examples:
        forest3d generate
        forest3d generate -d '{"tree": 100, "rock": 20}'
        forest3d generate --terrain-meta '{"num_rows": 10, "plant_spacing": 0.6}'
        forest3d generate -v
    """
    console = ctx.obj["console"]
    logger = ctx.obj["logger"]
    config = load_config(ctx.obj.get("config_path"))

    # Parse density JSON if provided
    parsed_density: Optional[Dict[str, int]] = None
    if density:
        try:
            parsed_density = json.loads(density)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON for density: {e}")

    # Parse terrain metadata JSON if provided
    # Seed placement geometry from the crop-rows config so the terrain mesh
    # and crop placement share one source of truth. --terrain-meta then
    # overrides individual keys for one-off runs.
    meta = None
    if terrain_type == "crop_rows":
        cr = config.terrain.crop_rows
        meta = {
            "num_rows": cr.num_rows,
            "row_width": cr.row_width,
            "furrow_width": cr.furrow_width,
            "headland_width": cr.headland_width,
            "plant_spacing": cr.plant_spacing,
            "stagger": cr.stagger,
        }
    if terrain_meta:
        try:
            overrides = json.loads(terrain_meta)
        except json.JSONDecodeError as e:
            raise click.ClickException(
                f"Invalid JSON for terrain-meta: {e}"
            )
        meta = {**meta, **overrides} if meta else overrides

    # Build density config: start from terrain-type defaults, then merge
    # --density overrides (instead of mutating config.density, so we stay
    # decoupled from the schema)
    if terrain_type:
        cls = get_terrain_class(terrain_type)
        density_config = dict(cls.default_densities())
    else:
        density_config = config.density.as_dict()

    if parsed_density:
        density_config.update(parsed_density)
        # Warn about unknown categories (with backward-compat schema check)
        if not terrain_type:
            for key in parsed_density:
                if not hasattr(config.density, key):
                    console.print(
                        f"[yellow]Warning:[/yellow] Unknown category '{key}'"
                    )
    project_base = Path(base_path) if base_path else Path.cwd()
    if not (project_base / "models").exists():
        raise click.ClickException(
            f"Models directory not found in {project_base}\n\n"
            "Make sure you're in a Forest3D project directory with:\n"
            "  - models/ground/  (terrain)\n"
            "  - models/tree/    (trees)\n"
            "  - models/crop/    (crops)\n"
            "  - etc."
        )

    # Display density configuration
    table = Table(title="Density Configuration", show_header=True)
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")

    for cat, count in density_config.items():
        if count > 0:
            table.add_row(cat, str(count))

    console.print(table)
    console.print()

    # Track progress
    progress_state = {"progress": None, "task": None}

    def progress_callback(percent, description):
        if progress_state["progress"] and progress_state["task"]:
            progress_state["progress"].update(
                progress_state["task"],
                completed=percent,
                description=description,
            )

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
    ) as progress:
        progress_state["progress"] = progress
        progress_state["task"] = progress.add_task(
            "Generating world...", total=100
        )

        try:
            populator = WorldPopulator(
                base_path=project_base,
                progress_callback=progress_callback,
                terrain_type=terrain_type,
                seed=seed,
            )

            world_path = populator.create_forest_world(
                density_config=density_config,
                terrain_meta=meta,
                output_path=Path(output) if output else None,
            )

            stats = populator.get_model_statistics()

        except FileNotFoundError as e:
            raise click.ClickException(str(e))
        except Exception as e:
            logger.error(f"World generation failed: {e}")
            raise click.ClickException(str(e))

    # Display results
    console.print()
    console.print(
        f"[green]Success![/green] World created at: {world_path}"
    )
    console.print(
        f"\nTotal models placed: [bold]{stats['total_models']}[/bold]"
    )

    results_table = Table(show_header=True)
    results_table.add_column("Category", style="cyan")
    results_table.add_column("Requested", justify="right")
    results_table.add_column("Placed", justify="right")
    results_table.add_column("Success", justify="right")

    total_requested = 0
    total_placed = 0
    for cat, requested in sorted(density_config.items()):
        if requested == 0:
            continue
        placed = stats["by_category"].get(cat, 0)
        total_requested += requested
        total_placed += placed

        if requested > 0:
            success_rate = f"{(placed / requested * 100):.0f}%"
            color = (
                "green"
                if placed == requested
                else "yellow" if placed > 0 else "red"
            )
            results_table.add_row(
                cat,
                str(requested),
                str(placed),
                f"[{color}]{success_rate}[/{color}]",
            )

    console.print(results_table)

    failed = total_requested - total_placed
    if failed > 0:
        console.print(
            f"\n[yellow]Note:[/yellow] {failed} models couldn't be "
            f"placed (area too crowded or terrain constraints)"
        )

    if verbose and stats.get("scale_stats"):
        console.print()
        scale_table = Table(title="Scale Statistics", show_header=True)
        scale_table.add_column("Category", style="cyan")
        scale_table.add_column("Min", justify="right")
        scale_table.add_column("Max", justify="right")
        scale_table.add_column("Mean", justify="right")

        for cat, scale_info in stats["scale_stats"].items():
            scale_table.add_row(
                cat,
                f"{scale_info['min']:.2f}",
                f"{scale_info['max']:.2f}",
                f"{scale_info['mean']:.2f}",
            )

        console.print(scale_table)

    console.print(f"\n[dim]To launch in Gazebo:[/dim]")
    console.print(
        f"  export "
        f"GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:"
        f"{project_base}/models"
    )
    console.print(f"  gz sim {world_path}")
