"""Soil type management CLI subcommand."""

import click

from forest3d.core.soil_bridge import get_available_soils, write_terrain_mapping


@click.command()
@click.argument("soil_name", required=False, default=None)
@click.option(
    "--set", "set_soil", type=str, default=None,
    help="Write terrain_mapping.yaml for this soil type without re-running terrain"
)
@click.pass_context
def soil(ctx, soil_name, set_soil):
    """List and inspect terramechanics soil types.

    \b
    Examples:
        forest3d soil                    # List all available soil types
        forest3d soil Sand_marsSim       # Show full params for one soil
        forest3d soil --set Clay         # Set terrain mapping to Clay
    """
    console = ctx.obj["console"]

    try:
        soils = get_available_soils()
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    # --set mode: write terrain_mapping and exit
    if set_soil:
        try:
            mapping_file = write_terrain_mapping(set_soil)
            console.print(f"[green]Terrain mapping:[/green] 'terrain' → {set_soil}")
            console.print(f"  Written to: {mapping_file}")
        except ValueError as e:
            raise click.ClickException(str(e))
        return

    # Show details for a specific soil
    if soil_name:
        if soil_name not in soils:
            available = ", ".join(soils.keys())
            raise click.ClickException(
                f"Unknown soil type '{soil_name}'. Available: {available}"
            )

        params = soils[soil_name]
        console.print(f"[bold]{soil_name}[/bold]\n")
        for key, value in params.items():
            console.print(f"  {key:<10s} {value}")
        return

    # List all soils with key params
    console.print("[bold]Available soil types[/bold]\n")
    for name, params in soils.items():
        phi = params.get("phi", "?")
        c = params.get("c", "?")
        K = params.get("K", "?")
        console.print(f"  {name:<25s} phi={phi}\u00b0  c={c} Pa  K={K} m")
    console.print(f"\n[dim]Default: Sand_marsSim[/dim]")
    console.print(f"[dim]Use 'forest3d soil <name>' for full parameters[/dim]")
