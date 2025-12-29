"""Launch Gazebo with generated world CLI subcommand."""

import os
import shutil
import subprocess
import click
from pathlib import Path

from forest3d.config.loader import load_config


def find_gazebo() -> tuple[str, str]:
    """Find Gazebo executable (Harmonic or Classic).

    Returns:
        Tuple of (executable_path, version_name)
    """
    # Check for Gazebo Harmonic (gz sim)
    gz_sim = shutil.which("gz")
    if gz_sim:
        return gz_sim, "harmonic"

    # Check for Gazebo Classic
    gazebo_classic = shutil.which("gazebo")
    if gazebo_classic:
        return gazebo_classic, "classic"

    return None, None


@click.command()
@click.option(
    "--world", "-w", "world_path", type=click.Path(exists=True),
    help="Path to world file (default: worlds/forest_world.world)"
)
@click.option(
    "--base-path", "-b", type=click.Path(exists=True),
    help="Project base path containing models/ and worlds/"
)
@click.option(
    "--verbose", "-v", is_flag=True,
    help="Show Gazebo output"
)
@click.pass_context
def launch(ctx, world_path, base_path, verbose):
    """Launch Gazebo with the generated forest world.

    Opens Gazebo Harmonic (or Classic) with the forest world file,
    automatically setting up the model path.

    \b
    Examples:
        forest3d launch
        forest3d launch --world worlds/custom.world
        forest3d launch -b ./my-project

    \b
    Requirements:
        - Gazebo Harmonic (gz-harmonic) or Gazebo Classic
        - X11 display for GUI (use DISPLAY env variable in Docker)

    \b
    Docker with GUI:
        docker run -e DISPLAY=$DISPLAY \\
                   -v /tmp/.X11-unix:/tmp/.X11-unix \\
                   -v $(pwd):/workspace \\
                   forest3d launch
    """
    console = ctx.obj["console"]
    logger = ctx.obj["logger"]

    # Find Gazebo
    gz_path, gz_version = find_gazebo()

    if not gz_path:
        raise click.ClickException(
            "Gazebo not found. Please install Gazebo Harmonic or Gazebo Classic.\n\n"
            "Install Gazebo Harmonic:\n"
            "  sudo apt install gz-harmonic\n\n"
            "Or use Docker:\n"
            "  docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \\\n"
            "             -v $(pwd):/workspace forest3d launch"
        )

    # Determine paths
    project_base = Path(base_path) if base_path else Path.cwd()
    models_path = project_base / "models"
    worlds_path = project_base / "worlds"

    if not models_path.exists():
        raise click.ClickException(f"Models directory not found: {models_path}")

    # Determine world file
    if world_path:
        world_file = Path(world_path)
    else:
        world_file = worlds_path / "forest_world.world"
        if not world_file.exists():
            raise click.ClickException(
                f"World file not found: {world_file}\n\n"
                "Generate a world first:\n"
                "  forest3d generate"
            )

    console.print(f"[bold]Launching Gazebo {gz_version.title()}[/bold]")
    console.print(f"  World: [cyan]{world_file}[/cyan]")
    console.print(f"  Models: [cyan]{models_path}[/cyan]")
    console.print()

    # Set up environment
    env = os.environ.copy()

    if gz_version == "harmonic":
        # Gazebo Harmonic uses GZ_SIM_RESOURCE_PATH
        existing_path = env.get("GZ_SIM_RESOURCE_PATH", "")
        if existing_path:
            env["GZ_SIM_RESOURCE_PATH"] = f"{models_path}:{existing_path}"
        else:
            env["GZ_SIM_RESOURCE_PATH"] = str(models_path)

        # Build command for Gazebo Harmonic
        cmd = [gz_path, "sim", str(world_file)]

        if verbose:
            cmd.insert(2, "-v4")
    else:
        # Gazebo Classic uses GAZEBO_MODEL_PATH
        existing_path = env.get("GAZEBO_MODEL_PATH", "")
        if existing_path:
            env["GAZEBO_MODEL_PATH"] = f"{models_path}:{existing_path}"
        else:
            env["GAZEBO_MODEL_PATH"] = str(models_path)

        # Build command for Gazebo Classic
        cmd = [gz_path, str(world_file)]

        if verbose:
            cmd.append("--verbose")

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        console.print("[dim]Starting Gazebo... (press Ctrl+C to exit)[/dim]")

        # Run Gazebo
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(project_base),
        )

        if result.returncode != 0:
            logger.warning(f"Gazebo exited with code {result.returncode}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Gazebo closed[/yellow]")
    except Exception as e:
        raise click.ClickException(f"Failed to launch Gazebo: {e}")
