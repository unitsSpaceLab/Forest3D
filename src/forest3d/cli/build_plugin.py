"""Build the gz_terramechanics Gazebo plugin CLI subcommand."""

import subprocess
import click

from forest3d.core.soil_bridge import get_plugin_source_dir


@click.command("build-plugin", context_settings=dict(ignore_unknown_options=True))
@click.option("-j", "--jobs", type=int, default=4, help="Parallel build jobs")
@click.argument("cmake_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def build_plugin(ctx, jobs, cmake_args):
    """Build the gz_terramechanics Gazebo plugin.

    Runs cmake and make to compile the C++ terramechanics plugin.
    The built library (libTerramechanicsSystem.so) is placed in
    plugins/gz_terramechanics/build/.

    \b
    Examples:
        forest3d build-plugin          # Build with 4 jobs
        forest3d build-plugin -j 8     # Build with 8 jobs

    \b
    Requirements:
        - cmake
        - A C++ compiler (g++ or clang++)
        - Gazebo Harmonic development libraries
    """
    console = ctx.obj["console"]

    try:
        plugin_dir = get_plugin_source_dir()
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    build_dir = plugin_dir / "build"
    build_dir.mkdir(exist_ok=True)

    console.print("[bold]Building gz_terramechanics plugin[/bold]")
    console.print(f"  Source: [cyan]{plugin_dir}[/cyan]")
    console.print(f"  Build:  [cyan]{build_dir}[/cyan]")
    console.print()

    # CMake configure
    console.print("[dim]Running cmake...[/dim]")
    try:
        cmake_cmd = ["cmake", ".."] + list(cmake_args)
        subprocess.run(
            cmake_cmd,
            cwd=str(build_dir),
            check=True,
        )
    except FileNotFoundError:
        raise click.ClickException(
            "cmake not found. Install with: sudo apt install cmake"
        )
    except subprocess.CalledProcessError as e:
        raise click.ClickException(f"cmake configuration failed (exit code {e.returncode})")

    # Make build
    console.print(f"[dim]Running make -j{jobs}...[/dim]")
    try:
        subprocess.run(
            ["make", f"-j{jobs}"],
            cwd=str(build_dir),
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise click.ClickException(f"Build failed (exit code {e.returncode})")

    # Find the built library
    lib_path = build_dir / "libTerramechanicsSystem.so"
    if lib_path.exists():
        console.print(f"\n[green]Success![/green] Plugin built at:")
        console.print(f"  {lib_path}")
    else:
        # Check for any .so file
        so_files = list(build_dir.glob("*.so"))
        if so_files:
            console.print(f"\n[green]Success![/green] Plugin built at:")
            for so in so_files:
                console.print(f"  {so}")
        else:
            console.print(
                "\n[yellow]Warning:[/yellow] Build completed but no .so file found. "
                "Check build output above."
            )
