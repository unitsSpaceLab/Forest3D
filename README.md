# Forest3D - Terrain and Forest Generation for Gazebo

Forest3D is an advanced terrain generation and environmental modeling toolkit that transforms real-world geographical data into photorealistic Gazebo simulation environments. The system processes Digital Elevation Model (DEM) data through sophisticated interpolation and smoothing algorithms to create detailed 3D terrain meshes, while automatically converting Blender assets into optimized simulation-ready models with proper collision geometry.

![Forest3D Environment](https://github.com/khalidbourr/Forest3D/blob/main/blender2Gazebo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Terrain Generation**: DEM processing with resolution enhancement and Gaussian smoothing
- **Asset Processing**: Automatic Blender to Gazebo conversion with optimized collision meshes
- **Forest Population**: Intelligent procedural placement with natural clustering patterns
- **Unified CLI**: Simple `forest3d` command with subcommands for each operation
- **Docker Support**: Pre-built images with GDAL for easy deployment

## Quick Start

### Option 1: Docker (Recommended)

The Docker image includes everything you need: Python, GDAL, Blender 4.2, and Gazebo Harmonic.

```bash
# Build the image (downloads Blender + Gazebo, ~2GB)
cd Forest3D
docker build -t forest3d -f docker/Dockerfile .

# Generate a forest world
docker run -v $(pwd):/workspace forest3d generate

# Convert Blender assets to Gazebo models
docker run -v $(pwd):/workspace forest3d convert \
  -i /workspace/Blender-Assets -o /workspace/models -c tree

# Launch Gazebo to view the world (requires X11)
xhost +local:docker  # Allow Docker to access display
docker run -e DISPLAY=$DISPLAY \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -v $(pwd):/workspace \
           --network host \
           forest3d launch
```

### Option 2: pip install

```bash
# Clone and install
git clone https://github.com/khalidbourr/Forest3D.git
cd Forest3D
pip install -e .

# For terrain generation, also install GDAL:
# Ubuntu/Debian:
sudo apt install python3-gdal gdal-bin libgdal-dev
pip install "pygdal==$(gdal-config --version).*"
```

## Usage

### Generate Forest World

```bash
# Use default settings
forest3d generate

# Custom density
forest3d generate --density '{"tree": 100, "rock": 20, "bush": 30}'

# Use a preset configuration
forest3d -c configs/examples/dense_forest.yaml generate
```

### Generate Terrain from DEM

```bash
forest3d terrain --dem models/ground/dem/terrain.tif

# With options
forest3d terrain --dem terrain.tif --scale 2.0 --smooth 1.5 --enhance
```

### Convert Blender Assets

```bash
forest3d convert --input ./Blender-Assets --output ./models --category tree
```

### Launch Gazebo

```bash
# Using the CLI (auto-configures model path)
forest3d launch

# Or manually with Gazebo Harmonic
export GZ_SIM_RESOURCE_PATH=$(pwd)/models
gz sim worlds/forest_world.world

# Or with Gazebo Classic
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(pwd)/models
gazebo worlds/forest_world.world
```

## CLI Reference

```
forest3d --help                    # Show all commands
forest3d terrain --help            # Terrain generation help
forest3d convert --help            # Asset conversion help
forest3d generate --help           # Forest generation help
forest3d launch --help             # Launch Gazebo help

# Global options
forest3d -v ...                    # Verbose output
forest3d -vv ...                   # Debug output
forest3d -c config.yaml ...        # Use config file
```

## Configuration

Create `forest3d.yaml` in your project directory:

```yaml
terrain:
  scale_factor: 1.0
  smooth_sigma: 1.0
  enhance: false

density:
  tree: 50
  bush: 10
  rock: 5
  grass: 50
  sand: 5

blender:
  visual_decimation: 0.1
  collision_decimation: 0.01
```

See `configs/examples/` for preset configurations.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `FOREST3D_BLENDER_PATH` | Path to Blender executable |
| `FOREST3D_BASE_PATH` | Project base directory |
| `FOREST3D_MODELS_PATH` | Models output directory |

## Project Structure

```
Forest3D/
├── src/forest3d/          # Python package
│   ├── cli/               # Command-line interface
│   ├── core/              # Core modules (terrain, converter, forest)
│   ├── config/            # Configuration handling
│   └── utils/             # Logging and progress utilities
├── models/                # Gazebo models
│   ├── ground/            # Terrain model
│   ├── tree/, rock/, etc. # Asset models
├── worlds/                # Generated world files
├── configs/               # Configuration presets
├── docker/                # Docker files
└── Blender-Assets/        # Source .blend files
```

## Asset Categories

| Category | Description | Default Count |
|----------|-------------|---------------|
| tree | Large vegetation | 50 |
| bush | Small vegetation/shrubs | 10 |
| rock | Rock formations | 5 |
| grass | Ground cover | 50 |
| sand | Sand dunes/patches | 5 |

## Adding Custom Assets

1. Place `.blend` files in `Blender-Assets/`
2. Convert to Gazebo format:
   ```bash
   forest3d convert -i ./Blender-Assets -o ./models -c tree
   ```
3. Models will be available for forest generation

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Lint
pylint src/forest3d/
```

## Docker Compose

```bash
# Development environment (with Blender + GDAL + Gazebo)
docker compose -f docker/docker-compose.yml run forest3d-dev

# Convert Blender assets
docker compose -f docker/docker-compose.yml run convert \
  -i /workspace/Blender-Assets -o /workspace/models -c tree

# Generate terrain from DEM
docker compose -f docker/docker-compose.yml run terrain --dem terrain.tif

# Generate forest world
docker compose -f docker/docker-compose.yml run generate

# Launch Gazebo to view world (requires X11)
xhost +local:docker
docker compose -f docker/docker-compose.yml run launch
```

## Troubleshooting

### GDAL Not Found
Use Docker or install GDAL system packages:
```bash
# Ubuntu/Debian
sudo apt install python3-gdal gdal-bin libgdal-dev
pip install "pygdal==$(gdal-config --version).*"
```

### Blender Not Found
Set the path explicitly:
```bash
export FOREST3D_BLENDER_PATH=/path/to/blender
# or
forest3d convert --blender /path/to/blender ...
```

### Model Path Issues
Ensure Gazebo can find models:
```bash
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(pwd)/models
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
