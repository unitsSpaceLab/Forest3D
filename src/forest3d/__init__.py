"""Forest3D - Terrain and forest generation for Gazebo simulation."""

__version__ = "0.1.0"
__author__ = "AI4Forest"

from forest3d.core.terrain import TerrainGenerator
from forest3d.core.converter import AssetExporter
from forest3d.core.forest import WorldPopulator

__all__ = ["TerrainGenerator", "AssetExporter", "WorldPopulator", "__version__"]
