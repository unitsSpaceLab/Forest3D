"""Forest3D - Terrain and world generation for Gazebo simulation."""

__version__ = "0.2.0"
__author__ = "AI4Forest"

from forest3d.core.terrain import DemTerrain, TerrainGenerator, GDAL_AVAILABLE
from forest3d.core.terrain_crop import CropRowTerrain, CropRowParams
from forest3d.core.terrain_base import (
    BaseTerrain,
    get_terrain_class,
    list_terrain_types,
    register_terrain,
)
from forest3d.core.converter import AssetExporter
from forest3d.core.forest import WorldPopulator

__all__ = [
    "BaseTerrain",
    "DemTerrain",
    "TerrainGenerator",
    "CropRowTerrain",
    "CropRowParams",
    "AssetExporter",
    "WorldPopulator",
    "get_terrain_class",
    "list_terrain_types",
    "register_terrain",
    "GDAL_AVAILABLE",
    "__version__",
]
