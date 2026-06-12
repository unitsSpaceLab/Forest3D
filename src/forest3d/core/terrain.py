"""DEM-based terrain generation."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
from scipy.ndimage import gaussian_filter

from forest3d.config.schema import TerrainConfig
from forest3d.core.terrain_base import (
    BaseTerrain,
    register_terrain,
)

try:
    from osgeo import gdal

    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

logger = logging.getLogger("forest3d.terrain")


@register_terrain
class DemTerrain(BaseTerrain):
    """Generate terrain from a Digital Elevation Model (GeoTIFF)."""

    TERRAIN_TYPE = "dem"

    def __init__(
        self,
        tif_path: Path,
        output_path: Optional[Path] = None,
        config: Optional[TerrainConfig] = None,
        blender_path: Optional[Path] = None,
    ):
        if not GDAL_AVAILABLE:
            raise ImportError(
                "GDAL is required for DEM terrain. Install with: pip install GDAL"
            )

        self.tif_path = Path(tif_path)
        if not self.tif_path.exists():
            raise FileNotFoundError(f"DEM file not found: {self.tif_path}")

        self.config = config or TerrainConfig()
        self._material_name = self.config.material_name

        if output_path:
            terrain_path = Path(output_path)
        else:
            terrain_path = self.tif_path.parent.parent

        super().__init__(terrain_path, config, blender_path)

    @classmethod
    def cli_options(cls) -> list:
        return [
            click.Option(
                ["--dem", "-d"], required=True, type=click.Path(exists=True),
                help="Path to DEM file (GeoTIFF)",
            ),
            click.Option(
                ["--scale", "-s"], type=float, default=None,
                help="Scale factor (default: 1.0)",
            ),
            click.Option(
                ["--smooth"], type=float, default=None,
                help="Gaussian smoothing sigma (default: 1.0)",
            ),
            click.Option(
                ["--enhance/--no-enhance"], default=None,
                help="Enable DEM resolution enhancement",
            ),
            click.Option(
                ["--texture"], type=click.Path(exists=True),
                help="Path to Blender file for terrain texture",
            ),
            click.Option(
                ["--blender"], type=click.Path(exists=True),
                help="Path to Blender executable (auto-detected if not set)",
            ),
        ]

    @classmethod
    def cli_create(cls, output_path: Path, config, **kwargs):
        blender = kwargs.get("blender")
        return cls(
            tif_path=kwargs["dem"],
            output_path=output_path,
            config=config,
            blender_path=Path(blender) if blender else None,
        )

    @classmethod
    def cli_apply_overrides(cls, config, kwargs) -> None:
        tc = config.terrain
        if kwargs.get("scale") is not None:
            tc.scale_factor = kwargs["scale"]
        if kwargs.get("smooth") is not None:
            tc.dem.smooth_sigma = kwargs["smooth"]
        if kwargs.get("enhance") is not None:
            tc.dem.enhance = kwargs["enhance"]
        if kwargs.get("texture") is not None:
            tc.texture_blend = Path(kwargs["texture"])
        if kwargs.get("blender") is not None:
            config.blender.path = Path(kwargs["blender"])

    @classmethod
    def default_categories(cls) -> Dict:
        return {
            "tree": {
                "scale_range": (0.8, 1.5),
                "min_distance": 8.0,
                "zone_weights": {"edge": 0.2, "center": 0.8},
                "rotation": {
                    "roll_range": (-0.05, 0.05),
                    "pitch_range": (-0.05, 0.05),
                    "yaw_range": (0, 6.2832),
                },
            },
            "rock": {
                "scale_range": (0.5, 2.0),
                "min_distance": 4.0,
                "zone_weights": {"edge": 0.8, "center": 0.2},
                "rotation": {
                    "roll_range": (-0.15, 0.15),
                    "pitch_range": (-0.15, 0.15),
                    "yaw_range": (0, 6.2832),
                },
            },
            "bush": {
                "scale_range": (0.3, 1.0),
                "min_distance": 2.0,
                "zone_weights": {"edge": 0.4, "center": 0.6},
                "rotation": {
                    "roll_range": (-0.03, 0.03),
                    "pitch_range": (-0.03, 0.03),
                    "yaw_range": (0, 6.2832),
                },
            },
            "grass": {
                "scale_range": (0.2, 0.6),
                "min_distance": 0.5,
                "zone_weights": {"edge": 0.5, "center": 0.5},
                "rotation": {
                    "roll_range": (0.0, 0.0),
                    "pitch_range": (0.0, 0.0),
                    "yaw_range": (0, 6.2832),
                },
            },
            "sand": {
                "scale_range": (1.0, 2.5),
                "min_distance": 3.0,
                "zone_weights": {"edge": 0.7, "center": 0.3},
                "rotation": {
                    "roll_range": (0.0, 0.0),
                    "pitch_range": (0.0, 0.0),
                    "yaw_range": (0, 6.2832),
                },
            },
        }

    @classmethod
    def default_strategies(cls) -> Dict:
        from forest3d.core.placement import (
            ClusteredPlacement,
            EdgePlacement,
        )

        return {
            "sand": EdgePlacement(
                edge_weight=0.7,
                jitter=1.0,
                height_offset_range=(-0.2, 0.0),
            ),
            "rock": EdgePlacement(
                edge_weight=0.8,
                jitter=2.0,
                height_offset_range=(-0.1, 0.1),
            ),
            "tree": ClusteredPlacement(
                anchor_categories=("tree",),
                cluster_prob=0.7,
                scale_aware=True,
                radius_scale_mult=2.0,
                radius_add=0.0,
                avoid_categories=("sand",),
                height_offset_range=(-0.08, 0.08),
            ),
            "bush": ClusteredPlacement(
                anchor_categories=("tree",),
                cluster_prob=0.6,
                scale_aware=True,
                radius_scale_mult=1.0,
                radius_add=3.0,
                height_offset_range=(-0.08, 0.08),
            ),
            "grass": ClusteredPlacement(
                anchor_categories=("tree", "bush"),
                cluster_prob=0.5,
                scale_aware=False,
                radius_min_fixed=1.0,
                radius_max_fixed=5.0,
                height_offset_range=(-0.05, 0.05),
            ),
        }

    @classmethod
    def default_cross_distances(cls) -> Dict:
        return {
            ("tree", "tree"): 8.0,
            ("tree", "bush"): 1.5,
            ("tree", "rock"): 2.0,
            ("tree", "grass"): 0.5,
            ("tree", "sand"): 4.0,
            ("bush", "bush"): 2.0,
            ("bush", "rock"): 1.5,
            ("bush", "grass"): 0.3,
            ("bush", "sand"): 2.0,
            ("rock", "rock"): 4.0,
            ("rock", "grass"): 0.5,
            ("rock", "sand"): 2.0,
            ("grass", "grass"): 0.5,
            ("grass", "sand"): 0.5,
            ("sand", "sand"): 3.0,
        }

    @classmethod
    def default_category_order(cls) -> List[str]:
        return ["sand", "rock", "tree", "bush", "grass"]

    @classmethod
    def default_densities(cls) -> Dict[str, int]:
        return {"tree": 50, "rock": 5, "bush": 10, "grass": 50, "sand": 5}

    def enhance_dem(self, scale_factor: float = 6.0) -> Path:
        output_tiff = self.tif_path.parent / "terrain_enhanced.tif"
        ds = gdal.Open(str(self.tif_path))
        if ds is None:
            raise RuntimeError(f"Failed to open {self.tif_path}")
        gdal.Warp(
            str(output_tiff),
            ds,
            width=int(ds.RasterXSize * scale_factor),
            height=int(ds.RasterYSize * scale_factor),
            resampleAlg=gdal.GRA_CubicSpline,
            options=["COMPRESS=LZW"],
        )
        logger.info(f"Enhanced DEM saved to: {output_tiff}")
        return output_tiff

    def generate_terrain_mesh(
        self,
        scale_factor: Optional[float] = None,
        z_scale: Optional[float] = None,
        smooth_sigma: Optional[float] = None,
        enhance: Optional[bool] = None,
        uv_tile_scale: float = 10.0,
    ) -> Tuple[Path, dict]:
        """Create terrain meshes from DEM data."""
        scale_factor = (
            scale_factor if scale_factor is not None else self.config.scale_factor
        )
        z_scale = z_scale if z_scale is not None else scale_factor
        smooth_sigma = (
            smooth_sigma if smooth_sigma is not None else self.config.smooth_sigma
        )
        enhance = enhance if enhance is not None else self.config.enhance

        if enhance:
            dem_file = self.enhance_dem(self.config.enhance_scale)
        else:
            dem_file = self.tif_path

        ds = gdal.Open(str(dem_file))
        if ds is None:
            raise RuntimeError(f"Failed to open {dem_file}")

        geotransform = ds.GetGeoTransform()
        pixel_width = abs(geotransform[1])
        pixel_height = abs(geotransform[5])
        elevation = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

        if smooth_sigma > 0:
            elevation = gaussian_filter(elevation, sigma=smooth_sigma)

        rows, cols = elevation.shape
        logger.info(f"Creating mesh from {rows}x{cols} DEM...")

        pixel_width *= scale_factor
        pixel_height *= scale_factor
        vertices, uvs, faces = self._build_mesh_from_heightmap(
            elevation, pixel_width, pixel_height, z_scale, uv_tile_scale
        )

        self._center_and_shift(vertices)

        normals = self._calculate_normals(vertices, faces)

        obj_path = self.mesh_path / "terrain.obj"
        self._write_obj(obj_path, vertices, uvs, normals, faces)
        logger.info(f"Created OBJ mesh: {obj_path}")

        stl_path = self.mesh_path / "terrain.stl"
        self._write_stl(stl_path, vertices, faces)
        logger.info(f"Created STL mesh: {stl_path}")

        stats = {
            "x_extent": float(np.ptp(vertices[:, 0])),
            "y_extent": float(np.ptp(vertices[:, 1])),
            "z_extent": float(np.ptp(vertices[:, 2])),
            "num_vertices": len(vertices),
            "num_faces": len(faces),
        }
        logger.info(
            f"Terrain: X={stats['x_extent']:.2f}, "
            f"Y={stats['y_extent']:.2f}, Z={stats['z_extent']:.2f}"
        )
        return stl_path, stats


TerrainGenerator = DemTerrain
