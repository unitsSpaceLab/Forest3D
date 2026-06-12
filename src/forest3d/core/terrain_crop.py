"""Procedural crop-row terrain generation for agricultural robotics."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np

from forest3d.config.schema import TerrainConfig
from forest3d.core.terrain_base import BaseTerrain, register_terrain

logger = logging.getLogger("forest3d.terrain_crop")


@dataclass
class CropRowParams:
    """Parameters for crop-row field generation."""

    field_length: float = 50.0        # X dimension (metres)
    field_width: float = 30.0         # Y dimension (metres)
    num_rows: int = 8                 # number of crop rows
    row_width: float = 1.0            # width of each raised bed (metres)
    furrow_width: float = 0.5         # width of each furrow (metres)
    row_height: float = 0.15          # height of row above furrow (metres)
    headland_width: float = 2.0       # unplanted border (metres)
    resolution: float = 0.25          # metres per grid cell
    row_profile: str = "rounded"      # "flat", "rounded", "trapezoid"


@register_terrain
class CropRowTerrain(BaseTerrain):
    """Procedural terrain with parallel crop rows for agricultural sim."""

    TERRAIN_TYPE = "crop_rows"

    def __init__(
        self,
        output_path: Path,
        config: Optional[TerrainConfig] = None,
        row_params: Optional[CropRowParams] = None,
        blender_path: Optional[Path] = None,
    ):
        super().__init__(output_path, config, blender_path)
        self.params = row_params or CropRowParams()

    @classmethod
    def cli_options(cls) -> list:
        return [
            click.Option(
                ["--field-length"], type=float, default=None,
                help="Field length in metres (default: 50)",
            ),
            click.Option(
                ["--field-width"], type=float, default=None,
                help="Field width in metres (default: 30)",
            ),
            click.Option(
                ["--num-rows"], type=int, default=None,
                help="Number of crop rows (default: 8)",
            ),
            click.Option(
                ["--row-width"], type=float, default=None,
                help="Crop row width in metres (default: 1.0)",
            ),
            click.Option(
                ["--furrow-width"], type=float, default=None,
                help="Furrow width in metres (default: 0.5)",
            ),
            click.Option(
                ["--row-height"], type=float, default=None,
                help="Row height above furrow in metres (default: 0.15)",
            ),
            click.Option(
                ["--row-profile"],
                type=click.Choice(["flat", "rounded", "trapezoid"]),
                default=None,
                help="Row cross-section profile (default: rounded)",
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
        from forest3d.core.terrain_crop import CropRowParams

        cr = config.terrain.crop_rows
        params = CropRowParams(
            field_length=kwargs.get("field_length") or cr.field_length,
            field_width=kwargs.get("field_width") or cr.field_width,
            num_rows=kwargs.get("num_rows") or cr.num_rows,
            row_width=kwargs.get("row_width") or cr.row_width,
            furrow_width=kwargs.get("furrow_width") or cr.furrow_width,
            row_height=kwargs.get("row_height") or cr.row_height,
            headland_width=cr.headland_width,
            resolution=cr.resolution,
            row_profile=kwargs.get("row_profile") or cr.row_profile,
        )
        blender = kwargs.get("blender")
        return cls(
            output_path=output_path,
            config=config,
            row_params=params,
            blender_path=Path(blender) if blender else None,
        )

    @classmethod
    def cli_apply_overrides(cls, config, kwargs) -> None:
        cr = config.terrain.crop_rows
        for k in (
            "field_length",
            "field_width",
            "num_rows",
            "row_width",
            "furrow_width",
            "row_height",
            "row_profile",
        ):
            if kwargs.get(k) is not None:
                setattr(cr, k, kwargs[k])
        if kwargs.get("texture") is not None:
            config.terrain.texture_blend = Path(kwargs["texture"])
        if kwargs.get("blender") is not None:
            config.blender.path = Path(kwargs["blender"])

    @classmethod
    def default_categories(cls) -> Dict:
        return {
            "crop": {
                "scale_range": (0.6, 1.2),
                "min_distance": 0.3,
                "zone_weights": {"edge": 0.0, "center": 1.0},
                "rotation": {
                    "roll_range": (0.0, 0.0),
                    "pitch_range": (0.0, 0.0),
                    "yaw_range": (0, 6.2832),
                },
            },
            "weed": {
                "scale_range": (0.2, 0.5),
                "min_distance": 0.2,
                "zone_weights": {"edge": 0.3, "center": 0.7},
                "rotation": {
                    "roll_range": (0.0, 0.0),
                    "pitch_range": (0.0, 0.0),
                    "yaw_range": (0, 6.2832),
                },
            },
            "irrigation": {
                "scale_range": (0.8, 1.2),
                "min_distance": 2.0,
                "zone_weights": {"edge": 0.0, "center": 1.0},
                "rotation": {
                    "roll_range": (0.0, 0.0),
                    "pitch_range": (0.0, 0.0),
                    "yaw_range": (0, 3.1416),
                },
            },
        }

    @classmethod
    def default_strategies(cls) -> Dict:
        from forest3d.core.placement import (
            ClusteredPlacement,
            RowPlacement,
        )

        return {
            "crop": RowPlacement(),
            "weed": ClusteredPlacement(
                anchor_categories=("crop",),
                cluster_prob=0.8,
                scale_aware=False,
                radius_min_fixed=0.3,
                radius_max_fixed=2.0,
                height_offset_range=(-0.02, 0.02),
            ),
            "irrigation": RowPlacement(),
        }

    @classmethod
    def default_cross_distances(cls) -> Dict:
        return {
            ("crop", "crop"): 0.3,
            ("crop", "weed"): 0.2,
            ("crop", "irrigation"): 0.5,
            ("weed", "weed"): 0.2,
            ("weed", "irrigation"): 0.5,
            ("irrigation", "irrigation"): 1.0,
        }

    @classmethod
    def default_category_order(cls) -> List[str]:
        return ["irrigation", "crop", "weed"]

    @classmethod
    def default_densities(cls) -> Dict[str, int]:
        return {"crop": 200, "weed": 50, "irrigation": 5}

    def _build_row_heightmap(self) -> Tuple[np.ndarray, float, float]:
        """Generate heightmap with alternating rows and furrows."""
        p = self.params

        cols = max(2, int(p.field_length / p.resolution))
        rows = max(2, int(p.field_width / p.resolution))

        elevation = np.zeros((rows, cols), dtype=np.float32)
        cycle = p.row_width + p.furrow_width
        half_row = p.row_width / 2.0

        for i in range(p.num_rows):
            centre = p.headland_width + half_row + i * cycle
            centre_idx = int(centre / p.resolution)
            half_row_idx = int(half_row / p.resolution)
            if centre_idx - half_row_idx >= rows:
                break
            start = max(0, centre_idx - half_row_idx)
            end = min(rows, centre_idx + half_row_idx)

            for r in range(start, end):
                dist_from_centre = abs(r * p.resolution - centre)
                if dist_from_centre >= half_row:
                    continue
                if p.row_profile == "flat":
                    height = p.row_height
                elif p.row_profile == "rounded":
                    t = dist_from_centre / half_row
                    height = p.row_height * np.cos(t * np.pi / 2)
                elif p.row_profile == "trapezoid":
                    # Flat top for ~60% of width, then slope
                    flat_frac = 0.6
                    if dist_from_centre < half_row * flat_frac:
                        height = p.row_height
                    else:
                        t = (dist_from_centre - half_row * flat_frac) / (
                            half_row * (1 - flat_frac)
                        )
                        height = p.row_height * max(0, 1 - t)
                else:
                    height = p.row_height
                elevation[r, :] = np.maximum(elevation[r, :], height)

        return elevation, p.resolution, p.resolution

    def generate_terrain_mesh(
        self,
        uv_tile_scale: float = 10.0,
        **kwargs,
    ) -> Tuple[Path, dict]:
        """Generate a crop-row field mesh."""
        elevation, dx, dy = self._build_row_heightmap()
        rows, cols = elevation.shape
        logger.info(
            f"Creating crop-row mesh: {rows}x{cols}, "
            f"{self.params.num_rows} rows"
        )

        vertices, uvs, faces = self._build_mesh_from_heightmap(
            elevation, dx, dy, 1.0, uv_tile_scale
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
            "num_rows": self.params.num_rows,
            "row_spacing": self.params.row_width + self.params.furrow_width,
        }
        logger.info(
            f"Field: X={stats['x_extent']:.2f}, Y={stats['y_extent']:.2f}, "
            f"Z={stats['z_extent']:.2f}, {stats['num_rows']} rows"
        )
        return stl_path, stats
