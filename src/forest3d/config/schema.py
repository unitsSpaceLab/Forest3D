"""Configuration schema using Pydantic for validation."""

from pathlib import Path
from typing import Dict, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


class BlenderConfig(BaseModel):
    """Blender configuration for asset conversion."""

    path: Optional[Path] = Field(
        default=None, description="Path to Blender executable (auto-detected if None)"
    )
    visual_decimation: float = Field(
        default=0.1, ge=0.01, le=1.0, description="Decimation ratio for visual mesh"
    )
    collision_decimation: float = Field(
        default=0.01,
        ge=0.001,
        le=0.5,
        description="Decimation ratio for collision mesh",
    )

    @field_validator("path", mode="before")
    @classmethod
    def expand_path(cls, v):
        if v is None:
            return v
        return Path(v).expanduser().resolve()


class DemConfig(BaseModel):
    """DEM-specific terrain settings."""

    smooth_sigma: float = Field(
        default=1.0, ge=0.0, le=10.0, description="Gaussian smoothing"
    )
    enhance: bool = Field(default=False, description="Enable DEM enhancement")
    enhance_scale: float = Field(
        default=6.0, ge=1.0, le=20.0, description="Enhancement scale"
    )


class CropRowConfig(BaseModel):
    """Crop-row terrain parameters for agricultural fields."""

    field_length: float = Field(
        default=50.0, ge=1.0, le=1000.0, description="Field length (X) in metres"
    )
    field_width: float = Field(
        default=30.0, ge=1.0, le=1000.0, description="Field width (Y) in metres"
    )
    num_rows: int = Field(
        default=8, ge=1, le=200, description="Number of crop rows"
    )
    row_width: float = Field(
        default=1.0, ge=0.2, le=5.0, description="Width of each raised bed (m)"
    )
    furrow_width: float = Field(
        default=0.5, ge=0.1, le=3.0, description="Width of each furrow (m)"
    )
    row_height: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Height of row above furrow (m)"
    )
    headland_width: float = Field(
        default=2.0, ge=0.0, le=20.0, description="Unplanted border width (m)"
    )
    resolution: float = Field(
        default=0.25, ge=0.05, le=1.0, description="Grid resolution (m/cell)"
    )
    row_profile: str = Field(
        default="rounded",
        description="Row cross-section: flat, rounded, trapezoid",
    )
    plant_spacing: float = Field(
        default=0.8,
        ge=0.05,
        le=10.0,
        description="Spacing between plants along a row (m); placement only",
    )
    stagger: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Offset of alternate rows as a fraction of plant_spacing; "
            "placement only"
        ),
    )


class TerrainConfig(BaseModel):
    """Terrain generation configuration."""

    type: str = Field(
        default="dem",
        description="Terrain type: 'dem', 'crop_rows', or custom registered type",
    )
    scale_factor: float = Field(
        default=1.0, ge=0.1, le=100.0, description="Scale factor"
    )
    material_name: str = Field(
        default="Terrain/Ground",
        description="Name for the generated material",
    )
    texture_blend: Optional[Path] = Field(
        default=None,
        description="Path to Blender file for terrain texture extraction",
    )
    dem: DemConfig = Field(default_factory=DemConfig)
    crop_rows: CropRowConfig = Field(default_factory=CropRowConfig)

    @field_validator("texture_blend", mode="before")
    @classmethod
    def expand_texture_blend(cls, v):
        if v is None:
            return v
        return Path(v).expanduser().resolve()

    # Backward compatibility: accept old flat fields and migrate to dem.*
    smooth_sigma: Optional[float] = Field(default=None)
    enhance: Optional[bool] = Field(default=None)
    enhance_scale: Optional[float] = Field(default=None)

    @model_validator(mode="after")
    def _migrate_legacy_fields(self):
        import logging
        if self.smooth_sigma is not None:
            logging.getLogger("forest3d.config").warning(
                "terrain.smooth_sigma is deprecated, use terrain.dem.smooth_sigma"
            )
            self.dem.smooth_sigma = self.smooth_sigma
        if self.enhance is not None:
            logging.getLogger("forest3d.config").warning(
                "terrain.enhance is deprecated, use terrain.dem.enhance"
            )
            self.dem.enhance = self.enhance
        if self.enhance_scale is not None:
            logging.getLogger("forest3d.config").warning(
                "terrain.enhance_scale is deprecated, use terrain.dem.enhance_scale"
            )
            self.dem.enhance_scale = self.enhance_scale
        # Strip legacy fields from the instance so they never appear in
        # serialized output.
        for f in ("smooth_sigma", "enhance", "enhance_scale"):
            try:
                delattr(self, f)
            except AttributeError:
                pass
        return self


class DensityConfig(BaseModel):
    """Forest/crop population density configuration."""

    tree: int = Field(default=50, ge=0, le=1000, description="Number of trees")
    bush: int = Field(default=10, ge=0, le=500, description="Number of bushes")
    rock: int = Field(default=5, ge=0, le=200, description="Number of rocks")
    grass: int = Field(
        default=50, ge=0, le=2000, description="Number of grass patches"
    )
    sand: int = Field(default=5, ge=0, le=100, description="Number of sand patches")
    crop: int = Field(
        default=0, ge=0, le=10000, description="Number of crop plants"
    )

    def as_dict(self) -> Dict[str, int]:
        return {
            "tree": self.tree,
            "bush": self.bush,
            "rock": self.rock,
            "grass": self.grass,
            "sand": self.sand,
            "crop": self.crop,
        }


class PathsConfig(BaseModel):
    """Project paths configuration."""

    base_path: Optional[Path] = Field(
        default=None, description="Project base path (auto-detected if None)"
    )
    models_path: Optional[Path] = Field(
        default=None, description="Models directory"
    )
    worlds_path: Optional[Path] = Field(
        default=None, description="Worlds directory"
    )

    @field_validator("base_path", "models_path", "worlds_path", mode="before")
    @classmethod
    def expand_path(cls, v):
        if v is None:
            return v
        return Path(v).expanduser().resolve()


class Forest3DConfig(BaseModel):
    """Main configuration schema for Forest3D."""

    blender: BlenderConfig = Field(default_factory=BlenderConfig)
    terrain: TerrainConfig = Field(default_factory=TerrainConfig)
    density: DensityConfig = Field(default_factory=DensityConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    model_config = {"extra": "ignore"}
