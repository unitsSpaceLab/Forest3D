"""Configuration schema using Pydantic for validation."""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class BlenderConfig(BaseModel):
    """Blender configuration for asset conversion."""

    path: Optional[Path] = Field(
        default=None, description="Path to Blender executable (auto-detected if None)"
    )
    visual_decimation: float = Field(
        default=0.1, ge=0.01, le=1.0, description="Decimation ratio for visual mesh"
    )
    collision_decimation: float = Field(
        default=0.01, ge=0.001, le=0.5, description="Decimation ratio for collision mesh"
    )

    @field_validator("path", mode="before")
    @classmethod
    def expand_path(cls, v):
        if v is None:
            return v
        return Path(v).expanduser().resolve()


class TerrainConfig(BaseModel):
    """Terrain generation configuration."""

    scale_factor: float = Field(default=1.0, ge=0.1, le=100.0, description="Scale factor")
    smooth_sigma: float = Field(default=1.0, ge=0.0, le=10.0, description="Gaussian smoothing")
    enhance: bool = Field(default=False, description="Enable DEM enhancement")
    enhance_scale: float = Field(default=6.0, ge=1.0, le=20.0, description="Enhancement scale")
    texture_blend: Optional[Path] = Field(
        default=None, description="Path to Blender file for terrain texture extraction"
    )
    material_name: str = Field(
        default="Terrain/Ground", description="Name for the generated material"
    )

    @field_validator("texture_blend", mode="before")
    @classmethod
    def expand_texture_path(cls, v):
        if v is None:
            return v
        return Path(v).expanduser().resolve()


class DensityConfig(BaseModel):
    """Forest population density configuration."""

    tree: int = Field(default=50, ge=0, le=1000, description="Number of trees")
    bush: int = Field(default=10, ge=0, le=500, description="Number of bushes")
    rock: int = Field(default=5, ge=0, le=200, description="Number of rocks")
    grass: int = Field(default=50, ge=0, le=2000, description="Number of grass patches")
    sand: int = Field(default=5, ge=0, le=100, description="Number of sand patches")


class PathsConfig(BaseModel):
    """Project paths configuration."""

    base_path: Optional[Path] = Field(
        default=None, description="Project base path (auto-detected if None)"
    )
    models_path: Optional[Path] = Field(default=None, description="Models directory")
    worlds_path: Optional[Path] = Field(default=None, description="Worlds directory")

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
