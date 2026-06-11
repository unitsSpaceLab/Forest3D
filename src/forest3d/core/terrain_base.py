"""Abstract base class for all terrain generators."""

import logging
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from forest3d.core.placement import PlacementStrategy

import numpy as np
from scipy.ndimage import gaussian_filter
from stl import mesh as stl_mesh

from forest3d.config.schema import TerrainConfig

logger = logging.getLogger("forest3d.terrain")


def find_blender() -> Optional[Path]:
    """Auto-detect Blender installation."""
    blender_in_path = shutil.which("blender")
    if blender_in_path:
        return Path(blender_in_path)

    common_paths = [
        Path("/usr/bin/blender"),
        Path("/usr/local/bin/blender"),
        Path("/snap/bin/blender"),
        Path("/opt/blender/blender"),
        Path.home() / "blender" / "blender",
    ]

    for base in [Path.home() / "Downloads", Path("/opt"), Path.home()]:
        if base.exists():
            try:
                for item in base.iterdir():
                    if item.is_dir() and item.name.lower().startswith("blender"):
                        blender_exec = item / "blender"
                        if blender_exec.exists() and blender_exec.is_file():
                            common_paths.append(blender_exec)
            except PermissionError:
                continue

    for path in common_paths:
        if path.exists() and path.is_file():
            return path

    return None


class BaseTerrain(ABC):
    """Base class for terrain generators.

    Every terrain type produces:
      - mesh/terrain.obj   (visual with UVs)
      - mesh/terrain.stl   (collision + height sampling)
      - model.sdf
      - model.config
      - test.world
    """

    TERRAIN_TYPE = "base"

    def __init__(
        self,
        output_path: Path,
        config: Optional[TerrainConfig] = None,
        blender_path: Optional[Path] = None,
    ):
        self.config = config or TerrainConfig()
        self._blender_path = blender_path
        self.output_path = Path(output_path)
        self.mesh_path = self.output_path / "mesh"
        self.material_path = self.output_path / "material"
        self.texture_path = self.output_path / "texture"
        self._setup_directories()

    def _setup_directories(self) -> None:
        for path in [self.mesh_path, self.material_path, self.texture_path]:
            path.mkdir(parents=True, exist_ok=True)

    # --- Subclasses must implement ---

    @abstractmethod
    def generate_terrain_mesh(
        self,
        **kwargs,
    ) -> Tuple[Path, dict]:
        """Generate OBJ (visual) and STL (collision) meshes.

        Returns:
            Tuple of (stl_path, stats_dict).
        """

    # --- CLI integration (override per type) ---

    @classmethod
    def cli_options(cls) -> list:
        """Return click.Option definitions for this terrain type.

        Used by the CLI to build the subcommand dynamically.
        Override in subclasses to declare type-specific arguments.
        """
        return []

    @classmethod
    def cli_create(cls, output_path: Path, config, **kwargs):
        """Build an instance from CLI keyword arguments.

        Override in subclasses whose ``__init__`` signature differs
        from ``cls(output_path=..., config=...)``.
        """
        return cls(output_path=output_path, config=config)

    @classmethod
    def cli_apply_overrides(cls, config, kwargs) -> None:
        """Push CLI flag values into the config object *in place*.

        Override to map CLI kwargs (dashed → underscored) to
        ``config.terrain.*`` fields before the terrain instance is built.
        The base implementation is a no-op.
        """

    @classmethod
    def cli_post_process(cls, instance, config) -> None:
        """Run after terrain generation, before the success message.

        Override for type-specific post-processing.  The base
        implementation handles shared logic like texture extraction.
        """
        if config.terrain.texture_blend:
            instance.extract_terrain_texture(config.terrain.texture_blend)

    # --- World-population defaults (override per terrain type) ---

    @classmethod
    def default_categories(cls) -> Dict[str, dict]:
        """Per-category configuration for ``WorldPopulator``.

        Each entry may contain: scale_range, min_distance, zone_weights, rotation.
        """
        return {}

    @classmethod
    def default_strategies(cls) -> Dict[str, "PlacementStrategy"]:
        """Placement strategies keyed by category name.

        Used by ``WorldPopulator`` to dispatch placement per category.
        Import ``PlacementStrategy`` lazily to avoid circular imports.
        """
        return {}

    @classmethod
    def default_cross_distances(cls) -> Dict[Tuple[str, str], float]:
        """Cross-category minimum distances (order-insensitive)."""
        return {}

    @classmethod
    def default_category_order(cls) -> List[str]:
        """Order in which categories are placed by ``WorldPopulator``."""
        return list(cls.default_categories().keys())

    @classmethod
    def default_densities(cls) -> Dict[str, int]:
        """Default model counts per category for this terrain type.
        Override to provide sensible defaults without requiring ``--density``.
        """
        return {}

    # --- Shared helpers ---

    def _build_mesh_from_heightmap(
        self,
        elevation: np.ndarray,
        pixel_width: float,
        pixel_height: float,
        z_scale: float = 1.0,
        uv_tile_scale: float = 10.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert a 2D heightmap into vertices, UVs, and faces."""
        rows, cols = elevation.shape
        vertices = []
        uvs = []
        for y in range(rows):
            for x in range(cols):
                wx = x * pixel_width
                wy = y * pixel_height
                wz = elevation[y, x] * z_scale
                vertices.append([wx, wy, wz])
                uvs.append([
                    (x / (cols - 1)) * uv_tile_scale,
                    (y / (rows - 1)) * uv_tile_scale,
                ])

        faces = []
        for y in range(rows - 1):
            for x in range(cols - 1):
                v0 = y * cols + x
                v1 = v0 + 1
                v2 = (y + 1) * cols + x
                v3 = v2 + 1
                faces.extend([[v0, v1, v2], [v1, v3, v2]])

        return np.array(vertices), np.array(uvs), np.array(faces)

    def _center_and_shift(self, vertices: np.ndarray) -> None:
        """Center XY and shift Z to zero, in-place."""
        center_xy = np.mean(vertices[:, :2], axis=0)
        vertices[:, 0] -= center_xy[0]
        vertices[:, 1] -= center_xy[1]
        vertices[:, 2] -= np.min(vertices[:, 2])

    def _calculate_normals(
        self, vertices: np.ndarray, faces: np.ndarray
    ) -> np.ndarray:
        """Calculate vertex normals."""
        normals = np.zeros_like(vertices)
        for face in faces:
            v0, v1, v2 = (
                vertices[face[0]],
                vertices[face[1]],
                vertices[face[2]],
            )
            fn = np.cross(v1 - v0, v2 - v0)
            length = np.linalg.norm(fn)
            if length > 0:
                fn /= length
            for idx in face:
                normals[idx] += fn
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths == 0] = 1
        normals /= lengths
        return normals

    def _write_obj(
        self,
        path: Path,
        vertices: np.ndarray,
        uvs: np.ndarray,
        normals: np.ndarray,
        faces: np.ndarray,
    ) -> None:
        """Write OBJ with UVs and normals."""
        with open(path, "w") as f:
            f.write("# Terrain mesh - Forest3D\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for uv in uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            for face in faces:
                f.write(
                    f"f {face[0]+1}/{face[0]+1}/{face[0]+1}"
                    f" {face[1]+1}/{face[1]+1}/{face[1]+1}"
                    f" {face[2]+1}/{face[2]+1}/{face[2]+1}\n"
                )

    def _write_stl(
        self, path: Path, vertices: np.ndarray, faces: np.ndarray
    ) -> None:
        """Write STL for collision and height sampling."""
        terrain = stl_mesh.Mesh(
            np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype)
        )
        for i, f in enumerate(faces):
            for j in range(3):
                terrain.vectors[i][j] = vertices[f[j]]
        terrain.save(str(path))

    def _find_textures(self) -> List[str]:
        textures = []
        if self.texture_path.exists():
            for f in self.texture_path.iterdir():
                if f.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    textures.append(f.name)
        return textures

    def _create_sdf_file(
        self, textures: Optional[List[str]] = None
    ) -> Path:
        """Create SDF with OBJ visual and STL collision."""
        albedo_map = normal_map = roughness_map = None

        if textures:
            for t in textures:
                tl = t.lower()
                if t.endswith(".exr"):
                    continue
                if any(k in tl for k in ["diff", "albedo", "base", "color"]):
                    albedo_map = t
                elif any(k in tl for k in ["normal", "nor", "nrm"]):
                    normal_map = t
                elif any(k in tl for k in ["rough"]):
                    roughness_map = t
            if not albedo_map:
                for t in textures:
                    if not t.endswith(".exr"):
                        albedo_map = t
                        break

        if albedo_map:
            pbr = f"""                <material>
                    <ambient>1.0 1.0 1.0 1</ambient>
                    <diffuse>1.0 1.0 1.0 1</diffuse>
                    <specular>0.1 0.1 0.1 1</specular>
                    <pbr>
                        <metal>
                            <albedo_map>model://ground/texture/{albedo_map}</albedo_map>"""
            if normal_map:
                pbr += f"""
                            <normal_map>model://ground/texture/{normal_map}</normal_map>"""
            if roughness_map:
                pbr += f"""
                            <roughness_map>model://ground/texture/{roughness_map}</roughness_map>"""
            pbr += """
                            <metalness>0.0</metalness>
                        </metal>
                    </pbr>
                </material>"""
        else:
            pbr = """                <material>
                    <ambient>0.6 0.6 0.6 1</ambient>
                    <diffuse>0.8 0.8 0.8 1</diffuse>
                </material>"""

        sdf = f"""<?xml version="1.0" ?>
<sdf version="1.8">
    <model name="terrain">
        <static>true</static>
        <link name="link">
            <collision name="collision">
                <geometry>
                    <mesh>
                        <uri>model://ground/mesh/terrain.stl</uri>
                    </mesh>
                </geometry>
            </collision>
            <visual name="visual">
                <geometry>
                    <mesh>
                        <uri>model://ground/mesh/terrain.obj</uri>
                    </mesh>
                </geometry>
{pbr}
            </visual>
        </link>
    </model>
</sdf>"""
        sdf_path = self.output_path / "model.sdf"
        sdf_path.write_text(sdf)
        return sdf_path

    def _create_config_file(self) -> Path:
        content = """<?xml version="1.0"?>
<model>
    <name>ground</name>
    <version>1.0</version>
    <sdf version="1.8">model.sdf</sdf>
    <author>
        <name>AI4Forest</name>
        <email>khalid.bourr@gmail.com</email>
    </author>
    <description>Terrain from Forest3D</description>
</model>"""
        path = self.output_path / "model.config"
        path.write_text(content)
        return path

    def _create_test_world(self) -> Path:
        content = """<?xml version="1.0" ?>
<sdf version="1.8">
    <world name="terrain_test">
        <scene>
            <ambient>0.6 0.6 0.6 1</ambient>
            <background>0.7 0.8 0.9 1</background>
        </scene>
        <physics name="1ms" type="ignored">
            <max_step_size>0.001</max_step_size>
            <real_time_factor>1.0</real_time_factor>
        </physics>
        <gravity>0 0 -9.8</gravity>
        <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
        <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
        <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
        <light name="sun" type="directional">
            <cast_shadows>true</cast_shadows>
            <pose>0 0 10 0 0 0</pose>
            <diffuse>1.0 1.0 1.0 1</diffuse>
            <specular>0.5 0.5 0.5 1</specular>
            <direction>-0.5 0.1 -0.9</direction>
        </light>
        <include>
            <name>terrain</name>
            <uri>model://ground</uri>
        </include>
    </world>
</sdf>"""
        path = self.output_path / "test.world"
        path.write_text(content)
        return path

    def extract_terrain_texture(self, blend_file: Path) -> List[Path]:
        """Extract textures from Blender file for PBR materials."""
        blend_file = Path(blend_file)
        if not blend_file.exists():
            raise FileNotFoundError(f"Blend file not found: {blend_file}")

        blender_path = self._blender_path or find_blender()
        if not blender_path:
            raise RuntimeError("Blender not found")

        script = f'''
import bpy, os, shutil
output_dir = "{self.texture_path}"
bpy.ops.wm.open_mainfile(filepath="{blend_file}")
for img in bpy.data.images:
    if img.source == 'FILE' and img.filepath:
        fp = bpy.path.abspath(img.filepath)
        if os.path.exists(fp):
            fn = os.path.basename(fp)
            if fn.lower().endswith('.exr'):
                img.file_format = 'PNG'
                fn = fn.rsplit('.', 1)[0] + '.png'
                img.save_render(os.path.join(output_dir, fn))
            else:
                shutil.copy2(fp, os.path.join(output_dir, fn))
            print(f"EXPORTED: {{fn}}")
    elif img.packed_file:
        fn = img.name if '.' in img.name else img.name + '.png'
        img.save_render(os.path.join(output_dir, fn))
        print(f"EXPORTED: {{fn}}")
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            subprocess.run(
                [str(blender_path), "--background", "--python", script_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
        finally:
            os.unlink(script_path)

        textures = self._find_textures()
        if textures:
            self._create_sdf_file(textures)
        return [self.texture_path / t for t in textures]

    def process_terrain(self, **kwargs) -> Path:
        """Full pipeline: generate mesh, SDF model, config, test world."""
        logger.info(f"Generating {self.TERRAIN_TYPE} terrain...")
        self.generate_terrain_mesh(**kwargs)
        textures = self._find_textures()
        self._create_sdf_file(textures)
        self._create_config_file()
        self._create_test_world()
        logger.info(f"Terrain complete: {self.output_path}")
        return self.output_path


# Registry of available terrain types
_TERRAIN_REGISTRY: Dict[str, type] = {}


def register_terrain(terrain_cls: type) -> type:
    """Register a terrain type so it can be looked up by name."""
    _TERRAIN_REGISTRY[terrain_cls.TERRAIN_TYPE] = terrain_cls
    return terrain_cls


def get_terrain_class(terrain_type: str) -> type:
    """Look up a terrain class by its type name."""
    if terrain_type not in _TERRAIN_REGISTRY:
        available = list(_TERRAIN_REGISTRY)
        raise ValueError(
            f"Unknown terrain type '{terrain_type}'. "
            f"Available: {available}"
        )
    return _TERRAIN_REGISTRY[terrain_type]


def list_terrain_types() -> List[str]:
    """Return all registered terrain type names."""
    return list(_TERRAIN_REGISTRY)
