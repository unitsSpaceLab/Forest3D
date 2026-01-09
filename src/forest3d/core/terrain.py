"""Terrain generation from DEM data."""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from stl import mesh as stl_mesh

from forest3d.config.schema import TerrainConfig

try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

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


class TerrainGenerator:
    """Generate terrain meshes from DEM data.

    Outputs:
        - terrain.obj: Visual mesh with UVs for PBR textures
        - terrain.stl: Collision mesh + height sampling for forest.py
    """

    def __init__(
            self,
            tif_path: Path,
            output_path: Optional[Path] = None,
            config: Optional[TerrainConfig] = None,
            blender_path: Optional[Path] = None,
    ):
        if not GDAL_AVAILABLE:
            raise ImportError("GDAL is required. Install with: pip install GDAL")

        self.tif_path = Path(tif_path)
        if not self.tif_path.exists():
            raise FileNotFoundError(f"DEM file not found: {self.tif_path}")

        self.config = config or TerrainConfig()
        self._blender_path = blender_path
        self._material_name = self.config.material_name

        if output_path:
            self.terrain_path = Path(output_path)
        else:
            self.terrain_path = self.tif_path.parent.parent

        self.mesh_path = self.terrain_path / "mesh"
        self.material_path = self.terrain_path / "material"
        self.texture_path = self.terrain_path / "texture"

        self._setup_directories()

    def _setup_directories(self) -> None:
        for path in [self.mesh_path, self.material_path, self.texture_path]:
            path.mkdir(parents=True, exist_ok=True)

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

    def create_terrain_mesh(
            self,
            scale_factor: Optional[float] = None,
            z_scale: Optional[float] = None,
            smooth_sigma: Optional[float] = None,
            enhance: Optional[bool] = None,
            uv_tile_scale: float = 10.0,
    ) -> Tuple[Path, dict]:
        """Create terrain meshes (OBJ for visual, STL for collision/height sampling)."""
        scale_factor = scale_factor if scale_factor is not None else self.config.scale_factor
        z_scale = z_scale if z_scale is not None else scale_factor
        smooth_sigma = smooth_sigma if smooth_sigma is not None else self.config.smooth_sigma
        enhance = enhance if enhance is not None else self.config.enhance

        # Load DEM
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

        # Generate vertices, UVs, faces
        vertices = []
        uvs = []
        for y in range(rows):
            for x in range(cols):
                world_x = x * pixel_width * scale_factor
                world_y = y * pixel_height * scale_factor
                world_z = elevation[y, x] * z_scale
                vertices.append([world_x, world_y, world_z])
                uvs.append([(x / (cols - 1)) * uv_tile_scale, (y / (rows - 1)) * uv_tile_scale])

        faces = []
        for y in range(rows - 1):
            for x in range(cols - 1):
                v0 = y * cols + x
                v1 = v0 + 1
                v2 = (y + 1) * cols + x
                v3 = v2 + 1
                faces.extend([[v0, v1, v2], [v1, v3, v2]])

        vertices = np.array(vertices)
        uvs = np.array(uvs)
        faces = np.array(faces)

        # Center XY, shift Z to 0
        center_xy = np.mean(vertices[:, :2], axis=0)
        vertices[:, 0] -= center_xy[0]
        vertices[:, 1] -= center_xy[1]
        vertices[:, 2] -= np.min(vertices[:, 2])

        # Calculate normals
        normals = self._calculate_normals(vertices, faces)

        # Save OBJ (visual with UVs)
        obj_path = self.mesh_path / "terrain.obj"
        self._write_obj(obj_path, vertices, uvs, normals, faces)
        logger.info(f"Created OBJ mesh: {obj_path}")

        # Save STL (collision + height sampling)
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
        logger.info(f"Terrain: X={stats['x_extent']:.2f}, Y={stats['y_extent']:.2f}, Z={stats['z_extent']:.2f}")
        return stl_path, stats

    def _calculate_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Calculate vertex normals."""
        normals = np.zeros_like(vertices)
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
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

    def _write_obj(self, path: Path, vertices: np.ndarray, uvs: np.ndarray, normals: np.ndarray, faces: np.ndarray) -> None:
        """Write OBJ with UVs and normals."""
        with open(path, 'w') as f:
            f.write("# Terrain mesh - Forest3D\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for uv in uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}/{face[2]+1}\n")

    def _write_stl(self, path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
        """Write STL for collision and height sampling."""
        terrain = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                terrain.vectors[i][j] = vertices[f[j]]
        terrain.save(str(path))

    def _create_sdf_file(self, textures: Optional[List[str]] = None) -> Path:
        """Create SDF with OBJ visual and STL collision."""
        albedo_map = normal_map = roughness_map = None

        if textures:
            for t in textures:
                tl = t.lower()
                if t.endswith('.exr'):
                    continue
                if any(k in tl for k in ["diff", "albedo", "base", "color"]):
                    albedo_map = t
                elif any(k in tl for k in ["normal", "nor", "nrm"]):
                    normal_map = t
                elif any(k in tl for k in ["rough"]):
                    roughness_map = t
            if not albedo_map:
                for t in textures:
                    if not t.endswith('.exr'):
                        albedo_map = t
                        break

        if albedo_map:
            pbr = f'''                <material>
                    <ambient>1.0 1.0 1.0 1</ambient>
                    <diffuse>1.0 1.0 1.0 1</diffuse>
                    <specular>0.1 0.1 0.1 1</specular>
                    <pbr>
                        <metal>
                            <albedo_map>model://ground/texture/{albedo_map}</albedo_map>'''
            if normal_map:
                pbr += f'''
                            <normal_map>model://ground/texture/{normal_map}</normal_map>'''
            if roughness_map:
                pbr += f'''
                            <roughness_map>model://ground/texture/{roughness_map}</roughness_map>'''
            pbr += '''
                            <metalness>0.0</metalness>
                        </metal>
                    </pbr>
                </material>'''
        else:
            pbr = '''                <material>
                    <ambient>0.6 0.6 0.6 1</ambient>
                    <diffuse>0.8 0.8 0.8 1</diffuse>
                </material>'''

        sdf = f'''<?xml version="1.0" ?>
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
</sdf>'''
        sdf_path = self.terrain_path / "model.sdf"
        sdf_path.write_text(sdf)
        return sdf_path

    def _create_config_file(self) -> Path:
        content = '''<?xml version="1.0"?>
<model>
    <name>ground</name>
    <version>1.0</version>
    <sdf version="1.8">model.sdf</sdf>
    <author>
        <name>AI4Forest</name>
        <email>khalid.bourr@gmail.com</email>
    </author>
    <description>Terrain from DEM with PBR materials</description>
</model>'''
        path = self.terrain_path / "model.config"
        path.write_text(content)
        return path

    def _create_test_world(self) -> Path:
        content = '''<?xml version="1.0" ?>
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
</sdf>'''
        path = self.terrain_path / "test.world"
        path.write_text(content)
        return path

    def _find_textures(self) -> List[str]:
        textures = []
        if self.texture_path.exists():
            for f in self.texture_path.iterdir():
                if f.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    textures.append(f.name)
        return textures

    def process_terrain(
            self,
            scale_factor: Optional[float] = None,
            z_scale: Optional[float] = None,
            smooth_sigma: Optional[float] = None,
            enhance: Optional[bool] = None,
            uv_tile_scale: float = 10.0,
    ) -> Path:
        """Full terrain pipeline."""
        logger.info("Starting terrain generation...")
        self.create_terrain_mesh(scale_factor, z_scale, smooth_sigma, enhance, uv_tile_scale)
        textures = self._find_textures()
        self._create_sdf_file(textures)
        self._create_config_file()
        self._create_test_world()
        logger.info(f"Terrain complete: {self.terrain_path}")
        return self.terrain_path

    def extract_terrain_texture(self, blend_file: Path) -> List[Path]:
        """Extract textures from Blender file."""
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
            subprocess.run([str(blender_path), "--background", "--python", script_path],
                           capture_output=True, text=True, timeout=120)
        finally:
            os.unlink(script_path)

        textures = self._find_textures()
        if textures:
            self._create_sdf_file(textures)
        return [self.texture_path / t for t in textures]