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
from stl import mesh

from forest3d.config.schema import TerrainConfig

# GDAL is optional - only required for terrain generation
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
    """Generate terrain meshes from DEM (Digital Elevation Model) data."""

    def __init__(
        self,
        tif_path: Path,
        output_path: Optional[Path] = None,
        config: Optional[TerrainConfig] = None,
        blender_path: Optional[Path] = None,
    ):
        if not GDAL_AVAILABLE:
            raise ImportError(
                "GDAL is required for terrain generation. "
                "Install with: pip install GDAL or use Docker image."
            )

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
        try:
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
        except Exception as e:
            raise RuntimeError(f"Failed to enhance DEM: {e}") from e

    def create_terrain_mesh(
        self,
        scale_factor: Optional[float] = None,
        smooth_sigma: Optional[float] = None,
        enhance: Optional[bool] = None,
    ) -> Tuple[Path, dict]:
        scale_factor = scale_factor if scale_factor is not None else self.config.scale_factor
        smooth_sigma = smooth_sigma if smooth_sigma is not None else self.config.smooth_sigma
        enhance = enhance if enhance is not None else self.config.enhance

        try:
            if enhance:
                dem_file = self.enhance_dem(self.config.enhance_scale)
                logger.info("Using enhanced DEM...")
            else:
                dem_file = self.tif_path
                logger.info("Using original DEM...")

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

            vertices = []
            faces = []

            for y in range(rows):
                for x in range(cols):
                    world_x = x * pixel_width * scale_factor
                    world_y = y * pixel_height * scale_factor
                    world_z = elevation[y, x] * scale_factor
                    vertices.append([world_x, world_y, world_z])

            for y in range(rows - 1):
                for x in range(cols - 1):
                    v0 = y * cols + x
                    v1 = v0 + 1
                    v2 = (y + 1) * cols + x
                    v3 = v2 + 1
                    faces.extend([[v0, v1, v2], [v1, v3, v2]])

            vertices = np.array(vertices)
            faces = np.array(faces)
            center = np.mean(vertices, axis=0)
            vertices -= center

            # Save as STL
            terrain = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                for j in range(3):
                    terrain.vectors[i][j] = vertices[f[j]]

            output_path = self.mesh_path / "terrain.stl"
            terrain.save(str(output_path))
            logger.info(f"Created terrain mesh at: {output_path}")

            stats = {
                "x_extent": float(np.ptp(vertices[:, 0])),
                "y_extent": float(np.ptp(vertices[:, 1])),
                "z_extent": float(np.ptp(vertices[:, 2])),
                "num_vertices": len(vertices),
                "num_faces": len(faces),
            }
            logger.info(f"Terrain dimensions: X={stats['x_extent']:.2f}, Y={stats['y_extent']:.2f}, Z={stats['z_extent']:.2f}")
            return output_path, stats

        except Exception as e:
            raise RuntimeError(f"Failed to create terrain mesh: {e}") from e

    def _create_sdf_file(self, textures: Optional[List[str]] = None) -> Path:
        """Create SDF file with PBR materials for Gazebo Harmonic."""
        # Find texture files
        albedo_map = None
        normal_map = None
        roughness_map = None

        if textures:
            for texture in textures:
                t_lower = texture.lower()
                if any(kw in t_lower for kw in ["diff", "albedo", "base", "color"]):
                    albedo_map = texture
                elif any(kw in t_lower for kw in ["normal", "nor", "nrm"]):
                    normal_map = texture
                elif any(kw in t_lower for kw in ["rough", "roughness"]):
                    roughness_map = texture

            if not albedo_map and textures:
                albedo_map = textures[0]

        # Build material section
        if albedo_map:
            pbr_content = f'''                <material>
                    <ambient>0.8 0.8 0.8 1</ambient>
                    <diffuse>1.0 1.0 1.0 1</diffuse>
                    <specular>0.2 0.2 0.2 1</specular>
                    <pbr>
                        <metal>
                            <albedo_map>model://ground/texture/{albedo_map}</albedo_map>'''
            if normal_map:
                pbr_content += f'''
                            <normal_map>model://ground/texture/{normal_map}</normal_map>'''
            if roughness_map:
                pbr_content += f'''
                            <roughness_map>model://ground/texture/{roughness_map}</roughness_map>'''
            pbr_content += '''
                        </metal>
                    </pbr>
                </material>'''
        else:
            pbr_content = '''                <material>
                    <ambient>0.5 0.5 0.5 1</ambient>
                    <diffuse>0.7 0.7 0.7 1</diffuse>
                </material>'''

        sdf_content = f'''<?xml version="1.0" ?>
<sdf version="1.7">
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
                        <uri>model://ground/mesh/terrain.stl</uri>
                    </mesh>
                </geometry>
{pbr_content}
            </visual>
        </link>
    </model>
</sdf>'''
        sdf_path = self.terrain_path / "model.sdf"
        sdf_path.write_text(sdf_content)
        return sdf_path

    def _create_config_file(self) -> Path:
        config_content = '''<?xml version="1.0"?>
<model>
    <name>ground</name>
    <version>1.0</version>
    <sdf version="1.7">model.sdf</sdf>
    <author>
        <name>AI4Forest</name>
        <email>khalid.bourr@gmail.com</email>
    </author>
    <description>Terrain model generated from DEM data</description>
</model>'''
        config_path = self.terrain_path / "model.config"
        config_path.write_text(config_content)
        return config_path

    def _create_test_world(self) -> Path:
        world_content = '''<?xml version="1.0" ?>
<sdf version="1.7">
    <world name="default">
        <gravity>0 0 -9.8</gravity>
        <light type="directional" name="sun">
            <cast_shadows>true</cast_shadows>
            <pose>0 0 10 0 0 0</pose>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
            <direction>-0.5 0.1 -0.9</direction>
        </light>
        <include>
            <name>terrain</name>
            <uri>model://ground</uri>
            <pose>0 0 0 0 0 0</pose>
        </include>
        <physics type="ode">
            <real_time_update_rate>1000.0</real_time_update_rate>
            <max_step_size>0.001</max_step_size>
            <real_time_factor>1</real_time_factor>
        </physics>
    </world>
</sdf>'''
        world_path = self.terrain_path / "test.world"
        world_path.write_text(world_content)
        return world_path

    def process_terrain(
        self,
        scale_factor: Optional[float] = None,
        smooth_sigma: Optional[float] = None,
        enhance: Optional[bool] = None,
    ) -> Path:
        logger.info("Starting terrain generation pipeline...")
        logger.info("Creating terrain mesh...")
        self.create_terrain_mesh(scale_factor=scale_factor, smooth_sigma=smooth_sigma, enhance=enhance)
        logger.info("Creating Gazebo model files...")
        self._create_sdf_file()
        self._create_config_file()
        self._create_test_world()
        logger.info(f"Terrain generation complete: {self.terrain_path}")
        return self.terrain_path

    def _find_textures(self) -> List[str]:
        textures = []
        for file in self.texture_path.iterdir():
            if file.suffix.lower() in [".jpg", ".png", ".jpeg", ".tga", ".tiff", ".exr"]:
                textures.append(file.name)
        return textures

    def _create_terrain_material(self, textures: List[str]) -> Path:
        """Create OGRE material file with texture scaling."""
        base_texture = None
        normal_texture = None

        for texture in textures:
            t_lower = texture.lower()
            if any(kw in t_lower for kw in ["base", "albedo", "diffuse", "color", "diff"]):
                base_texture = texture
            elif any(kw in t_lower for kw in ["normal", "nrm", "norm"]):
                normal_texture = texture

        if not base_texture and textures:
            base_texture = textures[0]

        if not base_texture:
            logger.warning("No textures found for material creation")
            return None

        self._material_name = "Terrain/Ground"

        material_content = f"""material {self._material_name}
{{
    technique
    {{
        pass
        {{
            ambient 0.8 0.8 0.8 1.0
            diffuse 0.9 0.9 0.9 1.0
            specular 0.1 0.1 0.1 1.0 10.0

            texture_unit baseMap
            {{
                texture ../texture/{base_texture}
                tex_coord_set 0
                filtering trilinear
                scale 0.01 0.01
            }}"""

        if normal_texture:
            material_content += f"""

            texture_unit normalMap
            {{
                texture ../texture/{normal_texture}
                tex_coord_set 0
                filtering trilinear
            }}"""

        material_content += """
        }
    }
}"""

        material_path = self.material_path / "terrain.material"
        material_path.write_text(material_content)
        logger.info(f"Created material file: {material_path}")
        logger.info(f"Using base texture: {base_texture}")
        return material_path

    def extract_terrain_texture(self, blend_file: Path) -> List[Path]:
        """Extract textures from Blender file and create OGRE material."""
        blend_file = Path(blend_file)
        if not blend_file.exists():
            raise FileNotFoundError(f"Blend file not found: {blend_file}")

        blender_path = self._blender_path or find_blender()
        if not blender_path:
            raise RuntimeError("Blender not found. Please specify path via --blender flag.")

        logger.info(f"Extracting textures from: {blend_file}")
        logger.info(f"Using Blender: {blender_path}")

        blender_script = f'''
import bpy
import os
import shutil

output_dir = "{self.texture_path}"
bpy.ops.wm.open_mainfile(filepath="{blend_file}")

for image in bpy.data.images:
    if image.source == 'FILE' and image.filepath:
        filepath = bpy.path.abspath(image.filepath)
        if os.path.exists(filepath):
            filename = os.path.basename(filepath)
            shutil.copy2(filepath, os.path.join(output_dir, filename))
            print(f"EXPORTED: {{filename}}")
    elif image.packed_file:
        filename = image.name
        if not any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
            filename += '.png'
        dst = os.path.join(output_dir, filename)
        image.save_render(dst)
        print(f"EXPORTED: {{filename}}")
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(blender_script)
            script_path = f.name

        try:
            result = subprocess.run(
                [str(blender_path), "--background", "--python", script_path],
                capture_output=True, text=True, timeout=120
            )
            exported = [line.replace("EXPORTED: ", "").strip()
                       for line in result.stdout.split('\n')
                       if line.startswith("EXPORTED: ")]
            if exported:
                logger.info(f"Extracted {len(exported)} texture(s)")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Blender texture extraction timed out")
        finally:
            os.unlink(script_path)

        textures = self._find_textures()
        if textures:
            self._create_terrain_material(textures)
            self._create_sdf_file(textures)  # Regenerate SDF with PBR materials
            logger.info(f"Created SDF with PBR materials: {', '.join(textures)}")
        else:
            logger.warning("No textures were extracted")

        return [self.texture_path / t for t in textures]