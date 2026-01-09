"""Blender to Gazebo asset converter."""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from forest3d.config.schema import BlenderConfig

logger = logging.getLogger("forest3d.converter")


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


class AssetExporter:
    """Export Blender assets to Gazebo models with glTF format."""

    def __init__(
        self,
        blender_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        config: Optional[BlenderConfig] = None,
    ):
        self.config = config or BlenderConfig()

        if blender_path:
            self._blender_path = Path(blender_path)
        elif self.config.path:
            self._blender_path = self.config.path
        else:
            detected = find_blender()
            if detected:
                self._blender_path = detected
                logger.info(f"Auto-detected Blender at: {detected}")
            else:
                raise RuntimeError("Blender not found. Install from https://www.blender.org/download/")

        if not self._blender_path.exists():
            raise FileNotFoundError(f"Blender not found at: {self._blender_path}")

        self.output_path = Path(output_path) if output_path else Path.cwd() / "models"
        self.visual_decimation = self.config.visual_decimation
        self.collision_decimation = self.config.collision_decimation

    def process_asset(
        self,
        blend_file: Path,
        category: str = "tree",
        progress_callback: Optional[callable] = None,
    ) -> Path:
        """Process a single Blender asset to Gazebo model."""
        blend_file = Path(blend_file)
        if not blend_file.exists():
            raise FileNotFoundError(f"Blend file not found: {blend_file}")

        base_name = blend_file.stem
        logger.info(f"Processing: {base_name}")

        asset_dir = self.output_path / category / base_name
        mesh_dir = asset_dir / "mesh"
        for d in [mesh_dir, asset_dir / "textures", asset_dir / "materials"]:
            d.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback(20, "Exporting glTF...")

        glb_path = mesh_dir / f"{base_name}.glb"
        collision_path = mesh_dir / f"{base_name}_collision.glb"
        self._export_glb(blend_file, glb_path, collision_path)

        if progress_callback:
            progress_callback(80, "Creating model files...")

        self._create_sdf_file(base_name, asset_dir)
        self._create_config_file(base_name, asset_dir)
        self._create_test_world(base_name, asset_dir, category)

        if progress_callback:
            progress_callback(100, "Complete")

        logger.info(f"Done: {base_name}")
        return asset_dir

    def _export_glb(self, blend_file: Path, output_path: Path, collision_path: Path) -> None:
        """Export glTF binary (.glb) for visual and collision meshes."""
        blender_script = f'''
import bpy

def prepare_and_export(filepath, decimate_ratio, include_textures=True):
    bpy.ops.object.select_all(action='DESELECT')

    mesh_objects = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.hide_set(False)
            obj.hide_viewport = False
            obj.hide_render = False
            if obj.name not in bpy.context.view_layer.objects:
                try:
                    bpy.context.collection.objects.link(obj)
                except:
                    pass
            mesh_objects.append(obj)

    for obj in mesh_objects:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        decimate = obj.modifiers.new(name="Decimate", type='DECIMATE')
        decimate.ratio = decimate_ratio
        bpy.ops.object.modifier_apply(modifier="Decimate")

    bpy.ops.export_scene.gltf(
        filepath=filepath,
        export_format='GLB',
        use_selection=False,
        export_apply=True,
        export_texcoords=include_textures,
        export_normals=True,
        export_materials='EXPORT' if include_textures else 'NONE',
        export_image_format='AUTO',
        export_yup=False,
    )

bpy.ops.wm.open_mainfile(filepath="{blend_file}")
prepare_and_export("{output_path}", {self.visual_decimation}, include_textures=True)

bpy.ops.wm.open_mainfile(filepath="{blend_file}")
prepare_and_export("{collision_path}", {self.collision_decimation}, include_textures=False)
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(blender_script)
            script_path = f.name

        try:
            result = subprocess.run(
                [str(self._blender_path), "--background", "--python", script_path],
                capture_output=True, text=True, timeout=300,
            )
            if not (output_path.exists() and collision_path.exists()):
                logger.error("glTF export failed")
                logger.debug(f"stdout: {result.stdout}")
                logger.debug(f"stderr: {result.stderr}")
                raise RuntimeError("Blender glTF export failed")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Blender export timed out")
        finally:
            os.unlink(script_path)

    def _create_sdf_file(self, model_name: str, model_dir: Path) -> Path:
        """Create SDF file using glTF meshes."""
        sdf_content = f'''<?xml version="1.0" ?>
<sdf version="1.8">
    <model name="{model_name}">
        <static>true</static>
        <link name="link">
            <collision name="collision">
                <geometry>
                    <mesh><uri>mesh/{model_name}_collision.glb</uri></mesh>
                </geometry>
            </collision>
            <visual name="visual">
                <geometry>
                    <mesh><uri>mesh/{model_name}.glb</uri></mesh>
                </geometry>
            </visual>
        </link>
    </model>
</sdf>'''
        sdf_path = model_dir / "model.sdf"
        sdf_path.write_text(sdf_content)
        return sdf_path

    def _create_config_file(self, model_name: str, model_dir: Path) -> Path:
        """Create model.config file."""
        config_content = f'''<?xml version="1.0"?>
<model>
    <name>{model_name}</name>
    <version>1.0</version>
    <sdf version="1.8">model.sdf</sdf>
    <author>
        <name>AI4Forest</name>
        <email>khalid.bourr@gmail.com</email>
    </author>
    <description>{model_name} model</description>
</model>'''
        config_path = model_dir / "model.config"
        config_path.write_text(config_content)
        return config_path

    def _create_test_world(self, model_name: str, model_dir: Path, category: str) -> Path:
        """Create test world file."""
        from xml.etree import ElementTree as ET
        from forest3d.utils.sdf import create_world_base, add_ground_plane, write_world_file

        sdf_root, world = create_world_base("asset_test")
        add_ground_plane(world)

        include = ET.SubElement(world, "include")
        ET.SubElement(include, "name").text = model_name
        ET.SubElement(include, "pose").text = "0 0 0 0 0 0"
        ET.SubElement(include, "uri").text = f"model://{category}/{model_name}"

        world_path = model_dir / "test.world"
        write_world_file(sdf_root, world_path)
        return world_path

    def process_directory(
        self,
        input_dir: Path,
        category: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> List[Path]:
        """Process all .blend files in a directory."""
        input_dir = Path(input_dir)
        blend_files = list(input_dir.glob("*.blend"))

        if not blend_files:
            logger.warning(f"No .blend files found in {input_dir}")
            return []

        logger.info(f"Found {len(blend_files)} .blend files")

        results = []
        for i, blend_file in enumerate(blend_files):
            try:
                if progress_callback:
                    progress_callback(int((i / len(blend_files)) * 100), f"Processing {blend_file.name}...")
                cat = category or "tree"
                result = self.process_asset(blend_file, cat)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed: {blend_file.name}: {e}")
        return results