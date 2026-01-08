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
    """Auto-detect Blender installation.

    Returns:
        Path to Blender executable if found, None otherwise.
    """
    # Check PATH first
    blender_in_path = shutil.which("blender")
    if blender_in_path:
        return Path(blender_in_path)

    # Check common installation locations
    common_paths = [
        Path("/usr/bin/blender"),
        Path("/usr/local/bin/blender"),
        Path("/snap/bin/blender"),
        Path("/opt/blender/blender"),
        Path.home() / "blender" / "blender",
    ]

    # Check for version-specific installations in common directories
    search_dirs = [
        Path.home() / "Downloads",
        Path("/opt"),
        Path.home(),
    ]

    for base in search_dirs:
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
    """Export Blender assets to Gazebo-compatible format.

    Converts .blend files into Gazebo models with separate visual
    and collision meshes, materials, and configuration files.
    """

    def __init__(
        self,
        blender_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        config: Optional[BlenderConfig] = None,
    ):
        """Initialize the asset exporter.

        Args:
            blender_path: Path to Blender executable.
            output_path: Base output directory for models.
            config: Blender configuration options.

        Raises:
            RuntimeError: If Blender is not found.
        """
        self.config = config or BlenderConfig()

        # Resolve Blender path
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
                raise RuntimeError(
                    "Blender not found. Please specify path via --blender flag, "
                    "config file, or FOREST3D_BLENDER_PATH environment variable. "
                    "Install Blender from https://www.blender.org/download/"
                )

        # Validate Blender exists
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
        """Process a single Blender asset.

        Args:
            blend_file: Path to .blend file.
            category: Model category (tree, bush, rock, grass, sand).
            progress_callback: Optional callback for progress updates.

        Returns:
            Path to the created model directory.

        Raises:
            FileNotFoundError: If blend file doesn't exist.
            RuntimeError: If export fails.
        """
        blend_file = Path(blend_file)
        if not blend_file.exists():
            raise FileNotFoundError(f"Blend file not found: {blend_file}")

        base_name = blend_file.stem
        logger.info(f"Processing asset: {base_name}")

        # Create directory structure
        category_dir = self.output_path / category
        asset_dir = category_dir / base_name
        mesh_dir = asset_dir / "mesh"
        textures_dir = asset_dir / "textures"
        materials_dir = asset_dir / "materials"

        for d in [mesh_dir, textures_dir, materials_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Export glTF files (better PBR material support than COLLADA)
        if progress_callback:
            progress_callback(20, "Exporting glTF files...")
        logger.info("Exporting glTF files...")

        glb_path = mesh_dir / f"{base_name}.glb"
        collision_path = mesh_dir / f"{base_name}_collision.glb"
        self._export_glb(blend_file, glb_path, collision_path)

        # glTF binary (.glb) embeds textures automatically
        if progress_callback:
            progress_callback(40, "Textures embedded in glTF...")
        logger.info("Textures embedded in glTF binary format")

        # Generate SDF and config files
        if progress_callback:
            progress_callback(80, "Creating model files...")
        logger.info("Creating SDF file...")
        self._create_sdf_file(base_name, asset_dir, category)

        logger.info("Creating model.config file...")
        self._create_config_file(base_name, asset_dir)

        logger.info("Creating test world file...")
        self._create_test_world(base_name, asset_dir, category)

        if progress_callback:
            progress_callback(100, "Complete")
        logger.info(f"Successfully processed: {base_name}")

        return asset_dir

    def _export_glb(self, blend_file: Path, output_path: Path, collision_path: Path) -> None:
        """Export optimized glTF binary (.glb) for visual and collision meshes.

        glTF format properly handles PBR materials and embeds textures automatically.
        This is the recommended format for Gazebo Sim.
        """
        blender_script = f'''
import bpy

def prepare_and_export(filepath, decimate_ratio, include_textures=True):
    """Prepare mesh objects and export to glTF binary."""
    # Deselect all first
    bpy.ops.object.select_all(action='DESELECT')

    mesh_objects = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            # Make sure object is visible and in view layer
            obj.hide_set(False)
            obj.hide_viewport = False
            obj.hide_render = False

            # Link to scene if not already linked
            if obj.name not in bpy.context.view_layer.objects:
                try:
                    bpy.context.collection.objects.link(obj)
                except:
                    pass

            mesh_objects.append(obj)

    # Select and process mesh objects
    for obj in mesh_objects:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Apply decimate modifier
        decimate = obj.modifiers.new(name="Decimate", type='DECIMATE')
        decimate.ratio = decimate_ratio
        bpy.ops.object.modifier_apply(modifier="Decimate")

    # Export to glTF binary format
    # glTF handles PBR materials natively and embeds textures in .glb
    bpy.ops.export_scene.gltf(
        filepath=filepath,
        export_format='GLB',
        use_selection=False,
        export_apply=True,
        export_texcoords=include_textures,  # Skip for collision
        export_normals=True,
        export_materials='EXPORT' if include_textures else 'NONE',  # Skip for collision
        export_image_format='AUTO',
        export_yup=False,
    )

# Load file and export visual mesh with textures
bpy.ops.wm.open_mainfile(filepath="{blend_file}")
prepare_and_export("{output_path}", {self.visual_decimation}, include_textures=True)

# Reload and export collision mesh (no textures needed)
bpy.ops.wm.open_mainfile(filepath="{blend_file}")
prepare_and_export("{collision_path}", {self.collision_decimation}, include_textures=False)
'''

        # Use a temporary file for the script
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(blender_script)
            script_path = f.name

        try:
            result = subprocess.run(
                [str(self._blender_path), "--background", "--python", script_path],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if output_path.exists() and collision_path.exists():
                logger.debug(f"Successfully exported: {output_path} and {collision_path}")
            else:
                logger.error(f"Failed to export glTF files")
                logger.debug(f"Blender stdout: {result.stdout}")
                logger.debug(f"Blender stderr: {result.stderr}")
                raise RuntimeError("Blender glTF export failed - output files not created")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Blender export timed out after 5 minutes")
        finally:
            os.unlink(script_path)

    def _find_textures(self, textures_dir: Path) -> List[str]:
        """Find all texture files in directory."""
        textures = []
        try:
            for file in textures_dir.iterdir():
                if file.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    textures.append(file.name)
                    logger.debug(f"Found texture: {file.name}")
        except Exception as e:
            logger.warning(f"Error scanning textures: {e}")
        return textures

    def _create_sdf_file(self, model_name: str, model_dir: Path, category: str) -> Path:
        """Create SDF file for the model.

        Uses glTF binary format (.glb) which properly handles PBR materials
        and embeds textures. This is the recommended format for Gazebo Sim.
        """
        sdf_content = f'''<?xml version="1.0" ?>
<sdf version="1.8">
    <model name="{model_name}">
        <static>true</static>
        <link name="link">
            <collision name="collision">
                <geometry>
                    <mesh>
                        <uri>mesh/{model_name}_collision.glb</uri>
                    </mesh>
                </geometry>
            </collision>
            <visual name="visual">
                <geometry>
                    <mesh>
                        <uri>mesh/{model_name}.glb</uri>
                    </mesh>
                </geometry>
            </visual>
        </link>
    </model>
</sdf>'''

        sdf_path = model_dir / "model.sdf"
        sdf_path.write_text(sdf_content)
        logger.debug(f"Created SDF file: {sdf_path}")
        return sdf_path

    def _create_material_file(
        self, model_name: str, textures: List[str], materials_dir: Path
    ) -> Path:
        """Create a material file with base color texture."""
        # Find the most appropriate base texture
        base_texture = None
        for texture in textures:
            if any(kw in texture.lower() for kw in ["base", "albedo", "diffuse", "color"]):
                base_texture = texture
                break

        # If no specific base texture found, use the first one
        if not base_texture and textures:
            base_texture = textures[0]

        if not base_texture:
            logger.warning("No textures found for material")
            return None

        material_content = f"""material {model_name}
{{
    technique
    {{
        pass
        {{
            ambient 0.8 0.8 0.8 1.0
            diffuse 0.7 0.7 0.7 1.0
            specular 0.3 0.3 0.3 1.0 20.0

            texture_unit
            {{
                texture ../textures/{base_texture}
                tex_coord_set 0
                filtering trilinear
                scale 1.0 1.0
            }}
        }}
    }}
}}"""

        material_path = materials_dir / f"{model_name}.material"
        material_path.write_text(material_content)
        logger.debug(f"Created material file using texture: {base_texture}")
        return material_path

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

    <description>
        {model_name} model for Gazebo simulation
    </description>
</model>'''

        config_path = model_dir / "model.config"
        config_path.write_text(config_content)
        logger.debug(f"Created config file: {config_path}")
        return config_path

    def _create_test_world(self, model_name: str, model_dir: Path, category: str) -> Path:
        """Create test world file for Gazebo Sim.

        Uses inline definitions for sun and ground plane instead of model includes
        for compatibility with Gazebo Sim (gz sim).
        """
        from xml.etree import ElementTree as ET
        from forest3d.utils.sdf import create_world_base, add_ground_plane, write_world_file

        sdf_root, world = create_world_base("asset_test")

        # Add ground plane for testing individual assets
        add_ground_plane(world)

        # Add the model being tested
        include = ET.SubElement(world, "include")
        ET.SubElement(include, "name").text = model_name
        ET.SubElement(include, "pose").text = "0 0 0 0 0 0"
        ET.SubElement(include, "uri").text = f"model://{category}/{model_name}"

        world_path = model_dir / "test.world"
        write_world_file(sdf_root, world_path)
        logger.debug(f"Created test world file: {world_path}")
        return world_path

    def process_directory(
        self,
        input_dir: Path,
        category: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> List[Path]:
        """Process all .blend files in a directory.

        Args:
            input_dir: Directory containing .blend files.
            category: Model category (auto-detected from filename if None).
            progress_callback: Optional callback for progress updates.

        Returns:
            List of paths to created model directories.
        """
        input_dir = Path(input_dir)
        blend_files = list(input_dir.glob("*.blend"))

        if not blend_files:
            logger.warning(f"No .blend files found in {input_dir}")
            return []

        logger.info(f"Found {len(blend_files)} .blend files to process")

        results = []
        for i, blend_file in enumerate(blend_files):
            try:
                if progress_callback:
                    progress_callback(
                        int((i / len(blend_files)) * 100),
                        f"Processing {blend_file.name}...",
                    )

                # Use provided category or default to "tree"
                cat = category or "tree"
                result = self.process_asset(blend_file, cat)
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process {blend_file.name}: {e}")
                continue

        return results
