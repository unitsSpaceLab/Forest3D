"""Terrain generation from DEM data."""

import logging
from pathlib import Path
from typing import Optional, Tuple

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


class TerrainGenerator:
    """Generate terrain meshes from DEM (Digital Elevation Model) data.

    This class processes GeoTIFF DEM files and creates Gazebo-compatible
    terrain models including STL meshes, SDF files, and configuration.
    """

    def __init__(
        self,
        tif_path: Path,
        output_path: Optional[Path] = None,
        config: Optional[TerrainConfig] = None,
    ):
        """Initialize the terrain generator.

        Args:
            tif_path: Path to the DEM file (GeoTIFF).
            output_path: Output directory for generated files.
            config: Terrain configuration options.

        Raises:
            ImportError: If GDAL is not available.
            FileNotFoundError: If the DEM file doesn't exist.
        """
        if not GDAL_AVAILABLE:
            raise ImportError(
                "GDAL is required for terrain generation. "
                "Install with: pip install GDAL or use Docker image."
            )

        self.tif_path = Path(tif_path)
        if not self.tif_path.exists():
            raise FileNotFoundError(f"DEM file not found: {self.tif_path}")

        self.config = config or TerrainConfig()

        # Determine output paths
        if output_path:
            self.terrain_path = Path(output_path)
        else:
            self.terrain_path = self.tif_path.parent.parent  # Assume models/ground structure

        self.mesh_path = self.terrain_path / "mesh"
        self.material_path = self.terrain_path / "material"
        self.texture_path = self.terrain_path / "texture"

        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create required output directories."""
        for path in [self.mesh_path, self.material_path, self.texture_path]:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created/verified directory: {path}")

    def enhance_dem(self, scale_factor: float = 6.0) -> Path:
        """Enhance DEM resolution using cubic spline interpolation.

        Args:
            scale_factor: Resolution multiplier.

        Returns:
            Path to enhanced DEM file.

        Raises:
            RuntimeError: If enhancement fails.
        """
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
        """Generate terrain mesh from DEM.

        Args:
            scale_factor: Scale factor for terrain (overrides config).
            smooth_sigma: Gaussian smoothing sigma (overrides config).
            enhance: Enable DEM enhancement (overrides config).

        Returns:
            Tuple of (path to STL file, statistics dict).

        Raises:
            RuntimeError: If mesh generation fails.
        """
        scale_factor = scale_factor if scale_factor is not None else self.config.scale_factor
        smooth_sigma = smooth_sigma if smooth_sigma is not None else self.config.smooth_sigma
        enhance = enhance if enhance is not None else self.config.enhance

        try:
            # Determine which DEM file to use
            if enhance:
                dem_file = self.enhance_dem(self.config.enhance_scale)
                logger.info("Using enhanced DEM...")
            else:
                dem_file = self.tif_path
                logger.info("Using original DEM...")

            # Open DEM file
            ds = gdal.Open(str(dem_file))
            if ds is None:
                raise RuntimeError(f"Failed to open {dem_file}")

            # Get geotransform information
            geotransform = ds.GetGeoTransform()
            pixel_width = abs(geotransform[1])
            pixel_height = abs(geotransform[5])

            # Read elevation data
            elevation = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

            # Apply smoothing if requested
            if smooth_sigma > 0:
                elevation = gaussian_filter(elevation, sigma=smooth_sigma)
                logger.debug(f"Applied Gaussian smoothing with sigma={smooth_sigma}")

            # Get dimensions
            rows, cols = elevation.shape

            # Create vertices
            logger.info(f"Creating mesh from {rows}x{cols} DEM...")
            vertices = []
            faces = []

            for y in range(rows):
                for x in range(cols):
                    world_x = x * pixel_width * scale_factor
                    world_y = y * pixel_height * scale_factor
                    world_z = elevation[y, x] * scale_factor
                    vertices.append([world_x, world_y, world_z])

            # Generate faces (triangles)
            for y in range(rows - 1):
                for x in range(cols - 1):
                    v0 = y * cols + x
                    v1 = v0 + 1
                    v2 = (y + 1) * cols + x
                    v3 = v2 + 1
                    faces.extend([[v0, v1, v2], [v1, v3, v2]])

            # Create the mesh
            vertices = np.array(vertices)
            faces = np.array(faces)

            # Center the mesh
            center = np.mean(vertices, axis=0)
            vertices -= center

            # Create the STL mesh
            terrain = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                for j in range(3):
                    terrain.vectors[i][j] = vertices[f[j]]

            # Save the mesh
            output_path = self.mesh_path / "terrain.stl"
            terrain.save(str(output_path))
            logger.info(f"Created terrain mesh at: {output_path}")

            # Calculate statistics
            stats = {
                "x_extent": float(np.ptp(vertices[:, 0])),
                "y_extent": float(np.ptp(vertices[:, 1])),
                "z_extent": float(np.ptp(vertices[:, 2])),
                "num_vertices": len(vertices),
                "num_faces": len(faces),
            }

            logger.info(
                f"Terrain dimensions: X={stats['x_extent']:.2f}, "
                f"Y={stats['y_extent']:.2f}, Z={stats['z_extent']:.2f}"
            )

            return output_path, stats

        except Exception as e:
            raise RuntimeError(f"Failed to create terrain mesh: {e}") from e

    def _create_sdf_file(self) -> Path:
        """Create SDF file for the terrain model."""
        sdf_content = '''<?xml version="1.0" ?>
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
                <material>
                    <script>
                        <uri>model://ground/material/terrain.material</uri>
                        <name>Terrain/Moss</name>
                    </script>
                </material>
            </visual>
        </link>
    </model>
</sdf>'''

        sdf_path = self.terrain_path / "model.sdf"
        sdf_path.write_text(sdf_content)
        logger.debug(f"Created SDF file: {sdf_path}")
        return sdf_path

    def _create_config_file(self) -> Path:
        """Create model.config file."""
        config_content = '''<?xml version="1.0"?>
<model>
    <name>ground</name>
    <version>1.0</version>
    <sdf version="1.7">model.sdf</sdf>

    <author>
        <name>AI4Forest</name>
        <email>khalid.bourr@gmail.com</email>
    </author>

    <description>
        Terrain model generated from DEM data for Gazebo simulation
    </description>
</model>'''

        config_path = self.terrain_path / "model.config"
        config_path.write_text(config_content)
        logger.debug(f"Created config file: {config_path}")
        return config_path

    def _create_test_world(self) -> Path:
        """Create test world file."""
        world_content = '''<?xml version="1.0" ?>
<sdf version="1.7">
    <world name="default">
        <include>
            <uri>model://sun</uri>
        </include>

        <include>
            <name>terrain</name>
            <uri>model://ground</uri>
            <pose>0 0 0 0 0 0</pose>
        </include>

        <physics type="ode">
            <real_time_update_rate>1000.0</real_time_update_rate>
            <max_step_size>0.001</max_step_size>
            <real_time_factor>1</real_time_factor>
            <gravity>0 0 -9.8</gravity>
        </physics>
    </world>
</sdf>'''

        world_path = self.terrain_path / "test.world"
        world_path.write_text(world_content)
        logger.debug(f"Created test world file: {world_path}")
        return world_path

    def process_terrain(
        self,
        scale_factor: Optional[float] = None,
        smooth_sigma: Optional[float] = None,
        enhance: Optional[bool] = None,
    ) -> Path:
        """Process complete terrain generation pipeline.

        Args:
            scale_factor: Scale factor for terrain.
            smooth_sigma: Gaussian smoothing sigma.
            enhance: Enable DEM enhancement.

        Returns:
            Path to the terrain model directory.
        """
        logger.info("Starting terrain generation pipeline...")

        logger.info("Creating terrain mesh...")
        stl_path, stats = self.create_terrain_mesh(
            scale_factor=scale_factor,
            smooth_sigma=smooth_sigma,
            enhance=enhance,
        )

        logger.info("Creating Gazebo model files...")
        self._create_sdf_file()
        self._create_config_file()
        self._create_test_world()

        logger.info(f"Terrain generation complete: {self.terrain_path}")
        return self.terrain_path
