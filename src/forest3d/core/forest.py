"""Forest world generation with procedural model placement."""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
from stl import mesh

from forest3d.config.schema import DensityConfig

logger = logging.getLogger("forest3d.forest")


class WorldPopulator:
    """Procedurally generate forest worlds with intelligent model placement.

    Places models on terrain using zone weighting, distance constraints,
    and natural clustering patterns.
    """

    # Scale ranges for each model category
    SCALE_RANGES = {
        "tree": (0.8, 1.5),
        "rock": (0.5, 2.0),
        "bush": (0.3, 1.0),
        "grass": (0.2, 0.6),
        "sand": (1.0, 2.5),
    }

    # Minimum distances between models of same category
    MIN_DISTANCES = {
        "tree": 3.0,
        "bush": 2.0,
        "rock": 4.0,
        "grass": 0.5,
        "sand": 3.0,
    }

    # Zone weights (edge vs center preference)
    ZONE_WEIGHTS = {
        "tree": {"edge": 0.2, "center": 0.8},
        "rock": {"edge": 0.8, "center": 0.2},
        "bush": {"edge": 0.4, "center": 0.6},
        "grass": {"edge": 0.5, "center": 0.5},
        "sand": {"edge": 0.7, "center": 0.3},
    }

    def __init__(
        self,
        base_path: Path,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        """Initialize the world populator.

        Args:
            base_path: Project base path containing models/ and worlds/.
            progress_callback: Optional callback for progress updates (percent, message).

        Raises:
            FileNotFoundError: If required paths don't exist.
        """
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models"
        self.worlds_path = self.base_path / "worlds"
        self.progress_callback = progress_callback

        self.placed_models: Dict[str, List[Tuple[float, float, float]]] = {
            "tree": [],
            "bush": [],
            "rock": [],
            "grass": [],
            "sand": [],
        }

        self._verify_paths()
        self.model_variants = self._get_model_variants()

    def _verify_paths(self) -> None:
        """Verify all required paths exist."""
        required_paths = [
            self.models_path / "ground",
            self.worlds_path,
        ]

        # Check for at least one model category
        categories = ["tree", "rock", "bush", "grass", "sand"]
        has_category = False
        for cat in categories:
            if (self.models_path / cat).exists():
                has_category = True
                required_paths.append(self.models_path / cat)

        if not has_category:
            logger.warning("No model categories found in models directory")

        missing_paths = []
        for path in required_paths:
            if not path.exists():
                missing_paths.append(str(path))

        if self.models_path / "ground" not in [Path(p) for p in missing_paths]:
            # Ground is required
            pass
        elif missing_paths:
            missing_str = "\n  - ".join(missing_paths)
            raise FileNotFoundError(f"Required paths not found:\n  - {missing_str}")

        # Create worlds directory if it doesn't exist
        self.worlds_path.mkdir(parents=True, exist_ok=True)

    def _get_model_variants(self) -> Dict[str, List[str]]:
        """Get available variants for each model category."""
        variants = {}
        categories = ["tree", "bush", "rock", "grass", "sand"]

        for category in categories:
            category_path = self.models_path / category
            if category_path.exists():
                variants[category] = []
                for d in category_path.iterdir():
                    if d.is_dir() and not d.name.startswith("."):
                        variants[category].append(d.name)
                if variants[category]:
                    logger.info(f"Found {len(variants[category])} variants for {category}")

        return variants

    def _get_terrain_mesh(self) -> mesh.Mesh:
        """Get terrain mesh for height sampling."""
        mesh_path = self.models_path / "ground" / "mesh" / "terrain.stl"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Terrain mesh not found at: {mesh_path}")
        return mesh.Mesh.from_file(str(mesh_path))

    def _get_random_variant(self, category: str) -> Optional[str]:
        """Get random variant for category."""
        variants = self.model_variants.get(category, [])
        if not variants:
            return None
        return np.random.choice(variants)

    def _check_distance_to_placed(self, x: float, y: float, category: str) -> bool:
        """Check if position is far enough from placed models."""
        min_distance = self.MIN_DISTANCES.get(category, 1.0)

        # Check distance to same category
        for px, py, _ in self.placed_models[category]:
            if np.sqrt((x - px) ** 2 + (y - py) ** 2) < min_distance:
                return False

        # Special checks for certain categories
        if category == "tree":
            # Trees should be far from rocks and sand
            for other_category in ["rock", "sand"]:
                for px, py, _ in self.placed_models[other_category]:
                    if np.sqrt((x - px) ** 2 + (y - py) ** 2) < self.MIN_DISTANCES[other_category]:
                        return False

        elif category == "bush":
            # Bushes should maintain some distance from sand
            for px, py, _ in self.placed_models["sand"]:
                if np.sqrt((x - px) ** 2 + (y - py) ** 2) < self.MIN_DISTANCES["sand"]:
                    return False

        return True

    def _get_random_position(
        self, terrain_mesh: mesh.Mesh, category: str, margin: float = 2.0
    ) -> Tuple[float, float, float]:
        """Get random position with intelligent placement."""
        bounds = terrain_mesh.vectors.reshape(-1, 3)
        min_x, max_x = np.min(bounds[:, 0]) + margin, np.max(bounds[:, 0]) - margin
        min_y, max_y = np.min(bounds[:, 1]) + margin, np.max(bounds[:, 1]) - margin

        max_attempts = 50

        for _ in range(max_attempts):
            is_edge = np.random.random() < self.ZONE_WEIGHTS[category]["edge"]

            if category == "sand":
                if is_edge:
                    edge = np.random.choice(["top", "bottom", "left", "right"])
                    if edge in ["top", "bottom"]:
                        x = np.random.uniform(min_x + margin, max_x - margin)
                        y = max_y - margin if edge == "top" else min_y + margin
                        y += np.random.uniform(-1, 1)
                    else:
                        x = max_x - margin if edge == "right" else min_x + margin
                        x += np.random.uniform(-1, 1)
                        y = np.random.uniform(min_y + margin, max_y - margin)
                else:
                    x = np.random.uniform(min_x + margin, max_x - margin)
                    y = np.random.uniform(min_y + margin, max_y - margin)

            elif category == "tree":
                if self.placed_models["tree"] and np.random.random() < 0.7:
                    # Cluster near existing trees
                    base_tree = self.placed_models["tree"][
                        np.random.randint(len(self.placed_models["tree"]))
                    ]
                    radius = np.random.uniform(
                        self.MIN_DISTANCES["tree"], self.MIN_DISTANCES["tree"] * 2
                    )
                    angle = np.random.uniform(0, 2 * np.pi)
                    x = base_tree[0] + radius * np.cos(angle)
                    y = base_tree[1] + radius * np.sin(angle)
                else:
                    # Avoid sand areas
                    valid_position = False
                    for _ in range(10):
                        x = np.random.uniform(min_x + margin, max_x - margin)
                        y = np.random.uniform(min_y + margin, max_y - margin)
                        if all(
                            np.sqrt((x - sx) ** 2 + (y - sy) ** 2) > self.MIN_DISTANCES["sand"] * 2
                            for sx, sy, _ in self.placed_models["sand"]
                        ):
                            valid_position = True
                            break
                    if not valid_position:
                        continue

            elif category == "rock":
                if is_edge:
                    edge = np.random.choice(["top", "bottom", "left", "right"])
                    if edge in ["top", "bottom"]:
                        x = np.random.uniform(min_x + margin, max_x - margin)
                        y = max_y - margin if edge == "top" else min_y + margin
                    else:
                        x = max_x - margin if edge == "right" else min_x + margin
                        y = np.random.uniform(min_y + margin, max_y - margin)
                else:
                    x = np.random.uniform(min_x + margin, max_x - margin)
                    y = np.random.uniform(min_y + margin, max_y - margin)

            elif category == "bush":
                if np.random.random() < 0.6 and self.placed_models["tree"]:
                    # Place near trees
                    base_tree = self.placed_models["tree"][
                        np.random.randint(len(self.placed_models["tree"]))
                    ]
                    radius = np.random.uniform(2.0, 4.0)
                    angle = np.random.uniform(0, 2 * np.pi)
                    x = base_tree[0] + radius * np.cos(angle)
                    y = base_tree[1] + radius * np.sin(angle)
                else:
                    x = np.random.uniform(min_x + margin, max_x - margin)
                    y = np.random.uniform(min_y + margin, max_y - margin)

            else:  # grass
                x = np.random.uniform(min_x + margin, max_x - margin)
                y = np.random.uniform(min_y + margin, max_y - margin)

            # Validate position
            if (
                min_x <= x <= max_x
                and min_y <= y <= max_y
                and self._check_distance_to_placed(x, y, category)
            ):
                # Sample height from terrain
                point = np.array([x, y])
                vectors = terrain_mesh.vectors
                distances = np.linalg.norm(vectors[:, :, :2] - point, axis=2)
                closest_tri = vectors[np.argmin(distances.min(axis=1))]
                z = np.mean(closest_tri[:, 2])

                # Category-specific height adjustments
                if category == "sand":
                    z += np.random.uniform(-0.2, 0)
                elif category == "grass":
                    z += np.random.uniform(-0.05, 0.05)
                elif category == "rock":
                    z += np.random.uniform(-0.1, 0.1)
                else:
                    z += np.random.uniform(-0.08, 0.08)

                self.placed_models[category].append((x, y, z))
                return x, y, z

        # Fallback position
        x = np.random.uniform(min_x + margin, max_x - margin)
        y = np.random.uniform(min_y + margin, max_y - margin)
        return x, y, 0

    def _add_lighting(self, world: ET.Element) -> None:
        """Add optimized lighting to the world."""
        # Add sun
        sun = ET.SubElement(world, "include")
        ET.SubElement(sun, "uri").text = "model://sun"

        # Add ambient directional light
        ambient = ET.SubElement(world, "light", {"name": "ambient", "type": "directional"})
        ET.SubElement(ambient, "cast_shadows").text = "false"
        ET.SubElement(ambient, "pose").text = "0 0 10 0 0 0"
        ET.SubElement(ambient, "diffuse").text = "0.8 0.8 0.8 1"
        ET.SubElement(ambient, "specular").text = "0.1 0.1 0.1 1"
        ET.SubElement(ambient, "direction").text = "0.1 0.1 -0.9"

        # Add point light
        point = ET.SubElement(world, "light", {"name": "point_light", "type": "point"})
        ET.SubElement(point, "cast_shadows").text = "false"
        ET.SubElement(point, "pose").text = "0 0 10 0 0 0"
        ET.SubElement(point, "diffuse").text = "0.3 0.3 0.3 1"
        ET.SubElement(point, "specular").text = "0.05 0.05 0.05 1"
        ET.SubElement(point, "attenuation")
        ET.SubElement(point, "range").text = "30"

    def create_forest_world(
        self,
        density_config: Optional[Dict[str, int]] = None,
    ) -> Path:
        """Create forest world with placed models.

        Args:
            density_config: Dict of category -> count. Uses defaults if None.

        Returns:
            Path to created world file.
        """
        # Reset placed models
        for category in self.placed_models:
            self.placed_models[category] = []

        # Use provided config or defaults
        if density_config is None:
            config = DensityConfig()
            density_config = {
                "tree": config.tree,
                "bush": config.bush,
                "rock": config.rock,
                "grass": config.grass,
                "sand": config.sand,
            }

        # Create world XML
        world_elem = ET.Element("sdf", version="1.7")
        world = ET.SubElement(world_elem, "world", name="forest_world")

        self._add_lighting(world)

        # Add terrain
        terrain = ET.SubElement(world, "include")
        ET.SubElement(terrain, "uri").text = "model://ground"
        ET.SubElement(terrain, "name").text = "terrain"
        ET.SubElement(terrain, "pose").text = "0 0 0 0 0 0"

        # Add physics
        physics = ET.SubElement(world, "physics")
        physics.set("type", "ode")
        ET.SubElement(physics, "real_time_update_rate").text = "1000.0"
        ET.SubElement(physics, "max_step_size").text = "0.001"
        ET.SubElement(physics, "real_time_factor").text = "1"
        ET.SubElement(physics, "gravity").text = "0 0 -9.8"

        terrain_mesh = self._get_terrain_mesh()

        # Process categories in specific order
        category_order = ["sand", "rock", "tree", "bush", "grass"]
        total_models = sum(density_config.get(c, 0) for c in category_order)
        models_placed = 0

        for category in category_order:
            if category not in density_config or category not in self.model_variants:
                continue

            count = density_config[category]
            if count == 0:
                continue

            logger.info(f"Adding {count} {category} models...")

            for i in range(count):
                try:
                    variant = self._get_random_variant(category)
                    if not variant:
                        continue

                    x, y, z = self._get_random_position(terrain_mesh, category)

                    # Scale and rotation
                    scale = np.random.uniform(*self.SCALE_RANGES[category])

                    # Category-specific rotations
                    if category == "sand":
                        roll = pitch = 0
                        yaw = np.random.uniform(0, 2 * np.pi)
                    elif category == "tree":
                        roll = pitch = np.random.uniform(-0.05, 0.05)
                        yaw = np.random.uniform(0, 2 * np.pi)
                    elif category == "rock":
                        roll = pitch = np.random.uniform(-0.15, 0.15)
                        yaw = np.random.uniform(0, 2 * np.pi)
                    else:
                        roll = pitch = 0
                        yaw = np.random.uniform(0, 2 * np.pi)

                    # Add model to world
                    include = ET.SubElement(world, "include")
                    ET.SubElement(include, "uri").text = f"model://{category}/{variant}"
                    ET.SubElement(include, "name").text = f"{category}_{i}"
                    ET.SubElement(include, "pose").text = f"{x} {y} {z} {roll} {pitch} {yaw}"
                    ET.SubElement(include, "scale").text = f"{scale} {scale} {scale}"

                    models_placed += 1

                    if self.progress_callback and total_models > 0:
                        progress = int((models_placed / total_models) * 100)
                        self.progress_callback(progress, f"Placing {category}...")

                except Exception as e:
                    logger.warning(f"Failed to add {category} model: {e}")
                    continue

        # Save the world file
        output_path = self.worlds_path / "forest_world.world"
        tree = ET.ElementTree(world_elem)

        # Pretty print XML
        try:
            ET.indent(tree, space="  ")
        except AttributeError:
            pass  # Python < 3.9

        tree.write(str(output_path), encoding="utf-8", xml_declaration=True)

        logger.info(f"World file created at: {output_path}")
        logger.info("Models placed:")
        for category in category_order:
            if category in self.placed_models:
                logger.info(f"  - {category}: {len(self.placed_models[category])}")

        return output_path

    def get_model_statistics(self) -> Dict:
        """Get statistics about placed models."""
        return {
            "total_models": sum(len(models) for models in self.placed_models.values()),
            "by_category": {
                category: len(models) for category, models in self.placed_models.items()
            },
            "variants_available": {
                category: len(variants) for category, variants in self.model_variants.items()
            },
        }
