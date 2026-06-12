"""Forest world generation with procedural model placement."""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
from stl import mesh

from forest3d.config.schema import DensityConfig
from forest3d.core.placement import PlacementContext
from forest3d.core.terrain_base import get_terrain_class

logger = logging.getLogger("forest3d.forest")


class WorldPopulator:
    """Procedural world generation with intelligent model placement."""

    def __init__(
        self,
        base_path: Path,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        terrain_type: Optional[str] = None,
        category_config: Optional[Dict[str, dict]] = None,
        cross_distances: Optional[Dict[Tuple[str, str], float]] = None,
        seed: Optional[int] = None,
    ):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models"
        self.worlds_path = self.base_path / "worlds"
        self.progress_callback = progress_callback
        self._seed = seed

        if terrain_type:
            cls = get_terrain_class(terrain_type)
            self.cat_config = deepcopy(cls.default_categories())
            self._strategies = dict(cls.default_strategies())
            self._category_order = list(cls.default_category_order())
            base_cross = dict(cls.default_cross_distances())
        else:
            from forest3d.core.terrain import DemTerrain
            self.cat_config = deepcopy(DemTerrain.default_categories())
            self._strategies = dict(DemTerrain.default_strategies())
            self._category_order = list(DemTerrain.default_category_order())
            base_cross = dict(DemTerrain.default_cross_distances())

        if category_config:
            for cat, cfg in category_config.items():
                if cat in self.cat_config:
                    self.cat_config[cat].update(cfg)
                else:
                    self.cat_config[cat] = cfg

        if cross_distances:
            base_cross.update(cross_distances)
        self._cross_distances = base_cross

        self.placed_models: Dict[str, List[Tuple[float, float, float, float]]] = {
            cat: [] for cat in self.cat_config
        }

        self._verify_paths()
        self.model_variants = self._get_model_variants()

    def _verify_paths(self) -> None:
        """Verify all required paths exist."""
        required_paths = [
            self.models_path / "ground",
            self.worlds_path,
        ]

        has_category = False
        for cat in self.cat_config:
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
            pass
        elif missing_paths:
            missing_str = "\n  - ".join(missing_paths)
            raise FileNotFoundError(f"Required paths not found:\n  - {missing_str}")

        self.worlds_path.mkdir(parents=True, exist_ok=True)

    def _get_model_variants(self) -> Dict[str, List[str]]:
        """Get available variants for each model category."""
        variants: Dict[str, List[str]] = {}
        for category in self.cat_config:
            category_path = self.models_path / category
            if category_path.exists():
                variants[category] = []
                for d in category_path.iterdir():
                    if d.is_dir() and not d.name.startswith("."):
                        variants[category].append(d.name)
                if variants[category]:
                    logger.info(
                        f"Found {len(variants[category])} variants for {category}"
                    )
        return variants

    def _get_terrain_mesh(self) -> mesh.Mesh:
        """Get terrain mesh for height sampling."""
        mesh_path = self.models_path / "ground" / "mesh" / "terrain.stl"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Terrain mesh not found at: {mesh_path}")
        return mesh.Mesh.from_file(str(mesh_path))

    def _add_scene_settings(self, world: ET.Element) -> None:
        scene = ET.SubElement(world, "scene")
        ET.SubElement(scene, "ambient").text = "0.4 0.4 0.4 1"
        ET.SubElement(scene, "background").text = "0.7 0.8 0.9 1"

    def _add_extra_lighting(self, world: ET.Element) -> None:
        ambient = ET.SubElement(
            world, "light", {"name": "ambient", "type": "directional"}
        )
        ET.SubElement(ambient, "cast_shadows").text = "false"
        ET.SubElement(ambient, "pose").text = "0 0 10 0 0 0"
        ET.SubElement(ambient, "diffuse").text = "0.6 0.6 0.6 1"
        ET.SubElement(ambient, "specular").text = "0.1 0.1 0.1 1"
        ET.SubElement(ambient, "direction").text = "0.1 0.1 -0.9"

        point = ET.SubElement(
            world, "light", {"name": "point_light", "type": "point"}
        )
        ET.SubElement(point, "cast_shadows").text = "false"
        ET.SubElement(point, "pose").text = "0 0 15 0 0 0"
        ET.SubElement(point, "diffuse").text = "0.3 0.3 0.3 1"
        ET.SubElement(point, "specular").text = "0.05 0.05 0.05 1"
        pa = ET.SubElement(point, "attenuation")
        ET.SubElement(pa, "range").text = "50"
        ET.SubElement(pa, "constant").text = "0.5"
        ET.SubElement(pa, "linear").text = "0.01"
        ET.SubElement(pa, "quadratic").text = "0.001"

    def create_forest_world(
        self,
        density_config: Optional[Dict[str, int]] = None,
        terrain_meta: Optional[Dict] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Create a world file with placed models."""
        from forest3d.utils.sdf import create_world_base, write_world_file

        for cat in self.placed_models:
            self.placed_models[cat] = []

        if density_config is None:
            cfg = DensityConfig()
            density_config = cfg.as_dict()

        world_elem, world = create_world_base("forest_world")
        self._add_scene_settings(world)
        self._add_extra_lighting(world)

        terrain = ET.SubElement(world, "include")
        ET.SubElement(terrain, "uri").text = "model://ground"
        ET.SubElement(terrain, "name").text = "terrain"
        ET.SubElement(terrain, "pose").text = "0 0 0 0 0 0"

        terrain_mesh = self._get_terrain_mesh()

        category_order = self._category_order
        total_models = sum(density_config.get(c, 0) for c in category_order)
        models_placed = 0
        models_failed = 0

        ctx = PlacementContext(
            terrain_mesh=terrain_mesh,
            world=world,
            placed_models=self.placed_models,
            cat_config=self.cat_config,
            cross_distances=self._cross_distances,
            model_variants=self.model_variants,
            progress_callback=self.progress_callback,
            total_models=total_models,
            models_placed=0,
            rng=np.random.default_rng(self._seed),
        )

        for category in category_order:
            count = density_config.get(category, 0)
            if count == 0:
                continue

            strategy = self._strategies.get(category)
            if strategy is None:
                logger.warning(f"No strategy for {category}, skipping")
                continue
            logger.info(
                f"Adding {count} {category} models "
                f"(strategy={type(strategy).__name__})..."
            )
            ctx.models_placed = models_placed

            try:
                n, models_placed = strategy.place(
                    category=category,
                    count=count,
                    ctx=ctx,
                    terrain_meta=terrain_meta,
                    start_index=0,
                )
                placed_here = n
                failed_here = max(0, count - n)
            except Exception as e:
                logger.warning(
                    f"Failed to place {category} models: {e}"
                )
                placed_here = 0
                failed_here = count
                models_failed += count

            logger.info(
                f"  {category}: placed {placed_here}/{count} "
                f"(failed: {failed_here})"
            )

        if output_path is None:
            output_path = self.worlds_path / "forest_world.world"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        write_world_file(world_elem, output_path)

        logger.info(f"World file created at: {output_path}")
        logger.info(
            f"Total models placed: {models_placed}/{total_models} "
            f"(failed: {models_failed})"
        )
        logger.info("Models placed by category:")
        for cat in category_order:
            if cat in self.placed_models:
                logger.info(f"  - {cat}: {len(self.placed_models[cat])}")

        return output_path

    def get_model_statistics(self) -> Dict:
        stats = {
            "total_models": sum(len(v) for v in self.placed_models.values()),
            "by_category": {
                cat: len(m) for cat, m in self.placed_models.items()
            },
            "variants_available": {
                cat: len(v) for cat, v in self.model_variants.items()
            },
        }
        stats["scale_stats"] = {}
        for cat, models in self.placed_models.items():
            if models:
                scales = [m[3] for m in models]
                stats["scale_stats"][cat] = {
                    "min": float(np.min(scales)),
                    "max": float(np.max(scales)),
                    "mean": float(np.mean(scales)),
                }
        return stats

    def get_placement_density_map(
        self, resolution: int = 50
    ) -> Dict[str, np.ndarray]:
        density_maps = {}
        all_positions = []
        for models in self.placed_models.values():
            all_positions.extend([(m[0], m[1]) for m in models])
        if not all_positions:
            return density_maps

        all_positions = np.array(all_positions)
        min_x, max_x = (
            np.min(all_positions[:, 0]),
            np.max(all_positions[:, 0]),
        )
        min_y, max_y = (
            np.min(all_positions[:, 1]),
            np.max(all_positions[:, 1]),
        )

        for cat, models in self.placed_models.items():
            if not models:
                density_maps[cat] = np.zeros((resolution, resolution))
                continue
            density = np.zeros((resolution, resolution))
            for x, y, z, sc in models:
                gx = int(
                    (x - min_x) / (max_x - min_x + 1e-6) * (resolution - 1)
                )
                gy = int(
                    (y - min_y) / (max_y - min_y + 1e-6) * (resolution - 1)
                )
                gx = np.clip(gx, 0, resolution - 1)
                gy = np.clip(gy, 0, resolution - 1)
                density[gy, gx] += 1
            density_maps[cat] = density
        return density_maps
