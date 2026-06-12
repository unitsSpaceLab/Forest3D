"""Placement strategies for procedural model placement."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
from stl import mesh


@dataclass
class PlacementContext:
    """Shared context for all placement strategies."""

    terrain_mesh: mesh.Mesh
    world: ET.Element
    placed_models: Dict[str, List[Tuple[float, float, float, float]]]
    cat_config: Dict[str, dict]
    cross_distances: Dict[Tuple[str, str], float]
    model_variants: Dict[str, List[str]]
    progress_callback: Optional[Callable[[int, str], None]] = None
    total_models: int = 0
    models_placed: int = 0
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng()
    )

    def _get_cross_distance(self, cat1: str, cat2: str) -> float:
        key = (cat1, cat2)
        if key in self.cross_distances:
            return self.cross_distances[key]
        key_rev = (cat2, cat1)
        if key_rev in self.cross_distances:
            return self.cross_distances[key_rev]
        return (
            self.cat_config.get(cat1, {}).get("min_distance", 1.0)
            + self.cat_config.get(cat2, {}).get("min_distance", 1.0)
        ) / 2

    def check_distance(
        self, x: float, y: float, category: str, scale: float = 1.0
    ) -> bool:
        for other_category, positions in self.placed_models.items():
            base_distance = self._get_cross_distance(category, other_category)
            for px, py, _, p_scale in positions:
                dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                scale_factor = (max(scale, 0.5) + max(p_scale, 0.5)) / 2
                if dist < base_distance * scale_factor:
                    return False
        return True

    def sample_height(self, x: float, y: float) -> float:
        point = np.array([x, y])
        vectors = self.terrain_mesh.vectors
        distances = np.linalg.norm(vectors[:, :, :2] - point, axis=2)
        closest_tri = vectors[np.argmin(distances.min(axis=1))]
        try:
            v0, v1, v2 = closest_tri
            denom = (
                (v1[1] - v2[1]) * (v0[0] - v2[0])
                + (v2[0] - v1[0]) * (v0[1] - v2[1])
            )
            if abs(denom) < 1e-10:
                return float(np.mean(closest_tri[:, 2]))
            w0 = (
                (v1[1] - v2[1]) * (x - v2[0])
                + (v2[0] - v1[0]) * (y - v2[1])
            ) / denom
            w1 = (
                (v2[1] - v0[1]) * (x - v2[0])
                + (v0[0] - v2[0]) * (y - v2[1])
            ) / denom
            w2 = 1.0 - w0 - w1
            return float(w0 * v0[2] + w1 * v1[2] + w2 * v2[2])
        except Exception:
            return float(np.mean(closest_tri[:, 2]))

    def get_bounds(
        self, margin: float = 0.0
    ) -> Tuple[float, float, float, float]:
        vectors = self.terrain_mesh.vectors.reshape(-1, 3)
        min_x = float(np.min(vectors[:, 0]) + margin)
        max_x = float(np.max(vectors[:, 0]) - margin)
        min_y = float(np.min(vectors[:, 1]) + margin)
        max_y = float(np.max(vectors[:, 1]) - margin)
        return min_x, max_x, min_y, max_y

    def random_variant(self, category: str) -> Optional[str]:
        variants = self.model_variants.get(category, [])
        if not variants:
            return None
        return str(self.rng.choice(variants))

    def rotation(self, category: str) -> Tuple[float, float, float]:
        cfg = self.cat_config.get(category, {})
        rot = cfg.get("rotation", {})
        rr = rot.get("roll_range", (0.0, 0.0))
        pr = rot.get("pitch_range", (0.0, 0.0))
        yr = rot.get("yaw_range", (0.0, 2 * np.pi))
        return (
            float(self.rng.uniform(*rr)),
            float(self.rng.uniform(*pr)),
            float(self.rng.uniform(*yr)),
        )

    def write_include(
        self,
        category: str,
        variant: str,
        idx: int,
        x: float,
        y: float,
        z: float,
        scale: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> None:
        include = ET.SubElement(self.world, "include")
        ET.SubElement(include, "uri").text = f"model://{category}/{variant}"
        ET.SubElement(include, "name").text = f"{category}_{idx}"
        ET.SubElement(include, "pose").text = (
            f"{x:.4f} {y:.4f} {z:.4f} {roll:.4f} {pitch:.4f} {yaw:.4f}"
        )
        ET.SubElement(include, "scale").text = (
            f"{scale:.3f} {scale:.3f} {scale:.3f}"
        )


class PlacementStrategy(ABC):
    """Places models of a single category on terrain."""

    @abstractmethod
    def place(
        self,
        category: str,
        count: int,
        ctx: PlacementContext,
        terrain_meta: Optional[Dict] = None,
        start_index: int = 0,
    ) -> Tuple[int, int]:
        """Place models and write XML includes.

        Returns (number_placed, updated_models_placed_counter).
        """


class EdgePlacement(PlacementStrategy):
    """Edge-biased random placement (sand, rock)."""

    def __init__(
        self,
        edge_weight: float = 0.7,
        jitter: float = 1.0,
        margin: float = 2.0,
        height_offset_range: Tuple[float, float] = (-0.08, 0.08),
    ):
        self.edge_weight = edge_weight
        self.jitter = jitter
        self.margin = margin
        self.height_offset_range = height_offset_range

    def place(
        self,
        category: str,
        count: int,
        ctx: PlacementContext,
        terrain_meta: Optional[Dict] = None,
        start_index: int = 0,
    ) -> Tuple[int, int]:
        if not ctx.model_variants.get(category):
            return 0, ctx.models_placed
        cfg = ctx.cat_config.get(category, {})
        sr = cfg.get("scale_range", (0.5, 1.0))
        min_x, max_x, min_y, max_y = ctx.get_bounds(margin=self.margin)

        placed = 0
        models_done = ctx.models_placed

        for _ in range(count):
            for _ in range(100):
                is_edge = ctx.rng.random() < self.edge_weight
                if is_edge:
                    x, y = self._edge_xy(min_x, max_x, min_y, max_y, ctx.rng)
                else:
                    x = ctx.rng.uniform(min_x, max_x)
                    y = ctx.rng.uniform(min_y, max_y)

                if not (min_x <= x <= max_x and min_y <= y <= max_y):
                    continue
                if not ctx.check_distance(x, y, category, scale=1.0):
                    continue

                z = ctx.sample_height(x, y)
                z += ctx.rng.uniform(*self.height_offset_range)

                variant = ctx.random_variant(category)
                scale = ctx.rng.uniform(*sr)
                roll, pitch, yaw = ctx.rotation(category)

                ctx.write_include(
                    category, variant, start_index + placed,
                    x, y, z, scale, roll, pitch, yaw,
                )
                ctx.placed_models[category].append((x, y, z, scale))
                placed += 1
                models_done += 1

                if ctx.progress_callback and ctx.total_models > 0:
                    pct = int((models_done / ctx.total_models) * 100)
                    ctx.progress_callback(pct, f"Placing {category}...")
                break

        return placed, models_done

    def _edge_xy(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        rng: np.random.Generator,
    ) -> Tuple[float, float]:
        edge = str(rng.choice(["top", "bottom", "left", "right"]))
        ev = rng.uniform(-self.jitter, self.jitter)
        if edge in ("top", "bottom"):
            x = rng.uniform(min_x, max_x)
            y = (max_y if edge == "top" else min_y) + ev
        else:
            x = (max_x if edge == "right" else min_x) + ev
            y = rng.uniform(min_y, max_y)
        return x, y


class ClusteredPlacement(PlacementStrategy):
    """Place models near anchor models (tree, bush, grass)."""

    def __init__(
        self,
        anchor_categories: Tuple[str, ...] = ("tree",),
        cluster_prob: float = 0.7,
        scale_aware: bool = True,
        radius_scale_mult: float = 1.0,
        radius_add: float = 3.0,
        radius_min_fixed: float = 1.0,
        radius_max_fixed: float = 5.0,
        avoid_categories: Tuple[str, ...] = (),
        margin: float = 2.0,
        height_offset_range: Tuple[float, float] = (-0.08, 0.08),
    ):
        self.anchor_categories = anchor_categories
        self.cluster_prob = cluster_prob
        self.scale_aware = scale_aware
        self.radius_scale_mult = radius_scale_mult
        self.radius_add = radius_add
        self.radius_min_fixed = radius_min_fixed
        self.radius_max_fixed = radius_max_fixed
        self.avoid_categories = avoid_categories
        self.margin = margin
        self.height_offset_range = height_offset_range

    def place(
        self,
        category: str,
        count: int,
        ctx: PlacementContext,
        terrain_meta: Optional[Dict] = None,
        start_index: int = 0,
    ) -> Tuple[int, int]:
        if not ctx.model_variants.get(category):
            return 0, ctx.models_placed
        cfg = ctx.cat_config.get(category, {})
        sr = cfg.get("scale_range", (0.5, 1.0))
        min_x, max_x, min_y, max_y = ctx.get_bounds(margin=self.margin)

        placed = 0
        models_done = ctx.models_placed

        for _ in range(count):
            for _ in range(100):
                x, y = self._pick_xy(
                    category, min_x, max_x, min_y, max_y, ctx, sr,
                )
                if x is None or y is None:
                    continue
                if not (min_x <= x <= max_x and min_y <= y <= max_y):
                    continue
                if not ctx.check_distance(x, y, category, scale=1.0):
                    continue
                if not self._avoid_clear(x, y, category, ctx):
                    continue

                z = ctx.sample_height(x, y)
                z += ctx.rng.uniform(*self.height_offset_range)

                variant = ctx.random_variant(category)
                scale = ctx.rng.uniform(*sr)
                roll, pitch, yaw = ctx.rotation(category)

                ctx.write_include(
                    category, variant, start_index + placed,
                    x, y, z, scale, roll, pitch, yaw,
                )
                ctx.placed_models[category].append((x, y, z, scale))
                placed += 1
                models_done += 1

                if ctx.progress_callback and ctx.total_models > 0:
                    pct = int((models_done / ctx.total_models) * 100)
                    ctx.progress_callback(pct, f"Placing {category}...")
                break

        return placed, models_done

    def _pick_xy(
        self,
        category: str,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        ctx: PlacementContext,
        scale_range: Tuple[float, float],
    ) -> Tuple[Optional[float], Optional[float]]:
        anchors = []
        for ac in self.anchor_categories:
            anchors.extend(ctx.placed_models.get(ac, []))

        if anchors and ctx.rng.random() < self.cluster_prob:
            base = anchors[ctx.rng.integers(len(anchors))]
            if self.scale_aware:
                cross = ctx._get_cross_distance(
                    category, self.anchor_categories[0]
                )
                min_radius = cross * max(1.0, base[3])
                max_radius = min_radius * self.radius_scale_mult + self.radius_add
            else:
                min_radius = self.radius_min_fixed
                max_radius = self.radius_max_fixed
            radius = ctx.rng.uniform(min_radius, max_radius)
            angle = ctx.rng.uniform(0, 2 * np.pi)
            return base[0] + radius * np.cos(angle), base[1] + radius * np.sin(angle)

        return ctx.rng.uniform(min_x, max_x), ctx.rng.uniform(min_y, max_y)

    def _avoid_clear(
        self,
        x: float,
        y: float,
        category: str,
        ctx: PlacementContext,
    ) -> bool:
        for ac in self.avoid_categories:
            for sx, sy, _, s_scale in ctx.placed_models.get(ac, []):
                dist = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
                min_d = ctx._get_cross_distance(category, ac) * max(1.0, s_scale)
                if dist < min_d:
                    return False
        return True


class RowPlacement(PlacementStrategy):
    """Regular row placement for agricultural crops."""

    def place(
        self,
        category: str,
        count: int,
        ctx: PlacementContext,
        terrain_meta: Optional[Dict] = None,
        start_index: int = 0,
    ) -> Tuple[int, int]:
        if not ctx.model_variants.get(category):
            return 0, ctx.models_placed
        min_x, max_x, min_y, max_y = ctx.get_bounds(margin=0)

        num_rows = 8
        row_width = 1.0
        furrow_width = 0.5
        headland = 2.0
        plant_spacing = 0.8
        stagger = 0.0

        if terrain_meta:
            num_rows = int(terrain_meta.get("num_rows", num_rows))
            row_width = float(terrain_meta.get("row_width", row_width))
            furrow_width = float(terrain_meta.get("furrow_width", furrow_width))
            headland = float(terrain_meta.get("headland_width", headland))
            plant_spacing = float(terrain_meta.get("plant_spacing", plant_spacing))
            stagger = float(terrain_meta.get("stagger", stagger))

        placed = 0
        models_done = ctx.models_placed
        cycle = row_width + furrow_width
        half_row = row_width / 2.0
        field_x_min = min_x + headland
        field_x_max = max_x - headland

        for row_i in range(num_rows):
            row_centre_y = min_y + headland + half_row + row_i * cycle
            if row_centre_y > max_y - headland:
                break

            row_length = field_x_max - field_x_min
            n_plants = max(1, int(row_length / plant_spacing))

            for pi in range(n_plants):
                if placed >= count:
                    return placed, models_done

                frac = pi / max(n_plants - 1, 1)
                x = field_x_min + frac * row_length
                if row_i % 2 == 1:
                    x += stagger * plant_spacing
                x += ctx.rng.uniform(-0.05, 0.05)
                y = row_centre_y + ctx.rng.uniform(-0.03, 0.03)

                if not (min_x <= x <= max_x and min_y <= y <= max_y):
                    continue
                if not ctx.check_distance(x, y, category, scale=1.0):
                    continue

                z = ctx.sample_height(x, y)
                z += ctx.rng.uniform(-0.02, 0.02)

                variant = ctx.random_variant(category)
                roll, pitch, yaw = ctx.rotation(category)

                ctx.write_include(
                    category, variant, start_index + placed,
                    x, y, z, 1.0, roll, pitch, yaw,
                )
                ctx.placed_models[category].append((x, y, z, 1.0))
                placed += 1
                models_done += 1

                if ctx.progress_callback and ctx.total_models > 0:
                    pct = int((models_done / ctx.total_models) * 100)
                    ctx.progress_callback(pct, f"Placing {category}...")

        return placed, models_done
