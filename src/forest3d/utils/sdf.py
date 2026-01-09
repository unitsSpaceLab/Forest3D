"""Shared SDF utilities for Gazebo Sim world generation."""

from typing import Tuple
from xml.etree import ElementTree as ET


def create_world_base(world_name: str) -> Tuple[ET.Element, ET.Element]:
    """Create SDF 1.8 world with plugins, physics, gravity, and sun."""
    sdf_root = ET.Element("sdf", version="1.8")
    world = ET.SubElement(sdf_root, "world", name=world_name)

    _add_plugins(world)

    physics = ET.SubElement(world, "physics", {"name": "1ms", "type": "ignored"})
    ET.SubElement(physics, "max_step_size").text = "0.001"
    ET.SubElement(physics, "real_time_factor").text = "1.0"

    ET.SubElement(world, "gravity").text = "0 0 -9.8"

    _add_lighting(world)

    return sdf_root, world


def _add_plugins(world: ET.Element) -> None:
    """Add Gazebo Sim system plugins."""
    plugins = [
        ("gz-sim-physics-system", "gz::sim::systems::Physics"),
        ("gz-sim-user-commands-system", "gz::sim::systems::UserCommands"),
        ("gz-sim-scene-broadcaster-system", "gz::sim::systems::SceneBroadcaster"),
    ]
    for filename, name in plugins:
        plugin = ET.SubElement(world, "plugin")
        plugin.set("filename", filename)
        plugin.set("name", name)


def _add_lighting(world: ET.Element) -> None:
    """Add directional sun light."""
    sun = ET.SubElement(world, "light", {"name": "sun", "type": "directional"})
    ET.SubElement(sun, "cast_shadows").text = "true"
    ET.SubElement(sun, "pose").text = "0 0 10 0 0 0"
    ET.SubElement(sun, "diffuse").text = "0.8 0.8 0.8 1"
    ET.SubElement(sun, "specular").text = "0.2 0.2 0.2 1"
    sun_attenuation = ET.SubElement(sun, "attenuation")
    ET.SubElement(sun_attenuation, "range").text = "1000"
    ET.SubElement(sun_attenuation, "constant").text = "0.9"
    ET.SubElement(sun_attenuation, "linear").text = "0.01"
    ET.SubElement(sun_attenuation, "quadratic").text = "0.001"
    ET.SubElement(sun, "direction").text = "-0.5 0.1 -0.9"


def add_ground_plane(world: ET.Element) -> None:
    """Add simple ground plane for testing."""
    ground = ET.SubElement(world, "model", {"name": "ground_plane"})
    ET.SubElement(ground, "static").text = "true"
    link = ET.SubElement(ground, "link", {"name": "link"})

    collision = ET.SubElement(link, "collision", {"name": "collision"})
    collision_geom = ET.SubElement(collision, "geometry")
    plane = ET.SubElement(collision_geom, "plane")
    ET.SubElement(plane, "normal").text = "0 0 1"
    ET.SubElement(plane, "size").text = "100 100"

    visual = ET.SubElement(link, "visual", {"name": "visual"})
    visual_geom = ET.SubElement(visual, "geometry")
    plane = ET.SubElement(visual_geom, "plane")
    ET.SubElement(plane, "normal").text = "0 0 1"
    ET.SubElement(plane, "size").text = "100 100"
    material = ET.SubElement(visual, "material")
    ET.SubElement(material, "ambient").text = "0.8 0.8 0.8 1"
    ET.SubElement(material, "diffuse").text = "0.8 0.8 0.8 1"
    ET.SubElement(material, "specular").text = "0.8 0.8 0.8 1"


def write_world_file(sdf_root: ET.Element, output_path) -> None:
    """Write SDF world to file."""
    tree = ET.ElementTree(sdf_root)
    try:
        ET.indent(tree, space="    ")
    except AttributeError:
        pass
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)