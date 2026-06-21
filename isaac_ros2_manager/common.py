from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from rosidl_runtime_py.convert import message_to_ordereddict
from sensor_msgs.msg import NavSatFix


# ---------------------------------------------------------------------------
# Namespace / name helpers
# ---------------------------------------------------------------------------

def clean_ns(value: str | None) -> str:
    text = str(value or "").strip()
    text = text.strip("/")
    return "/".join(part for part in text.split("/") if part)


def abs_name(value: str | None) -> str:
    ns = clean_ns(value)
    return "/" + ns if ns else ""


def join_name(*parts: str | None) -> str:
    return abs_name("/".join(clean_ns(part) for part in parts if clean_ns(part)))


# ---------------------------------------------------------------------------
# Parameter coercion helpers
# ---------------------------------------------------------------------------

def param_string_array(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            pass
        return [item.strip() for item in text.split(",") if item.strip()]
    return [str(item) for item in value]


def param_float_array(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = [item.strip() for item in text.split(",") if item.strip()]
        if isinstance(parsed, (list, tuple)):
            return [float(item) for item in parsed]
        return [float(parsed)]
    return [float(item) for item in value]


def param_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


# ---------------------------------------------------------------------------
# Pose / geometry helpers
# ---------------------------------------------------------------------------

def yaw_from_quat(q) -> float:
    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def set_quat_from_yaw(q, yaw: float) -> None:
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)


def pose2d_from_odom(msg: Odometry) -> Pose2D:
    pose = Pose2D()
    pose.x = float(msg.pose.pose.position.x)
    pose.y = float(msg.pose.pose.position.y)
    pose.theta = yaw_from_quat(msg.pose.pose.orientation)
    return pose


def odom_to_dict(msg: Odometry) -> dict:
    return dict(message_to_ordereddict(msg))


def gps_to_dict(msg: NavSatFix) -> dict:
    return dict(message_to_ordereddict(msg))


def distance_xy(a, b) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def lat_lon_from_xy(
    x: float,
    y: float,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float,
    z: float,
) -> tuple[float, float, float]:
    """Local ENU metres (x=East, y=North) -> GPS, equirectangular about the origin."""
    meters_per_deg_lat = 111_320.0
    lat = origin_lat + y / meters_per_deg_lat
    lon_scale = meters_per_deg_lat * max(math.cos(math.radians(origin_lat)), 0.01)
    lon = origin_lon + x / lon_scale
    return lat, lon, origin_alt + z


# ---------------------------------------------------------------------------
# Proto registry: Webots PROTO / agent kind -> Isaac spawn descriptor
# ---------------------------------------------------------------------------

@dataclass
class ProtoSpec:
    """How a Webots agent maps onto an Isaac world_manager spawn request."""

    uri: str                       # world_manager spawn type: "drone" | "carter"
    kind: str                      # "uav" | "ugv"
    resource: dict = field(default_factory=dict)  # base resource_string payload


# Keyed by lowercased proto name. `kind` is the fallback when a proto is unknown.
_PROTO_REGISTRY: dict[str, ProtoSpec] = {
    "mavic2prosimple": ProtoSpec(
        uri="drone",
        kind="uav",
        resource={"publish_pose_ros": True, "num_rotors": 4},
    ),
    "mavic2pro": ProtoSpec(
        uri="drone",
        kind="uav",
        resource={"publish_pose_ros": True, "num_rotors": 4},
    ),
    "scout": ProtoSpec(
        uri="carter",
        kind="ugv",
        resource={"control_mode": "ros2", "drive_backend": "cmd_vel", "speed": 1.0},
    ),
}

_KIND_FALLBACK: dict[str, ProtoSpec] = {
    "uav": ProtoSpec(uri="drone", kind="uav", resource={"publish_pose_ros": True, "num_rotors": 4}),
    "ugv": ProtoSpec(uri="carter", kind="ugv", resource={"control_mode": "ros2", "drive_backend": "cmd_vel", "speed": 1.0}),
}


def resolve_proto(agent: dict) -> ProtoSpec:
    """Pick the spawn descriptor for an agent, preferring its proto then its kind."""
    proto = str(agent.get("proto", "")).strip().lower()
    spec = _PROTO_REGISTRY.get(proto)
    if spec is not None:
        return ProtoSpec(uri=spec.uri, kind=spec.kind, resource=dict(spec.resource))
    kind = str(agent.get("kind", "ugv")).strip().lower()
    fallback = _KIND_FALLBACK.get(kind, _KIND_FALLBACK["ugv"])
    return ProtoSpec(uri=fallback.uri, kind=fallback.kind, resource=dict(fallback.resource))


# ---------------------------------------------------------------------------
# Mission / area model + named-location resolution
# ---------------------------------------------------------------------------

@dataclass
class Area:
    area_id: str
    offset: tuple[float, float]
    size: tuple[float, float]

    @property
    def bounds(self) -> list[list[float]]:
        x, y = self.offset
        sx, sy = self.size
        return [[x, y], [x + sx, y + sy]]

    def contains(self, x: float, y: float) -> bool:
        bounds = self.bounds
        return bounds[0][0] <= x <= bounds[1][0] and bounds[0][1] <= y <= bounds[1][1]

    def default_pose(self) -> Pose2D:
        pose = Pose2D()
        pose.x = self.offset[0] + min(5.0, self.size[0] * 0.2)
        pose.y = self.offset[1] + min(5.0, self.size[1] * 0.2)
        return pose


# Named locations look like "area_00_l_init" or "area_00_l_07".
_LOCATION_RE = re.compile(r"^(?P<area>area_\d+)_l_(?P<tag>.+)$")


@dataclass
class MissionModel:
    grid_size: int
    edge_size: float
    world_offset: tuple[float, float]
    areas: dict[str, Area] = field(default_factory=dict)

    @classmethod
    def build(cls, grid_size: int, edge_size: float, world_offset: list[float] | tuple[float, float]):
        grid_size = max(1, int(grid_size))
        edge_size = float(edge_size)
        if len(world_offset) >= 2:
            offset = (float(world_offset[0]), float(world_offset[1]))
        else:
            half = -0.5 * edge_size * grid_size
            offset = (half, half)
        model = cls(grid_size=grid_size, edge_size=edge_size, world_offset=offset)
        index = 0
        for row in range(grid_size):
            for col in range(grid_size):
                area_id = f"area_{index:02d}"
                model.areas[area_id] = Area(
                    area_id=area_id,
                    offset=(offset[0] + col * edge_size, offset[1] + row * edge_size),
                    size=(edge_size, edge_size),
                )
                index += 1
        return model

    def area_at(self, x: float, y: float) -> Area | None:
        for area in self.areas.values():
            if area.contains(x, y):
                return area
        return None

    def location_xy(self, location_id: str | None) -> tuple[float, float] | None:
        """Resolve a named location (e.g. "area_00_l_init") to local x,y metres.

        Only init/spawn locations are knowable at team-load time, so any
        location of an area resolves to that area's default spawn pose. Returns
        None when the location does not reference a known area, letting the
        caller fall back to deterministic index stacking.
        """
        if not location_id:
            return None
        match = _LOCATION_RE.match(str(location_id).strip())
        if match is None:
            return None
        area = self.areas.get(match.group("area"))
        if area is None:
            return None
        pose = area.default_pose()
        return pose.x, pose.y

    def as_area_db_entry(self) -> dict:
        world_size = self.edge_size * self.grid_size
        return {
            "grid_size": self.grid_size,
            "edge_size": self.edge_size,
            "world_size": world_size,
            "world_offset": list(self.world_offset),
            "coordinate_system": "local",
            "coordinate_origin": list(self.world_offset),
            "areas": {
                area_id: {
                    "area_id": area.area_id,
                    "offset": list(area.offset),
                    "bounds": area.bounds,
                }
                for area_id, area in self.areas.items()
            },
        }
