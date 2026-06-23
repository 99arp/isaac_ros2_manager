from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from typing import Any

from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


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


def best_effort_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        depth=depth,
        history=HistoryPolicy.KEEP_LAST,
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )


def _float_pair(value: Any, fallback: tuple[float, float]) -> tuple[float, float]:
    try:
        values = param_float_array(value)
    except Exception:
        values = []
    if len(values) >= 2:
        return float(values[0]), float(values[1])
    return fallback


def _positive_float(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except Exception:
        return fallback
    if math.isfinite(result) and result > 0.0:
        return result
    return fallback


@dataclass(frozen=True)
class PlanarFrameTransform:
    """Affine XY transform between Isaac-native metres and the Webots local grid."""

    local_offset: tuple[float, float] = (0.0, 0.0)
    local_edge_size: float = 1.0
    native_offset: tuple[float, float] = (0.0, 0.0)
    native_edge_size: float = 1.0

    @classmethod
    def from_values(
        cls,
        *,
        local_edge_size: Any,
        local_offset: Any,
        native_edge_size: Any,
        native_offset: Any,
    ) -> "PlanarFrameTransform":
        local_edge = _positive_float(local_edge_size, 1.0)
        native_edge = _positive_float(native_edge_size, local_edge)
        return cls(
            local_offset=_float_pair(local_offset, (0.0, 0.0)),
            local_edge_size=local_edge,
            native_offset=_float_pair(native_offset, (0.0, 0.0)),
            native_edge_size=native_edge,
        )

    @property
    def native_to_local_scale(self) -> float:
        return self.local_edge_size / self.native_edge_size

    @property
    def local_to_native_scale(self) -> float:
        return self.native_edge_size / self.local_edge_size

    def native_to_local_xy(self, x: float, y: float) -> tuple[float, float]:
        scale = self.native_to_local_scale
        return (
            self.local_offset[0] + (float(x) - self.native_offset[0]) * scale,
            self.local_offset[1] + (float(y) - self.native_offset[1]) * scale,
        )

    def local_to_native_xy(self, x: float, y: float) -> tuple[float, float]:
        scale = self.local_to_native_scale
        return (
            self.native_offset[0] + (float(x) - self.local_offset[0]) * scale,
            self.native_offset[1] + (float(y) - self.local_offset[1]) * scale,
        )

    def native_to_local_distance(self, distance: float) -> float:
        return abs(float(distance) * self.native_to_local_scale)

    def local_to_native_distance(self, distance: float) -> float:
        return abs(float(distance) * self.local_to_native_scale)

    def metadata(self) -> dict[str, Any]:
        return {
            "native_frame": {
                "offset": [self.native_offset[0], self.native_offset[1]],
                "edge_size": self.native_edge_size,
            },
            "local_frame": {
                "offset": [self.local_offset[0], self.local_offset[1]],
                "edge_size": self.local_edge_size,
            },
            "native_to_local_scale": self.native_to_local_scale,
        }


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
