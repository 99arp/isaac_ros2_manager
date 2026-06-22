from __future__ import annotations

import ast
import json
import math
from typing import Any


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
