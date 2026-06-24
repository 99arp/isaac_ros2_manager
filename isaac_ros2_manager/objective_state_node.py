from __future__ import annotations

import json
import math
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from auspex_msgs.srv import UpsertSubframe
from webots_ros2_manager_msgs.action import ObjectiveAction
from webots_ros2_manager_msgs.msg import Objective, StringKnowledge

from .common import PlanarFrameTransform, param_float_array, param_string_array, yaw_from_quat


@dataclass
class ObjectiveRecord:
    name: str
    location: str
    x: float
    y: float
    theta: float
    type: int
    status: int
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    detections: int = 1


class IsaacObjectiveStateNode(Node):
    """ROS-side owner for Isaac static objective state.

    This intentionally mirrors the Webots contract used by the skills:
    PoseStamped detections in, area_objectives KNOW data out, and per-agent
    ObjectiveAction detect/disarm servers.
    """

    def __init__(self) -> None:
        super().__init__("isaac_objective_state")
        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameter("team_namespace", "chipgt")
        self.declare_parameter("area_id", "area_00")
        self.declare_parameter("action_agents", ["mini1", "irobot2"])
        self.declare_parameter("objectives_file", "")
        self.declare_parameter("objectives_json", "")
        self.declare_parameter("team_file", "")
        self.declare_parameter("team_json", "")
        self.declare_parameter("detected_objectives_topic", "")
        self.declare_parameter("area_objectives_topic", "/auspex_know/knowledge_collector/area_objectives")
        self.declare_parameter("team_data_topic", "/auspex_know/knowledge_collector/team_data")
        self.declare_parameter("area_data_topic", "/auspex_know/knowledge_collector/area_data")
        self.declare_parameter("publish_team_data", True)
        self.declare_parameter("publish_area_data", True)
        self.declare_parameter("direct_know_write", True)
        self.declare_parameter("area_size", [250.0, 250.0])
        self.declare_parameter("area_offset", [0.0, 0.0])
        self.declare_parameter("edge_size", 0.0)
        self.declare_parameter("world_offset", [0.0, 0.0])
        self.declare_parameter("nav_edge_size", 0.0)
        self.declare_parameter("nav_world_offset", [0.0, 0.0])
        self.declare_parameter("coordinate_system", "local")
        self.declare_parameter("coordinate_origin", [0.0, 0.0, 0.0])
        self.declare_parameter("objectives_coordinate_frame", "native")
        self.declare_parameter("detections_coordinate_frame", "local")
        self.declare_parameter("local_frame_id", "local")
        self.declare_parameter("default_objective_type", "ground")
        self.declare_parameter("merge_radius_m", 1.0)
        self.declare_parameter("match_tolerance_m", 3.0)
        self.declare_parameter("publish_period_s", 2.0)

        self.team_namespace = str(self.get_parameter("team_namespace").value).strip("/")
        self.area_id = str(self.get_parameter("area_id").value or "area_00")
        self.default_objective_type = self._objective_type_from_value(
            self.get_parameter("default_objective_type").value
        )
        self.merge_radius_m = max(0.0, float(self.get_parameter("merge_radius_m").value))
        self.match_tolerance_m = max(0.0, float(self.get_parameter("match_tolerance_m").value))
        self.records: dict[str, ObjectiveRecord] = {}
        self.action_servers: list[ActionServer] = []
        self.team_config = self._load_team_config()
        self.local_frame_id = str(self.get_parameter("local_frame_id").value or "local")
        self.objectives_coordinate_frame = str(
            self.get_parameter("objectives_coordinate_frame").value or "native"
        ).strip().lower()
        self.detections_coordinate_frame = str(
            self.get_parameter("detections_coordinate_frame").value or "local"
        ).strip().lower()
        self.frame_transform = PlanarFrameTransform.from_values(
            local_edge_size=self._local_edge_size(),
            local_offset=self._area_offset(),
            native_edge_size=self._native_edge_size(),
            native_offset=self._native_offset(),
        )

        detected_topic = str(self.get_parameter("detected_objectives_topic").value or "")
        if not detected_topic:
            detected_topic = f"/{self.team_namespace}/team_manager/detected_objectives"
        area_objectives_topic = str(self.get_parameter("area_objectives_topic").value)
        team_data_topic = str(self.get_parameter("team_data_topic").value)
        area_data_topic = str(self.get_parameter("area_data_topic").value)

        self.area_objectives_pub = self.create_publisher(
            StringKnowledge,
            area_objectives_topic,
            10,
        )
        self.team_data_pub = self.create_publisher(StringKnowledge, team_data_topic, 10)
        self.area_data_pub = self.create_publisher(StringKnowledge, area_data_topic, 10)
        self.publish_team_data = bool(self.get_parameter("publish_team_data").value)
        self.publish_area_data = bool(self.get_parameter("publish_area_data").value)
        self.direct_know_write = bool(self.get_parameter("direct_know_write").value)
        self.know_upsert_client = self.create_client(UpsertSubframe, "/auspex_know/upsert_subframe")
        self._know_upsert_unavailable_logged = False
        self.create_subscription(
            PoseStamped,
            detected_topic,
            self._detected_objective_cb,
            10,
            callback_group=self.cb_group,
        )

        for agent in self._action_agents():
            self.action_servers.append(
                ActionServer(
                    self,
                    ObjectiveAction,
                    f"/{self.team_namespace}/{agent}/detect",
                    execute_callback=self._execute_detect,
                    callback_group=self.cb_group,
                )
            )
            self.action_servers.append(
                ActionServer(
                    self,
                    ObjectiveAction,
                    f"/{self.team_namespace}/{agent}/disarm",
                    execute_callback=self._execute_disarm,
                    callback_group=self.cb_group,
                )
            )

        self._load_initial_objectives()

        publish_period_s = max(0.0, float(self.get_parameter("publish_period_s").value))
        if publish_period_s > 0.0:
            self.create_timer(publish_period_s, self._publish_knowledge)
        self._publish_knowledge()

        self.get_logger().info(
            "Isaac objective state ready: "
            f"detections={detected_topic}; area_objectives={area_objectives_topic}; "
            f"team_data={team_data_topic}; area_data={area_data_topic}; "
            f"publish_team_data={self.publish_team_data}; publish_area_data={self.publish_area_data}; "
            f"objectives={len(self.records)}; agents={self._action_agents()}; "
            f"native->local scale={self.frame_transform.native_to_local_scale:.3f}"
        )

    def _action_agents(self) -> list[str]:
        seen = set()
        agents = []
        for agent in param_string_array(self.get_parameter("action_agents").value):
            agent = str(agent).strip("/")
            if not agent or agent in seen:
                continue
            seen.add(agent)
            agents.append(agent)
        return agents

    def _load_team_config(self) -> dict[str, Any]:
        payloads = []
        team_file = str(self.get_parameter("team_file").value or "").strip()
        if team_file:
            path = Path(team_file).expanduser()
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    payloads.append(json.load(f))
            else:
                self.get_logger().warning(f"Team config file not found: {path}")

        team_json = str(self.get_parameter("team_json").value or "").strip()
        if team_json:
            payloads.append(json.loads(team_json))

        for payload in payloads:
            if isinstance(payload, dict):
                return payload

        return {
            "name": self.team_namespace,
            "area": self.area_id,
            "area_size": self._area_size(),
            "area_offset": self._area_offset(),
            "agents": {agent: {"kind": "ugv", "location": f"{self.area_id}_l_init"} for agent in self._action_agents()},
        }

    def _load_initial_objectives(self) -> None:
        payloads = []
        objectives_file = str(self.get_parameter("objectives_file").value or "").strip()
        if objectives_file:
            path = Path(objectives_file).expanduser()
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    payloads.append(json.load(f))
            else:
                self.get_logger().warning(f"Objective config file not found: {path}")

        objectives_json = str(self.get_parameter("objectives_json").value or "").strip()
        if objectives_json:
            payloads.append(json.loads(objectives_json))

        for payload in payloads:
            for item in self._iter_objective_items(payload):
                self._add_initial_objective(item)

    def _iter_objective_items(self, payload: Any):
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item
            return

        if not isinstance(payload, dict):
            return

        area_id = str(payload.get("area_id", self.area_id) or self.area_id)
        objectives = payload.get("objectives")
        if isinstance(objectives, list):
            for item in objectives:
                if isinstance(item, dict):
                    item = dict(item)
                    item.setdefault("area_id", area_id)
                    yield item
            return

        if isinstance(objectives, dict):
            for name, entry in objectives.items():
                if not isinstance(entry, dict):
                    continue
                data = dict(entry.get("data", entry))
                data.setdefault("name", name)
                data.setdefault("area_id", area_id)
                if "location" in entry:
                    data.setdefault("location", entry["location"])
                yield data
            return

        yield payload

    def _add_initial_objective(self, item: dict[str, Any]) -> None:
        area_id = str(item.get("area_id", self.area_id) or self.area_id)
        if area_id != self.area_id:
            self.get_logger().warning(f"Ignoring objective for unsupported area {area_id!r}")
            return

        position = self._position_from_item(item)
        if position is None:
            self.get_logger().warning(f"Ignoring objective without position: {item!r}")
            return

        index = self._next_record_index()
        name = str(item.get("name") or f"{self.area_id}_objective_{index:02d}")
        if name in self.records:
            self.get_logger().warning(f"Ignoring duplicate objective {name!r}")
            return

        gps = item.get("gps_position") if isinstance(item.get("gps_position"), dict) else {}
        input_frame = str(item.get("coordinate_frame", self.objectives_coordinate_frame) or "native")
        x, y = self._xy_to_local(position[0], position[1], input_frame)
        self.records[name] = ObjectiveRecord(
            name=name,
            location=str(item.get("location") or f"{self.area_id}_l_{index:02d}"),
            x=x,
            y=y,
            theta=position[2],
            type=self._objective_type_from_value(item.get("type", self.default_objective_type)),
            status=self._status_from_value(item.get("status", Objective.ACTIVE)),
            latitude=float(gps.get("latitude", 0.0) or 0.0),
            longitude=float(gps.get("longitude", 0.0) or 0.0),
            altitude=float(gps.get("altitude", 0.0) or 0.0),
        )

    def _position_from_item(self, item: dict[str, Any]) -> tuple[float, float, float] | None:
        raw = item.get("position", item.get("init_pos", item.get("pose")))
        theta = float(item.get("theta", item.get("init_yaw", 0.0)) or 0.0)
        try:
            if isinstance(raw, dict):
                return (
                    float(raw["x"]),
                    float(raw["y"]),
                    float(raw.get("theta", theta) or 0.0),
                )
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                return float(raw[0]), float(raw[1]), theta
        except Exception:
            return None
        return None

    def _detected_objective_cb(self, msg: PoseStamped) -> None:
        x, y = self._xy_to_local(
            msg.pose.position.x,
            msg.pose.position.y,
            self._detection_frame(msg),
        )
        theta = yaw_from_quat(msg.pose.orientation)

        record = self._nearest_record_euclidean(x, y, self.merge_radius_m)
        if record is None:
            record = self._create_record(x, y, theta)
            self.get_logger().info(
                f"Created detected objective {record.name} at ({record.x:.2f}, {record.y:.2f})"
            )
        else:
            self._merge_record_position(record, x, y, theta)
            self.get_logger().info(
                f"Merged detected objective into {record.name} at ({record.x:.2f}, {record.y:.2f})"
            )
        self._publish_area_objectives()

    def _create_record(self, x: float, y: float, theta: float) -> ObjectiveRecord:
        index = self._next_record_index()
        name = f"{self.area_id}_objective_{index:02d}"
        record = ObjectiveRecord(
            name=name,
            location=f"{self.area_id}_l_{index:02d}",
            x=x,
            y=y,
            theta=theta,
            type=self.default_objective_type,
            status=int(Objective.ACTIVE),
        )
        self.records[name] = record
        return record

    def _next_record_index(self) -> int:
        index = 0
        while f"{self.area_id}_objective_{index:02d}" in self.records:
            index += 1
        return index

    def _merge_record_position(self, record: ObjectiveRecord, x: float, y: float, theta: float) -> None:
        n = max(1, record.detections)
        record.x = (record.x * n + x) / float(n + 1)
        record.y = (record.y * n + y) / float(n + 1)
        record.theta = (record.theta * n + theta) / float(n + 1)
        record.detections = n + 1

    def _execute_detect(self, goal_handle):
        request = goal_handle.request
        record = self._match_request(request)
        result = ObjectiveAction.Result()

        if record is None:
            goal_handle.succeed()
            result.success = True
            result.message = "no trap"
            result.type = int(ObjectiveAction.Result.NO_TRAP)
            return result

        goal_handle.succeed()
        result.success = True
        result.message = "ok"
        result.type = self._action_result_type(record.type)
        return result

    def _execute_disarm(self, goal_handle):
        request = goal_handle.request
        record = self._match_request(request)
        result = ObjectiveAction.Result()

        if record is None:
            goal_handle.abort()
            result.success = False
            result.message = "disarm failed"
            result.type = int(ObjectiveAction.Result.NO_TRAP)
            return result

        record.status = int(Objective.INACTIVE)
        self._publish_area_objectives()
        goal_handle.succeed()
        result.success = True
        result.message = "disarm successful"
        result.type = self._action_result_type(record.type)
        return result

    def _match_request(self, request) -> ObjectiveRecord | None:
        name = str(getattr(request, "name", "") or "").strip()
        if name and name in self.records:
            return self.records[name]

        position = getattr(request, "position", None)
        if position is None:
            return None

        try:
            x = float(position.x)
            y = float(position.y)
        except Exception:
            return None
        return self._nearest_record_webots_tolerance(x, y, self.match_tolerance_m)

    def _nearest_record_webots_tolerance(
        self,
        x: float,
        y: float,
        tolerance_m: float,
    ) -> ObjectiveRecord | None:
        best = None
        best_distance = None
        for record in self.records.values():
            if abs(record.x - x) > tolerance_m or abs(record.y - y) > tolerance_m:
                continue
            distance = math.hypot(record.x - x, record.y - y)
            if best_distance is None or distance < best_distance:
                best = record
                best_distance = distance
        return best

    def _nearest_record_euclidean(
        self,
        x: float,
        y: float,
        tolerance_m: float,
    ) -> ObjectiveRecord | None:
        best = None
        best_distance = None
        for record in self.records.values():
            distance = math.hypot(record.x - x, record.y - y)
            if distance <= tolerance_m and (best_distance is None or distance < best_distance):
                best = record
                best_distance = distance
        return best

    def _publish_area_objectives(self) -> None:
        msg = StringKnowledge()
        msg.instance_id = self.area_id
        msg.instance_data_json = json.dumps(self._area_objectives_payload())
        self.area_objectives_pub.publish(msg)
        self._upsert_know_subframe("area_objectives", msg)

    def _publish_knowledge(self) -> None:
        self._publish_area_objectives()
        if self.publish_team_data:
            self._publish_team_data()
        if self.publish_area_data:
            self._publish_area_data()

    def _publish_team_data(self) -> None:
        msg = StringKnowledge()
        msg.instance_id = self._team_id()
        msg.instance_data_json = json.dumps(self._team_payload())
        self.team_data_pub.publish(msg)
        self._upsert_know_subframe("team_data", msg)

    def _publish_area_data(self) -> None:
        msg = StringKnowledge()
        msg.instance_id = "area_data"
        msg.instance_data_json = json.dumps(self._area_data_payload())
        self.area_data_pub.publish(msg)
        self._upsert_know_subframe("area_data", msg)

    def _upsert_know_subframe(self, frame: str, msg: StringKnowledge) -> None:
        if not self.direct_know_write:
            return
        if not self.know_upsert_client.wait_for_service(timeout_sec=0.0):
            if not self._know_upsert_unavailable_logged:
                self.get_logger().warning(
                    "AUSPEX-KNOW upsert service is not available; "
                    "continuing with topic-only knowledge publishing."
                )
                self._know_upsert_unavailable_logged = True
            return

        request = UpsertSubframe.Request()
        request.frame = frame
        request.instance_id = msg.instance_id
        request.subframe = frame
        request.item = json.dumps({
            "instance_id": msg.instance_id,
            "instance_data_json": msg.instance_data_json,
        })
        future = self.know_upsert_client.call_async(request)
        future.add_done_callback(self._upsert_know_done)

    def _upsert_know_done(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warning(f"AUSPEX-KNOW upsert failed: {exc}")
            return
        if response is not None and not response.success:
            self.get_logger().warning("AUSPEX-KNOW upsert returned success=False")

    def _area_objectives_payload(self) -> dict[str, Any]:
        locations = {
            f"{self.area_id}_l_init": {
                "type": "Waypoint",
                "data": {"x": 0.0, "y": 0.0, "theta": 0.0},
            }
        }
        objectives = {}
        for record in self.records.values():
            objectives[record.name] = {
                "type": "Objective",
                "location": record.location,
                "data": {
                    "type": int(record.type),
                    "status": int(record.status),
                    "position": {
                        "x": float(record.x),
                        "y": float(record.y),
                        "theta": float(record.theta),
                    },
                    "gps_position": {
                        "latitude": float(record.latitude),
                        "longitude": float(record.longitude),
                        "altitude": float(record.altitude),
                    },
                    "name": record.name,
                },
            }
            locations[record.location] = {
                "type": "Waypoint",
                "data": {
                    "x": float(record.x),
                    "y": float(record.y),
                    "theta": float(record.theta),
                },
            }
        return {
            "area_id": self.area_id,
            "locations": locations,
            "objectives": objectives,
        }

    def _team_id(self) -> str:
        return str(self.team_config.get("name") or self.team_namespace)

    def _team_payload(self) -> dict[str, Any]:
        agents_cfg = self.team_config.get("agents")
        if not isinstance(agents_cfg, dict):
            agents_cfg = {}

        agent_names = list(agents_cfg.keys()) or self._action_agents()
        agents = {}
        for agent_name in agent_names:
            raw = agents_cfg.get(agent_name, {})
            agent_data = copy.deepcopy(raw) if isinstance(raw, dict) else {}
            agent_data.setdefault("kind", self._agent_kind(agent_data))
            agent_data.setdefault("location", f"{self.area_id}_l_init")
            agent_data["pose"] = self._default_odometry(agent_name)
            agent_data["gps"] = self._default_navsatfix()
            agents[agent_name] = agent_data

        return {
            "team_id": self._team_id(),
            "agents": agents,
            "data": {
                "area": {
                    "id": self.area_id,
                    "offset": self._area_offset(),
                    "bounds": self._area_bounds(),
                }
            },
        }

    def _area_data_payload(self) -> dict[str, Any]:
        area_size = self._area_size()
        edge_size = float(max(area_size)) if area_size else 0.0
        return {
            "grid_size": 1,
            "edge_size": edge_size,
            "world_size": edge_size,
            "world_offset": self._area_offset(),
            "coordinate_system": str(self.get_parameter("coordinate_system").value or "local"),
            "coordinate_origin": self._coordinate_origin(),
            "coordinate_transform": self.frame_transform.metadata(),
            "areas": {
                self.area_id: {
                    "area_id": self.area_id,
                    "offset": self._area_offset(),
                    "bounds": self._area_bounds(),
                }
            },
        }

    def _area_size(self) -> list[float]:
        try:
            edge_size = float(self.get_parameter("edge_size").value)
            if math.isfinite(edge_size) and edge_size > 0.0:
                return [edge_size, edge_size]
        except Exception:
            pass
        from_team = self.team_config.get("area_size") if hasattr(self, "team_config") else None
        raw = from_team if from_team is not None else self.get_parameter("area_size").value
        values = param_float_array(raw)
        if len(values) >= 2:
            return [float(values[0]), float(values[1])]
        return [250.0, 250.0]

    def _area_offset(self) -> list[float]:
        try:
            values = param_float_array(self.get_parameter("world_offset").value)
            if len(values) >= 2:
                return [float(values[0]), float(values[1])]
        except Exception:
            pass
        from_team = self.team_config.get("area_offset") if hasattr(self, "team_config") else None
        raw = from_team if from_team is not None else self.get_parameter("area_offset").value
        values = param_float_array(raw)
        if len(values) >= 2:
            return [float(values[0]), float(values[1])]
        return [0.0, 0.0]

    def _area_bounds(self) -> list[list[float]]:
        offset = self._area_offset()
        size = self._area_size()
        return [
            [offset[0], offset[0] + size[0]],
            [offset[1], offset[1] + size[1]],
        ]

    def _local_edge_size(self) -> float:
        try:
            value = float(self.get_parameter("edge_size").value)
            if math.isfinite(value) and value > 0.0:
                return value
        except Exception:
            pass
        size = self._area_size()
        return max(1.0, float(size[0]))

    def _native_edge_size(self) -> float:
        try:
            value = float(self.get_parameter("nav_edge_size").value)
            if math.isfinite(value) and value > 0.0:
                return value
        except Exception:
            pass
        return self._local_edge_size()

    def _native_offset(self) -> list[float]:
        values = param_float_array(self.get_parameter("nav_world_offset").value)
        if len(values) >= 2:
            return [float(values[0]), float(values[1])]
        return self._area_offset()

    def _xy_to_local(self, x: float, y: float, coordinate_frame: str) -> tuple[float, float]:
        frame = str(coordinate_frame or "local").strip().lower()
        if frame in ("native", "isaac", "isaacsim", "nav", "nav2", "world", "odom", "map"):
            return self.frame_transform.native_to_local_xy(float(x), float(y))
        return float(x), float(y)

    def _detection_frame(self, msg: PoseStamped) -> str:
        configured = self.detections_coordinate_frame
        if configured != "auto":
            return configured
        frame_id = str(msg.header.frame_id or "").strip().lower()
        if frame_id == self.local_frame_id.lower() or frame_id in ("local", "webots"):
            return "local"
        return "native"

    def _coordinate_origin(self) -> list[float]:
        values = param_float_array(self.get_parameter("coordinate_origin").value)
        if len(values) >= 3:
            return [float(values[0]), float(values[1]), float(values[2])]
        return [0.0, 0.0, 0.0]

    def _agent_kind(self, agent_data: dict[str, Any]) -> str:
        kind = str(agent_data.get("kind") or "").strip().lower()
        if kind:
            return kind
        manager = str(agent_data.get("manager") or "").strip().lower()
        if manager.startswith("uav_"):
            return "uav"
        if manager.startswith("ugv_"):
            return "ugv"
        agent_type = str(agent_data.get("type") or "").strip().lower()
        if "uav" in agent_type or "drone" in agent_type:
            return "uav"
        return "ugv"

    def _default_odometry(self, child_frame_id: str) -> dict[str, Any]:
        offset = self._area_offset()
        x = float(offset[0])
        y = float(offset[1])
        return {
            "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": self.local_frame_id},
            "child_frame_id": str(child_frame_id),
            "pose": {
                "pose": {
                    "position": {"x": x, "y": y, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "covariance": [0.0] * 36,
            },
            "twist": {
                "twist": {
                    "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
                },
                "covariance": [0.0] * 36,
            },
        }

    def _default_navsatfix(self) -> dict[str, Any]:
        return {
            "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
            "status": {"status": 0, "service": 0},
            "latitude": 0.0,
            "longitude": 0.0,
            "altitude": 0.0,
            "position_covariance": [0.0] * 9,
            "position_covariance_type": 0,
        }

    def _objective_type_from_value(self, value: Any) -> int:
        if isinstance(value, int):
            return int(value)
        text = str(value).strip().lower()
        mapping = {
            "unknown": Objective.UNKNOWN,
            "ground": Objective.IS_GROUND,
            "is_ground": Objective.IS_GROUND,
            "air": Objective.IS_AIR,
            "is_air": Objective.IS_AIR,
            "combined": Objective.IS_COMBINED,
            "combi": Objective.IS_COMBINED,
            "is_combined": Objective.IS_COMBINED,
            "none": Objective.NO_TRAP,
            "no_trap": Objective.NO_TRAP,
            "no-trap": Objective.NO_TRAP,
        }
        return int(mapping.get(text, self.default_objective_type if hasattr(self, "default_objective_type") else Objective.IS_GROUND))

    def _status_from_value(self, value: Any) -> int:
        if isinstance(value, int):
            return int(value)
        text = str(value).strip().lower()
        mapping = {
            "active": Objective.ACTIVE,
            "inactive": Objective.INACTIVE,
        }
        return int(mapping.get(text, Objective.ACTIVE))

    def _action_result_type(self, objective_type: int) -> int:
        mapping = {
            int(Objective.UNKNOWN): int(ObjectiveAction.Result.UNKNOWN),
            int(Objective.IS_GROUND): int(ObjectiveAction.Result.IS_GROUND),
            int(Objective.IS_AIR): int(ObjectiveAction.Result.IS_AIR),
            int(Objective.IS_COMBINED): int(ObjectiveAction.Result.IS_COMBINED),
            int(Objective.NO_TRAP): int(ObjectiveAction.Result.NO_TRAP),
        }
        return mapping.get(int(objective_type), int(ObjectiveAction.Result.NO_TRAP))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IsaacObjectiveStateNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
