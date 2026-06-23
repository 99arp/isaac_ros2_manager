from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import rclpy
import yaml
from ample_msgs.action import ExecutePlan
from auspex_msgs.srv import UpsertSubframe
from geometry_msgs.msg import Pose2D, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient, ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from webots_ros2_manager_msgs.action import MoveToArea, SweepArea
from webots_ros2_manager_msgs.msg import StringKnowledge

from .common import PlanarFrameTransform, best_effort_qos, param_float_array


@dataclass(frozen=True)
class Area:
    area_id: str
    offset: tuple[float, float]
    size: tuple[float, float]

    @property
    def bounds(self) -> list[list[float]]:
        return [
            [self.offset[0], self.offset[0] + self.size[0]],
            [self.offset[1], self.offset[1] + self.size[1]],
        ]

    def contains(self, x: float, y: float) -> bool:
        return (
            self.offset[0] <= x < self.offset[0] + self.size[0]
            and self.offset[1] <= y < self.offset[1] + self.size[1]
        )


class IsaacTeamManagerNode(Node):
    """Webots-compatible team facade for Isaac-backed robots."""

    def __init__(self) -> None:
        super().__init__("team_manager")
        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameter("team_file", "")
        self.declare_parameter("team_json", "")
        self.declare_parameter("grid_size", 1)
        self.declare_parameter("edge_size", 100.0)
        self.declare_parameter("world_offset", "")
        self.declare_parameter("nav_edge_size", 0.0)
        self.declare_parameter("nav_world_offset", "")
        self.declare_parameter("local_frame_id", "local")
        self.declare_parameter("execute_plan_action", "ample/execute_plan")
        self.declare_parameter("team_data_topic", "/auspex_know/knowledge_collector/team_data")
        self.declare_parameter("area_data_topic", "/auspex_know/knowledge_collector/area_data")
        self.declare_parameter("direct_know_write", True)
        self.declare_parameter("publish_period_s", 2.0)

        self.team_config = self._load_team_config()
        self.team_id = str(self.team_config.get("name") or self.get_namespace().strip("/") or "chipgt")
        self.current_area_id = str(self.team_config.get("area") or "area_00")
        self.areas = self._build_areas()
        if self.current_area_id not in self.areas:
            self.current_area_id = next(iter(self.areas))
        self.nav_areas = self._build_nav_areas()
        self.frame_transform = PlanarFrameTransform.from_values(
            local_edge_size=self._edge_size(),
            local_offset=self._world_offset(),
            native_edge_size=self._nav_edge_size(),
            native_offset=self._nav_world_offset(),
        )
        self.local_frame_id = str(self.get_parameter("local_frame_id").value or "local")

        self.agent_poses = {agent: self._default_odometry(agent) for agent in self._agents()}
        self.odom_subscriptions = []
        sensor_qos = best_effort_qos()
        for agent in self._agents():
            self.odom_subscriptions.append(
                self.create_subscription(
                    Odometry,
                    f"{agent}/odometry",
                    lambda msg, agent=agent: self._odom_cb(msg, agent),
                    sensor_qos,
                    callback_group=self.cb_group,
                )
            )
            self.odom_subscriptions.append(
                self.create_subscription(
                    PoseStamped,
                    f"{agent}/pose",
                    lambda msg, agent=agent: self._pose_cb(msg, agent),
                    sensor_qos,
                    callback_group=self.cb_group,
                )
            )

        team_data_topic = str(self.get_parameter("team_data_topic").value)
        area_data_topic = str(self.get_parameter("area_data_topic").value)
        self.team_data_pub = self.create_publisher(StringKnowledge, team_data_topic, 10)
        self.area_data_pub = self.create_publisher(StringKnowledge, area_data_topic, 10)
        self.direct_know_write = bool(self.get_parameter("direct_know_write").value)
        self.know_upsert_client = self.create_client(UpsertSubframe, "/auspex_know/upsert_subframe")
        self._know_upsert_unavailable_logged = False

        execute_plan_action = str(self.get_parameter("execute_plan_action").value or "ample/execute_plan")
        self.execute_plan_client = ActionClient(
            self,
            ExecutePlan,
            execute_plan_action,
            callback_group=self.cb_group,
        )
        self.move_to_area_action = ActionServer(
            self,
            MoveToArea,
            "move_to_area",
            self.move_to_area_callback,
            callback_group=self.cb_group,
        )
        self.sweep_area_action = ActionServer(
            self,
            SweepArea,
            "sweep_area",
            self.sweep_area_callback,
            callback_group=self.cb_group,
        )

        publish_period_s = max(0.0, float(self.get_parameter("publish_period_s").value))
        if publish_period_s > 0.0:
            self.create_timer(publish_period_s, self._publish_knowledge, callback_group=self.cb_group)
        self._publish_knowledge()

        self.get_logger().info(
            f"Isaac team manager ready for {self.team_id}: "
            f"areas={len(self.areas)} current={self.current_area_id} agents={list(self._agents())}; "
            f"native->local scale={self.frame_transform.native_to_local_scale:.3f} "
            f"native_offset={self.frame_transform.native_offset} "
            f"local_offset={self.frame_transform.local_offset}"
        )

    def move_to_area_callback(self, goal_handle):
        request = goal_handle.request
        target_area = self.areas.get(request.area_id)
        result = MoveToArea.Result()

        if target_area is None:
            goal_handle.abort()
            result.success = False
            result.message = "bad area id"
            return result

        target_pose_local = self._goal_pose_in_area(request.position_in_area, target_area)
        if not target_area.contains(target_pose_local.x, target_pose_local.y):
            goal_handle.abort()
            result.success = False
            result.message = "bad area position"
            return result

        if request.area_id == self.current_area_id:
            goal_handle.succeed()
            result.success = True
            result.message = "arrived"
            return result

        nav_target_area = self.nav_areas.get(request.area_id)
        target_pose = self._local_pose_to_native(target_pose_local)
        if nav_target_area is not None and not nav_target_area.contains(target_pose.x, target_pose.y):
            self.get_logger().warning(
                f"Transformed Nav2 goal ({target_pose.x:.2f}, {target_pose.y:.2f}) "
                f"is outside native bounds for {request.area_id}: {nav_target_area.bounds}"
            )

        plan_goal = ExecutePlan.Goal()
        plan_goal.plan_as_string, inputs = self._move_plan(target_pose)
        plan_goal.plan_type = "rl"
        plan_goal.plan_input_yaml = yaml.safe_dump(inputs, sort_keys=False)
        plan_goal.reset_autorun_when_done = False

        self.get_logger().info(
            f"Moving team {self.team_id} to {request.area_id} "
            f"via local ({target_pose_local.x:.2f}, {target_pose_local.y:.2f}) "
            f"-> native ({target_pose.x:.2f}, {target_pose.y:.2f}, {target_pose.theta:.2f})."
        )
        plan_result = self._send_execute_plan(plan_goal)
        if plan_result is not None and plan_result.success:
            self.current_area_id = request.area_id
            self._publish_knowledge()
            goal_handle.succeed()
            result.success = True
            result.message = "arrived in new area"
            return result

        goal_handle.abort()
        result.success = False
        result.message = "failed to move to new area"
        return result

    def sweep_area_callback(self, goal_handle):
        request = goal_handle.request
        result = SweepArea.Result()
        area_id = request.area_id if request.area_id else self.current_area_id
        area = self.areas.get(area_id)
        if area is None:
            goal_handle.abort()
            result.success = False
            result.message = "bad area id"
            return result
        agent_name = request.agent_id or self._first_uav()
        if not agent_name:
            goal_handle.abort()
            result.success = False
            result.message = "no uav in team"
            return result

        target_pose_local = Pose2D()
        target_pose_local.x = area.offset[0] + min(5.0, area.size[0] * 0.25)
        target_pose_local.y = area.offset[1] + min(5.0, area.size[1] * 0.25)
        target_pose_local.theta = float(request.altitude or 15.0)
        target_pose = self._local_pose_to_native(target_pose_local)

        plan_goal = ExecutePlan.Goal()
        plan_goal.plan_as_string, inputs = self._move_plan(target_pose)
        plan_goal.plan_type = "rl"
        plan_goal.plan_input_yaml = yaml.safe_dump(inputs, sort_keys=False)
        plan_goal.reset_autorun_when_done = False

        self.get_logger().info(
            f"Sweeping {area.area_id} with {agent_name}: "
            f"local ({target_pose_local.x:.2f}, {target_pose_local.y:.2f}) "
            f"-> native ({target_pose.x:.2f}, {target_pose.y:.2f})."
        )
        plan_result = self._send_execute_plan(plan_goal)
        if plan_result is not None and plan_result.success:
            goal_handle.succeed()
            result.success = True
            result.message = "swept area"
            return result

        goal_handle.abort()
        result.success = False
        result.message = "failed to sweep area"
        return result

    def _send_execute_plan(self, plan_goal: ExecutePlan.Goal):
        if not self.execute_plan_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().warning("AMPLE execute_plan action server is not available.")
            return None

        send_future = self.execute_plan_client.send_goal_async(plan_goal)
        if not self._wait_future(send_future, timeout_sec=10.0):
            self.get_logger().warning("Timed out waiting for AMPLE goal acceptance.")
            return None

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warning("AMPLE rejected execute_plan goal.")
            return None

        result_future = goal_handle.get_result_async()
        if not self._wait_future(result_future, timeout_sec=180.0):
            self.get_logger().warning("Timed out waiting for AMPLE execute_plan result.")
            return None

        response = result_future.result()
        return response.result if response is not None else None

    def _wait_future(self, future, timeout_sec: float) -> bool:
        done = Event()
        future.add_done_callback(lambda _future: done.set())
        return done.wait(timeout_sec)

    def _move_plan(self, target_pose: Pose2D) -> tuple[str, dict[str, dict[str, float]]]:
        sections = {"robot": "", "input": "", "body": ""}
        inputs = {}
        for index, (agent_name, agent) in enumerate(self._agents().items()):
            kind = self._agent_kind(agent)
            agent_type = str(agent.get("type") or f"{kind}_ranger")
            sections["robot"] += f"\t\t{agent_name}: {agent_type}\n"
            sections["input"] += f"\t\ttarget_{agent_name}: Waypoint\n"
            sections["body"] += (
                f"\t\tmove_{agent_name} {{ move_{kind}[{agent_name}](target_{agent_name}) "
                "success.succeeded }\n"
            )
            inputs[f"target_{agent_name}"] = {
                "x": float(target_pose.x),
                "y": float(target_pose.y) + (2.0 * index),
                "theta": float(target_pose.theta),
            }

        return self._format_plan("parallel", "move_to_area", sections), inputs

    def _sweep_plan(
        self,
        agent_name: str,
        area: Area,
        altitude: float,
    ) -> tuple[str, dict[str, dict[str, float]]]:
        agent = self._agents()[agent_name]
        agent_type = str(agent.get("type") or "uav_ranger")
        sections = {
            "robot": f"\t\t{agent_name}: {agent_type}\n",
            "input": "",
            "body": "",
        }
        inset = min(5.0, area.size[0] * 0.2, area.size[1] * 0.2)
        corners = [
            (area.offset[0] + inset, area.offset[1] + inset),
            (area.offset[0] + area.size[0] - inset, area.offset[1] + inset),
            (area.offset[0] + area.size[0] - inset, area.offset[1] + area.size[1] - inset),
            (area.offset[0] + inset, area.offset[1] + area.size[1] - inset),
        ]
        inputs = {}
        for index, (x, y) in enumerate(corners):
            wp_name = f"wp_{index}"
            sections["input"] += f"\t\t{wp_name}: Waypoint\n"
            sections["body"] += (
                f"\t\tmove_{index} {{ move_uav[{agent_name}]({wp_name}) "
                "success.succeeded }\n"
            )
            inputs[wp_name] = {"x": float(x), "y": float(y), "theta": float(altitude)}
        return self._format_plan("sequence", "sweep_area", sections), inputs

    def _format_plan(self, composite: str, name: str, sections: dict[str, str]) -> str:
        plan = f"{composite} {name} {{\n"
        for key, text in sections.items():
            plan += f"\t{key} {{\n{text}\t}}\n"
        plan += "}"
        return plan

    def _goal_pose_in_area(self, requested: Pose2D, area: Area) -> Pose2D:
        pose = Pose2D()
        pose.theta = float(requested.theta)
        if requested.x == 0.0 and requested.y == 0.0:
            inset = min(5.0, area.size[0] * 0.25, area.size[1] * 0.25)
            pose.x = area.offset[0] + inset
            pose.y = area.offset[1] + inset
            return pose
        pose.x = float(requested.x)
        pose.y = float(requested.y)
        return pose

    def _local_pose_to_native(self, pose: Pose2D) -> Pose2D:
        native_x, native_y = self.frame_transform.local_to_native_xy(pose.x, pose.y)
        result = Pose2D()
        result.x = native_x
        result.y = native_y
        result.theta = float(pose.theta)
        return result

    def _publish_knowledge(self) -> None:
        self._publish_team_data()
        self._publish_area_data()

    def _publish_team_data(self) -> None:
        msg = StringKnowledge()
        msg.instance_id = self.team_id
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

    def _team_payload(self) -> dict[str, Any]:
        current_area = self.areas[self.current_area_id]
        agents = {}
        for agent_name, raw in self._agents().items():
            agent_data = copy.deepcopy(raw)
            agent_data.setdefault("kind", self._agent_kind(agent_data))
            agent_data.setdefault("manager", f"{agent_data['kind']}_ranger_manager")
            agent_data["location"] = f"{self.current_area_id}_l_init"
            agent_data["pose"] = self.agent_poses.get(agent_name, self._default_odometry(agent_name))
            agent_data.setdefault("gps", self._default_navsatfix())
            agents[agent_name] = agent_data

        return {
            "team_id": self.team_id,
            "agents": agents,
            "data": {
                "area": {
                    "id": self.current_area_id,
                    "offset": [current_area.offset[0], current_area.offset[1]],
                    "bounds": current_area.bounds,
                }
            },
        }

    def _area_data_payload(self) -> dict[str, Any]:
        return {
            "grid_size": self._grid_size(),
            "edge_size": self._edge_size(),
            "world_size": self._grid_size() * self._edge_size(),
            "world_offset": list(self._world_offset()),
            "coordinate_system": "local",
            "coordinate_origin": [0.0, 0.0, 0.0],
            "coordinate_transform": self.frame_transform.metadata(),
            "areas": {
                area_id: {
                    "area_id": area.area_id,
                    "offset": [area.offset[0], area.offset[1]],
                    "bounds": area.bounds,
                }
                for area_id, area in self.areas.items()
            },
        }

    def _odom_cb(self, msg: Odometry, agent: str) -> None:
        self.agent_poses[agent] = self._odom_payload(msg, agent)
        self._update_current_area_from_agent_poses()

    def _pose_cb(self, msg: PoseStamped, agent: str) -> None:
        self.agent_poses[agent] = self._pose_payload(msg, agent)
        self._update_current_area_from_agent_poses()

    def _odom_payload(self, msg: Odometry, child_frame_id: str) -> dict[str, Any]:
        local_x, local_y = self.frame_transform.native_to_local_xy(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )
        velocity_scale = self.frame_transform.native_to_local_scale
        return {
            "header": {
                "stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)},
                "frame_id": self.local_frame_id,
            },
            "child_frame_id": str(msg.child_frame_id or child_frame_id),
            "pose": {
                "pose": {
                    "position": {
                        "x": float(local_x),
                        "y": float(local_y),
                        "z": float(msg.pose.pose.position.z),
                    },
                    "orientation": {
                        "x": float(msg.pose.pose.orientation.x),
                        "y": float(msg.pose.pose.orientation.y),
                        "z": float(msg.pose.pose.orientation.z),
                        "w": float(msg.pose.pose.orientation.w),
                    },
                },
                "covariance": [float(value) for value in msg.pose.covariance],
            },
            "twist": {
                "twist": {
                    "linear": {
                        "x": float(msg.twist.twist.linear.x) * velocity_scale,
                        "y": float(msg.twist.twist.linear.y) * velocity_scale,
                        "z": float(msg.twist.twist.linear.z),
                    },
                    "angular": {
                        "x": float(msg.twist.twist.angular.x),
                        "y": float(msg.twist.twist.angular.y),
                        "z": float(msg.twist.twist.angular.z),
                    },
                },
                "covariance": [float(value) for value in msg.twist.covariance],
            },
        }

    def _pose_payload(self, msg: PoseStamped, child_frame_id: str) -> dict[str, Any]:
        local_x, local_y = self.frame_transform.native_to_local_xy(
            msg.pose.position.x,
            msg.pose.position.y,
        )
        return {
            "header": {
                "stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)},
                "frame_id": self.local_frame_id,
            },
            "child_frame_id": str(child_frame_id),
            "pose": {
                "pose": {
                    "position": {
                        "x": float(local_x),
                        "y": float(local_y),
                        "z": float(msg.pose.position.z),
                    },
                    "orientation": {
                        "x": float(msg.pose.orientation.x),
                        "y": float(msg.pose.orientation.y),
                        "z": float(msg.pose.orientation.z),
                        "w": float(msg.pose.orientation.w),
                    },
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

    def _default_odometry(self, child_frame_id: str) -> dict[str, Any]:
        if hasattr(self, "frame_transform"):
            x, y = self.frame_transform.native_to_local_xy(0.0, 0.0)
        else:
            area = self.areas.get(self.current_area_id)
            if area is None:
                offset = self._world_offset()
                x = offset[0] + 5.0
                y = offset[1] + 5.0
            else:
                x = area.offset[0] + 5.0
                y = area.offset[1] + 5.0
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

    def _load_team_config(self) -> dict[str, Any]:
        for parameter_name in ("team_file", "team_json"):
            raw = str(self.get_parameter(parameter_name).value or "").strip()
            if not raw:
                continue
            path = Path(raw).expanduser()
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload

        return {
            "name": self.get_namespace().strip("/") or "chipgt",
            "area": "area_00",
            "area_offset": [0.0, 0.0],
            "agents": {},
        }

    def _build_areas(self) -> dict[str, Area]:
        return self._build_area_grid(self._edge_size(), self._world_offset())

    def _build_nav_areas(self) -> dict[str, Area]:
        return self._build_area_grid(self._nav_edge_size(), self._nav_world_offset())

    def _build_area_grid(self, edge_size: float, world_offset: tuple[float, float]) -> dict[str, Area]:
        areas = {}
        grid_size = self._grid_size()
        index = 0
        for x in range(grid_size):
            for y in range(grid_size):
                area_id = f"area_{index:02d}"
                areas[area_id] = Area(
                    area_id=area_id,
                    offset=(world_offset[0] + (x * edge_size), world_offset[1] + (y * edge_size)),
                    size=(edge_size, edge_size),
                )
                index += 1
        return areas

    def _area_at(self, x: float, y: float) -> Area | None:
        for area in self.areas.values():
            if area.contains(x, y):
                return area
        return None

    def _update_current_area_from_agent_poses(self) -> None:
        if not self.agent_poses:
            return
        areas_seen = set()
        for pose in self.agent_poses.values():
            try:
                position = pose["pose"]["pose"]["position"]
                area = self._area_at(float(position["x"]), float(position["y"]))
            except Exception:
                return
            if area is None:
                return
            areas_seen.add(area.area_id)
        if len(areas_seen) != 1:
            return
        area_id = next(iter(areas_seen))
        if area_id != self.current_area_id:
            self.current_area_id = area_id
            self.get_logger().info(f"Team {self.team_id} entered {self.current_area_id}.")
            self._publish_knowledge()

    def _agents(self) -> dict[str, dict[str, Any]]:
        agents = self.team_config.get("agents")
        return agents if isinstance(agents, dict) else {}

    def _first_uav(self) -> str:
        for agent_name, agent in self._agents().items():
            if self._agent_kind(agent) == "uav":
                return agent_name
        return ""

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

    def _grid_size(self) -> int:
        try:
            return max(1, int(self.get_parameter("grid_size").value))
        except Exception:
            return 1

    def _edge_size(self) -> float:
        try:
            value = float(self.get_parameter("edge_size").value)
            if math.isfinite(value) and value > 0.0:
                return value
        except Exception:
            pass
        area_size = self.team_config.get("area_size")
        values = param_float_array(area_size)
        if values:
            return max(1.0, float(values[0]) / float(self._grid_size()))
        return 100.0

    def _nav_edge_size(self) -> float:
        try:
            value = float(self.get_parameter("nav_edge_size").value)
            if math.isfinite(value) and value > 0.0:
                return value
        except Exception:
            pass
        return self._edge_size()

    def _world_offset(self) -> tuple[float, float]:
        values = param_float_array(self.get_parameter("world_offset").value)
        if len(values) >= 2:
            return float(values[0]), float(values[1])
        values = param_float_array(self.team_config.get("area_offset"))
        if len(values) >= 2:
            return float(values[0]), float(values[1])
        return 0.0, 0.0

    def _nav_world_offset(self) -> tuple[float, float]:
        values = param_float_array(self.get_parameter("nav_world_offset").value)
        if len(values) >= 2:
            return float(values[0]), float(values[1])
        return self._world_offset()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IsaacTeamManagerNode()
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
