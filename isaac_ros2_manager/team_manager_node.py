from __future__ import annotations

import json
import os
import re
import time

import rclpy
from ample_msgs.action import ExecutePlan
from ament_index_python.packages import get_package_share_directory
from ament_index_python.packages import PackageNotFoundError
from geometry_msgs.msg import Pose2D, PoseStamped, Twist
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient, ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rosidl_runtime_py.convert import message_to_ordereddict
from std_srvs.srv import Trigger
from webots_ros2_manager_msgs.action import MoveToArea, SweepArea
from yaml import dump

from .agent_bridge_node import AgentBridge, AgentBridgeConfig
from .common import (
    MissionModel,
    gps_to_dict,
    join_name,
    lat_lon_from_xy,
    odom_to_dict,
    param_bool,
    param_float_array,
    resolve_proto,
)

try:
    from . import auspex_compat
except Exception:
    auspex_compat = None

try:
    from simulation_interfaces.srv import SpawnEntity
except Exception:
    SpawnEntity = None


class IsaacTeamManager(Node):
    """Launch a Webots-style team into Isaac Sim and bridge per-agent control.

    Reads the same team JSON the Webots stack uses, maps each agent's proto/kind
    onto an Isaac world_manager spawn request, places it at its named location,
    and stands up an AgentBridge so the upstream managers can drive it.
    """

    def __init__(self):
        super().__init__("team_manager")
        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameter("team_json", "None")
        self.declare_parameter("grid_size", 1)
        self.declare_parameter("edge_size", 250.0)
        self.declare_parameter("world_offset", [0.0, 0.0], descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("spawn", True)
        self.declare_parameter("spawn_with_isaac", True)
        self.declare_parameter("auto_update_db", True)
        self.declare_parameter("goal_tolerance", 1.0)
        self.declare_parameter("goal_timeout_sec", 120.0)
        self.declare_parameter("origin_lat", 0.0)
        self.declare_parameter("origin_lon", 0.0)
        self.declare_parameter("origin_alt", 0.0)
        self.declare_parameter("spawn_alt", 1.0)
        self.declare_parameter("odom_timeout_sec", 1.0)
        self.declare_parameter("synthetic_odom", True)
        self.declare_parameter("spawn_confirm_timeout_sec", 180.0)
        self.declare_parameter("spawn_max_attempts", 0)
        self.declare_parameter("spawn_retry_period_sec", 2.0)
        self.declare_parameter("spawn_call_timeout_sec", 10.0)

        team_json = self._resolve_team_json(str(self.get_parameter("team_json").value))
        with open(team_json, "r", encoding="utf-8") as f:
            self.team_dict = json.load(f)
        self.team_json = team_json
        self.team_name = self.team_dict["name"]
        self.agents = self.team_dict.get("agents", {})

        self.grid_size = int(self.get_parameter("grid_size").value)
        self.edge_size = float(self.get_parameter("edge_size").value)
        self.world_offset = param_float_array(self.get_parameter("world_offset").value)
        if len(self.world_offset) < 2 and "area_offset" in self.team_dict:
            self.world_offset = list(self.team_dict["area_offset"])
        self.mission = MissionModel.build(self.grid_size, self.edge_size, self.world_offset)
        self.current_area_id = self.team_dict.get("area") or next(iter(self.mission.areas.keys()))

        self.origin_lat = float(self.get_parameter("origin_lat").value)
        self.origin_lon = float(self.get_parameter("origin_lon").value)
        self.origin_alt = float(self.get_parameter("origin_alt").value)
        self.spawn_alt = float(self.get_parameter("spawn_alt").value)
        self.spawn_confirm_timeout_sec = float(self.get_parameter("spawn_confirm_timeout_sec").value)
        self.spawn_max_attempts = max(0, int(self.get_parameter("spawn_max_attempts").value))
        self.spawn_retry_period_sec = max(0.2, float(self.get_parameter("spawn_retry_period_sec").value))
        self.spawn_call_timeout_sec = max(1.0, float(self.get_parameter("spawn_call_timeout_sec").value))
        self.has_geo_origin = self.origin_lat != 0.0 or self.origin_lon != 0.0

        self.spawn_requested = param_bool(self.get_parameter("spawn").value)
        self.spawn_with_isaac = param_bool(self.get_parameter("spawn_with_isaac").value)
        self.auto_update_db = param_bool(self.get_parameter("auto_update_db").value)
        self.spawned_agents: set[str] = set()       # /world_manager/add returned OK (accepted, not yet confirmed)
        self.pending_spawn_agents: set[str] = set()
        self.confirmed_agents: set[str] = set()      # real Isaac odom seen -> entity is actually alive
        self.spawn_attempts: dict[str, int] = {}
        self.spawn_ack_monotonic: dict[str, float] = {}
        self.spawn_call_monotonic: dict[str, float] = {}
        self.last_spawn_wait_log_monotonic = 0.0
        self.agent_spawn_wait_log_monotonic: dict[str, float] = {}

        self.spawn_client = None
        if SpawnEntity is not None and self.spawn_with_isaac:
            self.spawn_client = self.create_client(SpawnEntity, "/world_manager/add", callback_group=self.cb_group)
        elif self.spawn_with_isaac:
            self.get_logger().warning("simulation_interfaces/srv/SpawnEntity is unavailable; using bridge-only mode")

        self.db_client = None
        if auspex_compat is not None and self.auto_update_db:
            self.db_client = auspex_compat.make_write_client(self, callback_group=self.cb_group)

        self.bridges: dict[str, AgentBridge] = {}
        self._create_agent_bridges()

        self.execute_plan_client = ActionClient(
            self,
            ExecutePlan,
            "ample/execute_plan",
            callback_group=self.cb_group,
        )

        self.create_service(Trigger, "~/spawn_team", self._spawn_team_cb, callback_group=self.cb_group)
        self.move_to_area_action = ActionServer(
            self,
            MoveToArea,
            "move_to_area",
            execute_callback=self._move_to_area_cb,
            callback_group=self.cb_group,
        )
        self.sweep_area_action = ActionServer(
            self,
            SweepArea,
            "sweep_area",
            execute_callback=self._sweep_area_cb,
            callback_group=self.cb_group,
        )

        if self.spawn_requested:
            self.spawn_retry_timer = self.create_timer(
                self.spawn_retry_period_sec,
                self._spawn_missing_agents,
                callback_group=self.cb_group,
            )
        else:
            self.spawn_retry_timer = None
        if self.db_client is not None:
            self.create_timer(1.0, self._update_db, callback_group=self.cb_group)

        self.get_logger().info(
            f"Isaac team_manager ready for team '{self.team_name}' with agents={list(self.agents.keys())};"
            f" spawn_with_isaac={self.spawn_with_isaac}; geo_origin={self.has_geo_origin}"
        )

    def _resolve_team_json(self, value: str) -> str:
        if value and value != "None" and os.path.isfile(value):
            return value
        if value and value != "None":
            for package_name in ("chipgt_bringup", "webots_ros2_manager"):
                try:
                    package_dir = get_package_share_directory(package_name)
                except PackageNotFoundError:
                    continue
                candidate = os.path.join(package_dir, "teams", value + ".json")
                if os.path.isfile(candidate):
                    return candidate
        raise ValueError("team_json must be an existing JSON file or an installed team name")

    def _agent_initial_pose(self, index: int, agent: dict) -> tuple[float, float, float]:
        """Local ENU metres for an agent, from its named location when known.

        Falls back to an explicit team start offset, then to deterministic
        per-index stacking inside the current area.
        """
        located = self.mission.location_xy(agent.get("location"))
        if located is not None:
            # Several agents commonly share a named init location; fan them out
            # deterministically by index so they do not spawn on top of each other.
            return located[0], located[1] + index * 2.0, 0.0
        if self.team_dict.get("webots__start_offset") is not None:
            base_x = float(self.team_dict["webots__start_offset"][0])
            base_y = float(self.team_dict["webots__start_offset"][1])
        else:
            area = self.mission.areas.get(self.current_area_id) or next(iter(self.mission.areas.values()))
            base_x = area.offset[0] + min(3.0, area.size[0] * 0.2)
            base_y = area.offset[1] + min(3.0, area.size[1] * 0.2)
        return base_x, base_y + index * 2.0, 0.0

    def _create_agent_bridges(self) -> None:
        for index, (agent_name, agent) in enumerate(self.agents.items()):
            ns = join_name(self.team_name, agent_name)
            x, y, z = self._agent_initial_pose(index, agent)
            kind = resolve_proto(agent).kind
            if kind == "ugv":
                odom_topics = (join_name(ns, "chassis", "odom"),)
                pose_topics = ()
            else:
                odom_topics = ()
                pose_topics = (join_name(ns, "pose"),)
            config = AgentBridgeConfig(
                agent_namespace=ns,
                kind=kind,
                isaac_cmd_vel_topic="/cmd_vel" if kind == "ugv" else join_name(ns, "cmd_vel"),
                isaac_odom_topics=odom_topics,
                isaac_pose_topics=pose_topics,
                set_target_topic=join_name(ns, "set_target"),
                initial_x=x,
                initial_y=y,
                initial_z=z,
                origin_lat=self.origin_lat,
                origin_lon=self.origin_lon,
                origin_alt=self.origin_alt,
                odom_timeout_sec=float(self.get_parameter("odom_timeout_sec").value),
                synthetic_odom=param_bool(self.get_parameter("synthetic_odom").value),
                goal_tolerance=float(self.get_parameter("goal_tolerance").value),
                goal_timeout_sec=float(self.get_parameter("goal_timeout_sec").value),
                aero_platform_id=self._safe_aero_platform_id(ns) if kind == "uav" else "",
            )
            self.bridges[agent_name] = AgentBridge(self, config)

    @staticmethod
    def _safe_aero_platform_id(value: str | None) -> str:
        text = str(value or "").strip().strip("/")
        text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        if text and text[0].isdigit():
            text = f"n_{text}"
        return text or "drone"

    def _spawn_team_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self.spawn_requested = True
        reset_agents = self._reset_unconfirmed_spawn_state()
        self._ensure_spawn_retry_timer()
        self._spawn_missing_agents()
        response.success = True
        response.message = (
            f"Spawn requested for team {self.team_name}; reset unconfirmed agents={reset_agents}; "
            "bridge topics are already active."
        )
        return response

    def _reset_unconfirmed_spawn_state(self) -> list[str]:
        reset_agents: list[str] = []
        for agent_name in self.agents:
            if self._agent_confirmed(agent_name):
                self.confirmed_agents.add(agent_name)
                continue
            reset_agents.append(agent_name)
            self.pending_spawn_agents.discard(agent_name)
            self.spawned_agents.discard(agent_name)
            self.spawn_attempts.pop(agent_name, None)
            self.spawn_ack_monotonic.pop(agent_name, None)
            self.spawn_call_monotonic.pop(agent_name, None)
            self.agent_spawn_wait_log_monotonic.pop(agent_name, None)
        if reset_agents:
            self.get_logger().warning(
                f"Resetting Isaac spawn retry state for unconfirmed agents: {reset_agents}"
            )
        return reset_agents

    def _ensure_spawn_retry_timer(self) -> None:
        if self.spawn_retry_timer is None:
            self.spawn_retry_timer = self.create_timer(
                self.spawn_retry_period_sec,
                self._spawn_missing_agents,
                callback_group=self.cb_group,
            )
            return
        self.spawn_retry_timer.reset()

    def _agent_confirmed(self, agent_name: str) -> bool:
        """True once the agent is publishing real Isaac telemetry.

        ``last_odom_monotonic`` only advances when the bridge receives an actual
        Isaac odom/pose message (synthetic integration never touches it), so a
        positive value means the spawned entity really exists in the sim -- a far
        stronger signal than the fire-and-forget /world_manager/add OK response.
        """
        bridge = self.bridges.get(agent_name)
        return bool(bridge is not None and getattr(bridge, "last_odom_monotonic", 0.0) > 0.0)

    def _spawn_missing_agents(self) -> None:
        if not self.spawn_with_isaac or self.spawn_client is None:
            return

        # Promote agents to "confirmed" once their telemetry is live.
        for agent_name in self.agents:
            if agent_name not in self.confirmed_agents and self._agent_confirmed(agent_name):
                self.confirmed_agents.add(agent_name)
                self.agent_spawn_wait_log_monotonic.pop(agent_name, None)
                self.get_logger().info(f"Confirmed Isaac agent {self.team_name}/{agent_name} (odom active)")

        if len(self.confirmed_agents) == len(self.agents):
            if self.spawn_retry_timer is not None:
                self.spawn_retry_timer.cancel()
            return

        now = time.monotonic()
        self._clear_stale_pending_spawn_calls(now)

        if not self.spawn_client.service_is_ready():
            try:
                self.spawn_client.wait_for_service(timeout_sec=0.0)
            except Exception:
                pass
        if not self.spawn_client.service_is_ready():
            if now - self.last_spawn_wait_log_monotonic >= 10.0:
                waiting = self._format_agent_wait_list(
                    name for name in self.agents if name not in self.confirmed_agents
                )
                self.get_logger().warning(
                    f"Waiting for /world_manager/add before spawning {waiting}"
                )
                self.last_spawn_wait_log_monotonic = now
            return

        for index, (agent_name, agent) in enumerate(self.agents.items()):
            if agent_name in self.confirmed_agents:
                continue
            if agent_name in self.pending_spawn_agents:
                started = self.spawn_call_monotonic.get(agent_name, now)
                self._log_agent_spawn_wait(
                    agent_name,
                    now,
                    f"Waiting for {self._agent_kind_label(agent_name)} spawn service response for "
                    f"{self.team_name}/{agent_name}; request in flight for {now - started:.0f}s",
                )
                continue
            # An accepted spawn that never produced telemetry within the confirm
            # window almost certainly failed in the background (e.g. the scene was
            # not ready). Drop the stale ack so it gets re-requested below.
            if agent_name in self.spawned_agents:
                accepted_for = now - self.spawn_ack_monotonic.get(agent_name, now)
                if accepted_for < self.spawn_confirm_timeout_sec:
                    self._log_agent_spawn_wait(
                        agent_name,
                        now,
                        f"Waiting for {self._agent_kind_label(agent_name)} spawn confirmation for "
                        f"{self.team_name}/{agent_name}; accepted {accepted_for:.0f}s ago; "
                        f"waiting for telemetry on {self._agent_confirmation_topic(agent_name)}",
                    )
                    continue
                self.spawned_agents.discard(agent_name)
                self.get_logger().warning(
                    f"Isaac agent {self.team_name}/{agent_name} accepted but no odom after "
                    f"{self.spawn_confirm_timeout_sec:.0f}s; re-requesting spawn"
                )
            if self.spawn_max_attempts > 0 and self.spawn_attempts.get(agent_name, 0) >= self.spawn_max_attempts:
                self._log_agent_spawn_wait(
                    agent_name,
                    now,
                    f"Waiting for manual spawn reset for {self._agent_kind_label(agent_name)} "
                    f"{self.team_name}/{agent_name}; max attempts reached "
                    f"({self.spawn_attempts.get(agent_name, 0)}/{self.spawn_max_attempts})",
                )
                continue
            request = self._spawn_request(index, agent_name, agent)
            self.pending_spawn_agents.add(agent_name)
            self.spawn_call_monotonic[agent_name] = now
            self.spawn_attempts[agent_name] = self.spawn_attempts.get(agent_name, 0) + 1
            self.get_logger().info(
                f"Requesting Isaac spawn for {self.team_name}/{agent_name} "
                f"({request.uri}, {self._spawn_attempt_text(agent_name)})"
            )
            future = self.spawn_client.call_async(request)
            future.add_done_callback(lambda fut, name=agent_name: self._spawn_done(name, fut))

        # In the default unlimited mode, keep the timer alive until every agent
        # has real Isaac telemetry. With an explicit finite max, preserve the old
        # stop behavior once all remaining agents are exhausted.
        if self.spawn_max_attempts > 0 and not self.pending_spawn_agents and all(
            name in self.confirmed_agents or self.spawn_attempts.get(name, 0) >= self.spawn_max_attempts
            for name in self.agents
        ):
            if self.spawn_retry_timer is not None:
                self.spawn_retry_timer.cancel()
            unconfirmed = [n for n in self.agents if n not in self.confirmed_agents]
            if unconfirmed:
                self.get_logger().error(
                    f"Giving up spawning Isaac agents after {self.spawn_max_attempts} attempts: {unconfirmed}"
                )

    def _clear_stale_pending_spawn_calls(self, now: float) -> None:
        for agent_name in list(self.pending_spawn_agents):
            started = self.spawn_call_monotonic.get(agent_name)
            if started is None or now - started < self.spawn_call_timeout_sec:
                continue
            self.pending_spawn_agents.discard(agent_name)
            self.spawn_call_monotonic.pop(agent_name, None)
            self.get_logger().warning(
                f"Isaac spawn service call for {self.team_name}/{agent_name} did not return after "
                f"{self.spawn_call_timeout_sec:.0f}s; retrying"
            )

    def _log_agent_spawn_wait(self, agent_name: str, now: float, message: str) -> None:
        last = self.agent_spawn_wait_log_monotonic.get(agent_name, 0.0)
        if now - last < 10.0:
            return
        self.agent_spawn_wait_log_monotonic[agent_name] = now
        self.get_logger().warning(message)

    def _format_agent_wait_list(self, agent_names) -> str:
        parts = [
            f"{self._agent_kind_label(name)} {self.team_name}/{name}"
            for name in agent_names
        ]
        return ", ".join(parts) if parts else "team agents"

    def _agent_kind_label(self, agent_name: str) -> str:
        agent = self.agents.get(agent_name, {})
        spec = resolve_proto(agent)
        if spec.kind == "uav":
            return "drone"
        if spec.uri == "carter":
            return "Carter"
        return spec.kind or spec.uri or "agent"

    def _agent_confirmation_topic(self, agent_name: str) -> str:
        agent = self.agents.get(agent_name, {})
        spec = resolve_proto(agent)
        ns = join_name(self.team_name, agent_name)
        if spec.kind == "uav":
            return join_name(ns, "pose")
        if spec.kind == "ugv":
            return join_name(ns, "chassis", "odom")
        return join_name(ns, "pose")

    def _spawn_attempt_text(self, agent_name: str) -> str:
        attempt = self.spawn_attempts.get(agent_name, 0)
        if self.spawn_max_attempts > 0:
            return f"attempt {attempt}/{self.spawn_max_attempts}"
        return f"attempt {attempt}, unlimited retries"

    def _spawn_request(self, index: int, agent_name: str, agent: dict) -> "SpawnEntity.Request":
        x, y, z = self._agent_initial_pose(index, agent)
        spec = resolve_proto(agent)
        namespace = f"{self.team_name}/{agent_name}"

        request = SpawnEntity.Request()
        request.name = f"{self.team_name}_{agent_name}"
        request.entity_namespace = namespace
        request.allow_renaming = True
        request.uri = spec.uri
        request.initial_pose.header.frame_id = "map"
        request.initial_pose.pose.orientation.w = 1.0

        payload = dict(spec.resource)
        payload.setdefault("spawn_in_glade", False)

        if self.has_geo_origin:
            # Place by GPS (what the world_manager spawners document); keep the
            # pose-message position at the origin so it is not double-applied.
            lat, lon, alt = lat_lon_from_xy(
                x, y, self.origin_lat, self.origin_lon, self.origin_alt, z
            )
            payload["lat"] = lat
            payload["lon"] = lon
            payload["alt"] = alt + (self.spawn_alt if spec.kind == "uav" else 0.0)
        else:
            # No geo-reference: hand the spawner local metres via initial_pose.
            request.initial_pose.pose.position.x = x
            request.initial_pose.pose.position.y = y
            request.initial_pose.pose.position.z = z + (self.spawn_alt if spec.kind == "uav" else 0.0)

        if spec.kind == "uav":
            payload.setdefault("aero_platform_id", self._safe_aero_platform_id(namespace))
            self._ensure_uav_camera_payload(payload, namespace)
            backend = dict(payload.get("px4_mavlink_backend") or {})
            backend.setdefault("vehicle_id", index + 1)
            backend.setdefault("px4_autolaunch", True)
            backend.setdefault("enable_lockstep", False)
            backend.setdefault("num_rotors", int(payload.pop("num_rotors", 4)))
            if not backend.get("px4_dir"):
                px4_dir = os.environ.get("PX4_DIR") or os.environ.get("PX4_AUTOPILOT_DIR")
                default_px4_dir = "/home/qnc/Desktop/PX4-Autopilot"
                if not px4_dir and os.path.isfile(os.path.join(default_px4_dir, "build", "px4_sitl_default", "bin", "px4")):
                    px4_dir = default_px4_dir
                if px4_dir:
                    backend["px4_dir"] = px4_dir
            payload["px4_mavlink_backend"] = backend
        elif spec.kind == "ugv":
            payload.setdefault("control_mode", "ros2")
            payload.setdefault("drive_backend", "cmd_vel")
            payload.setdefault("ros_namespace", namespace)
            payload.setdefault("ros_topic_identifier", namespace)
            payload.setdefault("cmd_vel_topic", "/cmd_vel")

        request.resource_string = json.dumps(payload)
        return request

    def _ensure_uav_camera_payload(self, payload: dict, namespace: str) -> None:
        sensors = payload.get("sensors")
        has_camera = isinstance(sensors, dict) and bool(sensors.get("cameras"))
        if has_camera:
            return

        camera_topic = join_name(namespace, "front_camera", "image_raw")
        camera_name = f"{self._safe_aero_platform_id(namespace)}_frontCamera"
        payload.setdefault("camera_defaults", {"FPS": 10, "Width": 640, "Height": 480})
        payload["sensors"] = {
            "publish_mode": "ros",
            "zmq_base_port": 5555,
            "cameras": [
                {
                    camera_name: {
                        "type": "camera",
                        "X": 0.15,
                        "Y": 0.0,
                        "Z": 0.0,
                        "Roll": 0.0,
                        "Pitch": -90.0,
                        "Yaw": 0.0,
                        "fov": 90.0,
                        "publish_rate": 10,
                        "publisher": "ros",
                        "ros2_topic": camera_topic,
                        "name": camera_name,
                    }
                }
            ],
        }

    def _spawn_done(self, agent_name: str, future) -> None:
        self.pending_spawn_agents.discard(agent_name)
        self.spawn_call_monotonic.pop(agent_name, None)
        try:
            response = future.result()
            ok = getattr(response.result, "result", 0) == response.result.RESULT_OK
            if ok:
                # OK only means the request was accepted; the actual spawn runs
                # asynchronously in the extension. Record the ack and wait for
                # real odom (_agent_confirmed) before declaring success.
                self.spawned_agents.add(agent_name)
                self.spawn_ack_monotonic[agent_name] = time.monotonic()
                self.get_logger().info(
                    f"Isaac spawn accepted for {self.team_name}/{agent_name}; awaiting odom confirmation"
                )
            else:
                self.get_logger().warning(
                    f"Isaac spawn failed for {self.team_name}/{agent_name}: {response.result.error_message}"
                )
        except Exception as exc:
            self.get_logger().warning(f"Isaac spawn call failed for {self.team_name}/{agent_name}: {exc}")

    def _move_to_area_cb(self, goal_handle):
        request = goal_handle.request
        area = self.mission.areas.get(request.area_id)
        if area is None:
            goal_handle.abort()
            result = MoveToArea.Result()
            result.success = False
            result.message = f"Unknown area {request.area_id!r}"
            return result

        target = request.position_in_area
        if target.x == 0.0 and target.y == 0.0:
            target = area.default_pose()

        if not area.contains(target.x, target.y):
            goal_handle.abort()
            result = MoveToArea.Result()
            result.success = False
            result.message = "bad area position"
            return result

        if area.area_id == self.current_area_id:
            goal_handle.succeed()
            result = MoveToArea.Result()
            result.success = True
            result.message = "arrived"
            return result

        if not self.execute_plan_client.wait_for_server(timeout_sec=5.0):
            goal_handle.abort()
            result = MoveToArea.Result()
            result.success = False
            result.message = "AMPLE not reached"
            return result

        plan_goal = ExecutePlan.Goal()
        plan_goal.plan_as_string, plan_goal.plan_input_yaml = self._move_to_area_plan(target)
        plan_goal.plan_type = "rl"

        plan_result, error = self._execute_plan(plan_goal)
        result = MoveToArea.Result()
        if plan_result is not None and plan_result.success:
            self.current_area_id = area.area_id
            goal_handle.succeed()
            result.success = True
            result.message = "arrived in new area"
            return result

        goal_handle.abort()
        result.success = False
        result.message = error or getattr(plan_result, "message", "") or "failed to move to new area"
        return result

    def _wait_future(self, future, timeout_sec: float):
        deadline = time.monotonic() + max(0.0, timeout_sec)
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                return None, "timed out"
            time.sleep(0.02)
        if not future.done():
            return None, "not completed"
        try:
            return future.result(), ""
        except Exception as exc:
            return None, str(exc)

    def _execute_plan(self, plan_goal: ExecutePlan.Goal):
        goal_handle, error = self._wait_future(self.execute_plan_client.send_goal_async(plan_goal), 5.0)
        if goal_handle is None:
            return None, f"execute_plan goal failed: {error}"
        if not goal_handle.accepted:
            return None, "execute_plan goal rejected"

        timeout_sec = max(30.0, len(self.agents) * float(self.get_parameter("goal_timeout_sec").value) + 10.0)
        result_response, error = self._wait_future(goal_handle.get_result_async(), timeout_sec)
        if result_response is None:
            return None, f"execute_plan result failed: {error}"
        return result_response.result, ""

    def _move_to_area_plan(self, target: Pose2D) -> tuple[str, str]:
        plan = "parallel move_to_area {\n"
        sections = {"robot": "", "input": "", "body": ""}
        inputs = {}

        for offset, (agent_name, agent) in enumerate(self.agents.items()):
            kind = str(agent.get("kind") or resolve_proto(agent).kind)
            agent_type = str(agent.get("type") or f"{kind}_ranger")
            target_pose = Pose2D()
            target_pose.x = float(target.x)
            target_pose.y = float(target.y) + offset * 2.0
            target_pose.theta = float(target.theta)

            sections["robot"] += f"\t\t{agent_name}: {agent_type}\n"
            sections["input"] += f"\t\ttarget_{agent_name}: Waypoint\n"
            sections["body"] += (
                f"\t\tmove_{agent_name} {{ "
                f"move_{kind}[{agent_name}](target_{agent_name}) success.succeeded"
                " }\n"
            )
            inputs[f"target_{agent_name}"] = dict(message_to_ordereddict(target_pose))

        for key, text in sections.items():
            plan += f"\t{key} {{\n{text}\t}}\n"
        plan += "}"
        return plan, dump(inputs)

    def _sweep_area_cb(self, goal_handle):
        request = goal_handle.request
        area = self.mission.areas.get(request.area_id) if request.area_id else self.mission.areas.get(self.current_area_id)
        if area is None:
            area = next(iter(self.mission.areas.values()))
        target_agent = request.agent_id or self._first_agent(kind="uav") or self._first_agent()
        bridge = self.bridges.get(target_agent)
        result = SweepArea.Result()
        if bridge is None:
            goal_handle.abort()
            result.success = False
            result.message = "No agent available for sweep"
            return result
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = area.offset[0] + area.size[0] * 0.5
        pose.pose.position.y = area.offset[1] + area.size[1] * 0.5
        pose.pose.position.z = float(request.altitude or 5.0)
        pose.pose.orientation.w = 1.0
        bridge._publish_target(pose)
        goal_handle.succeed()
        result.success = True
        result.message = f"Sent sweep target to {target_agent}"
        return result

    def _first_agent(self, kind: str | None = None) -> str | None:
        for name, agent in self.agents.items():
            if kind is None or resolve_proto(agent).kind == kind:
                return name
        return None

    def _send_team_target(self, x: float, y: float, z: float = 0.0) -> None:
        for offset, bridge in enumerate(self.bridges.values()):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y) + offset * 2.0
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = 1.0
            bridge._publish_target(pose)

    def _platform_entry(self) -> str:
        data = {"team_id": self.team_name, "agents": {}, "data": {}}
        for name, agent in self.agents.items():
            bridge = self.bridges[name]
            agent_data = dict(agent)
            agent_data["pose"] = odom_to_dict(bridge.current_odom)
            agent_data["gps"] = gps_to_dict(bridge.current_gps())
            data["agents"][name] = agent_data
        area = self.mission.areas.get(self.current_area_id)
        if area is not None:
            data["data"]["area"] = {
                "id": area.area_id,
                "offset": list(area.offset),
                "bounds": area.bounds,
            }
        return json.dumps(data)

    def _update_db(self) -> None:
        if self.db_client is None:
            return
        if not self.db_client.service_is_ready() and not self.db_client.wait_for_service(timeout_sec=0.01):
            return
        self.db_client.call_async(auspex_compat.write_request(
            "platform", instance_id=self.team_name, entity=self._platform_entry()))


def main(args=None):
    rclpy.init(args=args)
    node = IsaacTeamManager()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
