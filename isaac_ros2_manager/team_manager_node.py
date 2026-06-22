from __future__ import annotations

import json
import math
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
    from simulation_interfaces.srv import DeleteEntity, SpawnEntity
except Exception:
    DeleteEntity = None
    SpawnEntity = None


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


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
        self.declare_parameter("spawn_call_timeout_sec", 180.0)
        self.declare_parameter("spawn_telemetry_stale_timeout_sec", 10.0)
        self.declare_parameter("spawn_confirm_settle_sec", 8.0)
        self.declare_parameter("spawn_motion_probe_enabled", True)
        self.declare_parameter("spawn_motion_probe_duration_sec", 3.0)
        self.declare_parameter("spawn_motion_probe_linear_x", 0.35)
        self.declare_parameter("spawn_motion_probe_min_delta_m", 0.05)
        self.declare_parameter("uav_vehicle_id_base", env_int("AERO_VEHICLE_ID", 0))

        team_json = self._resolve_team_json(str(self.get_parameter("team_json").value))
        with open(team_json, "r", encoding="utf-8") as f:
            self.team_dict = json.load(f)
        self.team_json = team_json
        self.team_name = self.team_dict["name"]
        self.agents = self.team_dict.get("agents", {})
        self.uav_vehicle_id_base = max(0, int(self.get_parameter("uav_vehicle_id_base").value))
        self.uav_vehicle_ids = self._build_uav_vehicle_ids()

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
        self.spawn_telemetry_stale_timeout_sec = max(
            1.0,
            float(self.get_parameter("spawn_telemetry_stale_timeout_sec").value),
        )
        self.spawn_confirm_settle_sec = max(0.0, float(self.get_parameter("spawn_confirm_settle_sec").value))
        self.spawn_motion_probe_enabled = param_bool(self.get_parameter("spawn_motion_probe_enabled").value)
        self.spawn_motion_probe_duration_sec = max(
            0.2,
            float(self.get_parameter("spawn_motion_probe_duration_sec").value),
        )
        self.spawn_motion_probe_linear_x = max(
            0.05,
            float(self.get_parameter("spawn_motion_probe_linear_x").value),
        )
        self.spawn_motion_probe_min_delta_m = max(
            0.0,
            float(self.get_parameter("spawn_motion_probe_min_delta_m").value),
        )
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
        self.spawn_first_telemetry_monotonic: dict[str, float] = {}
        self.spawn_motion_probed_agents: set[str] = set()
        self.spawn_motion_probe_active_agents: set[str] = set()
        self.spawn_retry_not_before_monotonic: dict[str, float] = {}
        self.last_spawn_wait_log_monotonic = 0.0
        self.agent_spawn_wait_log_monotonic: dict[str, float] = {}

        self.spawn_client = None
        self.remove_client = None
        if SpawnEntity is not None and self.spawn_with_isaac:
            self.spawn_client = self.create_client(SpawnEntity, "/world_manager/add", callback_group=self.cb_group)
            if DeleteEntity is not None:
                self.remove_client = self.create_client(
                    DeleteEntity,
                    "/world_manager/remove",
                    callback_group=self.cb_group,
                )
        elif self.spawn_with_isaac:
            self.get_logger().warning("simulation_interfaces/srv/SpawnEntity is unavailable; using bridge-only mode")
        if (
            self.spawn_with_isaac
            and self.spawn_motion_probe_enabled
            and self.remove_client is None
            and DeleteEntity is None
        ):
            self.get_logger().warning(
                "simulation_interfaces/srv/DeleteEntity is unavailable; Carter motion probe cannot remove bad spawns"
            )

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
            f" spawn_with_isaac={self.spawn_with_isaac}; geo_origin={self.has_geo_origin};"
            f" uav_vehicle_ids={self.uav_vehicle_ids}"
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
                isaac_cmd_vel_topic=join_name(ns, "cmd_vel"),
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

    def _build_uav_vehicle_ids(self) -> dict[str, int]:
        vehicle_ids: dict[str, int] = {}
        for agent_name, agent in self.agents.items():
            if resolve_proto(agent).kind == "uav":
                vehicle_ids[agent_name] = self.uav_vehicle_id_base + len(vehicle_ids)
        return vehicle_ids

    def _spawn_team_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self.spawn_requested = True
        reset_agents, waiting_agents = self._reset_stale_unconfirmed_spawn_state()
        self._ensure_spawn_retry_timer()
        self._spawn_missing_agents()
        response.success = True
        response.message = (
            f"Spawn requested for team {self.team_name}; reset stale agents={reset_agents}; "
            f"already waiting={waiting_agents}; "
            "bridge topics are already active."
        )
        return response

    def _reset_stale_unconfirmed_spawn_state(self) -> tuple[list[str], list[str]]:
        reset_agents: list[str] = []
        waiting_agents: list[str] = []
        now = time.monotonic()
        for agent_name in self.agents:
            if self._agent_confirmed(agent_name, now=now) and self._agent_motion_confirmed(agent_name):
                self.confirmed_agents.add(agent_name)
                continue

            call_started = self.spawn_call_monotonic.get(agent_name)
            if agent_name in self.pending_spawn_agents and (
                call_started is None or now - call_started < self.spawn_call_timeout_sec
            ):
                waiting_agents.append(agent_name)
                continue

            ack_started = self.spawn_ack_monotonic.get(agent_name)
            if agent_name in self.spawned_agents and (
                ack_started is None or now - ack_started < self.spawn_confirm_timeout_sec
            ):
                waiting_agents.append(agent_name)
                continue

            reset_agents.append(agent_name)
            self._clear_agent_spawn_tracking(agent_name, clear_attempts=True)
        if reset_agents:
            self.get_logger().warning(
                f"Resetting stale Isaac spawn retry state for unconfirmed agents: {reset_agents}"
            )
        if waiting_agents:
            self.get_logger().info(
                f"Keeping active Isaac spawn waits for unconfirmed agents: {waiting_agents}"
            )
        return reset_agents, waiting_agents

    def _ensure_spawn_retry_timer(self) -> None:
        if self.spawn_retry_timer is None:
            self.spawn_retry_timer = self.create_timer(
                self.spawn_retry_period_sec,
                self._spawn_missing_agents,
                callback_group=self.cb_group,
            )
            return
        self.spawn_retry_timer.reset()

    def _clear_agent_spawn_tracking(
        self,
        agent_name: str,
        *,
        clear_attempts: bool = False,
        clear_retry_delay: bool = True,
        clear_bridge_telemetry: bool = True,
    ) -> None:
        self.confirmed_agents.discard(agent_name)
        self.pending_spawn_agents.discard(agent_name)
        self.spawned_agents.discard(agent_name)
        if clear_attempts:
            self.spawn_attempts.pop(agent_name, None)
        self.spawn_ack_monotonic.pop(agent_name, None)
        self.spawn_call_monotonic.pop(agent_name, None)
        self.spawn_first_telemetry_monotonic.pop(agent_name, None)
        self.spawn_motion_probed_agents.discard(agent_name)
        self.spawn_motion_probe_active_agents.discard(agent_name)
        if clear_retry_delay:
            self.spawn_retry_not_before_monotonic.pop(agent_name, None)
        self.agent_spawn_wait_log_monotonic.pop(agent_name, None)

        if not clear_bridge_telemetry:
            return
        bridge = self.bridges.get(agent_name)
        if bridge is None:
            return
        try:
            bridge.last_odom_monotonic = 0.0
            bridge._native_odom_origin = None
        except Exception:
            pass

    def _agent_confirmed(self, agent_name: str, *, now: float | None = None) -> bool:
        """True once the agent is publishing real Isaac telemetry.

        ``last_odom_monotonic`` only advances when the bridge receives an actual
        Isaac odom/pose message (synthetic integration never touches it), so a
        recent positive value means the spawned entity really exists in the sim
        -- a far stronger signal than the fire-and-forget /world_manager/add OK
        response. The freshness check lets the manager recover after Isaac Sim is
        restarted while this ROS launch keeps running.
        """
        bridge = self.bridges.get(agent_name)
        last_odom = float(getattr(bridge, "last_odom_monotonic", 0.0) if bridge is not None else 0.0)
        if last_odom <= 0.0:
            self.spawn_first_telemetry_monotonic.pop(agent_name, None)
            return False
        if now is None:
            now = time.monotonic()
        if now - last_odom > self.spawn_telemetry_stale_timeout_sec:
            self.spawn_first_telemetry_monotonic.pop(agent_name, None)
            return False
        if agent_name in self.confirmed_agents:
            return True
        first_telemetry = self.spawn_first_telemetry_monotonic.setdefault(agent_name, last_odom)
        return now - first_telemetry >= self.spawn_confirm_settle_sec

    def _agent_motion_confirmed(self, agent_name: str) -> bool:
        if not self._requires_motion_probe(agent_name):
            return True
        if agent_name in self.spawn_motion_probed_agents:
            return True
        if agent_name in self.spawn_motion_probe_active_agents:
            self._log_agent_spawn_wait(
                agent_name,
                time.monotonic(),
                f"Waiting for Carter drive probe to finish for {self.team_name}/{agent_name}",
            )
            return False
        return self._probe_carter_motion(agent_name)

    def _requires_motion_probe(self, agent_name: str) -> bool:
        if not self.spawn_motion_probe_enabled:
            return False
        agent = self.agents.get(agent_name, {})
        spec = resolve_proto(agent)
        return spec.kind == "ugv" and spec.uri == "carter"

    def _probe_carter_motion(self, agent_name: str) -> bool:
        bridge = self.bridges.get(agent_name)
        if bridge is None or getattr(bridge, "cmd_vel_pub", None) is None:
            self.get_logger().warning(
                f"Skipping Carter motion probe for {self.team_name}/{agent_name}; bridge publisher unavailable"
            )
            return True

        start = bridge.current_odom.pose.pose.position
        start_x = float(start.x)
        start_y = float(start.y)
        probe_twist = Twist()
        probe_twist.linear.x = self.spawn_motion_probe_linear_x
        duration = self.spawn_motion_probe_duration_sec

        self.spawn_motion_probe_active_agents.add(agent_name)
        try:
            self.get_logger().info(
                f"Validating Carter drive for {self.team_name}/{agent_name}: "
                f"{duration:.1f}s cmd_vel probe at {self.spawn_motion_probe_linear_x:.2f} m/s"
            )
            deadline = time.monotonic() + duration
            while rclpy.ok() and time.monotonic() < deadline:
                bridge._publish_cmd_vel(probe_twist)
                time.sleep(0.05)
            bridge._publish_cmd_vel(Twist())
            time.sleep(0.2)
        except Exception:
            self.spawn_motion_probe_active_agents.discard(agent_name)
            raise

        end = bridge.current_odom.pose.pose.position
        delta_xy = math.hypot(float(end.x) - start_x, float(end.y) - start_y)
        if delta_xy >= self.spawn_motion_probe_min_delta_m:
            self.spawn_motion_probed_agents.add(agent_name)
            self.spawn_motion_probe_active_agents.discard(agent_name)
            self.get_logger().info(
                f"Carter drive validated for {self.team_name}/{agent_name}: "
                f"delta_xy={delta_xy:.3f}m"
            )
            return True

        self.get_logger().warning(
            f"Carter drive probe failed for {self.team_name}/{agent_name}: "
            f"delta_xy={delta_xy:.3f}m < {self.spawn_motion_probe_min_delta_m:.3f}m; "
            "removing bad spawn and retrying"
        )
        self._remove_isaac_agent(agent_name)
        self._clear_agent_spawn_tracking(
            agent_name,
            clear_attempts=False,
            clear_retry_delay=False,
            clear_bridge_telemetry=True,
        )
        self.spawn_retry_not_before_monotonic[agent_name] = (
            time.monotonic() + max(1.0, self.spawn_retry_period_sec)
        )
        return False

    def _remove_isaac_agent(self, agent_name: str) -> bool:
        if self.remove_client is None or DeleteEntity is None:
            self.get_logger().warning(
                f"Cannot remove bad Isaac spawn for {self.team_name}/{agent_name}; "
                "/world_manager/remove is unavailable"
            )
            return False
        if not self.remove_client.service_is_ready():
            try:
                self.remove_client.wait_for_service(timeout_sec=1.0)
            except Exception:
                pass
        if not self.remove_client.service_is_ready():
            self.get_logger().warning(
                f"Cannot remove bad Isaac spawn for {self.team_name}/{agent_name}; "
                "/world_manager/remove is not ready"
            )
            return False

        request = DeleteEntity.Request()
        request.entity = f"{self.team_name}_{agent_name}"
        response, error = self._wait_future(
            self.remove_client.call_async(request),
            timeout_sec=max(2.0, self.spawn_retry_period_sec),
        )
        if response is None:
            self.get_logger().warning(
                f"Delete request for {self.team_name}/{agent_name} did not complete: {error}"
            )
            return False

        result = getattr(response, "result", None)
        code = int(getattr(result, "result", 0))
        ok_code = int(getattr(result, "RESULT_OK", 1))
        if code == ok_code:
            self.get_logger().info(f"Delete accepted for bad Isaac spawn {self.team_name}/{agent_name}")
            return True

        self.get_logger().warning(
            f"Delete request failed for bad Isaac spawn {self.team_name}/{agent_name}: "
            f"{getattr(result, 'error_message', '')}"
        )
        return False

    def _spawn_missing_agents(self) -> None:
        if not self.spawn_with_isaac or self.spawn_client is None:
            return

        now = time.monotonic()
        self._drop_stale_confirmed_agents(now)

        # Promote agents to "confirmed" once their telemetry is live.
        for agent_name in self.agents:
            if (
                agent_name not in self.confirmed_agents
                and self._agent_confirmed(agent_name, now=now)
                and self._agent_motion_confirmed(agent_name)
            ):
                self.confirmed_agents.add(agent_name)
                self.agent_spawn_wait_log_monotonic.pop(agent_name, None)
                self.spawn_retry_not_before_monotonic.pop(agent_name, None)
                self.get_logger().info(
                    f"Confirmed Isaac agent {self.team_name}/{agent_name} (odom and drive active)"
                )
            elif agent_name not in self.confirmed_agents and agent_name in self.spawn_first_telemetry_monotonic:
                first_telemetry = self.spawn_first_telemetry_monotonic[agent_name]
                self._log_agent_spawn_wait(
                    agent_name,
                    now,
                    f"Waiting for {self._agent_kind_label(agent_name)} telemetry to settle for "
                    f"{self.team_name}/{agent_name}; "
                    f"{now - first_telemetry:.0f}/{self.spawn_confirm_settle_sec:.0f}s after first odom",
                )

        if len(self.confirmed_agents) == len(self.agents):
            # Keep the timer alive after initial success. If Isaac Sim is
            # restarted while the rest of the stack stays up, the next timer tick
            # will notice stale telemetry and re-request the missing entity.
            return

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
            retry_not_before = self.spawn_retry_not_before_monotonic.get(agent_name)
            if retry_not_before is not None:
                if now < retry_not_before:
                    self._log_agent_spawn_wait(
                        agent_name,
                        now,
                        f"Waiting to retry {self._agent_kind_label(agent_name)} spawn for "
                        f"{self.team_name}/{agent_name}; delete is settling for "
                        f"{retry_not_before - now:.1f}s",
                    )
                    continue
                self.spawn_retry_not_before_monotonic.pop(agent_name, None)
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
            self.spawn_first_telemetry_monotonic.pop(agent_name, None)
            self.spawn_motion_probed_agents.discard(agent_name)
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

    def _drop_stale_confirmed_agents(self, now: float) -> None:
        stale_agents = [
            agent_name
            for agent_name in list(self.confirmed_agents)
            if not self._agent_confirmed(agent_name, now=now)
        ]
        for agent_name in stale_agents:
            bridge = self.bridges.get(agent_name)
            last_odom = float(getattr(bridge, "last_odom_monotonic", 0.0) if bridge is not None else 0.0)
            age = now - last_odom if last_odom > 0.0 else float("inf")
            self._clear_agent_spawn_tracking(agent_name)
            self.get_logger().warning(
                f"Lost Isaac telemetry for {self.team_name}/{agent_name} "
                f"(last odom {age:.1f}s ago); re-requesting spawn"
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
        request.allow_renaming = False
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
            backend.setdefault("vehicle_id", self.uav_vehicle_ids.get(agent_name, index))
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
            payload.setdefault("cmd_vel_topic", join_name(namespace, "cmd_vel"))

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
