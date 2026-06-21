from __future__ import annotations

import math
import re
import time
from copy import deepcopy
from dataclasses import dataclass

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, Float32MultiArray
from std_srvs.srv import Trigger
from webots_ros2_manager_msgs.action import ObjectiveAction
from webots_ros2_manager_msgs.srv import ObjectiveService

from .common import (
    abs_name,
    distance_xy,
    join_name,
    lat_lon_from_xy,
    param_bool,
    param_float_array,
    param_string_array,
    set_quat_from_yaw,
    yaw_from_quat,
)

try:
    from auspex_aero_msgs.action import Fly3D as AeroFly3D
    from auspex_aero_msgs.action import Land as AeroLand
    from auspex_aero_msgs.action import Takeoff as AeroTakeoff
except Exception:
    AeroFly3D = None
    AeroLand = None
    AeroTakeoff = None

try:
    from auspex_msgs.msg import PlatformState
except Exception:
    PlatformState = None


@dataclass
class AgentBridgeConfig:
    agent_namespace: str
    kind: str = "ugv"
    isaac_cmd_vel_topic: str = ""
    isaac_odom_topics: tuple[str, ...] | None = None
    isaac_pose_topics: tuple[str, ...] | None = None
    set_target_topic: str = ""
    initial_x: float = 0.0
    initial_y: float = 0.0
    initial_z: float = 0.0
    initial_yaw: float = 0.0
    origin_lat: float = 0.0
    origin_lon: float = 0.0
    origin_alt: float = 0.0
    odom_timeout_sec: float = 1.0
    synthetic_odom: bool = True
    goal_tolerance: float = 1.0
    goal_timeout_sec: float = 120.0
    publish_rate_hz: float = 20.0
    objective_service_timeout_sec: float = 5.0
    cmd_vel_linear_gain: float = 0.8
    cmd_vel_angular_gain: float = 1.8
    cmd_vel_max_linear: float = 1.0
    cmd_vel_max_angular: float = 1.2
    cmd_vel_heading_tolerance: float = 0.35
    aero_platform_id: str = ""
    aero_server_wait_sec: float = 10.0
    aero_action_timeout_sec: float = 180.0
    aero_takeoff_height_m: float = 2.5
    aero_speed_m_s: float = 3.0


class AgentBridge:
    """Make one Isaac agent speak the Webots/AUSPEX per-robot ROS contract.

    Subscribes the manager-facing cmd_vel/target topics and republishes onto the
    Isaac-native topics, while echoing Isaac odom back out as odom/pose/gps and
    serving navigate_to_pose / takeoff / land that the upstream stack expects.
    """

    def __init__(self, node: Node, config: AgentBridgeConfig):
        self.node = node
        self.config = config
        self.cb_group = ReentrantCallbackGroup()

        self.agent_ns = abs_name(config.agent_namespace)
        if not self.agent_ns:
            raise ValueError("agent_namespace must not be empty")

        default_cmd = join_name(self.agent_ns, "cmd_vel")
        default_set_target = join_name(self.agent_ns, "set_target")
        self.webots_cmd_vel_topic = default_cmd
        self.isaac_cmd_vel_topic = config.isaac_cmd_vel_topic or default_cmd
        self.set_target_topic = config.set_target_topic or default_set_target
        self.odom_output_topic = join_name(self.agent_ns, "odom")
        self.odometry_output_topic = join_name(self.agent_ns, "odometry")
        self.odom_matcher_output_topic = join_name(self.agent_ns, "odom_matcher")
        self.pose_output_topic = join_name(self.agent_ns, "pose")
        self.gps_output_topic = join_name(self.agent_ns, "gps")
        self.is_flying_topic = join_name(self.agent_ns, "isFlying")
        self.is_flying_snake_topic = join_name(self.agent_ns, "is_flying")

        odom_topics = (
            config.isaac_odom_topics
            if config.isaac_odom_topics is not None
            else (join_name(self.agent_ns, "chassis", "odom"),)
        )
        self.isaac_odom_topics = tuple(dict.fromkeys(odom_topics))
        pose_topics = (
            config.isaac_pose_topics
            if config.isaac_pose_topics is not None
            else (join_name(self.agent_ns, "pose"),)
        )
        self.isaac_pose_topics = tuple(dict.fromkeys(pose_topics))

        self.current_odom = self._make_odom(config.initial_x, config.initial_y, config.initial_z, config.initial_yaw)
        self.last_odom_monotonic = 0.0
        self.last_step_monotonic = time.monotonic()
        self.last_cmd = Twist()
        self.last_cmd_monotonic = 0.0
        self.is_uav = str(config.kind).strip().lower() == "uav"
        self._native_odom_origin = None
        self._native_odom_origin_yaw = 0.0
        self.is_flying = False
        self.aero_platform_id = config.aero_platform_id or self._safe_aero_platform_id(self.agent_ns)
        self.aero_gps: NavSatFix | None = None
        self.aero_status = ""
        self.last_aero_state_monotonic = 0.0
        self._active_aero_goal_handle = None

        self.cmd_vel_pub = None
        if self.is_uav:
            node.create_subscription(
                Twist,
                self.webots_cmd_vel_topic,
                self._cmd_vel_record_cb,
                10,
                callback_group=self.cb_group,
            )
        else:
            self.cmd_vel_pub = node.create_publisher(Twist, self.isaac_cmd_vel_topic, 10)
            if self.isaac_cmd_vel_topic != self.webots_cmd_vel_topic:
                node.create_subscription(
                    Twist,
                    self.webots_cmd_vel_topic,
                    self._cmd_vel_cb,
                    10,
                    callback_group=self.cb_group,
                )
            else:
                node.create_subscription(
                    Twist,
                    self.webots_cmd_vel_topic,
                    self._cmd_vel_record_cb,
                    10,
                    callback_group=self.cb_group,
                )

        self.set_target_pub = node.create_publisher(Float32MultiArray, self.set_target_topic, 10)
        self.goal_pose_pub = node.create_publisher(PoseStamped, join_name(self.agent_ns, "goal_pose"), 10)
        self.odom_pub = self._publisher_if_needed(Odometry, self.odom_output_topic, self.isaac_odom_topics)
        self.odometry_pub = self._publisher_if_needed(Odometry, self.odometry_output_topic, self.isaac_odom_topics)
        self.odom_matcher_pub = self._publisher_if_needed(Odometry, self.odom_matcher_output_topic, self.isaac_odom_topics)
        self.pose_pub = self._publisher_if_needed(PoseStamped, self.pose_output_topic, self.isaac_pose_topics)
        self.gps_pub = node.create_publisher(NavSatFix, self.gps_output_topic, 10)
        self.is_flying_pub = node.create_publisher(Bool, self.is_flying_topic, 10)
        self.is_flying_snake_pub = node.create_publisher(Bool, self.is_flying_snake_topic, 10)

        for topic in self.isaac_odom_topics:
            node.create_subscription(
                Odometry,
                topic,
                lambda msg, source=topic: self._odom_cb(msg, source),
                10,
                callback_group=self.cb_group,
            )
        for topic in self.isaac_pose_topics:
            if topic not in self.isaac_odom_topics:
                node.create_subscription(
                    PoseStamped,
                    topic,
                    lambda msg, source=topic: self._pose_cb(msg, source),
                    10,
                    callback_group=self.cb_group,
                )

        self.aero_takeoff_client = None
        self.aero_land_client = None
        self.aero_fly3d_client = None
        if self.is_uav and AeroTakeoff is not None and AeroLand is not None and AeroFly3D is not None:
            self.aero_takeoff_client = ActionClient(
                node,
                AeroTakeoff,
                join_name(self.aero_platform_id, "fm", "takeoff"),
                callback_group=self.cb_group,
            )
            self.aero_land_client = ActionClient(
                node,
                AeroLand,
                join_name(self.aero_platform_id, "fm", "land"),
                callback_group=self.cb_group,
            )
            self.aero_fly3d_client = ActionClient(
                node,
                AeroFly3D,
                join_name(self.aero_platform_id, "fm", "fly_3d"),
                callback_group=self.cb_group,
            )
        elif self.is_uav:
            node.get_logger().warning(
                f"[isaac bridge] {self.agent_ns}: auspex_aero_msgs unavailable;"
                " UAV movement skills cannot command AUSPEX-AERO"
            )

        if self.is_uav and PlatformState is not None:
            state_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
            )
            node.create_subscription(
                PlatformState,
                "/platform_state",
                self._aero_platform_state_cb,
                state_qos,
                callback_group=self.cb_group,
            )

        self.nav_action = ActionServer(
            node,
            NavigateToPose,
            join_name(self.agent_ns, "navigate_to_pose"),
            execute_callback=self._navigate_cb,
            cancel_callback=self._cancel_nav_cb,
            callback_group=self.cb_group,
        )
        self.takeoff_srv = node.create_service(
            Trigger,
            join_name(self.agent_ns, "takeoff"),
            self._takeoff_cb,
            callback_group=self.cb_group,
        )
        self.land_srv = node.create_service(
            Trigger,
            join_name(self.agent_ns, "land"),
            self._land_cb,
            callback_group=self.cb_group,
        )
        self.takeoff_bridge_srv = node.create_service(
            Trigger,
            join_name(self.agent_ns, "takeoff_bridge"),
            self._takeoff_cb,
            callback_group=self.cb_group,
        )
        self.detect_client = node.create_client(
            ObjectiveService,
            "/env_manager/get_objective_type",
            callback_group=self.cb_group,
        )
        self.disarm_client = node.create_client(
            ObjectiveService,
            "/env_manager/change_objective_state",
            callback_group=self.cb_group,
        )
        self.detect_action = ActionServer(
            node,
            ObjectiveAction,
            join_name(self.agent_ns, "detect"),
            execute_callback=self._detect_cb,
            callback_group=self.cb_group,
        )
        self.disarm_action = ActionServer(
            node,
            ObjectiveAction,
            join_name(self.agent_ns, "disarm"),
            execute_callback=self._disarm_cb,
            callback_group=self.cb_group,
        )

        period = 1.0 / max(config.publish_rate_hz, 1.0)
        self.timer = node.create_timer(period, self._timer_cb, callback_group=self.cb_group)
        command_detail = (
            f"AERO platform={self.aero_platform_id}"
            if self.is_uav
            else f"cmd_vel {self.webots_cmd_vel_topic} -> {self.isaac_cmd_vel_topic}"
        )
        node.get_logger().info(
            f"[isaac bridge] {self.agent_ns}: {command_detail};"
            f" odom sources={list(self.isaac_odom_topics)};"
            f" outputs={self.odom_output_topic},{self.odometry_output_topic},{self.odom_matcher_output_topic},"
            f"{self.pose_output_topic},{self.gps_output_topic};"
            f" set_target={self.set_target_topic};"
            f" skill_actions={join_name(self.agent_ns, 'navigate_to_pose')},"
            f"{join_name(self.agent_ns, 'detect')},{join_name(self.agent_ns, 'disarm')}"
        )

    def _publisher_if_needed(self, msg_type, topic: str, source_topics: tuple[str, ...]):
        if topic in source_topics:
            return None
        return self.node.create_publisher(msg_type, topic, 10)

    def _safe_publish(self, publisher, msg) -> None:
        if publisher is None or not rclpy.ok():
            return
        try:
            publisher.publish(msg)
        except Exception:
            pass

    @staticmethod
    def _safe_aero_platform_id(value: str | None) -> str:
        text = str(value or "").strip().strip("/")
        text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        if text and text[0].isdigit():
            text = f"n_{text}"
        return text or "drone"

    @staticmethod
    def _aero_key(value: str | None) -> str:
        key = re.sub(r"[^0-9A-Za-z_]+", "_", str(value or "").strip().strip("/").lower())
        return re.sub(r"_+", "_", key).strip("_")

    def _aero_platform_state_cb(self, msg) -> None:
        if self._aero_key(getattr(msg, "platform_id", "")) != self._aero_key(self.aero_platform_id):
            return
        gps_msg = NavSatFix()
        gps_msg.header = getattr(msg, "header", gps_msg.header)
        gps = getattr(msg, "platform_gps_position", None)
        if gps is not None:
            gps_msg.latitude = float(getattr(gps, "latitude", 0.0))
            gps_msg.longitude = float(getattr(gps, "longitude", 0.0))
            gps_msg.altitude = float(getattr(gps, "altitude", 0.0))
            self.aero_gps = gps_msg
            self.last_aero_state_monotonic = time.monotonic()
        status = str(getattr(msg, "platform_status", "") or "").upper()
        if status:
            self.aero_status = status
            if status == "AIRBORNE":
                self.is_flying = True
            elif status == "LANDED":
                self.is_flying = False

    def _make_odom(self, x: float, y: float, z: float, yaw: float) -> Odometry:
        msg = Odometry()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = float(z)
        set_quat_from_yaw(msg.pose.pose.orientation, float(yaw))
        return msg

    def _cmd_vel_record_cb(self, msg: Twist) -> None:
        self.last_cmd = msg
        self.last_cmd_monotonic = time.monotonic()

    def _cmd_vel_cb(self, msg: Twist) -> None:
        self._cmd_vel_record_cb(msg)
        self._safe_publish(self.cmd_vel_pub, msg)

    def _publish_cmd_vel(self, msg: Twist) -> None:
        self.last_cmd = msg
        self.last_cmd_monotonic = time.monotonic()
        self._safe_publish(self.cmd_vel_pub, msg)

    def _odom_cb(self, msg: Odometry, source: str) -> None:
        odom = self._rebase_isaac_odom(msg, source)
        self.current_odom = odom
        self.last_odom_monotonic = time.monotonic()
        self._publish_state(odom)

    def _pose_cb(self, msg: PoseStamped, source: str) -> None:
        odom = Odometry()
        odom.header = msg.header
        odom.header.frame_id = msg.header.frame_id or "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose = msg.pose
        self.current_odom = odom
        self.last_odom_monotonic = time.monotonic()
        self._publish_state(odom)

    def _rebase_isaac_odom(self, msg: Odometry, source: str) -> Odometry:
        """Convert Isaac Carter's local odom into the mission/world frame."""
        if self.is_uav or source not in self.isaac_odom_topics:
            return msg

        if self._native_odom_origin is None:
            self._native_odom_origin = deepcopy(msg.pose.pose.position)
            self._native_odom_origin_yaw = yaw_from_quat(msg.pose.pose.orientation)
            self.node.get_logger().info(
                f"[isaac bridge] {self.agent_ns}: rebasing Isaac odom source {source} "
                f"from local origin x={self._native_odom_origin.x:.3f}, "
                f"y={self._native_odom_origin.y:.3f}, yaw={self._native_odom_origin_yaw:.3f} "
                f"to spawn x={self.config.initial_x:.3f}, y={self.config.initial_y:.3f}, "
                f"yaw={self.config.initial_yaw:.3f}"
            )

        native = msg.pose.pose.position
        dx = float(native.x) - float(self._native_odom_origin.x)
        dy = float(native.y) - float(self._native_odom_origin.y)
        dz = float(native.z) - float(self._native_odom_origin.z)
        origin_yaw = float(self.config.initial_yaw)
        c = math.cos(origin_yaw)
        s = math.sin(origin_yaw)

        odom = deepcopy(msg)
        odom.pose.pose.position.x = float(self.config.initial_x) + c * dx - s * dy
        odom.pose.pose.position.y = float(self.config.initial_y) + s * dx + c * dy
        odom.pose.pose.position.z = float(self.config.initial_z) + dz

        native_yaw = yaw_from_quat(msg.pose.pose.orientation)
        world_yaw = origin_yaw + self._normalize_angle(native_yaw - self._native_odom_origin_yaw)
        set_quat_from_yaw(odom.pose.pose.orientation, world_yaw)

        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        odom.twist.twist.linear.x = c * vx - s * vy
        odom.twist.twist.linear.y = s * vx + c * vy
        return odom

    def _timer_cb(self) -> None:
        if not rclpy.ok():
            return
        now = time.monotonic()
        if self.config.synthetic_odom and now - self.last_odom_monotonic > self.config.odom_timeout_sec:
            self._integrate_synthetic(now)
        self._publish_state(self.current_odom)

    def _integrate_synthetic(self, now: float) -> None:
        dt = max(0.0, min(now - self.last_step_monotonic, 0.2))
        self.last_step_monotonic = now
        if now - self.last_cmd_monotonic > 1.0:
            return
        yaw = yaw_from_quat(self.current_odom.pose.pose.orientation)
        yaw += float(self.last_cmd.angular.z) * dt
        speed = float(self.last_cmd.linear.x)
        self.current_odom.pose.pose.position.x += math.cos(yaw) * speed * dt
        self.current_odom.pose.pose.position.y += math.sin(yaw) * speed * dt
        set_quat_from_yaw(self.current_odom.pose.pose.orientation, yaw)
        self.current_odom.twist.twist = self.last_cmd

    @staticmethod
    def _normalize_angle(value: float) -> float:
        return math.atan2(math.sin(value), math.cos(value))

    def _publish_state(self, odom: Odometry) -> None:
        if not rclpy.ok():
            return
        stamp = self.node.get_clock().now().to_msg()
        odom.header.stamp = stamp
        if not odom.header.frame_id:
            odom.header.frame_id = "odom"
        if not odom.child_frame_id:
            odom.child_frame_id = "base_link"

        if self.odom_pub is not None:
            self._safe_publish(self.odom_pub, odom)
        if self.odometry_pub is not None:
            self._safe_publish(self.odometry_pub, odom)
        if self.odom_matcher_pub is not None:
            self._safe_publish(self.odom_matcher_pub, odom)

        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self._safe_publish(self.pose_pub, pose)

        gps = NavSatFix()
        gps.header = odom.header
        lat, lon, alt = lat_lon_from_xy(
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            self.config.origin_lat,
            self.config.origin_lon,
            self.config.origin_alt,
            odom.pose.pose.position.z,
        )
        gps.latitude = lat
        gps.longitude = lon
        gps.altitude = alt
        self._safe_publish(self.gps_pub, gps)

        flying = Bool()
        flying.data = bool(self.is_flying)
        self._safe_publish(self.is_flying_pub, flying)
        self._safe_publish(self.is_flying_snake_pub, flying)

    def _publish_target(self, pose: PoseStamped) -> None:
        if not rclpy.ok():
            return
        msg = Float32MultiArray()
        msg.data = [
            float(pose.pose.position.x),
            float(pose.pose.position.y),
            float(pose.pose.position.z),
        ]
        self._safe_publish(self.set_target_pub, msg)
        self._safe_publish(self.goal_pose_pub, pose)

    def _cancel_nav_cb(self, goal_handle) -> CancelResponse:
        if self._active_aero_goal_handle is not None:
            try:
                self._active_aero_goal_handle.cancel_goal_async()
            except Exception:
                pass
        self._publish_cmd_vel(Twist())
        return CancelResponse.ACCEPT

    def _navigate_cb(self, goal_handle):
        goal = goal_handle.request.pose
        if not goal.header.frame_id:
            goal.header.frame_id = "map"

        if self.is_uav:
            return self._navigate_uav_with_aero(goal_handle, goal)

        start = time.monotonic()
        result = NavigateToPose.Result()
        feedback = NavigateToPose.Feedback()
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self._publish_cmd_vel(Twist())
                goal_handle.canceled()
                return result

            feedback.current_pose = self.current_pose()
            feedback.distance_remaining = float(distance_xy(self.current_odom.pose.pose.position, goal.pose.position))
            elapsed = int(time.monotonic() - start)
            feedback.navigation_time = Duration(sec=elapsed, nanosec=0)
            feedback.estimated_time_remaining = Duration(sec=max(0, int(feedback.distance_remaining)), nanosec=0)
            goal_handle.publish_feedback(feedback)
            if feedback.distance_remaining <= self.config.goal_tolerance:
                self._publish_cmd_vel(Twist())
                goal_handle.succeed()
                return result
            if time.monotonic() - start > self.config.goal_timeout_sec:
                self.node.get_logger().warning(
                    f"[isaac bridge] {self.agent_ns}: navigate_to_pose timed out"
                    f" at distance {feedback.distance_remaining:.2f}"
                )
                self._publish_cmd_vel(Twist())
                goal_handle.abort()
                return result
            self._drive_toward(goal)
            time.sleep(0.1)
        self._publish_cmd_vel(Twist())
        goal_handle.abort()
        return result

    def _navigate_uav_with_aero(self, goal_handle, goal: PoseStamped):
        result = NavigateToPose.Result()
        if self.aero_fly3d_client is None:
            self.node.get_logger().warning(
                f"[isaac bridge] {self.agent_ns}: cannot navigate UAV; AUSPEX-AERO action clients are unavailable"
            )
            goal_handle.abort()
            return result

        target_lat, target_lon, target_alt, target_z, goal_yaw_deg = self._aero_target_from_pose(goal)

        if not self.is_flying:
            takeoff_height = max(
                self.config.aero_takeoff_height_m,
                target_z,
                float(self.current_odom.pose.pose.position.z),
            )
            ok, message = self._run_aero_action_blocking(
                self.aero_takeoff_client,
                self._aero_takeoff_goal(takeoff_height),
                "takeoff",
                timeout_sec=max(self.config.aero_action_timeout_sec, 180.0),
            )
            if ok:
                self.is_flying = True
            else:
                self.node.get_logger().warning(
                    f"[isaac bridge] {self.agent_ns}: AERO takeoff before navigate failed: {message};"
                    " trying fly_3d anyway"
                )

        fly_goal = AeroFly3D.Goal()
        fly_goal.target_lat_deg = float(target_lat)
        fly_goal.target_lon_deg = float(target_lon)
        fly_goal.target_alt_amsl_m = float(target_alt)
        fly_goal.speed_m_s = float(max(0.1, self.config.aero_speed_m_s))
        fly_goal.heading_offset_deg = 0.0
        fly_goal.goal_yaw_deg = float(goal_yaw_deg)

        aero_goal_handle, error = self._send_aero_goal(self.aero_fly3d_client, fly_goal, "fly_3d")
        if aero_goal_handle is None:
            self.node.get_logger().warning(f"[isaac bridge] {self.agent_ns}: AERO fly_3d failed: {error}")
            goal_handle.abort()
            return result

        self._active_aero_goal_handle = aero_goal_handle
        result_future = aero_goal_handle.get_result_async()
        start = time.monotonic()
        feedback = NavigateToPose.Feedback()
        timeout_sec = max(self.config.goal_timeout_sec, self.config.aero_action_timeout_sec)
        try:
            while rclpy.ok() and not result_future.done():
                if goal_handle.is_cancel_requested:
                    try:
                        aero_goal_handle.cancel_goal_async()
                    except Exception:
                        pass
                    goal_handle.canceled()
                    return result

                feedback.current_pose = self.current_pose()
                feedback.distance_remaining = float(distance_xy(self.current_odom.pose.pose.position, goal.pose.position))
                elapsed = int(time.monotonic() - start)
                feedback.navigation_time = Duration(sec=elapsed, nanosec=0)
                feedback.estimated_time_remaining = Duration(
                    sec=max(0, int(feedback.distance_remaining / max(self.config.aero_speed_m_s, 0.1))),
                    nanosec=0,
                )
                goal_handle.publish_feedback(feedback)

                if time.monotonic() - start > timeout_sec:
                    try:
                        aero_goal_handle.cancel_goal_async()
                    except Exception:
                        pass
                    self.node.get_logger().warning(
                        f"[isaac bridge] {self.agent_ns}: AERO fly_3d timed out"
                        f" at distance {feedback.distance_remaining:.2f}"
                    )
                    goal_handle.abort()
                    return result
                time.sleep(0.1)

            result_response, error = self._wait_future(result_future, 1.0)
        finally:
            self._active_aero_goal_handle = None

        if result_response is None:
            self.node.get_logger().warning(f"[isaac bridge] {self.agent_ns}: AERO fly_3d result failed: {error}")
            goal_handle.abort()
            return result

        aero_result = getattr(result_response, "result", None)
        if bool(getattr(aero_result, "success", False)):
            self.is_flying = True
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def _aero_target_from_pose(self, goal: PoseStamped) -> tuple[float, float, float, float, float]:
        target_z = float(goal.pose.position.z)
        if target_z <= 0.1:
            target_z = max(float(self.current_odom.pose.pose.position.z), self.config.aero_takeoff_height_m)

        if self.aero_gps is not None and self.last_odom_monotonic > 0.0:
            cur = self.current_odom.pose.pose.position
            dx = float(goal.pose.position.x) - float(cur.x)
            dy = float(goal.pose.position.y) - float(cur.y)
            dz = target_z - float(cur.z)
            lat, lon, alt = self._gps_from_delta(
                self.aero_gps.latitude,
                self.aero_gps.longitude,
                self.aero_gps.altitude,
                dx,
                dy,
                dz,
            )
        else:
            lat, lon, alt = lat_lon_from_xy(
                float(goal.pose.position.x),
                float(goal.pose.position.y),
                self.config.origin_lat,
                self.config.origin_lon,
                self.config.origin_alt,
                target_z,
            )

        goal_yaw_deg = math.degrees(yaw_from_quat(goal.pose.orientation))
        return lat, lon, alt, target_z, goal_yaw_deg

    @staticmethod
    def _gps_from_delta(
        latitude: float,
        longitude: float,
        altitude: float,
        east_m: float,
        north_m: float,
        up_m: float,
    ) -> tuple[float, float, float]:
        meters_per_deg_lat = 111_320.0
        lat = float(latitude) + float(north_m) / meters_per_deg_lat
        lon_scale = meters_per_deg_lat * max(math.cos(math.radians(float(latitude))), 0.01)
        lon = float(longitude) + float(east_m) / lon_scale
        return lat, lon, float(altitude) + float(up_m)

    def _aero_takeoff_goal(self, height_agl_m: float):
        goal = AeroTakeoff.Goal()
        goal.height_agl_m = float(max(0.1, height_agl_m))
        return goal

    def _aero_land_goal(self):
        return AeroLand.Goal()

    def _send_aero_goal(self, client, goal, label: str):
        if client is None:
            return None, f"{label} action client is unavailable"
        if not client.wait_for_server(timeout_sec=self.config.aero_server_wait_sec):
            return None, f"{label} action server is unavailable for platform {self.aero_platform_id}"
        future = client.send_goal_async(goal)
        aero_goal_handle, error = self._wait_future(future, self.config.aero_server_wait_sec)
        if aero_goal_handle is None:
            return None, error or f"{label} goal send failed"
        if not getattr(aero_goal_handle, "accepted", False):
            return None, f"{label} goal rejected"
        return aero_goal_handle, ""

    def _run_aero_action_blocking(self, client, goal, label: str, timeout_sec: float | None = None) -> tuple[bool, str]:
        aero_goal_handle, error = self._send_aero_goal(client, goal, label)
        if aero_goal_handle is None:
            return False, error
        result_response, error = self._wait_future(
            aero_goal_handle.get_result_async(),
            timeout_sec if timeout_sec is not None else self.config.aero_action_timeout_sec,
        )
        if result_response is None:
            return False, error or f"{label} result timed out"
        aero_result = getattr(result_response, "result", None)
        if bool(getattr(aero_result, "success", False)):
            return True, "success"
        return False, f"{label} returned success=false"

    def _angle_error(self, target: float, current: float) -> float:
        return math.atan2(math.sin(target - current), math.cos(target - current))

    def _drive_toward(self, goal: PoseStamped) -> None:
        cur = self.current_odom.pose.pose
        dx = float(goal.pose.position.x) - float(cur.position.x)
        dy = float(goal.pose.position.y) - float(cur.position.y)
        distance = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        current_yaw = yaw_from_quat(cur.orientation)
        yaw_error = self._angle_error(desired_yaw, current_yaw)

        cmd = Twist()
        cmd.angular.z = max(
            -self.config.cmd_vel_max_angular,
            min(self.config.cmd_vel_max_angular, self.config.cmd_vel_angular_gain * yaw_error),
        )
        if abs(yaw_error) <= self.config.cmd_vel_heading_tolerance:
            cmd.linear.x = max(
                0.0,
                min(self.config.cmd_vel_max_linear, self.config.cmd_vel_linear_gain * distance),
            )
        self._publish_cmd_vel(cmd)

    def current_pose(self) -> PoseStamped:
        msg = PoseStamped()
        msg.header = self.current_odom.header
        msg.pose = self.current_odom.pose.pose
        return msg

    def current_gps(self) -> NavSatFix:
        msg = NavSatFix()
        cur = self.current_odom.pose.pose.position
        msg.header = self.current_odom.header
        lat, lon, alt = lat_lon_from_xy(
            cur.x,
            cur.y,
            self.config.origin_lat,
            self.config.origin_lon,
            self.config.origin_alt,
            cur.z,
        )
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = alt
        return msg

    def _takeoff_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self.is_uav and self.aero_takeoff_client is not None:
            height = max(
                self.config.aero_takeoff_height_m,
                float(self.current_odom.pose.pose.position.z),
            )
            ok, message = self._run_aero_action_blocking(
                self.aero_takeoff_client,
                self._aero_takeoff_goal(height),
                "takeoff",
                timeout_sec=max(self.config.aero_action_timeout_sec, 180.0),
            )
            response.success = ok
            response.message = f"AERO takeoff {message}"
            if ok:
                self.is_flying = True
            return response

        self.is_flying = True
        if self.current_odom.pose.pose.position.z < 1.0:
            self.current_odom.pose.pose.position.z = 1.0
        response.success = True
        response.message = "takeoff accepted by Isaac compatibility bridge"
        return response

    def _land_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self.is_uav and self.aero_land_client is not None:
            ok, message = self._run_aero_action_blocking(
                self.aero_land_client,
                self._aero_land_goal(),
                "land",
                timeout_sec=max(self.config.aero_action_timeout_sec, 180.0),
            )
            response.success = ok
            response.message = f"AERO land {message}"
            if ok:
                self.is_flying = False
            return response

        self.is_flying = False
        response.success = True
        response.message = "land accepted by Isaac compatibility bridge"
        return response

    def _objective_request(self, goal) -> ObjectiveService.Request:
        request = ObjectiveService.Request()
        request.position = goal.position
        request.name = goal.name
        return request

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

    def _call_objective_service(self, client, request: ObjectiveService.Request, service_name: str):
        if not client.service_is_ready() and not client.wait_for_service(timeout_sec=0.5):
            return None, f"{service_name} is unavailable"
        future = client.call_async(request)
        return self._wait_future(future, self.config.objective_service_timeout_sec)

    def _objective_action_cb(self, goal_handle, client, service_name: str):
        goal = goal_handle.request
        response, error = self._call_objective_service(client, self._objective_request(goal), service_name)
        result = ObjectiveAction.Result()
        if response is None:
            result.success = False
            result.message = error or f"{service_name} failed"
            goal_handle.abort()
            return result

        result.success = bool(response.success)
        result.message = response.message
        result.type = response.type
        if response.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def _detect_cb(self, goal_handle):
        return self._objective_action_cb(goal_handle, self.detect_client, "/env_manager/get_objective_type")

    def _disarm_cb(self, goal_handle):
        return self._objective_action_cb(goal_handle, self.disarm_client, "/env_manager/change_objective_state")


class IsaacAgentBridgeNode(Node):
    def __init__(self):
        super().__init__("isaac_agent_bridge")
        self.declare_parameter("agent_namespace", "/isaac/irobot2")
        self.declare_parameter("kind", "ugv")
        self.declare_parameter("isaac_cmd_vel_topic", "")
        self.declare_parameter("isaac_odom_topics", [])
        self.declare_parameter("isaac_pose_topics", [])
        self.declare_parameter("set_target_topic", "")
        self.declare_parameter("initial_pose", [0.0, 0.0, 0.0])
        self.declare_parameter("initial_yaw", 0.0)
        self.declare_parameter("origin_lat", 0.0)
        self.declare_parameter("origin_lon", 0.0)
        self.declare_parameter("origin_alt", 0.0)
        self.declare_parameter("odom_timeout_sec", 1.0)
        self.declare_parameter("synthetic_odom", True)
        self.declare_parameter("goal_tolerance", 1.0)
        self.declare_parameter("goal_timeout_sec", 120.0)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("objective_service_timeout_sec", 5.0)
        self.declare_parameter("cmd_vel_linear_gain", 0.8)
        self.declare_parameter("cmd_vel_angular_gain", 1.8)
        self.declare_parameter("cmd_vel_max_linear", 1.0)
        self.declare_parameter("cmd_vel_max_angular", 1.2)
        self.declare_parameter("cmd_vel_heading_tolerance", 0.35)
        self.declare_parameter("aero_platform_id", "")
        self.declare_parameter("aero_server_wait_sec", 10.0)
        self.declare_parameter("aero_action_timeout_sec", 180.0)
        self.declare_parameter("aero_takeoff_height_m", 2.5)
        self.declare_parameter("aero_speed_m_s", 3.0)

        initial_pose = param_float_array(self.get_parameter("initial_pose").value)
        while len(initial_pose) < 3:
            initial_pose.append(0.0)

        config = AgentBridgeConfig(
            agent_namespace=self.get_parameter("agent_namespace").value,
            kind=self.get_parameter("kind").value,
            isaac_cmd_vel_topic=self.get_parameter("isaac_cmd_vel_topic").value,
            isaac_odom_topics=tuple(param_string_array(self.get_parameter("isaac_odom_topics").value)) or None,
            isaac_pose_topics=tuple(param_string_array(self.get_parameter("isaac_pose_topics").value)) or None,
            set_target_topic=self.get_parameter("set_target_topic").value,
            initial_x=float(initial_pose[0]),
            initial_y=float(initial_pose[1]),
            initial_z=float(initial_pose[2]),
            initial_yaw=float(self.get_parameter("initial_yaw").value),
            origin_lat=float(self.get_parameter("origin_lat").value),
            origin_lon=float(self.get_parameter("origin_lon").value),
            origin_alt=float(self.get_parameter("origin_alt").value),
            odom_timeout_sec=float(self.get_parameter("odom_timeout_sec").value),
            synthetic_odom=param_bool(self.get_parameter("synthetic_odom").value),
            goal_tolerance=float(self.get_parameter("goal_tolerance").value),
            goal_timeout_sec=float(self.get_parameter("goal_timeout_sec").value),
            publish_rate_hz=float(self.get_parameter("publish_rate_hz").value),
            objective_service_timeout_sec=float(self.get_parameter("objective_service_timeout_sec").value),
            cmd_vel_linear_gain=float(self.get_parameter("cmd_vel_linear_gain").value),
            cmd_vel_angular_gain=float(self.get_parameter("cmd_vel_angular_gain").value),
            cmd_vel_max_linear=float(self.get_parameter("cmd_vel_max_linear").value),
            cmd_vel_max_angular=float(self.get_parameter("cmd_vel_max_angular").value),
            cmd_vel_heading_tolerance=float(self.get_parameter("cmd_vel_heading_tolerance").value),
            aero_platform_id=str(self.get_parameter("aero_platform_id").value),
            aero_server_wait_sec=float(self.get_parameter("aero_server_wait_sec").value),
            aero_action_timeout_sec=float(self.get_parameter("aero_action_timeout_sec").value),
            aero_takeoff_height_m=float(self.get_parameter("aero_takeoff_height_m").value),
            aero_speed_m_s=float(self.get_parameter("aero_speed_m_s").value),
        )
        self.bridge = AgentBridge(self, config)


def main(args=None):
    rclpy.init(args=args)
    node = IsaacAgentBridgeNode()
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
