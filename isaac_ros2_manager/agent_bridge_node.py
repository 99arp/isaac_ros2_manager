from __future__ import annotations

import math
import time
from dataclasses import dataclass

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, Float32MultiArray
from std_srvs.srv import Trigger

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


@dataclass
class AgentBridgeConfig:
    agent_namespace: str
    kind: str = "ugv"
    isaac_cmd_vel_topic: str = ""
    isaac_odom_topics: tuple[str, ...] = ()
    isaac_pose_topics: tuple[str, ...] = ()
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

        odom_topics = config.isaac_odom_topics or (join_name(self.agent_ns, "chassis", "odom"),)
        self.isaac_odom_topics = tuple(dict.fromkeys(odom_topics))
        pose_topics = config.isaac_pose_topics or (join_name(self.agent_ns, "pose"),)
        self.isaac_pose_topics = tuple(dict.fromkeys(pose_topics))

        self.current_odom = self._make_odom(config.initial_x, config.initial_y, config.initial_z, config.initial_yaw)
        self.last_odom_monotonic = 0.0
        self.last_step_monotonic = time.monotonic()
        self.last_cmd = Twist()
        self.last_cmd_monotonic = 0.0
        self.is_flying = config.kind == "uav"

        self.cmd_vel_pub = None
        if self.isaac_cmd_vel_topic != self.webots_cmd_vel_topic:
            self.cmd_vel_pub = node.create_publisher(Twist, self.isaac_cmd_vel_topic, 10)
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
        self.pose_pub = node.create_publisher(PoseStamped, self.pose_output_topic, 10)
        self.gps_pub = node.create_publisher(NavSatFix, self.gps_output_topic, 10)
        self.is_flying_pub = node.create_publisher(Bool, self.is_flying_topic, 10)

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

        period = 1.0 / max(config.publish_rate_hz, 1.0)
        self.timer = node.create_timer(period, self._timer_cb, callback_group=self.cb_group)
        node.get_logger().info(
            f"[isaac bridge] {self.agent_ns}: cmd_vel {self.webots_cmd_vel_topic}"
            f" -> {self.isaac_cmd_vel_topic}; odom sources={list(self.isaac_odom_topics)};"
            f" outputs={self.odom_output_topic},{self.odometry_output_topic},{self.odom_matcher_output_topic},"
            f"{self.pose_output_topic},{self.gps_output_topic};"
            f" set_target={self.set_target_topic}"
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

    def _odom_cb(self, msg: Odometry, source: str) -> None:
        self.current_odom = msg
        self.last_odom_monotonic = time.monotonic()
        self._publish_state(msg)

    def _pose_cb(self, msg: PoseStamped, source: str) -> None:
        odom = Odometry()
        odom.header = msg.header
        odom.header.frame_id = msg.header.frame_id or "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose = msg.pose
        self.current_odom = odom
        self.last_odom_monotonic = time.monotonic()
        self._publish_state(odom)

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
        stop = Float32MultiArray()
        cur = self.current_odom.pose.pose.position
        stop.data = [float(cur.x), float(cur.y), float(cur.z)]
        self._safe_publish(self.set_target_pub, stop)
        return CancelResponse.ACCEPT

    def _navigate_cb(self, goal_handle):
        goal = goal_handle.request.pose
        if not goal.header.frame_id:
            goal.header.frame_id = "map"
        self._publish_target(goal)

        start = time.monotonic()
        result = NavigateToPose.Result()
        feedback = NavigateToPose.Feedback()
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return result
            self._publish_target(goal)
            feedback.current_pose = self.current_pose()
            feedback.distance_remaining = float(distance_xy(self.current_odom.pose.pose.position, goal.pose.position))
            elapsed = int(time.monotonic() - start)
            feedback.navigation_time = Duration(sec=elapsed, nanosec=0)
            feedback.estimated_time_remaining = Duration(sec=max(0, int(feedback.distance_remaining)), nanosec=0)
            goal_handle.publish_feedback(feedback)
            if feedback.distance_remaining <= self.config.goal_tolerance:
                goal_handle.succeed()
                return result
            if time.monotonic() - start > self.config.goal_timeout_sec:
                self.node.get_logger().warning(
                    f"[isaac bridge] {self.agent_ns}: navigate_to_pose timed out"
                    f" at distance {feedback.distance_remaining:.2f}"
                )
                goal_handle.abort()
                return result
            time.sleep(0.2)
        goal_handle.abort()
        return result

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
        self.is_flying = True
        if self.current_odom.pose.pose.position.z < 1.0:
            self.current_odom.pose.pose.position.z = 1.0
        response.success = True
        response.message = "takeoff accepted by Isaac compatibility bridge"
        return response

    def _land_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self.is_flying = False
        response.success = True
        response.message = "land accepted by Isaac compatibility bridge"
        return response


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

        initial_pose = param_float_array(self.get_parameter("initial_pose").value)
        while len(initial_pose) < 3:
            initial_pose.append(0.0)

        config = AgentBridgeConfig(
            agent_namespace=self.get_parameter("agent_namespace").value,
            kind=self.get_parameter("kind").value,
            isaac_cmd_vel_topic=self.get_parameter("isaac_cmd_vel_topic").value,
            isaac_odom_topics=tuple(param_string_array(self.get_parameter("isaac_odom_topics").value)),
            isaac_pose_topics=tuple(param_string_array(self.get_parameter("isaac_pose_topics").value)),
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
