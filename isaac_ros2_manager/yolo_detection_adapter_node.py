from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from .common import PlanarFrameTransform, param_bool, param_string_array, yaw_from_quat


def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("-", "_").replace(" ", "_")


@dataclass
class TrackedDetection:
    x: float
    y: float
    count: int
    published: bool
    label: str
    last_seen: float


class IsaacYoloDetectionAdapter(Node):
    """Convert Isaac YOLO bbox JSON into the Webots team-manager trap pose contract."""

    def __init__(self):
        super().__init__("isaac_yolo_detection_adapter")
        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameter("detections_topic", "/chipgt/mini1/front_camera/image_raw/detections")
        self.declare_parameter("odom_topic", "/chipgt/mini1/odometry")
        self.declare_parameter("output_topic", "/chipgt/team_manager/detected_objectives")
        self.declare_parameter("camera_fov_deg", 45.0)
        self.declare_parameter("ground_z", 0.0)
        self.declare_parameter("camera_height_m", 0.0)
        self.declare_parameter("min_confidence", 0.25)
        self.declare_parameter("min_detections", 2)
        self.declare_parameter("merge_radius_m", 1.0)
        self.declare_parameter("publish_once", True)
        self.declare_parameter("allowed_labels", "trap,traps,bear_trap,bear-trap,bear trap,landmine,mine")
        self.declare_parameter("edge_size", 1.0)
        self.declare_parameter("world_offset", "0.0,0.0")
        self.declare_parameter("nav_edge_size", 1.0)
        self.declare_parameter("nav_world_offset", "0.0,0.0")
        self.declare_parameter("output_coordinate_frame", "local")
        self.declare_parameter("frame_id", "local")

        self.camera_fov_rad = math.radians(float(self.get_parameter("camera_fov_deg").value))
        self.ground_z = float(self.get_parameter("ground_z").value)
        self.camera_height_m = float(self.get_parameter("camera_height_m").value)
        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.min_detections = max(1, int(self.get_parameter("min_detections").value))
        self.merge_radius_m = max(0.0, float(self.get_parameter("merge_radius_m").value))
        self.publish_once = param_bool(self.get_parameter("publish_once").value)
        self.allowed_labels = {
            normalize_label(label)
            for label in param_string_array(self.get_parameter("allowed_labels").value)
            if label.strip()
        }
        self.output_coordinate_frame = str(
            self.get_parameter("output_coordinate_frame").value or "local"
        ).strip().lower()
        self.frame_transform = PlanarFrameTransform.from_values(
            local_edge_size=self.get_parameter("edge_size").value,
            local_offset=self.get_parameter("world_offset").value,
            native_edge_size=self.get_parameter("nav_edge_size").value,
            native_offset=self.get_parameter("nav_world_offset").value,
        )
        self.frame_id = str(self.get_parameter("frame_id").value or self.output_coordinate_frame)
        if self._publishes_local_coordinates():
            self.merge_radius_m = self.frame_transform.native_to_local_distance(self.merge_radius_m)

        self.odom: Odometry | None = None
        self.tracked: list[TrackedDetection] = []

        detections_topic = str(self.get_parameter("detections_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)

        self.create_subscription(String, detections_topic, self._detections_cb, 10, callback_group=self.cb_group)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10, callback_group=self.cb_group)
        self.pose_pub = self.create_publisher(PoseStamped, output_topic, 10)

        self.get_logger().info(
            "Isaac YOLO detection adapter ready: "
            f"detections={detections_topic}; odom={odom_topic}; output={output_topic}; "
            f"labels={sorted(self.allowed_labels) or ['*']}; "
            f"output_frame={self.output_coordinate_frame}; "
            f"native->local scale={self.frame_transform.native_to_local_scale:.3f}"
        )

    def _odom_cb(self, msg: Odometry) -> None:
        self.odom = msg

    def _detections_cb(self, msg: String) -> None:
        if self.odom is None:
            self.get_logger().debug("Ignoring YOLO detections until odom is available")
            return

        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warning(f"Ignoring invalid YOLO detection JSON: {exc}")
            return

        width = int(payload.get("width") or 0)
        height = int(payload.get("height") or 0)
        detections = payload.get("detections") or []
        if width <= 0 or height <= 0 or not isinstance(detections, list):
            return

        for detection in detections:
            if not isinstance(detection, dict):
                continue
            label = str(detection.get("label", "")).strip()
            label_key = normalize_label(label)
            if self.allowed_labels and label_key not in self.allowed_labels:
                continue
            if float(detection.get("conf", 0.0)) < self.min_confidence:
                continue
            try:
                center_u = (float(detection["x1"]) + float(detection["x2"])) * 0.5
                center_v = (float(detection["y1"]) + float(detection["y2"])) * 0.5
            except Exception:
                continue
            native_x, native_y = self._pixel_to_world(center_u, center_v, width, height)
            x, y = self._output_xy(native_x, native_y)
            self._track_detection(x, y, label or "trap")

    def _pixel_to_world(self, u: float, v: float, width: int, height: int) -> tuple[float, float]:
        assert self.odom is not None
        pose = self.odom.pose.pose
        altitude = self.camera_height_m
        if altitude <= 0.0:
            altitude = max(0.0, float(pose.position.z) - self.ground_z)
        if altitude <= 0.0:
            altitude = 1.0

        aspect_ratio = float(width) / max(float(height), 1.0)
        fov_v_rad = 2.0 * math.atan(math.tan(self.camera_fov_rad * 0.5) / aspect_ratio)
        visible_width_m = 2.0 * altitude * math.tan(self.camera_fov_rad * 0.5)
        visible_height_m = 2.0 * altitude * math.tan(fov_v_rad * 0.5)
        m_per_px_x = visible_width_m / float(width)
        m_per_px_y = visible_height_m / float(height)

        dx_px = float(u) - width * 0.5
        dy_px = float(v) - height * 0.5

        rel_x = -dy_px * m_per_px_y
        rel_y = -dx_px * m_per_px_x

        yaw = yaw_from_quat(pose.orientation)
        rot_x = rel_x * math.cos(yaw) - rel_y * math.sin(yaw)
        rot_y = rel_x * math.sin(yaw) + rel_y * math.cos(yaw)
        return float(pose.position.x) + rot_x, float(pose.position.y) + rot_y

    def _publishes_local_coordinates(self) -> bool:
        return self.output_coordinate_frame in ("local", "webots")

    def _output_xy(self, native_x: float, native_y: float) -> tuple[float, float]:
        if self._publishes_local_coordinates():
            return self.frame_transform.native_to_local_xy(native_x, native_y)
        return float(native_x), float(native_y)

    def _track_detection(self, x: float, y: float, label: str) -> None:
        now = time.monotonic()
        match = None
        for tracked in self.tracked:
            if math.hypot(x - tracked.x, y - tracked.y) <= self.merge_radius_m:
                match = tracked
                break

        if match is None:
            match = TrackedDetection(x=x, y=y, count=0, published=False, label=label, last_seen=now)
            self.tracked.append(match)

        n = match.count
        match.x = (match.x * n + x) / float(n + 1)
        match.y = (match.y * n + y) / float(n + 1)
        match.count += 1
        match.label = label
        match.last_seen = now

        if match.count < self.min_detections:
            return
        if self.publish_once and match.published:
            return
        self._publish_detection(match)
        match.published = True

    def _publish_detection(self, detection: TrackedDetection) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(detection.x)
        msg.pose.position.y = float(detection.y)
        msg.pose.position.z = self.ground_z
        msg.pose.orientation.w = 1.0
        self.pose_pub.publish(msg)
        self.get_logger().info(
            f"Published detected objective from YOLO label={detection.label!r} "
            f"at ({detection.x:.2f}, {detection.y:.2f}) after {detection.count} detections"
        )


def main(args=None):
    rclpy.init(args=args)
    node = IsaacYoloDetectionAdapter()
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
