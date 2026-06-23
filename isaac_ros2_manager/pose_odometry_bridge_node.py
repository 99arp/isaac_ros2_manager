from __future__ import annotations

import math
from typing import Iterable

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from .common import best_effort_qos


def _split_topics(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + (float(stamp.nanosec) * 1e-9)


class PoseOdometryBridgeNode(Node):
    """Publish Webots-compatible Odometry from an Isaac PoseStamped stream."""

    def __init__(self) -> None:
        super().__init__("pose_odometry_bridge")
        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameter("pose_topic", "")
        self.declare_parameter("pose_topics", "")
        self.declare_parameter("odom_topics", "")
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")

        pose_topic = str(self.get_parameter("pose_topic").value or "").strip()
        pose_topics = _split_topics(str(self.get_parameter("pose_topics").value or ""))
        if pose_topic:
            pose_topics.insert(0, pose_topic)
        pose_topics = list(dict.fromkeys(pose_topics))
        odom_topics = _split_topics(str(self.get_parameter("odom_topics").value or ""))
        self.frame_id = str(self.get_parameter("frame_id").value or "odom")
        self.child_frame_id = str(self.get_parameter("child_frame_id").value or "base_link")

        if not pose_topics:
            raise ValueError("pose_topic or pose_topics must be set")
        if not odom_topics:
            raise ValueError("odom_topics must include at least one output topic")

        self.odom_publishers = [
            self.create_publisher(Odometry, topic, 10)
            for topic in odom_topics
        ]
        self.previous_pose_by_topic: dict[str, PoseStamped] = {}
        pose_qos = best_effort_qos()

        for topic in pose_topics:
            self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, source_topic=topic: self._pose_cb(msg, source_topic),
                pose_qos,
                callback_group=self.cb_group,
            )
        self.get_logger().info(
            f"Pose odometry bridge ready: {pose_topics} -> {odom_topics}"
        )

    def _pose_cb(self, msg: PoseStamped, source_topic: str) -> None:
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = str(msg.header.frame_id or self.frame_id)
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose = msg.pose

        previous = self.previous_pose_by_topic.get(source_topic)
        if previous is not None:
            dt = _stamp_seconds(msg.header.stamp) - _stamp_seconds(previous.header.stamp)
            if math.isfinite(dt) and dt > 1e-6:
                odom.twist.twist.linear.x = (
                    float(msg.pose.position.x) - float(previous.pose.position.x)
                ) / dt
                odom.twist.twist.linear.y = (
                    float(msg.pose.position.y) - float(previous.pose.position.y)
                ) / dt
                odom.twist.twist.linear.z = (
                    float(msg.pose.position.z) - float(previous.pose.position.z)
                ) / dt

        self.previous_pose_by_topic[source_topic] = msg
        for publisher in self.odom_publishers:
            publisher.publish(odom)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoseOdometryBridgeNode()
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
