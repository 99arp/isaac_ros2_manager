from __future__ import annotations

import time

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class CmdVelStampedToTwistBridge(Node):
    """Forward Webots simple-controller TwistStamped commands as plain Twist."""

    def __init__(self) -> None:
        super().__init__("cmd_vel_stamped_to_twist_bridge")

        self.declare_parameter("source_topic", "")
        self.declare_parameter("target_topic", "")
        self.declare_parameter("log_period_sec", 5.0)

        self.source_topic = str(self.get_parameter("source_topic").value or "").strip()
        self.target_topic = str(self.get_parameter("target_topic").value or "").strip()
        self.log_period_sec = max(0.0, float(self.get_parameter("log_period_sec").value))
        self.last_log_time = 0.0

        if not self.source_topic:
            raise ValueError("source_topic must be set")
        if not self.target_topic:
            raise ValueError("target_topic must be set")

        self.publisher = self.create_publisher(Twist, self.target_topic, 10)
        self.create_subscription(TwistStamped, self.source_topic, self._cmd_vel_cb, 10)

        self.get_logger().info(
            f"Bridging {self.source_topic} TwistStamped -> {self.target_topic} Twist"
        )

    def _cmd_vel_cb(self, msg: TwistStamped) -> None:
        out = Twist()
        out.linear = msg.twist.linear
        out.angular = msg.twist.angular
        self.publisher.publish(out)

        if self.log_period_sec <= 0.0:
            return
        now = time.monotonic()
        if now - self.last_log_time >= self.log_period_sec:
            self.last_log_time = now
            self.get_logger().info(
                "Forwarded cmd_vel "
                f"linear.x={out.linear.x:.3f} angular.z={out.angular.z:.3f}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelStampedToTwistBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
