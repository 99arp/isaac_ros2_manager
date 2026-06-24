from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class TwistRelay(Node):
    """Relay Twist commands between the namespaced Nav2 contract and Isaac."""

    def __init__(self):
        super().__init__("isaac_twist_relay")

        self.declare_parameter("input_topic", "cmd_vel")
        self.declare_parameter("output_topic", "cmd_vel")

        self.input_topic = str(self.get_parameter("input_topic").value).strip()
        self.output_topic = str(self.get_parameter("output_topic").value).strip()
        if not self.input_topic or not self.output_topic:
            raise ValueError("input_topic and output_topic must be non-empty")

        self.publisher = self.create_publisher(Twist, self.output_topic, 10)
        self.subscription = self.create_subscription(
            Twist,
            self.input_topic,
            self._relay,
            10,
        )

        self.get_logger().info(
            f"Relaying Twist commands: {self.input_topic} -> {self.output_topic}"
        )

    def _relay(self, msg: Twist) -> None:
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TwistRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
