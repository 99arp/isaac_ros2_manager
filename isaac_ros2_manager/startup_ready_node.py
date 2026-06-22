from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool


class IsaacStartupReady(Node):
    """Report when the statically configured Isaac world is ready."""

    def __init__(self):
        super().__init__("isaac_startup_ready")

        self.declare_parameter("world_ready_topic", "/world_manager/ready")
        self.declare_parameter("ready_topic", "/isaac_integration/ready")
        self.declare_parameter("timeout_sec", 0.0)

        self.world_ready_topic = str(self.get_parameter("world_ready_topic").value)
        self.ready_topic = str(self.get_parameter("ready_topic").value)
        self.timeout_sec = max(0.0, float(self.get_parameter("timeout_sec").value))
        self.start_time = time.monotonic()
        self.ready = False

        ready_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.ready_pub = self.create_publisher(Bool, self.ready_topic, ready_qos)
        self.create_subscription(Bool, self.world_ready_topic, self._world_ready_cb, ready_qos)
        self.create_timer(1.0, self._timer_cb)

        self.get_logger().info(
            f"Waiting for Isaac static world readiness on {self.world_ready_topic}; "
            f"will publish {self.ready_topic}"
        )

    def _world_ready_cb(self, msg: Bool) -> None:
        if not msg.data:
            return
        if not self.ready:
            self.get_logger().info("Isaac static world is ready.")
        self.ready = True
        self.ready_pub.publish(Bool(data=True))

    def _timer_cb(self) -> None:
        if self.ready:
            self.ready_pub.publish(Bool(data=True))
            return
        if self.timeout_sec and time.monotonic() - self.start_time > self.timeout_sec:
            self.get_logger().error(
                f"Timed out waiting for Isaac readiness topic {self.world_ready_topic}"
            )
            raise SystemExit(1)


def main(args=None):
    rclpy.init(args=args)
    node = IsaacStartupReady()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
