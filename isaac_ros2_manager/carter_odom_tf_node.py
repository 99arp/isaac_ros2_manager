from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from .common import best_effort_qos, param_bool, set_quat_from_yaw, yaw_from_quat


def _stamp_is_zero(stamp) -> bool:
    return int(stamp.sec) == 0 and int(stamp.nanosec) == 0


def _valid_quaternion(q) -> bool:
    norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
    return math.isfinite(norm) and norm > 1e-12


class CarterOdomTf(Node):
    """Publish the Webots-style Carter odom/TF contract from Isaac native odom."""

    def __init__(self):
        super().__init__("isaac_carter_odom_tf")

        self.declare_parameter("native_odom_topic", "chassis/odom")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("map_frame_id", "map")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("base_footprint_frame_id", "base_footprint")
        self.declare_parameter("publish_map_to_odom", True)
        self.declare_parameter("publish_base_footprint", True)
        self.declare_parameter("force_2d", True)
        self.declare_parameter("use_native_odom_stamp", False)
        self.declare_parameter("sensor_parent_frame_id", "base_footprint")
        self.declare_parameter("sensor_frame_id", "front_3d_lidar")
        self.declare_parameter("sensor_x", 0.43)
        self.declare_parameter("sensor_y", 0.0)
        self.declare_parameter("sensor_z", 0.10)
        self.declare_parameter("sensor_roll", 0.0)
        self.declare_parameter("sensor_pitch", 0.0)
        self.declare_parameter("sensor_yaw", 0.0)

        self.native_odom_topic = str(self.get_parameter("native_odom_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.map_frame_id = str(self.get_parameter("map_frame_id").value)
        self.odom_frame_id = str(self.get_parameter("odom_frame_id").value)
        self.base_frame_id = str(self.get_parameter("base_frame_id").value)
        self.base_footprint_frame_id = str(self.get_parameter("base_footprint_frame_id").value)
        self.publish_map_to_odom = param_bool(self.get_parameter("publish_map_to_odom").value)
        self.publish_base_footprint = param_bool(self.get_parameter("publish_base_footprint").value)
        self.force_2d = param_bool(self.get_parameter("force_2d").value)
        self.use_native_odom_stamp = param_bool(self.get_parameter("use_native_odom_stamp").value)
        self.sensor_parent_frame_id = str(self.get_parameter("sensor_parent_frame_id").value)
        self.sensor_frame_id = str(self.get_parameter("sensor_frame_id").value)
        self.sensor_x = float(self.get_parameter("sensor_x").value)
        self.sensor_y = float(self.get_parameter("sensor_y").value)
        self.sensor_z = float(self.get_parameter("sensor_z").value)
        self.sensor_roll = float(self.get_parameter("sensor_roll").value)
        self.sensor_pitch = float(self.get_parameter("sensor_pitch").value)
        self.sensor_yaw = float(self.get_parameter("sensor_yaw").value)

        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.create_subscription(
            Odometry,
            self.native_odom_topic,
            self._odom_cb,
            best_effort_qos(),
        )

        self._published_first_odom = False
        self._publish_static_sensor_tf()
        self.get_logger().info(
            "Isaac Carter odom/TF adapter ready: "
            f"{self.native_odom_topic} -> {self.odom_topic}; "
            f"TF frames {self.map_frame_id}->{self.odom_frame_id}->{self.base_frame_id}"
        )

    def _stamp_for(self, msg: Odometry):
        if self.use_native_odom_stamp and not _stamp_is_zero(msg.header.stamp):
            return msg.header.stamp
        return self.get_clock().now().to_msg()

    def _pose_orientation(self, msg: Odometry):
        orientation = msg.pose.pose.orientation
        if not _valid_quaternion(orientation):
            set_quat_from_yaw(orientation, 0.0)
            return orientation
        if self.force_2d:
            yaw = yaw_from_quat(orientation)
            set_quat_from_yaw(orientation, yaw)
        return orientation

    def _odom_cb(self, msg: Odometry) -> None:
        stamp = self._stamp_for(msg)
        orientation = self._pose_orientation(msg)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame_id
        odom.child_frame_id = self.base_frame_id
        odom.pose.pose.position.x = float(msg.pose.pose.position.x)
        odom.pose.pose.position.y = float(msg.pose.pose.position.y)
        odom.pose.pose.position.z = 0.0 if self.force_2d else float(msg.pose.pose.position.z)
        odom.pose.pose.orientation = orientation
        odom.pose.covariance = msg.pose.covariance
        odom.twist.twist = msg.twist.twist
        if self.force_2d:
            odom.twist.twist.linear.z = 0.0
            odom.twist.twist.angular.x = 0.0
            odom.twist.twist.angular.y = 0.0
        odom.twist.covariance = msg.twist.covariance
        self.odom_pub.publish(odom)

        transforms = []
        if self.publish_map_to_odom:
            transforms.append(self._identity_tf(stamp, self.map_frame_id, self.odom_frame_id))

        odom_to_base = TransformStamped()
        odom_to_base.header.stamp = stamp
        odom_to_base.header.frame_id = self.odom_frame_id
        odom_to_base.child_frame_id = self.base_frame_id
        odom_to_base.transform.translation.x = odom.pose.pose.position.x
        odom_to_base.transform.translation.y = odom.pose.pose.position.y
        odom_to_base.transform.translation.z = odom.pose.pose.position.z
        odom_to_base.transform.rotation = orientation
        transforms.append(odom_to_base)

        if self.publish_base_footprint:
            transforms.append(self._identity_tf(stamp, self.base_frame_id, self.base_footprint_frame_id))

        self.tf_broadcaster.sendTransform(transforms)
        if not self._published_first_odom:
            self._published_first_odom = True
            self.get_logger().info(
                f"Publishing Carter odom/TF from first native odom sample on {self.native_odom_topic}."
            )

    def _identity_tf(self, stamp, parent: str, child: str) -> TransformStamped:
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = parent
        tf.child_frame_id = child
        tf.transform.rotation.w = 1.0
        return tf

    def _publish_static_sensor_tf(self) -> None:
        if not self.sensor_frame_id or not self.sensor_parent_frame_id:
            return
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.sensor_parent_frame_id
        tf.child_frame_id = self.sensor_frame_id
        tf.transform.translation.x = self.sensor_x
        tf.transform.translation.y = self.sensor_y
        tf.transform.translation.z = self.sensor_z
        set_quat_from_yaw(tf.transform.rotation, self.sensor_yaw)
        if self.sensor_roll or self.sensor_pitch:
            self.get_logger().warning(
                "sensor_roll and sensor_pitch are currently ignored; only sensor_yaw is applied."
            )
        self.static_tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = CarterOdomTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
