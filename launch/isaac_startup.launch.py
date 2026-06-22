#!/usr/bin/env python

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("world_ready_topic", default_value="/world_manager/ready"),
        DeclareLaunchArgument("ready_topic", default_value="/isaac_integration/ready"),
        DeclareLaunchArgument("timeout_sec", default_value="0.0"),
        Node(
            package="isaac_ros2_manager",
            executable="isaac_startup_ready_node",
            name="isaac_startup_ready",
            output="screen",
            parameters=[{
                "world_ready_topic": LaunchConfiguration("world_ready_topic"),
                "ready_topic": LaunchConfiguration("ready_topic"),
                "timeout_sec": LaunchConfiguration("timeout_sec"),
            }],
        ),
    ])
