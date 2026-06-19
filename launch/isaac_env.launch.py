#!/usr/bin/env python

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("grid_size", default_value="1"),
        DeclareLaunchArgument("edge_size", default_value="250.0"),
        DeclareLaunchArgument("world_offset", default_value="[]"),
        DeclareLaunchArgument("objectives_per_area", default_value="0"),
        DeclareLaunchArgument("auto_update_db", default_value="True"),
        Node(
            package="isaac_ros2_manager",
            executable="isaac_env_manager_node",
            name="env_manager",
            output="screen",
            parameters=[{
                "grid_size": LaunchConfiguration("grid_size"),
                "edge_size": LaunchConfiguration("edge_size"),
                "world_offset": LaunchConfiguration("world_offset"),
                "objectives_per_area": LaunchConfiguration("objectives_per_area"),
                "auto_update_db": LaunchConfiguration("auto_update_db"),
            }],
        ),
    ])
