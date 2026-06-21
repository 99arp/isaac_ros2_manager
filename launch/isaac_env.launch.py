#!/usr/bin/env python

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("grid_size", default_value="1"),
        DeclareLaunchArgument("edge_size", default_value="250.0"),
        DeclareLaunchArgument("world_offset", default_value="[0.0, 0.0]"),
        DeclareLaunchArgument("objectives_per_area", default_value="0"),
        DeclareLaunchArgument("auto_update_db", default_value="True"),
        DeclareLaunchArgument("spawn_with_isaac", default_value="True"),
        DeclareLaunchArgument("objective_model", default_value="bear_trap"),
        DeclareLaunchArgument("objective_z_offset", default_value="0.0"),
        DeclareLaunchArgument("objective_target_length_m", default_value="1.0"),
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
                "spawn_with_isaac": LaunchConfiguration("spawn_with_isaac"),
                "objective_model": LaunchConfiguration("objective_model"),
                "objective_z_offset": LaunchConfiguration("objective_z_offset"),
                "objective_target_length_m": LaunchConfiguration("objective_target_length_m"),
            }],
        ),
    ])
