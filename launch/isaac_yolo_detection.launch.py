#!/usr/bin/env python

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    camera_topic = LaunchConfiguration("camera_topic")
    detections_topic = LaunchConfiguration("detections_topic")
    weights = LaunchConfiguration("weights")
    confidence = LaunchConfiguration("confidence")

    resolved_detections_topic = PythonExpression([
        "'",
        detections_topic,
        "' if '",
        detections_topic,
        "' else '",
        camera_topic,
        "'.rstrip('/') + '/detections'",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("camera_topic", default_value="/chipgt/mini1/front_camera/image_raw"),
        DeclareLaunchArgument("detections_topic", default_value=""),
        DeclareLaunchArgument("weights", default_value="/home/qnc/runs/detect/topdown/weights/best.pt"),
        DeclareLaunchArgument("confidence", default_value="0.25"),
        DeclareLaunchArgument("odom_topic", default_value="/chipgt/mini1/odometry"),
        DeclareLaunchArgument("output_topic", default_value="/chipgt/team_manager/detected_objectives"),
        DeclareLaunchArgument("camera_fov_deg", default_value="45.0"),
        DeclareLaunchArgument("ground_z", default_value="0.0"),
        DeclareLaunchArgument("camera_height_m", default_value="0.0"),
        DeclareLaunchArgument("min_detections", default_value="2"),
        DeclareLaunchArgument("merge_radius_m", default_value="1.0"),
        DeclareLaunchArgument("allowed_labels", default_value="trap,traps,bear_trap,bear-trap,bear trap,landmine,mine"),
        DeclareLaunchArgument("edge_size", default_value="40.0"),
        DeclareLaunchArgument("world_offset", default_value="0.0,0.0"),
        DeclareLaunchArgument("nav_edge_size", default_value="40.0"),
        DeclareLaunchArgument("nav_world_offset", default_value="0.0,0.0"),
        DeclareLaunchArgument("output_coordinate_frame", default_value="local"),
        ExecuteProcess(
            name="isaac_yolo_camera_view",
            cmd=[
                "python3",
                "/home/qnc/Desktop/isaacsim_integration_project/yolo/yolo_camera_view.py",
                "--topic",
                camera_topic,
                "--weights",
                weights,
                "--conf",
                confidence,
                "--det-topic",
                resolved_detections_topic,
            ],
            output="screen",
        ),
        Node(
            package="isaac_ros2_manager",
            executable="isaac_yolo_detection_adapter_node",
            name="isaac_yolo_detection_adapter",
            output="screen",
            parameters=[{
                "detections_topic": resolved_detections_topic,
                "odom_topic": LaunchConfiguration("odom_topic"),
                "output_topic": LaunchConfiguration("output_topic"),
                "camera_fov_deg": LaunchConfiguration("camera_fov_deg"),
                "ground_z": LaunchConfiguration("ground_z"),
                "camera_height_m": LaunchConfiguration("camera_height_m"),
                "min_confidence": confidence,
                "min_detections": LaunchConfiguration("min_detections"),
                "merge_radius_m": LaunchConfiguration("merge_radius_m"),
                "allowed_labels": LaunchConfiguration("allowed_labels"),
                "edge_size": LaunchConfiguration("edge_size"),
                "world_offset": LaunchConfiguration("world_offset"),
                "nav_edge_size": LaunchConfiguration("nav_edge_size"),
                "nav_world_offset": LaunchConfiguration("nav_world_offset"),
                "output_coordinate_frame": LaunchConfiguration("output_coordinate_frame"),
                "frame_id": LaunchConfiguration("output_coordinate_frame"),
            }],
        ),
    ])
