#!/usr/bin/env python

import json
import os

from ament_index_python.packages import get_package_share_directory
from ament_index_python.packages import PackageNotFoundError
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _resolve_team_json(team: str) -> str:
    if os.path.isfile(team):
        return team
    for package_name in ("chipgt_bringup", "webots_ros2_manager"):
        try:
            package_dir = get_package_share_directory(package_name)
        except PackageNotFoundError:
            continue
        candidate = os.path.join(package_dir, "teams", team + ".json")
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Cannot resolve team JSON: {team}")


def _agent_kind(agent_dict: dict) -> str:
    kind = str(agent_dict.get("kind") or "").strip().lower()
    if kind:
        return kind
    manager = str(agent_dict.get("manager") or "").strip().lower()
    if manager.startswith("uav_"):
        return "uav"
    if manager.startswith("ugv_"):
        return "ugv"
    agent_type = str(agent_dict.get("type") or "").strip().lower()
    if "uav" in agent_type or "drone" in agent_type:
        return "uav"
    return "ugv"


def _skill_manager_node(team_ns: str, agent_name: str, agent_dict: dict):
    node_ns = f"/{team_ns}/{agent_name}"
    kind = _agent_kind(agent_dict)
    if kind == "uav":
        return Node(
            package="uav_ranger_manager",
            executable="uav_ranger_manager_node",
            namespace=node_ns,
            output="screen",
            remappings=[
                ("~/odometry", f"{node_ns}/odometry"),
                ("~/is_flying", f"{node_ns}/is_flying"),
                ("navigate_to_pose", f"{node_ns}/navigate_to_pose"),
                ("detect", f"{node_ns}/detect"),
                ("disarm", f"{node_ns}/disarm"),
                ("takeoff", f"{node_ns}/takeoff"),
                ("land", f"{node_ns}/land"),
            ],
            parameters=[{"use_sim_time": True}],
        )
    if kind == "ugv":
        return Node(
            package="ugv_ranger_manager",
            executable="ugv_ranger_manager_node",
            namespace=node_ns,
            output="screen",
            remappings=[
                ("~/odometry", f"{node_ns}/odom"),
                ("navigate_to_pose", f"{node_ns}/navigate_to_pose"),
                ("detect", f"{node_ns}/detect"),
                ("disarm", f"{node_ns}/disarm"),
            ],
            parameters=[{"use_sim_time": True}],
        )
    raise ValueError(f"Unsupported Isaac skill manager kind for {agent_name}: {kind}")


def _team_setup(context, *args, **kwargs):
    team_arg = LaunchConfiguration("team").perform(context)
    team_json = _resolve_team_json(team_arg)
    with open(team_json, "r", encoding="utf-8") as f:
        team_dict = json.load(f)

    team_ns = team_dict["name"]
    launch_agents = _as_bool(LaunchConfiguration("launch_agents").perform(context))
    launch_skill_managers = _as_bool(LaunchConfiguration("launch_skill_managers").perform(context))
    actions = []

    if launch_skill_managers:
        for agent_name, agent_dict in team_dict.get("agents", {}).items():
            actions.append(_skill_manager_node(team_ns, agent_name, agent_dict))

    if launch_agents:
        for agent_name, agent_dict in team_dict.get("agents", {}).items():
            agent_launch = agent_dict.get("launch")
            if not agent_launch:
                continue
            launch_pkg = agent_launch.get("package")
            launch_file = agent_launch.get("launch")
            if not os.path.isabs(launch_file):
                launch_file = os.path.join(
                    get_package_share_directory(launch_pkg),
                    "launch",
                    launch_file,
                )
            launch_arguments = {
                "ns": team_ns,
                "robot_name": agent_name,
            }
            for key, value in agent_launch.get("params", {}).items():
                launch_arguments[key] = str(value)
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(launch_file),
                launch_arguments=launch_arguments.items(),
            ))

    actions.append(Node(
        package="isaac_ros2_manager",
        executable="isaac_team_manager_node",
        name="team_manager",
        namespace=team_ns,
        output="screen",
        parameters=[{
            "team_json": team_json,
            "grid_size": LaunchConfiguration("grid_size"),
            "edge_size": LaunchConfiguration("edge_size"),
            "world_offset": LaunchConfiguration("world_offset"),
            "spawn": LaunchConfiguration("spawn"),
            "spawn_with_isaac": LaunchConfiguration("spawn_with_isaac"),
            "auto_update_db": LaunchConfiguration("auto_update_db"),
            "synthetic_odom": LaunchConfiguration("synthetic_odom"),
            "odom_timeout_sec": LaunchConfiguration("odom_timeout_sec"),
            "goal_tolerance": LaunchConfiguration("goal_tolerance"),
            "goal_timeout_sec": LaunchConfiguration("goal_timeout_sec"),
            "origin_lat": LaunchConfiguration("origin_lat"),
            "origin_lon": LaunchConfiguration("origin_lon"),
            "origin_alt": LaunchConfiguration("origin_alt"),
            "spawn_alt": LaunchConfiguration("spawn_alt"),
            "spawn_confirm_timeout_sec": LaunchConfiguration("spawn_confirm_timeout_sec"),
            "spawn_call_timeout_sec": LaunchConfiguration("spawn_call_timeout_sec"),
            "spawn_telemetry_stale_timeout_sec": LaunchConfiguration("spawn_telemetry_stale_timeout_sec"),
            "spawn_confirm_settle_sec": LaunchConfiguration("spawn_confirm_settle_sec"),
            "spawn_motion_probe_enabled": LaunchConfiguration("spawn_motion_probe_enabled"),
            "spawn_motion_probe_duration_sec": LaunchConfiguration("spawn_motion_probe_duration_sec"),
            "spawn_motion_probe_linear_x": LaunchConfiguration("spawn_motion_probe_linear_x"),
            "spawn_motion_probe_min_delta_m": LaunchConfiguration("spawn_motion_probe_min_delta_m"),
            "uav_vehicle_id_base": LaunchConfiguration("uav_vehicle_id_base"),
        }],
    ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("team", default_value="None"),
        DeclareLaunchArgument("grid_size", default_value="1"),
        DeclareLaunchArgument("edge_size", default_value="250.0"),
        DeclareLaunchArgument("world_offset", default_value="[0.0, 0.0]"),
        DeclareLaunchArgument("spawn", default_value="True"),
        DeclareLaunchArgument("spawn_with_isaac", default_value="True"),
        DeclareLaunchArgument("launch_agents", default_value="False"),
        DeclareLaunchArgument("launch_skill_managers", default_value="True"),
        DeclareLaunchArgument("auto_update_db", default_value="True"),
        DeclareLaunchArgument("synthetic_odom", default_value="True"),
        DeclareLaunchArgument("odom_timeout_sec", default_value="1.0"),
        DeclareLaunchArgument("goal_tolerance", default_value="1.0"),
        DeclareLaunchArgument("goal_timeout_sec", default_value="120.0"),
        DeclareLaunchArgument("origin_lat", default_value="0.0"),
        DeclareLaunchArgument("origin_lon", default_value="0.0"),
        DeclareLaunchArgument("origin_alt", default_value="0.0"),
        DeclareLaunchArgument("spawn_alt", default_value="1.0"),
        DeclareLaunchArgument("spawn_confirm_timeout_sec", default_value="180.0"),
        DeclareLaunchArgument("spawn_call_timeout_sec", default_value="180.0"),
        DeclareLaunchArgument("spawn_telemetry_stale_timeout_sec", default_value="10.0"),
        DeclareLaunchArgument("spawn_confirm_settle_sec", default_value="8.0"),
        DeclareLaunchArgument("spawn_motion_probe_enabled", default_value="True"),
        DeclareLaunchArgument("spawn_motion_probe_duration_sec", default_value="3.0"),
        DeclareLaunchArgument("spawn_motion_probe_linear_x", default_value="0.35"),
        DeclareLaunchArgument("spawn_motion_probe_min_delta_m", default_value="0.05"),
        DeclareLaunchArgument("uav_vehicle_id_base", default_value="0"),
        OpaqueFunction(function=_team_setup),
    ])
