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


def _format_action_topic(template: str, team_ns: str, agent_name: str, index: int) -> str:
    try:
        return str(template).format(team=team_ns, agent=agent_name, index=index)
    except Exception:
        return str(template)


def _aero_bridge_node(
    team_ns: str,
    agent_name: str,
    takeoff_action: str,
    land_action: str,
    fly_3d_action: str,
):
    node_ns = f"/{team_ns}/{agent_name}"
    return Node(
        package="uav_ranger_manager",
        executable="auspex_aero_bridge.py",
        namespace=node_ns,
        name="auspex_aero_bridge",
        output="screen",
        parameters=[{
            "takeoff_action": takeoff_action,
            "land_action": land_action,
            "fly_3d_action": fly_3d_action,
        }],
    )


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


def _objective_state_node(
    team_ns: str,
    area_id: str,
    agent_names: list[str],
    team_file: str,
    area_size: list[float],
    area_offset: list[float],
    objectives_file: str,
    area_objectives_topic: str,
    team_data_topic: str,
    area_data_topic: str,
):
    return Node(
        package="isaac_ros2_manager",
        executable="isaac_objective_state_node",
        name="isaac_objective_state",
        output="screen",
        parameters=[{
            "team_namespace": team_ns,
            "area_id": area_id,
            "action_agents": agent_names,
            "team_file": team_file,
            "area_size": area_size,
            "area_offset": area_offset,
            "objectives_file": objectives_file,
            "detected_objectives_topic": f"/{team_ns}/team_manager/detected_objectives",
            "area_objectives_topic": area_objectives_topic,
            "team_data_topic": team_data_topic,
            "area_data_topic": area_data_topic,
            "use_sim_time": False,
        }],
    )


def _team_setup(context, *args, **kwargs):
    team_arg = LaunchConfiguration("team").perform(context)
    team_json = _resolve_team_json(team_arg)
    with open(team_json, "r", encoding="utf-8") as f:
        team_dict = json.load(f)

    team_ns = team_dict["name"]
    launch_agents = _as_bool(LaunchConfiguration("launch_agents").perform(context))
    launch_skill_managers = _as_bool(LaunchConfiguration("launch_skill_managers").perform(context))
    launch_aero_bridges = _as_bool(LaunchConfiguration("launch_aero_bridges").perform(context))
    launch_objective_state = _as_bool(LaunchConfiguration("launch_objective_state").perform(context))
    objectives_file = LaunchConfiguration("objectives_file").perform(context)
    area_objectives_topic = LaunchConfiguration("area_objectives_topic").perform(context)
    team_data_topic = LaunchConfiguration("team_data_topic").perform(context)
    area_data_topic = LaunchConfiguration("area_data_topic").perform(context)
    aero_takeoff_action = LaunchConfiguration("aero_takeoff_action").perform(context)
    aero_land_action = LaunchConfiguration("aero_land_action").perform(context)
    aero_fly_3d_action = LaunchConfiguration("aero_fly_3d_action").perform(context)
    actions = []
    agent_names = list(team_dict.get("agents", {}).keys())

    if launch_objective_state:
        actions.append(_objective_state_node(
            team_ns,
            str(team_dict.get("area") or "area_00"),
            agent_names,
            team_json,
            list(team_dict.get("area_size") or [250.0, 250.0]),
            list(team_dict.get("area_offset") or [-125.0, -125.0]),
            objectives_file,
            area_objectives_topic,
            team_data_topic,
            area_data_topic,
        ))

    uav_index = 0
    if launch_aero_bridges:
        for agent_name, agent_dict in team_dict.get("agents", {}).items():
            if _agent_kind(agent_dict) != "uav":
                continue
            uav_index += 1
            actions.append(_aero_bridge_node(
                team_ns,
                agent_name,
                _format_action_topic(aero_takeoff_action, team_ns, agent_name, uav_index),
                _format_action_topic(aero_land_action, team_ns, agent_name, uav_index),
                _format_action_topic(aero_fly_3d_action, team_ns, agent_name, uav_index),
            ))

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

    return actions


def generate_launch_description():
    default_objectives_file = os.path.join(
        get_package_share_directory("isaac_ros2_manager"),
        "config",
        "integration_objectives.json",
    )
    return LaunchDescription([
        DeclareLaunchArgument("team", default_value="None"),
        DeclareLaunchArgument("launch_agents", default_value="False"),
        DeclareLaunchArgument("launch_skill_managers", default_value="True"),
        DeclareLaunchArgument("launch_aero_bridges", default_value="True"),
        DeclareLaunchArgument("launch_objective_state", default_value="True"),
        DeclareLaunchArgument(
            "objectives_file",
            default_value=os.environ.get("ISAAC_OBJECTIVES_FILE", default_objectives_file),
        ),
        DeclareLaunchArgument(
            "area_objectives_topic",
            default_value="/auspex_know/knowledge_collector/area_objectives",
        ),
        DeclareLaunchArgument(
            "team_data_topic",
            default_value="/auspex_know/knowledge_collector/team_data",
        ),
        DeclareLaunchArgument(
            "area_data_topic",
            default_value="/auspex_know/knowledge_collector/area_data",
        ),
        DeclareLaunchArgument(
            "aero_takeoff_action",
            default_value=os.environ.get("ISAAC_AERO_TAKEOFF_ACTION", "/{team}_{agent}/fm/takeoff"),
        ),
        DeclareLaunchArgument(
            "aero_land_action",
            default_value=os.environ.get("ISAAC_AERO_LAND_ACTION", "/{team}_{agent}/fm/land"),
        ),
        DeclareLaunchArgument(
            "aero_fly_3d_action",
            default_value=os.environ.get("ISAAC_AERO_FLY_3D_ACTION", "/{team}_{agent}/fm/fly_3d"),
        ),
        OpaqueFunction(function=_team_setup),
    ])
