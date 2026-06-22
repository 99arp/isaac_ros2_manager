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


def _same_type_relay_node(
    source_topic: str,
    target_topic: str,
    output_type: str,
    expression: str,
    name: str,
    qos_reliability: str = "reliable",
):
    return Node(
        package="topic_tools",
        executable="relay_field",
        name=name,
        output="screen",
        arguments=[
            source_topic,
            target_topic,
            output_type,
            expression,
            "--wait-for-start",
            "--qos-reliability",
            qos_reliability,
        ],
        parameters=[{"use_sim_time": False}],
    )


def _carter_native_bridge_nodes(team_ns: str, agent_name: str, controller_type: str):
    node_ns = f"/{team_ns}/{agent_name}"
    controller_type = str(controller_type or "").strip().lower()
    if controller_type == "nav2":
        cmd_vel_type = "geometry_msgs/msg/Twist"
        cmd_vel_expression = "{linear: m.linear, angular: m.angular}"
    else:
        cmd_vel_type = "geometry_msgs/msg/TwistStamped"
        cmd_vel_expression = "{linear: m.twist.linear, angular: m.twist.angular}"
    nodes = [
        _same_type_relay_node(
            "/chassis/odom",
            f"{node_ns}/odometry",
            "nav_msgs/msg/Odometry",
            "{header: m.header, child_frame_id: m.child_frame_id, pose: m.pose, twist: m.twist}",
            f"{agent_name}_isaac_odom_to_odometry",
        ),
        _same_type_relay_node(
            "/chassis/odom",
            f"{node_ns}/odom_matcher",
            "nav_msgs/msg/Odometry",
            "{header: m.header, child_frame_id: m.child_frame_id, pose: m.pose, twist: m.twist}",
            f"{agent_name}_isaac_odom_to_odom_matcher",
        ),
        _same_type_relay_node(
            "/chassis/imu",
            f"{node_ns}/imu",
            "sensor_msgs/msg/Imu",
            (
                "{header: m.header, orientation: m.orientation, "
                "orientation_covariance: m.orientation_covariance, "
                "angular_velocity: m.angular_velocity, "
                "angular_velocity_covariance: m.angular_velocity_covariance, "
                "linear_acceleration: m.linear_acceleration, "
                "linear_acceleration_covariance: m.linear_acceleration_covariance}"
            ),
            f"{agent_name}_isaac_imu",
        ),
        _same_type_relay_node(
            f"{node_ns}/cmd_vel",
            "/cmd_vel",
            "geometry_msgs/msg/Twist",
            cmd_vel_expression,
            f"{agent_name}_isaac_cmd_vel",
        ),
    ]
    if controller_type == "nav2":
        nodes.extend([
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name=f"{agent_name}_front_lidar_tf",
                namespace=node_ns,
                output="screen",
                remappings=[
                    ("/tf", f"{node_ns}/tf"),
                    ("/tf_static", f"{node_ns}/tf_static"),
                ],
                arguments=[
                    "--x", "0.15",
                    "--z", "0.25",
                    "--frame-id", "base_link",
                    "--child-frame-id", "front_3d_lidar",
                ],
            ),
            Node(
                package="pointcloud_to_laserscan",
                executable="pointcloud_to_laserscan_node",
                name=f"{agent_name}_pointcloud_to_scan",
                namespace=node_ns,
                output="screen",
                remappings=[
                    ("cloud_in", "/front_3d_lidar/lidar_points"),
                    ("scan", f"{node_ns}/scan_raw"),
                    ("/tf", f"{node_ns}/tf"),
                    ("/tf_static", f"{node_ns}/tf_static"),
                ],
                parameters=[{
                    "target_frame": "",
                    "transform_tolerance": 0.05,
                    "min_height": -0.5,
                    "max_height": 1.5,
                    "angle_min": -3.14159,
                    "angle_max": 3.14159,
                    "angle_increment": 0.0087,
                    "scan_time": 0.1,
                    "range_min": 0.05,
                    "range_max": 5.0,
                    "use_inf": True,
                    "inf_epsilon": 1.0,
                    "use_sim_time": False,
                }],
            ),
            _same_type_relay_node(
                f"{node_ns}/scan_raw",
                f"{node_ns}/scan",
                "sensor_msgs/msg/LaserScan",
                (
                    "{header: {stamp: now, frame_id: m.header.frame_id}, "
                    "angle_min: m.angle_min, angle_max: m.angle_max, "
                    "angle_increment: m.angle_increment, time_increment: m.time_increment, "
                    "scan_time: m.scan_time, range_min: m.range_min, range_max: m.range_max, "
                    "ranges: m.ranges, intensities: m.intensities}"
                ),
                f"{agent_name}_restamp_scan",
                qos_reliability="best_effort",
            ),
        ])
    return nodes


def _agent_launch_include(team_ns: str, agent_name: str, agent_dict: dict):
    agent_launch = agent_dict.get("launch")
    if not agent_launch:
        return None
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
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file),
        launch_arguments=launch_arguments.items(),
    )


def _ugv_movement_actions(team_ns: str, agent_name: str, controller_type: str):
    node_ns = f"/{team_ns}/{agent_name}"
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("webots_cmd"),
                "launch",
                "robot.launch.py",
            )
        ),
        launch_arguments={
            "ns": team_ns,
            "robot_name": agent_name,
            "params_file": os.path.join(
                get_package_share_directory("webots_cmd"),
                "params",
                "irobot_nav2.yaml",
            ),
            "controller_type": controller_type,
            "yolo_detection": "False",
            "use_sim_time": "False",
        }.items(),
    )
    manager = Node(
        package="ugv_ranger_manager",
        executable="ugv_ranger_manager_node",
        namespace=node_ns,
        output="screen",
        remappings=[
            ("~/odometry", f"{node_ns}/odom_matcher"),
            ("navigate_to_pose", f"{node_ns}/navigate_to_pose"),
            ("detect", f"{node_ns}/detect"),
            ("disarm", f"{node_ns}/disarm"),
        ],
        parameters=[{"use_sim_time": False}],
    )
    return [robot_launch, manager]


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
    publish_team_data: bool,
    publish_area_data: bool,
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
            "publish_team_data": publish_team_data,
            "publish_area_data": publish_area_data,
            "use_sim_time": False,
        }],
    )


def _team_manager_node(
    team_ns: str,
    team_file: str,
    grid_size: str,
    edge_size: str,
    world_offset: str,
    team_data_topic: str,
    area_data_topic: str,
):
    grid_size_value = max(1, int(float(grid_size)))
    edge_size_value = float(edge_size)
    return Node(
        package="isaac_ros2_manager",
        executable="isaac_team_manager_node",
        name="team_manager",
        namespace=team_ns,
        output="screen",
        parameters=[{
            "team_file": team_file,
            "grid_size": grid_size_value,
            "edge_size": edge_size_value,
            "world_offset": world_offset,
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
    launch_ugv_agent_launches = _as_bool(LaunchConfiguration("launch_ugv_agent_launches").perform(context))
    launch_carter_native_bridge = _as_bool(LaunchConfiguration("launch_carter_native_bridge").perform(context))
    ugv_controller_type = LaunchConfiguration("ugv_controller_type").perform(context)
    launch_skill_managers = _as_bool(LaunchConfiguration("launch_skill_managers").perform(context))
    launch_aero_bridges = _as_bool(LaunchConfiguration("launch_aero_bridges").perform(context))
    launch_objective_state = _as_bool(LaunchConfiguration("launch_objective_state").perform(context))
    launch_team_manager = _as_bool(LaunchConfiguration("launch_team_manager").perform(context))
    objectives_file = LaunchConfiguration("objectives_file").perform(context)
    area_objectives_topic = LaunchConfiguration("area_objectives_topic").perform(context)
    team_data_topic = LaunchConfiguration("team_data_topic").perform(context)
    area_data_topic = LaunchConfiguration("area_data_topic").perform(context)
    grid_size = LaunchConfiguration("grid_size").perform(context)
    edge_size = LaunchConfiguration("edge_size").perform(context)
    world_offset = LaunchConfiguration("world_offset").perform(context)
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
            not launch_team_manager,
            not launch_team_manager,
        ))

    if launch_team_manager:
        actions.append(_team_manager_node(
            team_ns,
            team_json,
            grid_size,
            edge_size,
            world_offset,
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

    launched_agent_launches = set()
    if launch_ugv_agent_launches:
        for agent_name, agent_dict in team_dict.get("agents", {}).items():
            if _agent_kind(agent_dict) != "ugv":
                continue
            actions.extend(_ugv_movement_actions(team_ns, agent_name, ugv_controller_type))
            launched_agent_launches.add(agent_name)
            if launch_carter_native_bridge:
                actions.extend(_carter_native_bridge_nodes(team_ns, agent_name, ugv_controller_type))

    if launch_agents:
        for agent_name, agent_dict in team_dict.get("agents", {}).items():
            if agent_name in launched_agent_launches:
                continue
            agent_launch = _agent_launch_include(team_ns, agent_name, agent_dict)
            if agent_launch is None:
                continue
            actions.append(agent_launch)
            launched_agent_launches.add(agent_name)

    if launch_skill_managers:
        for agent_name, agent_dict in team_dict.get("agents", {}).items():
            if agent_name in launched_agent_launches:
                continue
            actions.append(_skill_manager_node(team_ns, agent_name, agent_dict))

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
        DeclareLaunchArgument("launch_ugv_agent_launches", default_value="True"),
        DeclareLaunchArgument("launch_carter_native_bridge", default_value="True"),
        DeclareLaunchArgument("ugv_controller_type", default_value=os.environ.get("ISAAC_UGV_CONTROLLER_TYPE", "nav2")),
        DeclareLaunchArgument("launch_skill_managers", default_value="True"),
        DeclareLaunchArgument("launch_aero_bridges", default_value="True"),
        DeclareLaunchArgument("launch_objective_state", default_value="True"),
        DeclareLaunchArgument("launch_team_manager", default_value="True"),
        DeclareLaunchArgument("grid_size", default_value=os.environ.get("GRID_SIZE", "4")),
        DeclareLaunchArgument("edge_size", default_value=os.environ.get("EDGE_SIZE", "40.0")),
        DeclareLaunchArgument("world_offset", default_value=os.environ.get("ISAAC_WORLD_OFFSET", "")),
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
