#!/usr/bin/env python

import json
import os

from ament_index_python.packages import get_package_share_directory
from ament_index_python.packages import PackageNotFoundError
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap


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


def _resolve_package_launch_file(launch_dict: dict) -> str:
    launch_file = str(launch_dict.get("launch") or "").strip()
    if not launch_file:
        raise ValueError("Agent launch stanza is missing 'launch'")
    if os.path.isabs(launch_file):
        return launch_file

    launch_pkg = str(launch_dict.get("package") or "").strip()
    if not launch_pkg:
        raise ValueError(f"Relative launch file {launch_file!r} needs a package")
    return os.path.join(get_package_share_directory(launch_pkg), "launch", launch_file)


def _float_launch_param(params: dict, key: str, default: float) -> float:
    try:
        return float(params.pop(key, default))
    except Exception:
        return float(default)


def _carter_odom_tf_node(
    team_ns: str,
    agent_name: str,
    native_odometry_topic: str,
    sensor_parent_frame_id: str,
    sensor_frame_id: str,
    sensor_x: float,
    sensor_y: float,
    sensor_z: float,
    use_native_odom_stamp: bool,
):
    node_ns = f"/{team_ns}/{agent_name}"
    return Node(
        package="isaac_ros2_manager",
        executable="isaac_carter_odom_tf_node",
        name="isaac_carter_odom_tf",
        namespace=node_ns,
        output="screen",
        remappings=[
            ("/tf", f"{node_ns}/tf"),
            ("/tf_static", f"{node_ns}/tf_static"),
        ],
        parameters=[{
            "native_odom_topic": native_odometry_topic,
            "odom_topic": f"{node_ns}/odom",
            "map_frame_id": "map",
            "odom_frame_id": "odom",
            "base_frame_id": "base_link",
            "base_footprint_frame_id": "base_footprint",
            "publish_map_to_odom": True,
            "publish_base_footprint": True,
            "force_2d": True,
            "use_native_odom_stamp": use_native_odom_stamp,
            "sensor_parent_frame_id": sensor_parent_frame_id,
            "sensor_frame_id": sensor_frame_id,
            "sensor_x": sensor_x,
            "sensor_y": sensor_y,
            "sensor_z": sensor_z,
            "use_sim_time": False,
        }],
    )


def _twist_relay_node(
    team_ns: str,
    agent_name: str,
    input_topic: str,
    output_topic: str,
):
    node_ns = f"/{team_ns}/{agent_name}"
    return Node(
        package="isaac_ros2_manager",
        executable="isaac_twist_relay_node",
        name="isaac_carter_cmd_vel_relay",
        namespace=node_ns,
        output="screen",
        parameters=[{
            "input_topic": input_topic,
            "output_topic": output_topic,
        }],
    )


def _ugv_agent_launch(team_ns: str, agent_name: str, agent_dict: dict):
    agent_launch = agent_dict.get("launch") or {}
    launch_file = _resolve_package_launch_file(agent_launch)
    node_ns = f"/{team_ns}/{agent_name}"
    launch_params = dict(agent_launch.get("params") or {})

    native_odometry_topic = str(
        launch_params.pop("isaac_native_odometry_topic", f"{node_ns}/chassis/odom")
    ).strip()
    native_cmd_vel_topic = str(
        launch_params.pop("isaac_native_cmd_vel_topic", "")
    ).strip()
    scan_topic = str(launch_params.pop("isaac_scan_topic", f"{node_ns}/scan")).strip()
    publish_odom_tf = _as_bool(launch_params.pop("isaac_publish_odom_tf", "True"))
    use_native_odom_stamp = _as_bool(launch_params.pop("isaac_odom_tf_use_native_stamp", "False"))
    sensor_parent_frame_id = str(
        launch_params.pop("isaac_lidar_parent_frame", "base_footprint")
    ).strip()
    sensor_frame_id = str(
        launch_params.pop("isaac_lidar_frame", "front_3d_lidar")
    ).strip()
    sensor_x = _float_launch_param(launch_params, "isaac_lidar_x", 0.43)
    sensor_y = _float_launch_param(launch_params, "isaac_lidar_y", 0.0)
    sensor_z = _float_launch_param(launch_params, "isaac_lidar_z", 0.10)

    launch_arguments = {
        "ns": team_ns,
        "robot_name": agent_name,
        "objective_actions": "False",
        "cmd_vel_stamped": "False",
        "launch_control_node": "False" if publish_odom_tf else "True",
    }
    launch_arguments.update({str(k): str(v) for k, v in launch_params.items()})

    actions = []
    if publish_odom_tf and native_odometry_topic:
        actions.append(_carter_odom_tf_node(
            team_ns,
            agent_name,
            native_odometry_topic,
            sensor_parent_frame_id,
            sensor_frame_id,
            sensor_x,
            sensor_y,
            sensor_z,
            use_native_odom_stamp,
        ))
    elif native_odometry_topic and native_odometry_topic != f"{node_ns}/odometry":
        actions.append(SetRemap(src=f"{node_ns}/odometry", dst=native_odometry_topic))
    if scan_topic and scan_topic != f"{node_ns}/scan":
        actions.append(SetRemap(src=f"{node_ns}/scan", dst=scan_topic))
    if native_cmd_vel_topic and native_cmd_vel_topic != f"{node_ns}/cmd_vel":
        actions.append(_twist_relay_node(
            team_ns,
            agent_name,
            f"{node_ns}/cmd_vel",
            native_cmd_vel_topic,
        ))

    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file),
        launch_arguments=launch_arguments.items(),
    ))
    return GroupAction(actions=actions)


def _parse_xy(value: str, fallback: list[float]) -> list[float]:
    try:
        parts = json.loads(str(value))
        if isinstance(parts, list) and len(parts) >= 2:
            return [float(parts[0]), float(parts[1])]
    except Exception:
        pass
    try:
        parts = [float(part.strip()) for part in str(value).split(",") if part.strip()]
        if len(parts) >= 2:
            return [parts[0], parts[1]]
    except Exception:
        pass
    return list(fallback)


def _area_offset_for_id(area_id: str, grid_size: int, edge_size: float, world_offset: list[float]) -> list[float]:
    try:
        index = int(str(area_id).split("_")[-1])
    except Exception:
        index = 0
    x_index = index // max(1, grid_size)
    y_index = index % max(1, grid_size)
    return [
        float(world_offset[0]) + x_index * float(edge_size),
        float(world_offset[1]) + y_index * float(edge_size),
    ]


def _aero_bridge_node(
    team_ns: str,
    agent_name: str,
    takeoff_action: str,
    land_action: str,
    fly_3d_action: str,
    local_waypoints: bool,
    local_origin_lat_deg: float,
    local_origin_lon_deg: float,
    local_origin_alt_amsl_m: float,
    local_yaw_deg: float,
    default_cruise_height_agl_m: float,
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
            "local_waypoints": local_waypoints,
            "local_origin_lat_deg": local_origin_lat_deg,
            "local_origin_lon_deg": local_origin_lon_deg,
            "local_origin_alt_amsl_m": local_origin_alt_amsl_m,
            "local_yaw_deg": local_yaw_deg,
            "default_cruise_height_agl_m": default_cruise_height_agl_m,
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
        if agent_dict.get("launch"):
            return _ugv_agent_launch(team_ns, agent_name, agent_dict)
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
            parameters=[{"use_sim_time": False}],
        )
    raise ValueError(f"Unsupported Isaac skill manager kind for {agent_name}: {kind}")


def _objective_state_node(
    team_ns: str,
    area_id: str,
    agent_names: list[str],
    team_file: str,
    grid_size: str,
    edge_size: str,
    world_offset: str,
    nav_edge_size: str,
    nav_world_offset: str,
    objectives_file: str,
    objectives_coordinate_frame: str,
    area_objectives_topic: str,
    team_data_topic: str,
    area_data_topic: str,
    publish_team_data: bool,
    publish_area_data: bool,
):
    grid_size_value = max(1, int(float(grid_size)))
    edge_size_value = float(edge_size)
    world_offset_value = _parse_xy(world_offset, [0.0, 0.0])
    nav_edge_size_value = float(nav_edge_size) if str(nav_edge_size).strip() else edge_size_value
    nav_world_offset_value = _parse_xy(nav_world_offset, world_offset_value)
    area_offset = _area_offset_for_id(area_id, grid_size_value, edge_size_value, world_offset_value)
    nav_area_offset = _area_offset_for_id(area_id, grid_size_value, nav_edge_size_value, nav_world_offset_value)
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
            "area_size": [edge_size_value, edge_size_value],
            "area_offset": area_offset,
            "edge_size": edge_size_value,
            "world_offset": area_offset,
            "nav_edge_size": nav_edge_size_value,
            "nav_world_offset": nav_area_offset,
            "objectives_coordinate_frame": objectives_coordinate_frame,
            "detections_coordinate_frame": "local",
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
    nav_edge_size: str,
    nav_world_offset: str,
    team_data_topic: str,
    area_data_topic: str,
    uav_cruise_height_agl_m: str,
):
    grid_size_value = max(1, int(float(grid_size)))
    edge_size_value = float(edge_size)
    nav_edge_size_value = float(nav_edge_size) if str(nav_edge_size).strip() else 0.0
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
            "nav_edge_size": nav_edge_size_value,
            "nav_world_offset": nav_world_offset,
            "team_data_topic": team_data_topic,
            "area_data_topic": area_data_topic,
            "uav_cruise_height_agl_m": float(uav_cruise_height_agl_m),
            "use_sim_time": False,
        }],
    )


def _team_setup(context, *args, **kwargs):
    team_arg = LaunchConfiguration("team").perform(context)
    team_json = _resolve_team_json(team_arg)
    with open(team_json, "r", encoding="utf-8") as f:
        team_dict = json.load(f)

    team_ns = team_dict["name"]
    launch_skill_managers = _as_bool(LaunchConfiguration("launch_skill_managers").perform(context))
    launch_aero_bridges = _as_bool(LaunchConfiguration("launch_aero_bridges").perform(context))
    launch_objective_state = _as_bool(LaunchConfiguration("launch_objective_state").perform(context))
    launch_team_manager = _as_bool(LaunchConfiguration("launch_team_manager").perform(context))
    objectives_file = LaunchConfiguration("objectives_file").perform(context)
    objectives_coordinate_frame = LaunchConfiguration("objectives_coordinate_frame").perform(context)
    area_objectives_topic = LaunchConfiguration("area_objectives_topic").perform(context)
    team_data_topic = LaunchConfiguration("team_data_topic").perform(context)
    area_data_topic = LaunchConfiguration("area_data_topic").perform(context)
    grid_size = LaunchConfiguration("grid_size").perform(context)
    edge_size = LaunchConfiguration("edge_size").perform(context)
    world_offset = LaunchConfiguration("world_offset").perform(context)
    nav_edge_size = LaunchConfiguration("nav_edge_size").perform(context)
    nav_world_offset = LaunchConfiguration("nav_world_offset").perform(context)
    aero_takeoff_action = LaunchConfiguration("aero_takeoff_action").perform(context)
    aero_land_action = LaunchConfiguration("aero_land_action").perform(context)
    aero_fly_3d_action = LaunchConfiguration("aero_fly_3d_action").perform(context)
    aero_local_waypoints = _as_bool(LaunchConfiguration("aero_local_waypoints").perform(context))
    aero_local_origin_lat_deg = float(LaunchConfiguration("aero_local_origin_lat_deg").perform(context))
    aero_local_origin_lon_deg = float(LaunchConfiguration("aero_local_origin_lon_deg").perform(context))
    aero_local_origin_alt_amsl_m = float(
        LaunchConfiguration("aero_local_origin_alt_amsl_m").perform(context)
    )
    aero_local_yaw_deg = float(LaunchConfiguration("aero_local_yaw_deg").perform(context))
    aero_default_cruise_height_agl_m = float(LaunchConfiguration(
        "aero_default_cruise_height_agl_m"
    ).perform(context))
    actions = []
    agent_names = list(team_dict.get("agents", {}).keys())

    if launch_objective_state:
        actions.append(_objective_state_node(
            team_ns,
            str(team_dict.get("area") or "area_00"),
            agent_names,
            team_json,
            grid_size,
            edge_size,
            world_offset,
            nav_edge_size,
            nav_world_offset,
            objectives_file,
            objectives_coordinate_frame,
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
            nav_edge_size,
            nav_world_offset,
            team_data_topic,
            area_data_topic,
            aero_default_cruise_height_agl_m,
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
                aero_local_waypoints,
                aero_local_origin_lat_deg,
                aero_local_origin_lon_deg,
                aero_local_origin_alt_amsl_m,
                aero_local_yaw_deg,
                aero_default_cruise_height_agl_m,
            ))

    if launch_skill_managers:
        for agent_name, agent_dict in team_dict.get("agents", {}).items():
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
        DeclareLaunchArgument("launch_skill_managers", default_value="True"),
        DeclareLaunchArgument("launch_aero_bridges", default_value="True"),
        DeclareLaunchArgument("launch_objective_state", default_value="True"),
        DeclareLaunchArgument(
            "launch_team_manager",
            default_value=os.environ.get("ISAAC_LAUNCH_TEAM_MANAGER", "True"),
        ),
        DeclareLaunchArgument("grid_size", default_value=os.environ.get("GRID_SIZE", "4")),
        DeclareLaunchArgument("edge_size", default_value=os.environ.get("EDGE_SIZE", "40.0")),
        DeclareLaunchArgument("world_offset", default_value=os.environ.get("ISAAC_WORLD_OFFSET", "0.0,0.0")),
        DeclareLaunchArgument("nav_edge_size", default_value=os.environ.get("ISAAC_NAV_EDGE_SIZE", os.environ.get("EDGE_SIZE", "40.0"))),
        DeclareLaunchArgument("nav_world_offset", default_value=os.environ.get("ISAAC_NAV_WORLD_OFFSET", os.environ.get("ISAAC_WORLD_OFFSET", "0.0,0.0"))),
        DeclareLaunchArgument(
            "objectives_file",
            default_value=os.environ.get("ISAAC_OBJECTIVES_FILE", default_objectives_file),
        ),
        DeclareLaunchArgument(
            "objectives_coordinate_frame",
            default_value=os.environ.get("ISAAC_OBJECTIVES_FRAME", "native"),
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
        DeclareLaunchArgument(
            "aero_local_waypoints",
            default_value=os.environ.get("ISAAC_AERO_LOCAL_WAYPOINTS", "True"),
        ),
        DeclareLaunchArgument(
            "aero_local_origin_lat_deg",
            default_value=os.environ.get("ISAAC_AERO_LOCAL_ORIGIN_LAT_DEG", "47.836262"),
        ),
        DeclareLaunchArgument(
            "aero_local_origin_lon_deg",
            default_value=os.environ.get("ISAAC_AERO_LOCAL_ORIGIN_LON_DEG", "11.614310"),
        ),
        DeclareLaunchArgument(
            "aero_local_origin_alt_amsl_m",
            default_value=os.environ.get("ISAAC_AERO_LOCAL_ORIGIN_ALT_AMSL_M", "0.0"),
        ),
        DeclareLaunchArgument(
            "aero_local_yaw_deg",
            default_value=os.environ.get("ISAAC_AERO_LOCAL_YAW_DEG", "0.0"),
        ),
        DeclareLaunchArgument(
            "aero_default_cruise_height_agl_m",
            default_value=os.environ.get("ISAAC_AERO_DEFAULT_CRUISE_HEIGHT_AGL_M", "15.0"),
        ),
        OpaqueFunction(function=_team_setup),
    ])
