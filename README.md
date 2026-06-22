# isaac_ros2_manager

Isaac-side helper package for the APE/Webots ROS contract.

This package is **not** the Isaac platform adapter. The platform adapter lives in
the Isaac Sim extension config/code:

`/home/qnc/Desktop/pip_isaacsim/env_isaacsim/lib/python3.11/site-packages/isaacsim/extsUser/auspex.platform_interface/auspex_platform_interface_python/config/config.jsonnet`

The active `integration_config` statically spawns the Isaac scene. Isaac owns
static world readiness and simulator/sensor output. Real platform adapters, such
as the AUSPEX-AERO bridge in `uav_ranger_manager`, provide control
services/actions consumed by the platform-agnostic skills.

Static objective state is the one simulator contract kept in this ROS package:
`isaac_objective_state_node` loads known trap/objective positions from
`config/integration_objectives.json`, publishes the Webots-compatible
`area_objectives` KNOW document, merges YOLO detections, and serves the
per-agent `detect`/`disarm` actions used by the skill managers.

## What Remains Here

| Executable | Role |
|---|---|
| `isaac_startup_ready_node` | Passive startup reporter. It republishes `/world_manager/ready` as `/isaac_integration/ready`; it does not create map, objectives, clock, or KNOW data. |
| `isaac_objective_state_node` | Loads configured static objectives, publishes `/auspex_know/knowledge_collector/area_objectives`, consumes `/<team>/team_manager/detected_objectives`, and serves `/<team>/<agent>/{detect,disarm}`. |
| `isaac_team_manager_node` | Webots-compatible team facade for Isaac. It publishes team/area KNOW state and exposes `/<team>/move_to_area` / `/<team>/sweep_area`, forwarding movement to `/<team>/ample/execute_plan`. |
| `isaac_yolo_detection_adapter_node` | Converts Isaac camera detections into the existing trap-detection pose topic. |

`isaac_team.launch.py` starts the objective-state node, the Isaac team manager,
and the existing UAV/UGV skill managers. For UAVs it also starts the AUSPEX-AERO
platform adapter that backs `takeoff`, `land`, and `navigate_to_pose`. It does
not spawn agents.

## Removed

The previous synthetic `AgentBridge` path has been removed. There is no
`/world_manager/add` fallback for agents and no synthetic odometry bridge. If a
required endpoint is missing, fix the component that owns that endpoint: Isaac
for static scene/sensors, AUSPEX-AERO or another platform adapter for flight
control, and KNOW/mission publishers for area/objective state.

## Expected Isaac Contract

For `TEAM=chipgt`:

- UAV `mini1`: `/chipgt/mini1/odometry`, `/chipgt/mini1/is_flying`,
  `/chipgt/mini1/takeoff`, `/chipgt/mini1/land`,
  `/chipgt/mini1/navigate_to_pose`, `/chipgt/mini1/detect`,
  `/chipgt/mini1/disarm`, `/chipgt/mini1/front_camera/image_raw`.
  `takeoff`, `land`, and `navigate_to_pose` may be provided by the
  AUSPEX-AERO adapter rather than the Isaac extension.
- UGV `irobot2`: `/chipgt/irobot2/odom`, `/chipgt/irobot2/cmd_vel`,
  `/chipgt/irobot2/navigate_to_pose`, `/chipgt/irobot2/detect`,
  `/chipgt/irobot2/disarm`
- KNOW collectors: `/auspex_know/knowledge_collector/team_data`,
  `/auspex_know/knowledge_collector/area_data`,
  `/auspex_know/knowledge_collector/area_objectives`

`/world_manager/ready` remains the startup barrier. It means the static Isaac
scene is loaded and the rest of APE can start.
