# isaac_ros2_manager

Isaac-side helper package for the APE/Webots ROS contract.



The active `integration_config` statically spawns the Isaac scene. Isaac owns
static world readiness and simulator/sensor output. 

AUSPEX-AERO bridge in `uav_ranger_manager`, provide control
services/actions


`isaac_objective_state_node` loads known trap/objective positions from
`config/integration_objectives.json`, publishes the Webots-compatible
`area_objectives` KNOW document, merges YOLO detections, and serves the
per-agent `detect`/`disarm` actions used by the skill managers.

The APE oracle still calls `/<team>/move_to_area`, so `isaac_team_manager_node`
remains as a compatibility facade for team-level actions. 

Carter I/O in normal path should be in non-ros mode self contained. this is just a compatiblity layer tat we introduce. 
It must not spawn
Isaac agents and it should not bridge Carter I/O in the normal static-extension
path.

## Coordinate Frames

The ROS facade publishes Webots-compatible KNOW data in the local oracle frame.
Native Isaac/Nav2 coordinates are converted with:

`local = world_offset + (native - nav_world_offset) * edge_size / nav_edge_size`

`WORLD=isaac` defaults `nav_edge_size` and `nav_world_offset` to the same values
as the backend local frame. The active Isaac `integration_config` centers the
terrain at the backend origin, so Isaac native XY is already backend/local XY.
Override `ISAAC_WORLD_OFFSET`, `ISAAC_NAV_EDGE_SIZE`,
`ISAAC_NAV_WORLD_OFFSET`, or `ISAAC_OBJECTIVES_FRAME` only when the Isaac scene
uses a different native frame.



## How I launch this in startup 


  ros2 run isaac_ros2_manager isaac_team_manager_node --ros-args \
    -r __ns:=/chipgt \
    -p team_file:=/home/qnc/Desktop/isaacsim_integration_project/chipgt_bringup/
    teams/chipgt_isaac.json \
    -p grid_size:=4 \
    -p edge_size:=40.0 \
    -p world_offset:="0.0,0.0" \
    -p nav_edge_size:=40.0 \
    -p nav_world_offset:="0.0,0.0" \
    -p uav_cruise_height_agl_m:=15.0



## What is not yet yet tested:

