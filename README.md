# isaac_ros2_manager

Isaac Sim counterpart to `webots_ros2_manager`. It launches a **Webots-style team
into Isaac Sim** and bridges the per-agent ROS contracts so the existing
AUSPEX / CHIP-GT stack (planner, knowledge DB, web app, agent managers) drives
the Isaac entities exactly as it drives Webots ones.

It is the productionised successor to the prototype `isaac_webots_compat`
package, with two gaps closed:

- **Proto-aware spawning** — each agent's Webots `proto` (e.g. `Mavic2ProSimple`,
  `Scout`) selects the Isaac spawn type and defaults via a registry in
  `common.py::_PROTO_REGISTRY`, instead of mapping only by `kind`.
- **Named-location placement** — each agent's `location` (e.g. `area_00_l_init`)
  is resolved to a pose via the mission/area model, instead of index stacking.

## Nodes

| Executable | Node | Role |
|---|---|---|
| `isaac_team_manager_node` | `team_manager` | Reads the team JSON, spawns each agent through `/world_manager/add` (`simulation_interfaces/srv/SpawnEntity`), runs a per-agent `AgentBridge`, serves `move_to_area` / `sweep_area`, mirrors platform state into the knowledge DB. |
| `isaac_env_manager_node` | `env_manager` | Areas, objectives, occupancy map, `/clock`, and the objective/area services the upstream stack expects. |
| `isaac_agent_bridge_node` | `isaac_agent_bridge` | Standalone single-agent bridge (the same `AgentBridge` the team manager hosts inline). |

## How a team becomes an Isaac world

```
teams/isaac.json ──► team_manager
   per agent:
     proto/kind   ──► resolve_proto() ──► uri (drone|carter) + resource defaults
     location     ──► MissionModel.location_xy() ──► local ENU x,y
     (origin_*)   ──► lat_lon_from_xy() ──► GPS lat/lon/alt in resource_string
   ──► SpawnEntity ──► /world_manager/add  (auspex world_manager.py)
   ──► AgentBridge: cmd_vel/set_target/navigate_to_pose ⇄ Isaac topics
```

## Coordinate frames — set the origin

The Webots stack works in **local ENU metres**; the Isaac `world_manager`
spawners are documented in **GPS**. Set `origin_lat` / `origin_lon` / `origin_alt`
to the scene's geo-anchor so agents land in the right place. When the origin is
left at `0,0`, the manager falls back to handing the spawner local metres via the
`SpawnEntity` initial pose instead of emitting null-island GPS.

## Launch

```bash
# Whole team into a running Isaac Sim (world_manager must be up)
ros2 launch isaac_ros2_manager isaac_team.launch.py \
  team:=isaac origin_lat:=47.836292 origin_lon:=11.614310 edge_size:=250.0

# Environment/objectives services only
ros2 launch isaac_ros2_manager isaac_env.launch.py grid_size:=1 edge_size:=250.0
```

## Reset semantics

Teardown of live Isaac entities (especially drones with PX4/AERO) is **not**
handled here by design — reset the world by restarting Isaac Sim, then relaunch.
