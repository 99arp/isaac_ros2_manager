from __future__ import annotations

import ast
import json
import random
import time

import rclpy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rosgraph_msgs.msg import Clock
from rosidl_runtime_py.convert import message_to_ordereddict
from std_srvs.srv import Trigger
from webots_ros2_manager_msgs.msg import Objective
from webots_ros2_manager_msgs.srv import ObjectiveService, SendString

from .common import MissionModel, param_bool, param_float_array

try:
    from . import auspex_compat
except Exception:
    auspex_compat = None

try:
    from simulation_interfaces.srv import DeleteEntity, SpawnEntity
except Exception:
    DeleteEntity = None
    SpawnEntity = None


class IsaacEnvironmentManager(Node):
    """Isaac-side stand-in for the Webots env_manager: areas, objectives, map, clock."""

    def __init__(self):
        super().__init__("env_manager")
        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameter("grid_size", 1)
        self.declare_parameter("edge_size", 250.0)
        self.declare_parameter("world_offset", [0.0, 0.0], descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("ground_z", 0.0)
        self.declare_parameter("objectives_per_area", 0)
        self.declare_parameter("objective_seed", 1)
        self.declare_parameter("spawn_objectives", True)
        self.declare_parameter("auto_update_db", True)
        self.declare_parameter("publish_clock", True)
        self.declare_parameter("map_rate_hz", 0.2)
        self.declare_parameter("clock_rate_hz", 20.0)
        self.declare_parameter("spawn_with_isaac", True)
        self.declare_parameter("objective_model", "bear_trap")
        self.declare_parameter("objective_z_offset", 0.0)
        self.declare_parameter("objective_target_length_m", 1.0)

        self.grid_size = int(self.get_parameter("grid_size").value)
        self.edge_size = float(self.get_parameter("edge_size").value)
        self.world_offset = param_float_array(self.get_parameter("world_offset").value)
        self.ground_z = float(self.get_parameter("ground_z").value)
        self.auto_update_db = param_bool(self.get_parameter("auto_update_db").value)
        self.publish_clock = param_bool(self.get_parameter("publish_clock").value)
        self.spawn_with_isaac = param_bool(self.get_parameter("spawn_with_isaac").value)
        self.objective_model = str(self.get_parameter("objective_model").value)
        self.objective_z_offset = float(self.get_parameter("objective_z_offset").value)
        self.objective_target_length_m = float(self.get_parameter("objective_target_length_m").value)
        self.mission = MissionModel.build(self.grid_size, self.edge_size, self.world_offset)
        self.objectives: dict[str, Objective] = {}
        self.teams: set[str] = set()
        self.start_time = time.monotonic()
        self.spawned_objective_entities: set[str] = set()
        self.pending_spawn_objective_entities: set[str] = set()
        self.pending_delete_objective_entities: set[str] = set()
        self.removed_objective_entities: set[str] = set()

        if param_bool(self.get_parameter("spawn_objectives").value):
            self._create_default_objectives()

        self.db_client = None
        if auspex_compat is not None and self.auto_update_db:
            self.db_client = auspex_compat.make_write_client(self)

        self.spawn_client = None
        self.delete_client = None
        if self.spawn_with_isaac and SpawnEntity is not None:
            self.spawn_client = self.create_client(SpawnEntity, "/world_manager/add", callback_group=self.cb_group)
        if self.spawn_with_isaac and DeleteEntity is not None:
            self.delete_client = self.create_client(DeleteEntity, "/world_manager/remove", callback_group=self.cb_group)
        if self.spawn_with_isaac and (self.spawn_client is None or self.delete_client is None):
            self.get_logger().warning("simulation_interfaces world-manager services unavailable; objective USD sync disabled")

        self.create_service(SendString, "~/add_controller", self._add_controller_cb, callback_group=self.cb_group)
        self.create_service(Trigger, "~/grid_size", self._grid_size_cb, callback_group=self.cb_group)
        self.create_service(SendString, "~/get_map_height", self._get_map_height_cb, callback_group=self.cb_group)
        self.create_service(ObjectiveService, "~/change_objective_state", self._change_objective_state_cb, callback_group=self.cb_group)
        self.create_service(ObjectiveService, "~/get_objective_type", self._get_objective_type_cb, callback_group=self.cb_group)
        self.create_service(Trigger, "~/get_heightmap_json", self._heightmap_json_cb, callback_group=self.cb_group)
        self.create_service(SendString, "~/spawn_heightmap", self._ok_string_cb("heightmap spawn is a no-op in Isaac compat"), callback_group=self.cb_group)
        self.create_service(SendString, "~/spawn_obstacle_map", self._ok_string_cb("obstacle map spawn is a no-op in Isaac compat"), callback_group=self.cb_group)
        self.create_service(SendString, "~/save_map_image", self._ok_string_cb("map image save is a no-op in Isaac compat"), callback_group=self.cb_group)
        self.create_service(SendString, "~/spawn_traps_in_area", self._spawn_traps_cb, callback_group=self.cb_group)
        self.create_service(SendString, "~/remove_traps_in_area", self._remove_traps_cb, callback_group=self.cb_group)

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
        self.map_pub = self.create_publisher(OccupancyGrid, "~/map", qos_profile=qos)
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)

        map_rate = max(float(self.get_parameter("map_rate_hz").value), 0.01)
        self.create_timer(1.0 / map_rate, self._publish_map, callback_group=self.cb_group)
        if self.publish_clock:
            clock_rate = max(float(self.get_parameter("clock_rate_hz").value), 1.0)
            self.create_timer(1.0 / clock_rate, self._publish_clock, callback_group=self.cb_group)
        if self.db_client is not None:
            self.create_timer(3.0, self._update_db, callback_group=self.cb_group)
        if self.spawn_with_isaac and self.spawn_client is not None:
            self.create_timer(2.0, self._sync_isaac_objectives, callback_group=self.cb_group)

        self._publish_map()
        self._update_db()
        self.get_logger().info(
            f"Isaac env_manager ready: grid={self.grid_size}, edge={self.edge_size},"
            f" offset={self.mission.world_offset}, objectives={len(self.objectives)}"
        )

    def _create_default_objectives(self) -> None:
        count = max(0, int(self.get_parameter("objectives_per_area").value))
        rng = random.Random(int(self.get_parameter("objective_seed").value))
        for area in self.mission.areas.values():
            for idx in range(count):
                obj = Objective()
                obj.name = f"{area.area_id}_objective_{idx:02d}"
                obj.status = Objective.ACTIVE
                obj.type = Objective.UNKNOWN
                obj.position.x = area.offset[0] + rng.uniform(0.2, 0.8) * area.size[0]
                obj.position.y = area.offset[1] + rng.uniform(0.2, 0.8) * area.size[1]
                obj.position.theta = 0.0
                self.objectives[obj.name] = obj

    def _add_controller_cb(self, request: SendString.Request, response: SendString.Response) -> SendString.Response:
        self.teams.add(request.data)
        response.success = True
        response.message = f"Controller '{request.data}' registered in Isaac compat."
        return response

    def _grid_size_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        response.success = True
        response.message = str(self.grid_size)
        return response

    def _get_map_height_cb(self, request: SendString.Request, response: SendString.Response) -> SendString.Response:
        try:
            data = ast.literal_eval(request.data) if request.data else [0.0, 0.0]
            if len(data) < 2:
                raise ValueError("expected [x, y]")
            response.success = True
            response.message = str(self.ground_z)
        except Exception as exc:
            response.success = False
            response.message = f"Could not parse requested XY data {request.data!r}: {exc}"
        return response

    def _find_objective(self, request: ObjectiveService.Request) -> Objective | None:
        name = request.name.strip()
        if name in self.objectives:
            return self.objectives[name]
        if name and not name.startswith("area_"):
            suffix = name.split("_")[-1]
            suffixes = {f"_{suffix}"}
            if suffix.isdecimal():
                suffixes.add(f"_{int(suffix):02d}")
            for obj in self.objectives.values():
                if any(obj.name.endswith(candidate) for candidate in suffixes):
                    return obj
        for obj in self.objectives.values():
            if abs(obj.position.x - request.position.x) <= 3.0 and abs(obj.position.y - request.position.y) <= 3.0:
                return obj
        return None

    def _change_objective_state_cb(self, request: ObjectiveService.Request, response: ObjectiveService.Response) -> ObjectiveService.Response:
        obj = self._find_objective(request)
        if obj is None:
            response.success = False
            response.message = "objective not found"
            return response
        was_active = obj.status == Objective.ACTIVE
        obj.status = Objective.INACTIVE
        response.success = True
        response.message = "status changed" if was_active else "status already inactive"
        response.status = obj.status
        response.type = obj.type
        self._update_db()
        self._sync_isaac_objectives()
        return response

    def _get_objective_type_cb(self, request: ObjectiveService.Request, response: ObjectiveService.Response) -> ObjectiveService.Response:
        obj = self._find_objective(request)
        response.success = True
        if obj is None:
            response.message = str(Objective.NO_TRAP)
            response.type = Objective.NO_TRAP
            response.status = Objective.INACTIVE
            return response
        if obj.type == Objective.UNKNOWN:
            obj.type = random.choice([Objective.IS_GROUND, Objective.IS_AIR, Objective.IS_COMBINED])
        response.message = str(obj.type)
        response.type = obj.type
        response.status = obj.status
        self._update_db()
        return response

    def _heightmap_json_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        response.success = True
        response.message = json.dumps({
            "dimensions": [self.grid_size + 1, self.grid_size + 1],
            "spacing": [self.edge_size, self.edge_size],
            "height": [[self.ground_z for _ in range(self.grid_size + 1)] for _ in range(self.grid_size + 1)],
        })
        return response

    def _ok_string_cb(self, message: str):
        def callback(request: SendString.Request, response: SendString.Response) -> SendString.Response:
            response.success = True
            response.message = message
            return response
        return callback

    def _spawn_traps_cb(self, request: SendString.Request, response: SendString.Response) -> SendString.Response:
        area = self.mission.areas.get(request.data)
        if area is None:
            response.success = False
            response.message = f"Unknown area {request.data!r}"
            return response
        existing = [name for name in self.objectives if name.startswith(area.area_id + "_objective_")]
        if not existing:
            obj = Objective()
            obj.name = f"{area.area_id}_objective_00"
            obj.status = Objective.ACTIVE
            obj.type = Objective.UNKNOWN
            pose = area.default_pose()
            obj.position = pose
            self.objectives[obj.name] = obj
        response.success = True
        response.message = f"Spawned traps in {area.area_id}"
        self._update_db()
        self._sync_isaac_objectives()
        return response

    def _remove_traps_cb(self, request: SendString.Request, response: SendString.Response) -> SendString.Response:
        prefix = request.data + "_objective_"
        for name in list(self.objectives):
            if name.startswith(prefix):
                self.removed_objective_entities.add(name)
                self._delete_objective_entity(name)
                del self.objectives[name]
        response.success = True
        response.message = f"Removed traps in {request.data}"
        self._update_db()
        return response

    def _sync_isaac_objectives(self) -> None:
        if not self.spawn_with_isaac:
            return
        for name in list(self.removed_objective_entities):
            if name not in self.pending_delete_objective_entities:
                self._delete_objective_entity(name)
        for name in list(self.spawned_objective_entities):
            if name not in self.objectives and name not in self.pending_delete_objective_entities:
                self.removed_objective_entities.add(name)
                self._delete_objective_entity(name)
        for obj in self.objectives.values():
            if (
                obj.status == Objective.ACTIVE
                and obj.name not in self.spawned_objective_entities
                and obj.name not in self.pending_spawn_objective_entities
            ):
                self._spawn_objective_entity(obj)
            elif (
                obj.status != Objective.ACTIVE
                and obj.name in self.spawned_objective_entities
                and obj.name not in self.pending_delete_objective_entities
            ):
                self._delete_objective_entity(obj.name)

    def _spawn_objective_entity(self, obj: Objective) -> None:
        if self.spawn_client is None:
            return
        if not self.spawn_client.service_is_ready():
            return

        self.pending_spawn_objective_entities.add(obj.name)
        request = SpawnEntity.Request()
        request.name = obj.name
        request.uri = "object"
        request.entity_namespace = obj.name
        request.allow_renaming = False
        request.initial_pose.header.frame_id = "map"
        request.initial_pose.pose.position.x = float(obj.position.x)
        request.initial_pose.pose.position.y = float(obj.position.y)
        request.initial_pose.pose.position.z = float(self.ground_z + self.objective_z_offset)
        request.initial_pose.pose.orientation.w = 1.0
        resource = {
            "model": self.objective_model,
            "stage_prefix": obj.name,
            "init_pos": [
                float(obj.position.x),
                float(obj.position.y),
                float(self.ground_z + self.objective_z_offset),
            ],
            "snap_to_terrain": True,
            "z_offset_m": float(self.objective_z_offset),
            "publish_pose_ros": True,
            "ros_topic_identifier": obj.name,
            "pose_frame_id": "map",
        }
        if self.objective_target_length_m > 0.0:
            resource["target_length_m"] = float(self.objective_target_length_m)
        request.resource_string = json.dumps(resource)
        future = self.spawn_client.call_async(request)
        future.add_done_callback(lambda fut, name=obj.name: self._spawn_objective_done(name, fut))

    def _spawn_objective_done(self, name: str, future) -> None:
        self.pending_spawn_objective_entities.discard(name)
        try:
            response = future.result()
            ok = getattr(response.result, "result", 0) == response.result.RESULT_OK
            if ok:
                self.spawned_objective_entities.add(name)
                self.get_logger().info(f"Spawned Isaac objective {name}")
            else:
                self.get_logger().warning(f"Isaac objective spawn failed for {name}: {response.result.error_message}")
        except Exception as exc:
            self.get_logger().warning(f"Isaac objective spawn call failed for {name}: {exc}")

    def _delete_objective_entity(self, name: str) -> None:
        if self.delete_client is None:
            self.spawned_objective_entities.discard(name)
            self.removed_objective_entities.discard(name)
            return
        if not self.delete_client.service_is_ready():
            return

        self.pending_delete_objective_entities.add(name)
        request = DeleteEntity.Request()
        request.entity = name
        future = self.delete_client.call_async(request)
        future.add_done_callback(lambda fut, entity=name: self._delete_objective_done(entity, fut))

    def _delete_objective_done(self, name: str, future) -> None:
        self.pending_delete_objective_entities.discard(name)
        try:
            response = future.result()
            ok = getattr(response.result, "result", 0) == response.result.RESULT_OK
            if ok:
                self.spawned_objective_entities.discard(name)
                self.removed_objective_entities.discard(name)
                self.get_logger().info(f"Removed Isaac objective {name}")
            else:
                self.get_logger().warning(f"Isaac objective remove failed for {name}: {response.result.error_message}")
        except Exception as exc:
            self.get_logger().warning(f"Isaac objective remove call failed for {name}: {exc}")

    def _publish_map(self) -> None:
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.resolution = float(self.edge_size)
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size
        msg.info.origin.position.x = float(self.mission.world_offset[0])
        msg.info.origin.position.y = float(self.mission.world_offset[1])
        msg.info.origin.orientation.w = 1.0
        msg.data = [0 for _ in range(self.grid_size * self.grid_size)]
        self.map_pub.publish(msg)

    def _publish_clock(self) -> None:
        msg = Clock()
        elapsed = time.monotonic() - self.start_time
        msg.clock.sec = int(elapsed)
        msg.clock.nanosec = int((elapsed - int(elapsed)) * 1e9)
        self.clock_pub.publish(msg)

    def _area_object_db_entries(self) -> list[dict]:
        by_area: dict[str, dict] = {}
        for area_id, area in self.mission.areas.items():
            init_pose = area.default_pose()
            by_area[area_id] = {
                "area_id": area_id,
                "locations": {
                    f"{area_id}_l_init": {
                        "type": "Waypoint",
                        "data": message_to_ordereddict(init_pose),
                    }
                },
                "objectives": {},
            }
        for obj in self.objectives.values():
            area = self.mission.area_at(obj.position.x, obj.position.y)
            area_id = area.area_id if area is not None else obj.name.split("_objective_")[0]
            location = obj.name.replace("_objective_", "_l_")
            by_area.setdefault(area_id, {"area_id": area_id, "locations": {}, "objectives": {}})
            by_area[area_id]["objectives"][obj.name] = {
                "type": "Objective",
                "location": location,
                "data": message_to_ordereddict(obj),
            }
            by_area[area_id]["locations"][location] = {
                "type": "Waypoint",
                "data": message_to_ordereddict(obj.position),
            }
        return list(by_area.values())

    def _update_db(self) -> None:
        if self.db_client is None:
            return
        if not self.db_client.service_is_ready() and not self.db_client.wait_for_service(timeout_sec=0.01):
            return

        self.db_client.call_async(auspex_compat.write_request(
            "area", instance_id="default", entity=self.mission.as_area_db_entry()))

        for entry in self._area_object_db_entries():
            self.db_client.call_async(auspex_compat.write_request(
                "object", instance_id=entry["area_id"], entity=entry))


def main(args=None):
    rclpy.init(args=args)
    node = IsaacEnvironmentManager()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
