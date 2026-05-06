import multiprocessing as mp
import os
from copy import deepcopy
import time
import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os.path as osp
from pathlib import Path
import numpy as np
from transforms3d.euler import euler2quat
import itertools
from typing import Callable
import toppra as ta
import mplib
from mplib.sapien_utils.conversion import convert_object_name
from mplib.collision_detection.fcl import CollisionGeometry
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
from mplib.collision_detection.fcl import Convex, CollisionObject, FCLObject
from mplib.collision_detection import fcl
from mplib.sapien_utils.urdf_exporter import export_kinematic_chain_urdf
from mplib.sapien_utils.srdf_exporter import export_srdf
from mplib.urdf_utils import generate_srdf, replace_urdf_package_keyword

import sapien
import sapien.physx as physx
from sapien import Entity
from sapien.physx import (
    PhysxArticulation,
    PhysxArticulationLinkComponent,
    PhysxCollisionShapeConvexMesh,
    PhysxRigidBaseComponent
)


from typing import Literal, Optional, Sequence, Union
import sys
import trimesh
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.examples.motionplanning.panda.solutions import solvePushCube, solvePickCube, solveStackCube, solvePegInsertionSide, solvePlugCharger, solvePullCubeTool, solveLiftPegUpright, solvePullCube
from mani_skill.envs.tasks import PickCubeEnv
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver

from mani_skill.utils import common
from mani_skill.utils.structs import Actor

from dsynth.assets.ss_assets import WIDTH, DEPTH

BAD_ENV_ERROR_CODE = -1234

def attach_object(  # type: ignore
    planning_world: SapienPlanningWorld,
    obj: Union[Entity, str],
    articulation: Union[PhysxArticulation, str],
    link: Union[PhysxArticulationLinkComponent, int],
    pose: Optional[mplib.Pose] = None,
    *,
    touch_links: Optional[list[Union[PhysxArticulationLinkComponent, str]]] = None,
    obj_geom: Optional[CollisionGeometry] = None,
) -> None:
    """
    Attaches given non-articulated object to the specified link of articulation.

    Updates ``acm_`` to allow collisions between attached object and touch_links.

    :param obj: the non-articulated object (or its name) to attach
    :param articulation: the planned articulation (or its name) to attach to
    :param link: the link of the planned articulation (or its index) to attach to
    :param pose: attached pose (relative pose from attached link to object).
        If ``None``, attach the object at its current pose.
    :param touch_links: links (or their names) that the attached object touches.
        When ``None``,

        * if the object is not currently attached, touch_links are set to the name
        of articulation links that collide with the object in the current state.

        * if the object is already attached, touch_links of the attached object
        is preserved and ``acm_`` remains unchanged.
    :param obj_geom: a CollisionGeometry object representing the attached object.
        If not ``None``, pose must be not ``None``.

    .. raw:: html

        <details>
        <summary><a>Overloaded
        <code class="docutils literal notranslate">
        <span class="pre">PlanningWorld.attach_object()</span>
        </code>
        methods</a></summary>
    .. automethod:: mplib.PlanningWorld.attach_object
        :no-index:
    .. raw:: html
        </details>
    """
    kwargs = {"name": obj, "art_name": articulation, "link_id": link}
    if pose is not None:
        kwargs["pose"] = pose
    if touch_links is not None:
        kwargs["touch_links"] = [
            l.name if isinstance(l, PhysxArticulationLinkComponent) else l
            for l in touch_links  # noqa: E741
        ]
    if obj_geom is not None:
        kwargs["obj_geom"] = obj_geom

    if isinstance(obj, Entity):
        kwargs["name"] = convert_object_name(obj)
    if isinstance(articulation, PhysxArticulation):
        kwargs["art_name"] = articulation = convert_object_name(articulation)
    if isinstance(link, PhysxArticulationLinkComponent):
        kwargs["link_id"] = (
            planning_world.get_articulation(articulation)
            .get_pinocchio_model()
            .get_link_names()
            .index(link.name)
        )

    planning_world.attach_object(**kwargs)


def get_fcl_object_name(entity):
    component = entity._objs[0].find_component_by_type(physx.PhysxRigidBaseComponent)
    return convert_object_name(component.entity)

def rodrigues_rotation(v, axis, theta):
    axis = axis / np.linalg.norm(axis)  # Ensure unit vector
    
    # Rodrigues' formula for vector rotation
    v_rot = (v * np.cos(theta) + 
             np.cross(axis, v) * np.sin(theta) + 
             axis * np.dot(axis, v) * (1 - np.cos(theta)))
    return v_rot

def get_tcp_pose(env):
    return env.agent.tcp.pose.sp

def get_tcp_matrix(env):
    tcp_pose = get_tcp_pose(env)
    return tcp_pose.to_transformation_matrix()

def get_base_pose(env):
        return env.agent.base_link.pose

def get_shoulder_pan_pose(env):
    return env.agent.shoulder_pan_link.pose

def get_base_shift_tcp_to_target(env, target_pos: np.ndarray):
    dist_to_target = get_tcp_pose(env).p - target_pos
    dist_to_target[2] = 0
    return np.linalg.norm(dist_to_target)

def get_direction_to_shelf(env):
    actor_shelf_name = env.active_shelves[0][0]
    shelf_pos = env.actors["fixtures"]["shelves"][actor_shelf_name].pose.sp.p
    shelf_direction = env.directions_to_shelf[0]
    return np.abs((get_tcp_pose(env).p - shelf_pos) @ shelf_direction) - DEPTH / 2

def compute_cylinder_grasp_info(
    actor: Actor,
    target_closing=None,
    ee_direction=None,
    depth=0.0,
    ortho=True,
    n_grasps_central = 1,
    n_grasps_lateral = 1,
    central_angle_range = [-np.pi/4, np.pi/4],
    lateral_angle_range = [-np.pi/4, np.pi/4],
):
    approaching = ee_direction.copy()
    approaching[2] = 0.
    approaching = common.np_normalize_vector(approaching)
    lateral_direction = np.cross(approaching, [0, 0, 1])

    mesh = get_component_mesh(
        actor._objs[0].find_component_by_type(physx.PhysxRigidDynamicComponent),
        to_world_frame=True,
    )
    assert mesh is not None, "can not get actor mesh for {}".format(actor)

    cylinder: trimesh.primitives.Cylinder = mesh.bounding_cylinder
    cylinder_obb: trimesh.primitives.Box = cylinder.bounding_box_oriented
    half_size = cylinder_obb.primitive.extents[0] / 2
    center = cylinder_obb.primitive.transform[:3, 3]

    central_angles = [0.]
    if n_grasps_central > 1:
        central_angles.extend(np.linspace(central_angle_range[0], central_angle_range[1], n_grasps_central - 1))

    lateral_angles = [0.]
    if n_grasps_lateral > 1:
        lateral_angles.extend(np.linspace(lateral_angle_range[0], lateral_angle_range[1], n_grasps_lateral - 1))

    grasps = []
    for central_angle, lateral_angle in itertools.product(central_angles, lateral_angles):
        approach_vector = rodrigues_rotation(approaching, [0, 0, 1], lateral_angle)
        approach_vector = rodrigues_rotation(approach_vector, lateral_direction, central_angle)
        
        closing = np.cross(approach_vector, [0, 0, 1])
        closing = common.np_normalize_vector(closing)

        half_size_rot = half_size / np.cos(central_angle)   
        origin = center + approaching * (-half_size_rot + min(depth, half_size_rot))
        
        grasp_info = dict(
            approaching=approach_vector, closing=closing, center=origin,
        )
        grasps.append(grasp_info)

    return grasps
    


def compute_box_grasp_thin_side_info(
    obb: trimesh.primitives.Box,
    target_closing=None,
    ee_direction=None,
    depth=0.0,
    ortho=True,
    n_grasps = 1,
    approaching_angles_range = [-np.pi/4, np.pi/4],
):
    """Compute grasp info given an oriented bounding box.
    The grasp info includes axes to define grasp frame, namely approaching, closing, orthogonal directions and center.

    Args:
        obb: oriented bounding box to grasp
        approaching: direction to approach the object
        target_closing: target closing direction, used to select one of multiple solutions
        depth: displacement from hand to tcp along the approaching vector. Usually finger length.
        ortho: whether to orthogonalize closing  w.r.t. approaching.
    """
    # NOTE(jigu): DO NOT USE `x.extents`, which is inconsistent with `x.primitive.transform`!
    extents = np.array(obb.primitive.extents)
    T = np.array(obb.primitive.transform)

    for i in range(3):
        if np.abs(T[:3, i] @ [0, 0, 1]) > 1e-1:
            height_extent_ids = i
            break

    inds = np.argsort(extents)
    inds = inds[inds != height_extent_ids]
    short_base_side_ind = inds[0]
    long_base_side_ind = inds[1]

    # height = extents[2]

    approaching_angles = [0.]
    if n_grasps > 1:
        approaching_angles.extend(np.linspace(approaching_angles_range[0], approaching_angles_range[1], n_grasps - 1))
    
    approaching_vectors = []
    default_approaching = np.array(T[:3, long_base_side_ind])
    default_approaching = common.np_normalize_vector(default_approaching)

    if ee_direction @ default_approaching < 0:
        default_approaching = -default_approaching

    normal_direction = np.cross(default_approaching, [0, 0, 1])

    for approaching_angle in approaching_angles:
        approaching = rodrigues_rotation(default_approaching, normal_direction, approaching_angle)
        approaching_vectors.append(approaching)

    closing = np.array(T[:3, short_base_side_ind])
    if target_closing is not None and target_closing @ closing < 0:
        closing = -closing

    # Find the origin on the surface
    center = T[:3, 3]
    half_size = extents[long_base_side_ind] / 2

    grasps = []
    for approaching, approaching_angle in zip(approaching_vectors, approaching_angles):
        half_size_rot = half_size / np.cos(approaching_angle)   
        origin = center + approaching * (-half_size_rot + min(depth, half_size_rot))

        if ortho:
            cur_closing = closing - (approaching @ closing) * approaching
            cur_closing = common.np_normalize_vector(cur_closing)

        grasp_info = dict(
            approaching=approaching, closing=cur_closing, center=origin, extents=extents
        )
        grasps.append(grasp_info)

    if len(grasps) == 1:
        return grasps[0]
    return grasps

def convert_actor_convex_mesh_to_fcl(actor: Actor):
    component = actor._objs[0].find_component_by_type(physx.PhysxRigidBaseComponent)
    assert component is not None, (
        f"No PhysxRigidBaseComponent found in {actor.name}: "
        f"{actor.components=}"
    )
    assert len(component.collision_shapes) == 1
    shape = component.collision_shapes[0]
    assert isinstance(shape, physx.PhysxCollisionShapeConvexMesh)

    # tranform vertices, so that scale == 1.0
    vertices = shape.vertices
    vertices[:, 0] *= shape.scale[0]
    vertices[:, 1] *= shape.scale[1]
    vertices[:, 2] *= shape.scale[2]
    c_geom = Convex(vertices=vertices, faces=shape.triangles)
    collision_shape = CollisionObject(c_geom)

    return FCLObject(
        convert_object_name(component.entity),
        component.entity.pose,
        [collision_shape],
        [mplib.Pose(shape.local_pose)],
    )

def is_mesh_cylindrical(actor, to_world_frame=True, thresh=5e-3):
    mesh = get_component_mesh(
        actor._objs[0].find_component_by_type(physx.PhysxRigidDynamicComponent),
        to_world_frame=to_world_frame,
    )
    assert mesh is not None, "can not get actor mesh for {}".format(actor)

    obb: trimesh.primitives.Box = mesh.bounding_box_oriented
    cylinder: trimesh.primitives.Cylinder = mesh.bounding_cylinder
    cylinder_obb: trimesh.primitives.Box = cylinder.bounding_box_oriented

    h_obb, w_obb = obb.primitive.extents[:2]
    h_c_obb, w_c_obb = cylinder_obb.primitive.extents[:2]

    #if extents are equal up to the permutation then the mesh is cylindrical
    if np.abs(h_obb * w_obb - h_c_obb * w_c_obb) < thresh and \
        np.abs(h_obb + w_obb - h_c_obb - w_c_obb) < thresh:
        return True
    return False
    

class SapienPlanningWorldV2(SapienPlanningWorld):
    """
    Patched version of SapienPlanningWorld for meshes with scale
    """
    def __init__(
        self,
        sim_scene: sapien.Scene,
        user_link_names: Sequence[str] = [],
        user_joint_names: Sequence[str] = [],
        planned_articulations: list[PhysxArticulation] = [],  # noqa: B006
        planned_urdf_paths: list[str] = [], # if populated with [None, None, ...,] then it loads urdfs from articulations
        disable_actors_collision=False,
        new_package_keyword: str = "",
        use_convex: bool = False,
        verbose: bool = False,
    ):
        """
        Creates an mplib.PlanningWorld from a sapien.Scene.

        :param planned_articulations: list of planned articulations.
        """
        assert len(planned_articulations) == len(planned_urdf_paths), "planned_articulations and planned_urdf_paths must have the same length"
        self.planned_articulations = planned_articulations
        mplib.PlanningWorld.__init__(self, [])
        self._sim_scene = sim_scene
        self.disable_actors_collision = disable_actors_collision

        articulations: list[PhysxArticulation] = sim_scene.get_all_articulations()
        actors: list[Entity] = sim_scene.get_all_actors()

        for articulation in articulations:
            if not self.disable_actors_collision:
                if articulation in planned_articulations:
                    continue # skip planned articulations
                urdf_str = export_kinematic_chain_urdf(articulation)
                srdf_str = export_srdf(articulation)

                # Convert all links to FCLObject
                collision_links = [
                    fcl_obj
                    for link in articulation.links
                    if (fcl_obj := self.convert_physx_component(link)) is not None
                ]

                articulated_model = mplib.ArticulatedModel.create_from_urdf_string(
                    urdf_str,
                    srdf_str,
                    collision_links=collision_links,
                    gravity=sim_scene.get_physx_system().config.gravity,  # type: ignore
                    link_names=[link.name for link in articulation.links],
                    joint_names=[j.name for j in articulation.active_joints],
                    verbose=False,
                )
                articulated_model.set_base_pose(articulation.root_pose)  # type: ignore
                articulated_model.set_qpos(
                    articulation.qpos,  # type: ignore
                    full=True,
                )  # update qpos
                self.add_articulation(articulated_model)

        for articulation, urdf_path in zip(planned_articulations, planned_urdf_paths):
            if urdf_path is not None:
                urdf_path = Path(urdf_path)
                if (srdf_path := urdf_path.with_suffix(".srdf")).is_file() or (
                    srdf_path := urdf_path.with_name(urdf_path.stem + "_mplib.srdf")
                ).is_file():
                    print(f"No SRDF file provided but found {srdf_path}")
                else:
                    srdf_path = generate_srdf(urdf_path, new_package_keyword, verbose=True)
                urdf_path = replace_urdf_package_keyword(urdf_path, new_package_keyword)
                articulated_model = mplib.ArticulatedModel(
                    str(urdf_path),
                    str(srdf_path),
                    name=convert_object_name(articulation),
                    link_names=user_link_names,  # type: ignore
                    joint_names=user_joint_names,  # type: ignore
                    convex=use_convex,
                    verbose=verbose,
                )
            else:
                urdf_str = export_kinematic_chain_urdf(articulation)
                srdf_str = export_srdf(articulation)

                # Convert all links to FCLObject
                collision_links = [
                    fcl_obj
                    for link in articulation.links
                    if (fcl_obj := self.convert_physx_component(link)) is not None
                ]

                articulated_model = mplib.ArticulatedModel.create_from_urdf_string(
                    urdf_str,
                    srdf_str,
                    collision_links=collision_links,
                    gravity=sim_scene.get_physx_system().config.gravity,  # type: ignore
                    link_names=[link.name for link in articulation.links],
                    joint_names=[j.name for j in articulation.active_joints],
                    verbose=False,
                )
            articulated_model.set_base_pose(articulation.root_pose)  # type: ignore
            articulated_model.set_qpos(
                articulation.qpos[4:],  # type: ignore
                full=True,
            )  # update qpos
            self.add_articulation(articulated_model)
            self.set_articulation_planned(convert_object_name(articulation), True)
        
        if not self.disable_actors_collision:
            for entity in actors:
                # if self.disable_actors_collision and 'food' in entity.name:
                #     continue
                component = entity.find_component_by_type(sapien.physx.PhysxRigidBaseComponent)
                assert component is not None, (
                    f"No PhysxRigidBaseComponent found in {entity.name}: "
                    f"{entity.components=}"
                )

                # Convert collision shapes at current global pose
                if (fcl_obj := self.convert_physx_component(component)) is not None:  # type: ignore
                    self.add_object(fcl_obj)

    @staticmethod
    def convert_physx_component(comp: physx.PhysxRigidBaseComponent) -> FCLObject | None:
        """
        Converts a SAPIEN physx.PhysxRigidBaseComponent to an FCLObject.
        All shapes in the returned FCLObject are already set at their world poses.

        :param comp: a SAPIEN physx.PhysxRigidBaseComponent.
        :return: an FCLObject containing all collision shapes in the Physx component.
            If the component has no collision shapes, return ``None``.
        """
        shapes: list[CollisionObject] = []
        shape_poses: list[mplib.Pose] = []
        for shape in comp.collision_shapes:
            shape_poses.append(mplib.Pose(shape.local_pose))  # type: ignore

            if isinstance(shape, physx.PhysxCollisionShapeBox):
                c_geom = fcl.Box(side=shape.half_size * 2)
            elif isinstance(shape, physx.PhysxCollisionShapeCapsule):
                c_geom = fcl.Capsule(radius=shape.radius, lz=shape.half_length * 2)
                # NOTE: physx Capsule has x-axis along capsule height
                # FCL Capsule has z-axis along capsule height
                shape_poses[-1] *= mplib.Pose(q=euler2quat(0, np.pi / 2, 0))
            elif isinstance(shape, PhysxCollisionShapeConvexMesh):
                # assert np.allclose(
                #     shape.scale, 1.0
                # ), f"Not unit scale {shape.scale}, need to rescale vertices?"

                # Scale vertices!
                vertices = shape.vertices
                vertices[:, 0] *= shape.scale[0]
                vertices[:, 1] *= shape.scale[1]
                vertices[:, 2] *= shape.scale[2]
                c_geom = Convex(vertices=vertices, faces=shape.triangles)
            elif isinstance(shape, physx.PhysxCollisionShapeCylinder):
                c_geom = fcl.Cylinder(radius=shape.radius, lz=shape.half_length * 2)
                # NOTE: physx Cylinder has x-axis along cylinder height
                # FCL Cylinder has z-axis along cylinder height
                shape_poses[-1] *= mplib.Pose(q=euler2quat(0, np.pi / 2, 0))
            elif isinstance(shape, physx.PhysxCollisionShapePlane):
                # PhysxCollisionShapePlane are actually a halfspace
                # https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/docs/Geometry.html#planes
                # PxPlane's Pose determines its normal and offert (normal is +x)
                n = shape_poses[-1].to_transformation_matrix()[:3, 0]
                d = n.dot(shape_poses[-1].p)
                c_geom = fcl.Halfspace(n=n, d=d)
                shape_poses[-1] = mplib.Pose()
            elif isinstance(shape, physx.PhysxCollisionShapeSphere):
                c_geom = fcl.Sphere(radius=shape.radius)
            elif isinstance(shape, physx.PhysxCollisionShapeTriangleMesh):
                # c_geom = None
                c_geom = fcl.BVHModel()
                c_geom.begin_model()
                c_geom.add_sub_model(vertices=shape.vertices, faces=shape.triangles)  # type: ignore
                c_geom.end_model()
            else:
                raise TypeError(f"Unknown shape type: {type(shape)}")
            if c_geom is not None:
                shapes.append(CollisionObject(c_geom))
            
        if len(shapes) == 0:
            return None

        return FCLObject(
            comp.name
            if isinstance(comp, PhysxArticulationLinkComponent)
            else convert_object_name(comp.entity),
            comp.entity.pose,  # type: ignore
            shapes,
            shape_poses,
        )

    def update_from_simulation(self, *, update_attached_object: bool = True) -> None:
        """
        Updates PlanningWorld's articulations/objects pose with current Scene state.
        Note that shape's local_pose is not updated.
        If those are changed, please recreate a new SapienPlanningWorld instance.

        :param update_attached_object: whether to update the attached pose of
            all attached objects
        """
        for articulation in self._sim_scene.get_all_articulations():
            if art := self.get_articulation(convert_object_name(articulation)):
                if articulation in self.planned_articulations:
                    root_pose = articulation.links[7].pose
                    assert articulation.links[7].name == 'scene-0-ds_fetch_basket_torso_lift_link'
                    art.set_base_pose(root_pose)  # type: ignore
                    # set_qpos to update poses
                    art.set_qpos(articulation.qpos[4:], full=True)  # type: ignore
                else:
                    art.set_base_pose(articulation.root_pose)  # type: ignore
                    # set_qpos to update poses
                    art.set_qpos(articulation.qpos, full=True)  # type: ignore
            else:
                raise RuntimeError(
                    f"Articulation {articulation.name} not found in PlanningWorld! "
                    "The scene might have changed since last update."
                )

        for entity in self._sim_scene.get_all_actors():
            object_name = convert_object_name(entity)

            # If entity is an attached object
            if attached_body := self.get_attached_object(object_name):
                if update_attached_object:  # update attached pose
                    attached_body.pose = (
                        attached_body.get_attached_link_global_pose().inv()
                        * entity.pose  # type: ignore
                    )
                attached_body.update_pose()
            elif fcl_obj := self.get_object(object_name):
                # Overwrite the object
                self.add_object(
                    FCLObject(
                        object_name,
                        entity.pose,  # type: ignore
                        fcl_obj.shapes,
                        fcl_obj.shape_poses,
                    )
                )
            elif (
                len(
                    entity.find_component_by_type(
                        physx.PhysxRigidBaseComponent
                    ).collision_shapes  # type: ignore
                )
                > 0
            ):
                raise RuntimeError(
                    f"Entity {entity.name} not found in PlanningWorld! "
                    "The scene might have changed since last update."
                )
    
# class SapienPlannerV2(SapienPlanner):
#     # plan_screw ankor
#     def plan_screw(
#         self,
#         goal_pose: mplib.Pose,
#         current_qpos: np.ndarray,
#         *,
#         qpos_step: float = 0.1,
#         time_step: float = 0.1,
#         wrt_world: bool = True,
#         masked_joints: list = None,
#         verbose: bool = False,
#     ) -> dict[str, str | np.ndarray | np.float64]:
#         # plan_screw ankor end
#         """
#         Plan from a start configuration to a goal pose of the end-effector using
#         screw motion

#         Args:
#             goal_pose: pose of the goal
#             current_qpos: current joint configuration (either full or move_group joints)
#             qpos_step: size of the random step
#             time_step: time step for the discretization
#             wrt_world: if True, interpret the target pose with respect to the
#                 world frame instead of the base frame
#             verbose: if True, will print the log of TOPPRA
#         """
#         current_qpos = self.pad_move_group_qpos(current_qpos.copy())
#         self.robot.set_qpos(current_qpos, True)

#         if wrt_world:
#             goal_pose = self._transform_goal_to_wrt_base(goal_pose)

#         def skew(vec):
#             return np.array([
#                 [0, -vec[2], vec[1]],
#                 [vec[2], 0, -vec[0]],
#                 [-vec[1], vec[0], 0],
#             ])

#         def pose2exp_coordinate(pose: mplib.Pose) -> tuple[np.ndarray, float]:
#             def rot2so3(rotation: np.ndarray):
#                 assert rotation.shape == (3, 3)
#                 if np.isclose(rotation.trace(), 3):
#                     return np.zeros(3), 1
#                 if np.isclose(rotation.trace(), -1):
#                     return np.zeros(3), -1e6
#                 theta = np.arccos((rotation.trace() - 1) / 2)
#                 omega = (
#                     1
#                     / 2
#                     / np.sin(theta)
#                     * np.array([
#                         rotation[2, 1] - rotation[1, 2],
#                         rotation[0, 2] - rotation[2, 0],
#                         rotation[1, 0] - rotation[0, 1],
#                     ]).T
#                 )
#                 return omega, theta

#             pose_mat = pose.to_transformation_matrix()
#             omega, theta = rot2so3(pose_mat[:3, :3])
#             if theta < -1e5:
#                 return omega, theta
#             ss = skew(omega)
#             inv_left_jacobian = (
#                 np.eye(3) / theta
#                 - 0.5 * ss
#                 + (1.0 / theta - 0.5 / np.tan(theta / 2)) * ss @ ss
#             )
#             v = inv_left_jacobian @ pose_mat[:3, 3]
#             return np.concatenate([v, omega]), theta

#         self.pinocchio_model.compute_forward_kinematics(current_qpos)
#         ee_index = self.link_name_2_idx[self.move_group]
#         # relative_pose = T_base_goal * T_base_link.inv()
#         relative_pose = goal_pose * self.pinocchio_model.get_link_pose(ee_index).inv()
#         omega, theta = pose2exp_coordinate(relative_pose)

#         if theta < -1e4:
#             return {"status": "screw plan failed."}
#         omega = omega.reshape((-1, 1)) * theta

#         move_joint_idx = self.move_group_joint_indices
#         path = [np.copy(current_qpos[move_joint_idx])]

#         while True:
#             self.pinocchio_model.compute_full_jacobian(current_qpos)
#             J = self.pinocchio_model.get_link_jacobian(ee_index, local=False)
#             mask = np.ones_like(J)
#             if masked_joints is not None:
#                 mask = np.tile(masked_joints, (mask.shape[0], 1)).astype(np.int32)
#             J *= mask
#             delta_q = np.linalg.pinv(J) @ omega
#             delta_q *= qpos_step / (np.linalg.norm(delta_q))
#             delta_twist = J @ delta_q

#             flag = False
#             if np.linalg.norm(delta_twist) > np.linalg.norm(omega):
#                 ratio = np.linalg.norm(omega) / np.linalg.norm(delta_twist)
#                 delta_q = delta_q * ratio
#                 delta_twist = delta_twist * ratio
#                 flag = True

#             current_qpos += delta_q.reshape(-1)
#             omega -= delta_twist

#             def check_joint_limit(q):
#                 n = len(q)
#                 for i in range(n):
#                     if (
#                         q[i] < self.joint_limits[i][0] - 1e-3
#                         or q[i] > self.joint_limits[i][1] + 1e-3
#                     ):
#                         return False
#                 return True

#             within_joint_limit = check_joint_limit(current_qpos)
#             self.planning_world.set_qpos_all(current_qpos[move_joint_idx])
#             collide = self.planning_world.is_state_colliding()

#             if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
#                 return {"status": "screw plan failed"}

#             path.append(np.copy(current_qpos[move_joint_idx]))

#             if flag:
#                 if verbose:
#                     ta.setup_logging("INFO")
#                 else:
#                     ta.setup_logging("WARNING")
#                 times, pos, vel, acc, duration = self.TOPP(np.vstack(path), time_step)
#                 return {
#                     "status": "Success",
#                     "time": times,
#                     "position": pos,
#                     "velocity": vel,
#                     "acceleration": acc,
#                     "duration": duration,
#                 }


#     def plan_pose(
#         self,
#         goal_pose: mplib.Pose,
#         current_qpos: np.ndarray,
#         mask: Optional[list[bool] | np.ndarray] = None,
#         *,
#         time_step: float = 0.1,
#         rrt_range: float = 0.1,
#         planning_time: float = 1,
#         fix_joint_limits: bool = True,
#         fixed_joint_indices: Optional[list[int]] = None,
#         wrt_world: bool = True,
#         simplify: bool = True,
#         constraint_function: Optional[Callable] = None,
#         constraint_jacobian: Optional[Callable] = None,
#         constraint_tolerance: float = 1e-3,
#         verbose: bool = False,
#         n_init_qpos: int = 20
#     ) -> dict[str, str | np.ndarray | np.float64]:
#         """
#         plan from a start configuration to a goal pose of the end-effector

#         Args:
#             goal_pose: pose of the goal
#             current_qpos: current joint configuration (either full or move_group joints)
#             mask: if the value at a given index is True, the joint is *not* used in the
#                 IK
#             time_step: time step for TOPPRA (time parameterization of path)
#             rrt_range: step size for RRT
#             planning_time: time limit for RRT
#             fix_joint_limits: if True, will clip the joint configuration to be within
#                 the joint limits
#             wrt_world: if true, interpret the target pose with respect to
#                 the world frame instead of the base frame
#             verbose: if True, will print the log of OMPL and TOPPRA
#         """
#         if mask is None:
#             mask = []

#         if fix_joint_limits:
#             current_qpos = np.clip(
#                 current_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1]
#             )
#         current_qpos = self.pad_move_group_qpos(current_qpos)

#         if wrt_world:
#             goal_pose = self._transform_goal_to_wrt_base(goal_pose)

#         # we need to take only the move_group joints when planning
#         # idx = self.move_group_joint_indices

#         ik_status, goal_qpos = self.IK(goal_pose, current_qpos, mask, n_init_qpos=n_init_qpos, verbose=True)
#         if ik_status != "Success":
#             return {"status": ik_status}

#         if verbose:
#             print("IK results:")
#             for i in range(len(goal_qpos)):  # type: ignore
#                 print(goal_qpos[i])  # type: ignore

#         return self.plan_qpos(
#             goal_qpos,  # type: ignore
#             current_qpos,
#             time_step=time_step,
#             rrt_range=rrt_range,
#             planning_time=planning_time,
#             fix_joint_limits=fix_joint_limits,
#             fixed_joint_indices=fixed_joint_indices,
#             simplify=simplify,
#             constraint_function=constraint_function,
#             constraint_jacobian=constraint_jacobian,
#             constraint_tolerance=constraint_tolerance,
#             verbose=verbose,
#         )

