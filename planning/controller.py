import io
import contextlib
import numpy as np
import sapien
from typing import Sequence
from mani_skill.utils import common
from mani_skill.examples.motionplanning.base_motionplanner import utils as maniskill_mp_utils

from dsynth.planning import motionplanner
from dsynth.planning import utils as dsynth_mp_utils


class Controller:
    FINGER_LENGTH = 0.04

    def __init__(self, env, debug=False, vis=False):
        self.env = env.unwrapped

        self.solver = motionplanner.FetchMotionPlanningSapienSolver(
            env,
            debug=debug,
            vis=vis,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=vis,
            print_env_info=False,
            disable_actors_collision=False,
            verbose=debug,
        )

        self.neutral_tcp_pose_wrt_to_base = ( 
            self._base_pose.inv() * self._tcp_pose * sapien.Pose(p=[0.1, 0, 0.1])
        )

    def move_ee_to_neutral_pose(self):
        """Move end-effector to a neutral pose"""
        target_pose = self._base_pose * self.neutral_tcp_pose_wrt_to_base
        return self._run(
            self.solver.static_manipulation,
            target_tcp_pose=target_pose,
            n_init_qpos=400,
            disable_lift_joint=False,
        )


    def move_ee_to_pregrasp_pose(self, item_id: int):
        """Move end-effector to an approach position near the product.

        Positions the end-effector at the product's height and slightly forward. 

        Args:
            item_id (int): ID of the product to align with.
        """
        product_center = self._product_center(item_id)
        neutral_pose = self._base_pose * self.neutral_tcp_pose_wrt_to_base
        aligned_pos = neutral_pose.p
        aligned_pos[2] = product_center[2]
        aligned_pose = sapien.Pose(p=aligned_pos, q=neutral_pose.q)
        target_pose = aligned_pose * sapien.Pose([0, 0, 0.2])
        return self._run(
            self.solver.static_manipulation,
            target_tcp_pose=target_pose,
            n_init_qpos=50,
            disable_lift_joint=False,
        )


    def move_ee_to_grasp_pose(self, item_id: int):
        """Move end-effector to the grasp pose for an item.

        Prerequisites:
        - Use move_ee_to_pregrasp_pose first to get into a good starting position.

        Args:
            item_id (int): ID of the product to grasp.
        """
        product_actor = self._product_actor(item_id)
        
        grasp_pose = self._compute_grasp_pose(product_actor)

        self.solver.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
            dsynth_mp_utils.get_fcl_object_name(product_actor), True
        )

        return self._run(
            self.solver.static_manipulation,
            target_tcp_pose=grasp_pose,
            n_init_qpos=400,
            disable_lift_joint=False,
        )

    def move_ee_to_drop_pose(self):
        """Move end-effector to a drop pose above the robot's basket"""
        goal_center = self.env.calc_target_pose().sp.p
        goal_approaching = np.array([0, 0.0, -1.0])
        goal_closing = -self._base_pose.to_transformation_matrix()[:3, 1]

        goal_pose = self.env.agent.build_grasp_pose(
            goal_approaching, goal_closing, goal_center
        )
        goal_pose = goal_pose * sapien.Pose(p=[-0.03, 0.0, -0.35])

        return self._run(
            self.solver.static_manipulation,
            target_tcp_pose=goal_pose,
            n_init_qpos=100,
            disable_lift_joint=False,
        )

    def move_ee_by(self, delta: Sequence[float]):
        """Move end-effector by a relative offset.

        Args:
            delta (Sequence[float]): Relative offset [dx, dy, dz] in meters in 
                end-effector frame:
                - X: up/down
                - Y: left/right
                - Z: forward/backward
        """
        target_pose = self._tcp_pose * sapien.Pose(delta)
        return self._run(
            self.solver.static_manipulation,
            target_tcp_pose=target_pose,
            n_init_qpos=200,
            disable_lift_joint=False,
        )

    def grasp(self, item_id: int):
        """Close gripper to pick up a product.

         Prerequisites:
        - End-effector must already be positioned at the grasp pose.

        Args:
            item_id (int): ID of the product to grasp.
        """
        product_actor = self._product_actor(item_id)
        result = self._run(self.solver.close_gripper)
        self.solver.planner.planning_world.attach_object(
            name=dsynth_mp_utils.get_fcl_object_name(product_actor),
            art_name="scene-0_ds_fetch_basket_1",
            link_id=self.solver.planner.move_group_link_id,
        )
        self.solver.planner.update_from_simulation()
        return result

    def release(self):
        """Open gripper to release the currently held item."""
        return self._run(self.solver.open_gripper)

    def drive_to_shelf(self, distance: float):
        """Drive the robot base to approach the center of the active shelf.

        Args:
            distance (float): Distance in meters to stop from the shelf
        """
        shelf_name = self.env.active_shelves[0][0]
        shelf_pose = self.env.actors["fixtures"]["shelves"][shelf_name].pose.sp
        target = shelf_pose.p - distance * self.env.directions_to_shelf[0]
        return self._run(self.solver.drive_base, target_pos=target)

    def drive_to_product(self, item_id: int, distance: float):
        """Drive the robot base toward a specific product.

        Args:
            item_id (int): ID of the product to drive toward.
            distance (float): Distance in meters to stop from the product.
        """
        product_actor = self._product_actor(item_id)
        product_pose = product_actor.pose.sp
        direction = product_pose.to_transformation_matrix()[:3, 1]
        target_pos = product_pose.p - distance * direction
        return self._run(self.solver.drive_base, target_pos=target_pos)

    def move_base_forward(self, delta: float):
        """Move the robot base forward or backward by a delta distance.

        Args:
            delta (float): Distance in meters. Positive moves forward, negative moves
                backward.
        """
        return self._run(self.solver.move_forward_delta, delta=delta)

    def rotate_base(self, delta: float):
        """Rotate the robot base around its Z axis.

        Args:
            delta (float): Rotation angle in radians. Positive is counterclockwise,
                negative is clockwise.
        """
        return self._run(self.solver.rotate_z_delta, delta=delta)

    def align_to_product(self, item_id: int):
        """Rotate the robot base to face a product.

        Args:
            item_id (int): ID of the product to align toward.
        """
        direction = self._product_center(item_id) - self._base_pose.p
        direction[2] = 0.0
        return self._run(self.solver.rotate_base_z, new_direction=direction)

    def _run(self, fn, **kwargs):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = fn(**kwargs)
        stdout = self._remove_duplicates(out.getvalue())
        self.last_stdout = stdout
        if result != -1:
            self.solver.planner.update_from_simulation()
        return result

    def _remove_duplicates(self, text: str) -> str:
        seen = set()
        result = []

        for line in text.splitlines():
            if line not in seen:
                seen.add(line)
                result.append(line)

        return "\n".join(result)

    @property
    def _base_pose(self):
        return self.env.agent.base_link.pose.sp

    @property
    def _tcp_pose(self):
        return self.env.agent.tcp.pose.sp

    @property
    def _tcp_matrix(self):
        return self.env.agent.tcp.pose.to_transformation_matrix()[0].cpu().numpy()

    @property
    def _tcp_center(self):
        return self._tcp_matrix[:3, 3]

    def _obb_center(self, obb):
        T = np.array(obb.primitive.transform)
        return T[:3, 3]

    def _product_actor(self, item_id):
        return [
            item
            for item in self.env.actors["products"].values()
            if item.per_scene_id[0].item() == item_id
        ][0]

    def _product_center(self, item_id):
        product_actor = self._product_actor(item_id)
        product_obb = maniskill_mp_utils.get_actor_obb(product_actor)
        product_center = self._obb_center(product_obb)
        return product_center
    
    def _compute_grasp_pose(self, product_actor):
        product_obb = maniskill_mp_utils.get_actor_obb(product_actor)
        product_center = self._obb_center(product_obb)

        if dsynth_mp_utils.is_mesh_cylindrical(product_actor):
            grasp_approaching = self.env.directions_to_shelf[0].copy()
            grasp_approaching[2] = 0.0
            grasp_approaching = common.np_normalize_vector(grasp_approaching)
            grasp_closing = np.cross(grasp_approaching, [0.0, 0.0, 1.0])
            grasp_center = product_center
        else:
            grasp_info = maniskill_mp_utils.compute_grasp_info_by_obb(
                product_obb,
                approaching=self._tcp_matrix[:3, 2],
                target_closing=self._tcp_matrix[:3, 1],
                depth=self.FINGER_LENGTH,
            )
            grasp_closing, grasp_center, grasp_approaching = (
                grasp_info["closing"],
                grasp_info["center"],
                grasp_info["approaching"],
            )

        grasp_pose = self.env.agent.build_grasp_pose(
            grasp_approaching, grasp_closing, grasp_center
        )

        return grasp_pose
