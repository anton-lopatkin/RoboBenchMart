import io
import contextlib
import numpy as np
import sapien
import time
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

    def navigate(self, item_id: int):
        """Drive to and face a product, ready for picking.

        Effect:
        - Robot base moves to a safe approach distance in front of the target
          product and rotates to face it directly.
        - Arm and gripper state unchanged.

        Args:
            item_id (int): ID of the product to navigate toward.
        """
        return self._run_sequence(
            lambda: self._drive_to_product(item_id, 1.2),
            lambda: self._align_to_product(item_id),
        )

    def pick(self, item_id: int):
        """Execute the full pick sequence for a product.

        Preconditions:
        - Robot must already be navigated to the product (use navigate first).

        Effect:
        - End-effector moves to pre-grasp standoff, base advances toward the
          shelf, end-effector moves to final grasp pose, and gripper closes
          around the product.

        Args:
            item_id (int): ID of the product to pick.
        """
        return self._run_sequence(
            lambda: self._move_ee_to_pregrasp_pose(item_id),
            lambda: self._move_base_forward(0.3),
            lambda: self._move_ee_to_grasp_pose(item_id),
            lambda: self._grasp(item_id),
            lambda: self._move_ee_by([0.05, 0, 0]),
            lambda: self._move_ee_to_neutral_pose()
        )

    def place_to_basket(self):
        """Execute the full place sequence to drop the held item into the basket.

        Preconditions:
        - Gripper must be holding a product.

        Effect:
        - Robot base reverses away from the shelf, end-effector moves to the
          drop pose above the basket, and gripper opens to release the product.
        """
        return self._run_sequence(
            lambda: self._move_base_forward(-0.4),
            lambda: self._move_ee_to_drop_pose(),
            lambda: self._release(),
        )

    def place_to_shelf(self, description: str, camera: str):
        """Place the held product at a shelf location identified by a textual description.

        Preconditions:
        - Gripper must be holding a product.

        Effect:
        - End-effector aligns to the target height, advances to the shelf surface
          at the described location, and releases the product.
        - Robot base unchanged.

        Args:
            description (str): Natural language description of the target shelf
                location (e.g., "empty spot on the left side of the middle shelf,
                next to the juice").
            camera (str): Camera with the best view of the target shelf location.
                - "left_base_camera_link": wide view of the shelf from the left side of the base
                - "right_base_camera_link": wide view of the shelf from the right side of the base
                - "fetch_hand": close-up view from the wrist, facing downward — rarely useful for shelf placement
        """
        pass

    def _move_ee_to_neutral_pose(self):
        """Move end-effector to a neutral pose.
        
        Effect:
        - End-effector moves to a predefined neutral pose in front of the robot's
          body (arm is retracted, gripper is roughly at chest height, facing 
          forward).
        - Gripper remains in its current open/closed state.
        - Robot base unchanged.
        """
        target_pose = self._base_pose * self.neutral_tcp_pose_wrt_to_base
        return self._run(
            self.solver.static_manipulation,
            target_tcp_pose=target_pose,
            n_init_qpos=400,
            disable_lift_joint=False,
        )


    def _move_ee_to_pregrasp_pose(self, item_id: int):
        """Move end-effector to an approach position near the product.

        Effect:
        - End-effector moves to an approach position near the target product,
          aligned exactly with its height and slightly forward (pre-grasp 
          standoff).
        - Gripper remains in its current open/closed state.
        - Robot base unchanged.
        
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
            n_init_qpos=100,
            disable_lift_joint=False,
        )

    def _move_ee_to_shelf_bbox(self, bbox: dict, camera: str):
        """Move end-effector to a bbox-identified shelf point at 10 cm depth.

        Effect:
        - End-effector moves to the shelf-surface position at the centre of the
          given bounding box, offset 10 cm inward along the shelf normal; gripper
          orientation unchanged.
        - Robot base unchanged.

        Args:
            bbox (dict): Bounding box with keys x_min, y_min, x_max, y_max in [0, 1],
            camera (str): Sensor used to capture the image from which bbox was derived.
        """
        point = self._bbox_to_shelf_point(bbox, camera)
        return self._move_ee_to_shelf_point(point)

    def _place_to_shelf(self, bbox: dict, camera: str):
        point = self._bbox_to_shelf_point(bbox, camera)
        return self._run_sequence(
            lambda: self._move_ee_to_height(point[2]),
            lambda: self._move_base_forward(0.1),
            lambda: self._move_ee_to_shelf_point(point),
            lambda: self._release(),
        )

    def _bbox_to_shelf_point(self, bbox: dict, camera: str) -> np.ndarray:
        cx = (bbox["x_min"] + bbox["x_max"]) / 2
        cy = (bbox["y_min"] + bbox["y_max"]) / 2

        cam = self.env.agent.scene.sensors[camera].camera
        u = cx * cam.width
        v = cy * cam.height

        return self._unproject_shelf_point(u, v, camera)

    def _move_ee_to_shelf_point(self, point: np.ndarray):
        shelf_name = self.env.active_shelves[0][0]
        shelf_pose = self.env.actors["fixtures"]["shelves"][shelf_name].pose.sp
        shelf_normal = shelf_pose.to_transformation_matrix()[:3, 1]
        shelf_normal[2] = 0.0
        shelf_normal = common.np_normalize_vector(shelf_normal)

        target_pose = sapien.Pose(p=point + 0.10 * shelf_normal, q=self._tcp_pose.q)
        return self._run(
            self.solver.static_manipulation,
            target_tcp_pose=target_pose,
            n_init_qpos=200,
            disable_lift_joint=False,
        )

    def _move_ee_to_height(self, z: float):
        current = self._tcp_pose
        target_pose = sapien.Pose(p=[current.p[0], current.p[1], z], q=current.q) * sapien.Pose([0, 0, 0.2])
        return self._run(
            self.solver.static_manipulation,
            target_tcp_pose=target_pose,
            n_init_qpos=200,
            disable_lift_joint=False,
        )

    def _move_ee_to_grasp_pose(self, item_id: int):
        """Move end-effector to the grasp pose for an item.

        Preconditions:
        - Use move_ee_to_pregrasp_pose first to get into a good starting position.

        Effect:
        - End-effector moves precisely to the computed grasp pose around the target
          product; fingers remain fully open and aligned for closing.
        - Robot base unchanged.

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

    def _move_ee_to_drop_pose(self):
        """Move end-effector to a drop pose above the robot's basket.
        
        Effect:
        - End-effector moves to a drop pose directly above the robot's basket;
          gripper is positioned low over the basket opening, ready to release.
        - Robot base unchanged.
        """
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

    def _move_ee_by(self, delta: Sequence[float]):
        """Move end-effector by a relative offset.

        Effect:
        - End-effector performs a pure translation by the relative offset [dx, dy, dz]
          in its local coordinate frame:
            - X-axis: up (+) / down (−)
            - Y-axis: left (−) / right (+)
            - Z-axis: forward (+) / backward (−)
        - Gripper orientation does not change.
        - Fingers stay in their current open/closed state.
        - Robot base unchanged.

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

    def _grasp(self, item_id: int):
        """Close the gripper to grasp the target product.

        Preconditions:
        - End-effector must already be positioned at the grasp pose.

        Effect:
        - Gripper fingers close around the target product and secure it.
        - Robot base and arm pose unchanged.

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

    def _release(self):
        """Open gripper to release the currently held item.
        
        Effect:
        - Gripper fingers open fully to their maximum width.
        - Any previously held product is detached from the gripper and becomes
          free (visually the product separates from the fingers and may fall
          or stay supported by the environment).
        - Robot base, arm pose and end-effector position unchanged.
        """
        return self._run(self.solver.open_gripper)

    def _drive_to_shelf(self, distance: float):
        """Drive the robot base to approach the center of the active shelf.

        Effect:
        - Robot base translates to a position directly in front of the active
          shelf, stopping at the exact specified distance from its center.
        - Base orientation can be any.
        - Arm and gripper state unchanged (relative to base).

        Args:
            distance (float): Distance in meters to stop from the shelf
        """
        shelf_name = self.env.active_shelves[0][0]
        shelf_pose = self.env.actors["fixtures"]["shelves"][shelf_name].pose.sp
        target = shelf_pose.p - distance * self.env.directions_to_shelf[0]
        return self._run(self.solver.drive_base, target_pos=target)

    def _drive_to_product(self, item_id: int, distance: float):
        """Drive the robot base toward a specific product.

        Effect:
        - Robot base translates to a position at the specified distance directly
          in front of the target product.
        - Base orientation can be any.
        - Arm and gripper state unchanged (relative to base).

        Args:
            item_id (int): ID of the product to drive toward.
            distance (float): Distance in meters to stop from the product.
        """
        product_actor = self._product_actor(item_id)
        product_pose = product_actor.pose.sp
        direction = product_pose.to_transformation_matrix()[:3, 1]
        target_pos = product_pose.p - distance * direction
        return self._run(self.solver.drive_base, target_pos=target_pos)

    def _move_base_forward(self, delta: float):
        """Move the robot base forward or backward by a delta distance.

        Effect:
        - Robot base performs a pure translation forward (positive delta) or
          backward (negative delta) along its current heading by the given distance.
        - Base orientation, arm and gripper state unchanged.

        Args:
            delta (float): Distance in meters. Positive moves forward, negative moves
                backward.
        """
        return self._run(self.solver.move_forward_delta, delta=delta)

    def _rotate_base(self, delta: float):
        """Rotate the robot base around its Z axis.

        Effect:
        - Robot base rotates in place around its vertical (Z) axis by the
          specified angle (positive = counterclockwise, negative = clockwise).
        - Arm and gripper state unchanged (relative to base).

        Args:
            delta (float): Rotation angle in radians. Positive is counterclockwise,
                negative is clockwise.
        """
        return self._run(self.solver.rotate_z_delta, delta=delta)

    def _align_to_product(self, item_id: int):
        """Rotate the robot base to face a product.

        Effect:
        - Robot base rotates in place so that its forward direction points directly
          toward the target product (on the horizontal plane).
        - Arm and gripper state unchanged (relative to base).
        Args:
            item_id (int): ID of the product to align toward.
        """
        direction = self._product_center(item_id) - self._base_pose.p
        direction[2] = 0.0
        return self._run(self.solver.rotate_base_z, new_direction=direction)

    def _run_sequence(self, *steps):
        for step in steps:
            if step() == -1:
                return -1

    def _run(self, fn, **kwargs):
        start = time.time()
        print(f"[controller] running...")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = fn(**kwargs)
        stdout = self._remove_duplicates(out.getvalue())
        self.last_stdout = stdout
        elapsed = time.time() - start
        print(f"[controller] finished ({elapsed:.1f}s)")
        if result == -1:
            print(f"[controller] motion planning failed")
            print(stdout)
            return result
        print(f"[controller] motion planning succeed")
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

    def _unproject_shelf_point(self, u, v, camera="right_base_camera_link"):
        shelf_depth = 0.5

        cam = self.env.agent.scene.sensors[camera].camera

        cam_pose = cam.global_pose.sp * sapien.Pose(p=[0, 0, 0], q=[0.5004, -0.5, 0.5, -0.5])
        T = cam_pose.to_transformation_matrix()
        R_cam, t_cam = T[:3, :3], T[:3, 3]

        shelf_name = self.env.active_shelves[0][0]
        shelf_pose = self.env.actors["fixtures"]["shelves"][shelf_name].pose.sp
        direction_to_shelf = shelf_pose.to_transformation_matrix()[:3, 1]

        t_plane_world = shelf_pose.p - 0.5 * shelf_depth * direction_to_shelf
        n_plane_cam = R_cam.T @ direction_to_shelf
        t_plane_cam = R_cam.T @ (t_plane_world - t_cam)

        A = np.array([
            [cam.fx, 0,      cam.cx - u],
            [0,      cam.fy, cam.cy - v],
            n_plane_cam,
        ])
        b = np.array([0.0, 0.0, np.dot(n_plane_cam, t_plane_cam)])
        return R_cam @ np.linalg.solve(A, b) + t_cam