import contextlib
import io
import time

import numpy as np
import sapien
from mani_skill.utils import common

from dsynth.planning import fetch_skills
from dsynth.planning import motionplanner


class Controller:
    FINGER_LENGTH = 0.04

    def __init__(self, env, debug=False, vis=False):
        self.env = env.unwrapped

        self.solver = motionplanner.FetchMotionPlanningSapienSolver(
            env,
            debug=debug,
            vis=vis,
            visualize_target_grasp_pose=vis,
            print_env_info=False,
            verbose=debug,
        )

        self.neutral_tcp_pose_wrt_to_base = (
            self._base_pose.inv() * self._tcp_pose * sapien.Pose(p=[0.1, 0, 0.1])
        )

    def navigate_to_product(self, item_id: int):
        """Navigate to a product, ready for picking.

        Effect:
        - Robot base moves to a safe approach distance in front of the target
          product and rotates to face it directly.
        - Arm and gripper state unchanged.

        Args:
            item_id (int): ID of the product to navigate toward.
        """
        return fetch_skills.align_to_target_product(
            env=self.env, 
            planner=self.solver, 
            target_product_actor=self._product_actor(item_id),
        )

    def navigate_to_shelf(self, description: str, camera: str):
        """Navigate to a shelf location identified by a textual description.

        Effect:
        - Robot base moves to a safe approach distance in front of the described
          shelf location and orients toward it.
        - Arm and gripper state unchanged.

        Args:
            description (str): Natural language description of the target shelf
                location (e.g., "empty spot on the left side of the middle shelf").
            camera (str): Camera with the best view of the target shelf location.
                - "left_base_camera_link": wide view of the shelf from the left
                  side of the base
                - "right_base_camera_link": wide view of the shelf from the right
                  side of the base
                - "fetch_hand": close-up view from the wrist, facing downward —
                  rarely useful for shelf navigation
        """
        pass

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
        return fetch_skills.fetch_object_from_shelf(
            env=self.env, 
            planner=self.solver,
            target_product_actor=self._product_actor(item_id),
            n_grasps=6,
            num_tries=5,
        )

    def place_to_basket(self):
        """Execute the full place sequence to drop the held item into the basket.

        Preconditions:
        - Gripper must be holding a product.

        Effect:
        - Robot base reverses away from the shelf, end-effector moves to the
          drop pose above the basket, and gripper opens to release the product.
        """
        return fetch_skills.drop_to_basket(
            env=self.env,
            planner=self.solver,
        )

    def place_to_shelf(self, description: str, camera: str):
        """Place the held product at a shelf location identified by a textual
        description.

        Preconditions:
        - Gripper must be holding a product.

        Effect:
        - End-effector aligns to the target height, advances to the shelf surface
          at the described location, and releases the product.
        - Robot base unchanged.pr

        Args:
            description (str): Natural language description of the target shelf
                location (e.g., "empty spot on the left side of the middle shelf,
                next to the juice").
            camera (str): Camera with the best view of the target shelf location.
                - "left_base_camera_link": wide view of the shelf from the left
                  side of the base
                - "right_base_camera_link": wide view of the shelf from the right
                  side of the base
                - "fetch_hand": close-up view from the wrist, facing downward —
                  rarely useful for shelf placement
        """
        pass

    def done(self):
        """Signal that the task has been successfully completed.

        Call this when all task objectives have been achieved and no further
        actions are needed.
        """
        return "done"

    def fail(self):
        """Signal that the task cannot be completed.

        Call this when the task is impossible to complete given the current
        state (e.g. required item is missing, repeated motion planning
        failures, goal is already achieved by someone else).
        """
        return "fail"

    def _place_to_shelf(self, bbox: dict, camera: str):
        point = self._bbox_to_shelf_point(bbox, camera)
        return self._run_sequence(
            lambda: self._move_ee_to_neutral_pose(),
            lambda: self._drive_to_shelf_point(point, 1.1),
            lambda: self._align_to_shelf_point(point),
            lambda: self._move_ee_to_height(point[2]),
            lambda: self._move_base_forward(0.3),
            lambda: self._move_ee_to_shelf_point(point),
            lambda: self._release(),
        )

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
        target_pose = sapien.Pose(
            p=[current.p[0], current.p[1], z], q=current.q
        ) * sapien.Pose([0, 0, 0.2])
        return self._run(
            self.solver.static_manipulation,
            target_tcp_pose=target_pose,
            n_init_qpos=200,
            disable_lift_joint=False,
        )

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

    def _drive_to_shelf_bbox(self, bbox: dict, camera: str):
        point = self._bbox_to_shelf_point(bbox, camera)
        return self._run_sequence(
            lambda: self._drive_to_shelf_point(point, 1.2),
            lambda: self._align_to_shelf_point(point),
        )

    def _drive_to_shelf_point(self, point: np.ndarray, distance: float):
        direction = point - self._base_pose.p
        direction[2] = 0.0
        direction = common.np_normalize_vector(direction)
        target = point - distance * direction
        return self._run(self.solver.drive_base, target_pos=target)

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

    def _align_to_shelf_point(self, point: np.ndarray):
        direction = point - self._base_pose.p
        direction[2] = 0.0
        return self._run(self.solver.rotate_base_z, new_direction=direction)

    def _run_sequence(self, *steps):
        for step in steps:
            if step() == -1:
                return -1

    def _run(self, fn, **kwargs):
        start = time.time()
        print("[controller] running...")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = fn(**kwargs)
        stdout = self._remove_duplicates(out.getvalue())
        self.last_stdout = stdout
        elapsed = time.time() - start
        print(f"[controller] finished ({elapsed:.1f}s)")
        if result == -1:
            print("[controller] motion planning failed")
            print(stdout)
            return result
        print("[controller] motion planning succeed")
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

    def _product_actor(self, item_id):
        return next(
            item
            for item in self.env.actors["products"].values()
            if item.per_scene_id[0].item() == item_id
        )

    def _bbox_to_shelf_point(self, bbox: dict, camera: str) -> np.ndarray:
        cx = (bbox["x_min"] + bbox["x_max"]) / 2
        cy = (bbox["y_min"] + bbox["y_max"]) / 2

        cam = self.env.agent.scene.sensors[camera].camera
        u = cx * cam.width
        v = cy * cam.height

        return self._unproject_shelf_point(u, v, camera)

    def _unproject_shelf_point(self, u, v, camera):
        shelf_depth = 0.5

        cam = self.env.agent.scene.sensors[camera].camera

        cam_pose = cam.global_pose.sp * sapien.Pose(
            p=[0, 0, 0], q=[0.5004, -0.5, 0.5, -0.5]
        )
        T = cam_pose.to_transformation_matrix()
        R_cam, t_cam = T[:3, :3], T[:3, 3]

        shelf_name = self.env.active_shelves[0][0]
        shelf_pose = self.env.actors["fixtures"]["shelves"][shelf_name].pose.sp
        direction_to_shelf = shelf_pose.to_transformation_matrix()[:3, 1]

        t_plane_world = shelf_pose.p - 0.5 * shelf_depth * direction_to_shelf
        n_plane_cam = R_cam.T @ direction_to_shelf
        t_plane_cam = R_cam.T @ (t_plane_world - t_cam)

        A = np.array(
            [
                [cam.fx, 0, cam.cx - u],
                [0, cam.fy, cam.cy - v],
                n_plane_cam,
            ]
        )
        b = np.array([0.0, 0.0, np.dot(n_plane_cam, t_plane_cam)])
        return R_cam @ np.linalg.solve(A, b) + t_cam
