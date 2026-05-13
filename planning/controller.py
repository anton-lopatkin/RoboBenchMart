import contextlib
import io
import time

import numpy as np
import sapien

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
        self.last_stdout = ""

    def pick(self, item_id: int):
        """Grasp a product from the shelf.

        Preconditions:
        - Robot must be positioned in front of the target shelf.

        Effect:
        - Arm aligns to the product, robot approaches the shelf, grasps the
          product, and retreats.

        Args:
            item_id (int): ID of the product to pick.
        """
        return self._run(
            lambda: fetch_skills.align_to_target_product(
                env=self.env, 
                planner=self.solver, 
                target_product_actor=self._product_actor(item_id),
            ),
            lambda: fetch_skills.fetch_object_from_shelf(
                env=self.env, 
                planner=self.solver,
                target_product_actor=self._product_actor(item_id),
                n_grasps=10,
                num_tries=5,
            ),
        )

    def place_to_basket(self):
        """Drop the held product into the basket.

        Preconditions:
        - Gripper must be holding a product.

        Effect:
        - Arm lifts and moves to the basket, gripper opens to release the product.
        """
        return self._run(
            lambda: fetch_skills.drop_to_basket(
                env=self.env,
                planner=self.solver,
            )
        )
    
    def place_to_shelf(self, description: str, camera: str):
        """Place the held product at a shelf location identified by a textual description.

        Preconditions:
        - Gripper must be holding a product.

        Effect:
        - Arm moves to the described shelf location and releases the product.

        Args:
            description (str): Natural language description of the target shelf
                location (e.g., "empty spot on the left side of the middle shelf,
                next to the juice").
            camera (str): Camera with the best view of the target shelf location.
                - "left_base_camera_link": wide view of the shelf from the left side
                - "right_base_camera_link": wide view of the shelf from the right side
                - "fetch_hand": close-up wrist view
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
        state (e.g. required item is missing, repeated motion planning failures).
        """
        return "fail"

    def _place_to_shelf(self, bbox: dict, camera: str):
        point = self._bbox_to_shelf_point(bbox, camera)
        direction = self.env.directions_to_shelf[0]
        closing = np.cross(direction, [0., 0., 1.])
        pose = self.env.agent.build_grasp_pose(direction, closing, point)

        return self._run(
            lambda: fetch_skills.align_to_target_pose(
                env=self.env, 
                planner=self.solver, 
                pose=pose,
            ),
            lambda: fetch_skills.place_object_to_pos(
                env=self.env,
                planner=self.solver,
                target_center_pos=point,
                target_ee_direction=direction,
                n_grasps=10,
            )
        )

    def _run(self, *steps):
        start = time.time()
        print("[controller] running...")

        for step in steps:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                result = step()
            stdout = self._remove_duplicates(out.getvalue())
            self.last_stdout = stdout
            if result == -1:
                print("[controller] motion planning failed")
                print(stdout)
                return result
            print("[controller] motion planning succeed")
            self.solver.planner.update_from_simulation()
        
        elapsed = time.time() - start
        print(f"[controller] finished ({elapsed:.1f}s)")
        return result

    def _remove_duplicates(self, text: str) -> str:
        seen = set()
        result = []

        for line in text.splitlines():
            if line not in seen:
                seen.add(line)
                result.append(line)

        return "\n".join(result)

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
