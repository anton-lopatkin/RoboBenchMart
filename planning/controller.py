import numpy as np
import sapien
from mani_skill.utils import common

from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    get_actor_obb,
    compute_grasp_info_by_obb
)

from dsynth.planning.motionplanner import FetchMotionPlanningSapienSolver
from dsynth.planning.utils import (
    get_fcl_object_name,
    is_mesh_cylindrical
)


class Controller:
    
    FINGER_LENGTH = 0.04

    def __init__(self, env, debug=False, vis=False):
        self.env = env.unwrapped

        self.solver = FetchMotionPlanningSapienSolver(
            env,
            debug=debug,
            vis=vis,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=vis,
            print_env_info=False,
            disable_actors_collision=False,
            verbose=debug
        )

        self.neutral_ee_pose_wrt_to_base = self.get_base_pose().sp.inv() * self.get_tcp_pose().sp * sapien.Pose(p=[0.1, 0, 0.1])
        

    def move_ee_to_neutral_pose(self):
        neutral_pose = self.get_base_pose().sp * self.neutral_ee_pose_wrt_to_base
        result = self.solver.static_manipulation(neutral_pose, n_init_qpos=400, disable_lift_joint=False)
        if result == -1:
            return result
        self.solver.planner.update_from_simulation()

    def move_ee_to_product_height(self, item_id):
        target_product_actor = [item for item in self.env.actors['products'].values() if item.per_scene_id[0].item() == item_id][0]
        target_product_obb = get_actor_obb(target_product_actor)
        target_product_center = self.get_obb_center(target_product_obb)
        
        lift_ee_pos = self.get_tcp_pose().sp.p
        lift_ee_pos[2] = target_product_center[2]
        lift_ee_pose = sapien.Pose(p=lift_ee_pos, q=self.get_tcp_pose().sp.q)
        lift_ee_pose = lift_ee_pose * sapien.Pose(p=[0, 0, 0.3])

        result = self.solver.static_manipulation(lift_ee_pose, n_init_qpos=50, disable_lift_joint=False)
        if result == -1:
            return result
        self.solver.planner.update_from_simulation()

    def lift_ee(self, delta):
        lift_pose = self.get_tcp_pose().sp * sapien.Pose([delta, 0., 0.])
        result = self.solver.static_manipulation(lift_pose, n_init_qpos=200, disable_lift_joint=False)
        if result == -1:
            return result
        self.solver.planner.update_from_simulation()

    def drive_to_shelf(self):        
        actor_shelf_name = self.env.active_shelves[0][0]
        shelf_pose = self.env.actors["fixtures"]["shelves"][actor_shelf_name].pose.sp
        origin = shelf_pose.p - 1.4 * self.env.directions_to_shelf[0]

        result = self.solver.drive_base(origin)
        if result == -1:
            return result
        self.solver.planner.update_from_simulation()

    def drive_to_product(self, item_id):
        target_product_actor = [item for item in self.env.actors['products'].values() if item.per_scene_id[0].item() == item_id][0]
        product_pose = target_product_actor.pose.sp 
        direction_to_product = product_pose.to_transformation_matrix()[:3, 1]
        target_pose = product_pose.p - 1.2 * direction_to_product

        result = self.solver.drive_base(target_pose)
        if result == -1:
            return result
        self.solver.planner.update_from_simulation()

    def move_base_forward(self, delta):
        result = self.solver.move_forward_delta(delta)
        if result == -1:
            return result
        self.solver.planner.update_from_simulation()

    def rotate_base(self, delta):
        result = self.solver.rotate_z_delta(delta)
        if result == -1:
            return result
        self.solver.planner.update_from_simulation()

    def align_to_product(self, item_id):
        target_product_actor = [item for item in self.env.actors['products'].values() if item.per_scene_id[0].item() == item_id][0]
        target_product_obb = get_actor_obb(target_product_actor)
        target_product_center = self.get_obb_center(target_product_obb)

        view_to_target = target_product_center - self.get_base_pose().sp.p
        view_to_target[2] = 0.

        result = self.solver.rotate_base_z(view_to_target)
        if result == -1:
            return result
        self.solver.planner.update_from_simulation()

    def move_base_towards_product(self, item_id):
        target_product_actor = [item for item in self.env.actors['products'].values() if item.per_scene_id[0].item() == item_id][0]
        target_product_obb = get_actor_obb(target_product_actor)
        target_product_center = self.get_obb_center(target_product_obb)

        pre_grasp_base_translation = target_product_center - self.get_tcp_center()
        pre_grasp_base_translation[2] = 0.

        # move base to position 0.15m in fornt of the target object
        base_target_pos = self.get_base_pose().sp.p + \
            (1 - 0.2 / np.linalg.norm(pre_grasp_base_translation)) * pre_grasp_base_translation
        
        result = self.solver.drive_base(base_target_pos)
        if result == -1:
            return result
        self.solver.planner.update_from_simulation()

    def grasp_product(self, item_id):
        target_product_actor = [item for item in self.env.actors['products'].values() if item.per_scene_id[0].item() == item_id][0]
        target_product_obb = get_actor_obb(target_product_actor)
        target_product_center = self.get_obb_center(target_product_obb)

        if is_mesh_cylindrical(target_product_actor):
            grasp_approaching = self.env.directions_to_shelf[0].copy()
            grasp_approaching[2] = 0.
            grasp_approaching = common.np_normalize_vector(grasp_approaching)

            grasp_closing = np.cross(grasp_approaching, [0., 0., 1.])
            grasp_center = target_product_center

        else: 
            grasp_info = compute_grasp_info_by_obb(
                target_product_obb,
                approaching=self.get_tcp_matrix()[:3, 2],
                target_closing=self.get_tcp_matrix()[:3, 1],
                depth=self.FINGER_LENGTH
            )
            grasp_closing, grasp_center, grasp_approaching = grasp_info["closing"], grasp_info["center"], grasp_info["approaching"]

        grasp_pose = self.env.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)

        self.solver.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
            get_fcl_object_name(target_product_actor), True
        )
        result = self.solver.static_manipulation(grasp_pose, n_init_qpos=400, disable_lift_joint=False)
        if result == -1:
            return result
        
        result = self.solver.close_gripper()
        if result == -1:
            return result
        
        kwargs = {"name": get_fcl_object_name(target_product_actor), "art_name": 'scene-0_ds_fetch_basket_1', "link_id": self.solver.planner.move_group_link_id}
        self.solver.planner.planning_world.attach_object(**kwargs)
        self.solver.planner.update_from_simulation()

    def drop_to_basket(self):
        goal_center = self.env.calc_target_pose().sp.p
        goal_approaching = np.array([0, 0., -1.])
        goal_closing = - self.get_base_pose().sp.to_transformation_matrix()[:3, 1]

        goal_pose = self.env.agent.build_grasp_pose(goal_approaching, goal_closing, goal_center)
        goal_pose = goal_pose * sapien.Pose(p=[-0.03, 0., -0.35])

        result = self.solver.static_manipulation(goal_pose, n_init_qpos=100, disable_lift_joint=False)
        if result == -1:
            return result
        
        result = self.solver.open_gripper()
        if result == -1:
            return result
        self.solver.planner.update_from_simulation()


    def get_obb_center(self, obb):
        T = np.array(obb.primitive.transform)
        return T[:3, 3]

    def get_base_pose(self):
        return self.env.agent.base_link.pose
        
    def get_tcp_pose(self):
        return self.env.agent.tcp.pose

    def get_tcp_matrix(self):
        tcp_pose = self.get_tcp_pose()
        return tcp_pose.to_transformation_matrix()[0].cpu().numpy()
        
    def get_tcp_center(self):
        return self.get_tcp_matrix()[:3, 3]