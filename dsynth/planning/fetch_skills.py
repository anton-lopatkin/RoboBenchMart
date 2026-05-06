import sapien
import numpy as np

from mani_skill.utils import common
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

from dsynth.planning.utils import (
    get_tcp_pose,
    get_tcp_matrix,
    get_base_pose,
    get_shoulder_pan_pose,
    get_fcl_object_name, 
    compute_box_grasp_thin_side_info,
    compute_cylinder_grasp_info,
    is_mesh_cylindrical,
    get_base_shift_tcp_to_target,
    get_direction_to_shelf
)
from dsynth.planning.motionplanner import FetchMotionPlanningSapienSolver

def align_ee_to_target_pos(env, planner: FetchMotionPlanningSapienSolver, target_pos: np.ndarray):
    shoulder_pos = get_shoulder_pan_pose(env).sp.p
    shoulder_to_tcp_distance = np.linalg.norm(shoulder_pos - get_tcp_pose(env).p)
    shoulder_to_item_direction = (target_pos - shoulder_pos)
    shoulder_to_item_direction = common.np_normalize_vector(shoulder_to_item_direction)
    target_tcp_pos = shoulder_pos + shoulder_to_item_direction * shoulder_to_tcp_distance

    closing = np.cross(shoulder_to_item_direction, [0., 0., 1.])
    closing = common.np_normalize_vector(closing)
    target_tcp_pose = env.agent.build_grasp_pose(shoulder_to_item_direction, closing, target_tcp_pos)

    res = planner.static_manipulation(target_tcp_pose)
    return res

def align_ee_to_target_product(env, planner: FetchMotionPlanningSapienSolver, target_product_actor):
    obb = get_actor_obb(target_product_actor)
    item_center = np.array(obb.primitive.transform)[:3, 3]
    res = align_ee_to_target_pos(env, planner, item_center)
    return res

def align_to_target_pose(env, planner: FetchMotionPlanningSapienSolver, pose: sapien.Pose):

    reset_arm_actions = planner.plan_reset_arm()
    if reset_arm_actions == -1:
        reset_arm_actions = None

    dir_to_shelf = env.directions_to_shelf[0]
    # base_to_tcp_distance = np.linalg.norm(get_tcp_pose(env).p - get_base_pose(env).sp.p)
    base_pos_near_target = pose.p - 1.35 * dir_to_shelf

    base_pos_near_target[2] = 0
    res = planner.drive_base(base_pos_near_target, dir_to_shelf, arm_actions=reset_arm_actions)
    if res == -1:
        return res

    delta_h = pose.p[2] - get_tcp_pose(env).p[2]
    res = planner.lift_body(delta_h)
    return res


def align_to_target_product(env, planner: FetchMotionPlanningSapienSolver, target_product_actor):
    return align_to_target_pose(env, planner, target_product_actor.pose.sp)

def approach_and_manipulate_to_pose_in_shelf(
    env, 
    planner: FetchMotionPlanningSapienSolver,
    target_center_pos: np.ndarray,
    target_poses_list: list[sapien.Pose],
    num_tries = 5,
    last_resort_pregrasp_approach_distance = 0.15,
):
    success = False
    # delta_approach = 0

    res = align_ee_to_target_pos(env, planner, target_center_pos)
    if res == -1:
        return res

    for tries in range(num_tries):
        ik_solvable_graps = []

        for grasp in target_poses_list:
            planner._update_grasp_visual(grasp)
            planner.update_torso_pose()
            # planner.render_wait()
            if planner.check_IK(grasp):
                # dist_to_target = get_base_shift_tcp_to_target(env, target_center_pos)
                res = planner.static_manipulation(grasp)
                if res != -1:
                    # delta_approach += dist_to_target
                    success = True
                    break
                ik_solvable_graps.append(grasp)

        if success:
            break

        if len(ik_solvable_graps) > 0: # try to approach closer if ik is solvable
            for solvable_grasp in ik_solvable_graps:
                last_resort_approach_pose = solvable_grasp * sapien.Pose([0, 0, -last_resort_pregrasp_approach_distance])
                if planner.check_IK(last_resort_approach_pose):
                    # dist_to_target = get_base_shift_tcp_to_target(env, target_center_pos)
                    res = planner.static_manipulation(last_resort_approach_pose)
                    if res != -1:
                        # delta_approach += dist_to_target
                        break

        dist_to_shelf = get_direction_to_shelf(env)
        res, _ = planner.move_base_forward_delta(dist_to_shelf / 2)
        if res == -1:
            return res
        # delta_approach += dist_to_shelf / 2

        res = align_ee_to_target_pos(env, planner, target_center_pos)
        if res == -1:
            return res
    if not success:
        return -1

    return res

def fetch_object_from_shelf(
    env, 
    planner: FetchMotionPlanningSapienSolver, 
    target_product_actor, 
    n_grasps=5, 
    num_tries = 5, 
    approach_distance = 0.1,
    last_resort_pregrasp_approach_distance = 0.15,
):
    FINGER_LENGTH = 0.03
    obb = get_actor_obb(target_product_actor)
    target_center_pos = np.array(obb.primitive.transform)[:3, 3]

    if is_mesh_cylindrical(target_product_actor):
        grasp_infos = compute_cylinder_grasp_info(
            target_product_actor,
            target_closing=get_tcp_matrix(env)[:3, 1],
            ee_direction=get_tcp_matrix(env)[:3, 2],
            depth=FINGER_LENGTH,
            n_grasps_central=n_grasps,
            n_grasps_lateral=n_grasps,
            central_angle_range=[-np.pi/4, np.pi/4],
            lateral_angle_range=[-np.pi/4, np.pi/4],
        )
    else:   
        grasp_infos = compute_box_grasp_thin_side_info(
            obb,
            target_closing=get_tcp_matrix(env)[:3, 1],
            ee_direction=get_tcp_matrix(env)[:3, 2],
            depth=FINGER_LENGTH,
            n_grasps=n_grasps,
        )
    grasps = []
    for grasp_info in grasp_infos:
        grasp_closing, grasp_center, grasp_approaching = grasp_info["closing"], grasp_info["center"], grasp_info["approaching"]
        grasp_pose = env.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)
        grasps.append(grasp_pose)

    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        get_fcl_object_name(target_product_actor), True
    )
    planner.planner.update_from_simulation()

    start_tcp_pos = get_tcp_pose(env).p

    res = approach_and_manipulate_to_pose_in_shelf(
        env, 
        planner, 
        target_center_pos, 
        grasps, 
        num_tries=num_tries, 
        last_resort_pregrasp_approach_distance=last_resort_pregrasp_approach_distance,
    )
    if res == -1:
        return res

    res = planner.close_gripper()
    if res == -1:
        return res
    res = planner.lift_body(0.05)
    if res == -1:
        return res

    delta_approach = start_tcp_pos - get_tcp_pose(env).p
    delta_approach[2] = 0
    delta_approach = np.linalg.norm(delta_approach)

    res = planner.move_base_forward_delta(-delta_approach)
    if res == -1:
        return res

    return res

def drop_to_basket(env, planner: FetchMotionPlanningSapienSolver):
    goal_center = env.calc_target_pose().sp.p
    goal_center = goal_center + np.array([0.05, 0., 0.4]) # add shift from base to basket

    goal_approaching = np.array([0, 0., -1.])
    goal_closing = - get_base_pose(env).sp.to_transformation_matrix()[:3, 1]

    goal_pose = env.agent.build_grasp_pose(goal_approaching, goal_closing, goal_center)

    res = planner.lift_body(0.3)
    if res == -1:
        return res
    
    res = planner.static_manipulation(goal_pose)
    if res == -1:
        return res
    
    res = planner.open_gripper()
    if res == -1:
        return res

    res = planner.idle_steps(t=10)
    if res == -1:
        return res
    return res