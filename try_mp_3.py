import multiprocessing as mp
import os
from copy import deepcopy
import time
import argparse
import torch
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os.path as osp
from pathlib import Path
from transforms3d.euler import euler2quat
import sapien
from mani_skill.utils.building.ground import build_ground
from mani_skill import PACKAGE_DIR
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.examples.motionplanning.panda.solutions import solvePushCube, solvePickCube, solveStackCube, solvePegInsertionSide, solvePlugCharger, solvePullCubeTool, solveLiftPegUpright, solvePullCube, solveDrawTriangle, solveDrawSVG, solvePlaceSphere, solveStackPyramid
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda, WidowXAI, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
import mplib
import numpy as np
import sapien
import trimesh

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.examples.motionplanning.two_finger_gripper.motionplanner import build_two_finger_gripper_grasp_pose_visual

from mani_skill.envs.tasks import PickCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)


from dsynth.robots.ds_fetch import DSFetchBasket
from dsynth.planning.motionplanner import FetchMotionPlanningSapienSolver
from dsynth.envs import *
from dsynth.planning.utils import (
    get_fcl_object_name, 
    compute_box_grasp_thin_side_info,
)

def solve(env, seed=None, debug=True, vis=False):
    env.reset(seed=seed, options={'reconfigure': True})
    planner = FetchMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        # base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        verbose=True,
    )

    FINGER_LENGTH = 0.1
    env = env.unwrapped

    def get_tcp_pose():
        return env.agent.tcp.pose

    def get_tcp_matrix():
        tcp_pose = get_tcp_pose()
        return tcp_pose.to_transformation_matrix()[0].cpu().numpy()

    # planner.add_box_collision(env.wall.get_collision_meshes()[0].extents, env.wall.pose.sp)
    target_actor = env.actors['products']['[ENV#0]_food.dairy_products.milkCarton:0:2:4:0']
    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(target_actor)

    grasp_info = compute_box_grasp_thin_side_info(
        obb,
        target_closing=get_tcp_matrix()[:3, 1],
        ee_direction=get_tcp_matrix()[:3, 2],
        depth=FINGER_LENGTH,
    )
    grasp_closing, grasp_center, grasp_approaching = grasp_info["closing"], grasp_info["center"], grasp_info["approaching"]
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_pose = env.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)
    pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.15])

    dir_to_shelf = env.directions_to_shelf[0]
    

    # -------------------------------------------------------------------------- # # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    base_pos_near_target = pre_grasp_pose.p - 1 * dir_to_shelf
    base_pos_near_target[2] = 0
    res = planner.drive_base(base_pos_near_target, dir_to_shelf)
    if res == -1:
        return res

    delta_h = pre_grasp_pose.p[2] - get_tcp_pose().sp.p[2]
    planner.lift_body(delta_h)
    
    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    # reach_pose = env.agent.tcp.pose.sp * sapien.Pose([-0.15, 0, -0.0])
    # planner.move_to_pose_with_screw(reach_pose)
    planner.static_manipulation(pre_grasp_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    # planner.move_to_pose_with_screw(grasp_pose)
    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        get_fcl_object_name(target_actor), True
    )

    for i in range(3):
        if not planner.check_IK(grasp_pose):
            res = planner.move_base_forward_delta(0.05)
            delta_h += 0.05
            if res == -1:
                return res
        else:
            break
    planner.static_manipulation(grasp_pose)
    planner.close_gripper()

    res = planner.move_base_forward_delta(-delta_h)
    if res == -1:
        return res

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_center = env.calc_target_pose().sp.p
    goal_center = goal_center + np.array([0.1, 0., 0.2]) # add shift from base to basket

    goal_approaching = np.array([0, 0., -1.])
    goal_closing = - get_base_pose().sp.to_transformation_matrix()[:3, 1]

    goal_pose = env.agent.build_grasp_pose(goal_approaching, goal_closing, goal_center)
    planner.open_gripper()
    # planner.close()
    return res



# def parse_args(args=None):
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1", help=f"Environment to run motion planning solver on. Available options are {list(MP_SOLUTIONS.keys())}")
#     parser.add_argument("-o", "--obs-mode", type=str, default="none", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
#     parser.add_argument("-n", "--num-traj", type=int, default=10, help="Number of trajectories to generate.")
#     parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
#     parser.add_argument("--reward-mode", type=str)
#     parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
#     parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
#     parser.add_argument("--vis", action="store_true", help="whether or not to open a GUI to visualize the solution live")
#     parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
#     parser.add_argument("--traj-name", type=str, help="The name of the trajectory .h5 file that will be created.")
#     parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
#     parser.add_argument("--record-dir", type=str, default="demos", help="where to save the recorded trajectories")
#     parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
#     return parser.parse_args()

def _main(proc_id: int = 0, start_seed: int = 0) -> str:
    env_id = 'DarkstoreContinuousBaseEnv'
    env = gym.make(
        env_id,
        obs_mode='none',
        robot_uids='ds_fetch_basket',
        config_dir_path = 'generated_envs/ds_small_scene/',
        control_mode="pd_joint_pos",
        render_mode="human",
        sensor_configs=dict(shader_pack='default'),
        human_render_camera_configs=dict(shader_pack='default'),
        viewer_camera_configs=dict(shader_pack='default'),
        sim_backend='auto',
    )
    # if env_id not in MP_SOLUTIONS:
    #     raise RuntimeError(f"No already written motion planning solutions for {env_id}. Available options are {list(MP_SOLUTIONS.keys())}")

    # if not args.traj_name:
    #     new_traj_name = time.strftime("%Y%m%d_%H%M%S")
    # else:
    #     new_traj_name = args.traj_name
    new_traj_name = time.strftime("%Y%m%d_%H%M%S")
    save_video = False
    env = RecordEpisode(
        env,
        output_dir=osp.join('demos', env_id, "motionplanning"),
        trajectory_name=new_traj_name, save_video=save_video,
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        video_fps=30,
        record_reward=False,
        save_on_reset=False
    )
    output_h5_path = env._h5_file.filename
    # solve = MP_SOLUTIONS[env_id]
    print(f"Motion Planning Running on {env_id}")
    num_traj = 10
    vis = True
    only_count_success = True
    pbar = tqdm(range(num_traj), desc=f"proc_id: {proc_id}")
    seed = start_seed
    successes = []
    solution_episode_lengths = []
    failed_motion_plans = 0
    passed = 0
    while True:
        #try:
        res = solve(env, seed=seed, debug=True, vis=True if vis else False)
        # except Exception as e:
        #     print(f"Cannot find valid solution because of an error in motion planning solution: {e}")
        #     res = -1

        if res == -1:
            success = False
            failed_motion_plans += 1
        else:
            success = res[-1]["success"].item()
            elapsed_steps = res[-1]["elapsed_steps"].item()
            solution_episode_lengths.append(elapsed_steps)
        successes.append(success)
        if only_count_success and not success:
            seed += 1
            env.flush_trajectory(save=False)
            if save_video:
                env.flush_video(save=False)
            continue
        else:
            env.flush_trajectory()
            if save_video:
                env.flush_video()
            pbar.update(1)
            pbar.set_postfix(
                dict(
                    success_rate=np.mean(successes),
                    failed_motion_plan_rate=failed_motion_plans / (seed + 1),
                    avg_episode_length=np.mean(solution_episode_lengths),
                    max_episode_length=np.max(solution_episode_lengths),
                    # min_episode_length=np.min(solution_episode_lengths)
                )
            )
            seed += 1
            passed += 1
            if passed == num_traj:
                break
    env.close()
    return output_h5_path

def main():
    _main()

if __name__ == "__main__":
    # start = time.time()
    mp.set_start_method("spawn")
    main()
    # print(f"Total time taken: {time.time() - start}")