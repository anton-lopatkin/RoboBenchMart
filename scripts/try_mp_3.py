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
import sys
sys.path.append('.')
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
from dsynth.planning.fetch_skills import *

def solve_dummy(env, seed=None, debug=True, vis=False):
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
    target_actor = env.actors['products']['[ENV#0]_food.ENERGY_DRINKS.MonsterEnergyDrink:0:0:3:0']
    res = align_to_target_product(env, planner, target_actor)
    if res == -1:
        return res

    # res = align_ee_to_target_product(env, planner, target_actor)
    return res

def solve_new(env, seed=None, debug=True, vis=False):
    env.reset(seed=seed, options={'reconfigure': True})
    planner = FetchMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        # base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=True,
        verbose=True,
    )

    env = env.unwrapped
    target_actors = [
        # env.actors['products']['[ENV#0]_food.dairy_products.milkCarton:0:3:2:0'],
        env.actors['products']['[ENV#0]_food.CEREALS.NestleFitnessChocolateCereals:0:2:1:0'],
        env.actors['products']['[ENV#0]_food.CRACKERS_COOKIES.OreoLemonCremeSandwichCookies:0:1:2:0'],
        env.actors['products']['[ENV#0]_food.BEER.DuffBeerCan:0:0:3:0'],
    ]

    for target_actor in target_actors:
        obb = get_actor_obb(target_actor)
        target_actor_pose_init = sapien.Pose(p=target_actor.pose.sp.p, q=target_actor.pose.sp.q)
        target_center = np.array(obb.primitive.transform)[:3, 3]
        target_center[2] += 0.47
        direction = env.directions_to_shelf[0]
        closing = np.cross(direction, [0., 0., 1.])
        target_pose = env.agent.build_grasp_pose(direction, closing, target_center)
        # target_pose = sapien.Pose(p=target_center, q=target_actor.pose.sp.q)

        res = align_to_target_product(env, planner, target_actor)
        if res == -1:
            return res
        
        res = fetch_object_from_shelf(env, planner, target_actor, n_grasps=10, num_tries=5)
        if res == -1:
            return res

        res = align_to_target_pose(env, planner, target_pose)
        if res == -1:
            planner.render_wait()
            return res

        res = place_object_to_pos(env, planner, target_center, env.directions_to_shelf[0], n_grasps=10)
        if res == -1:
            return res

def solve(env, seed=None, debug=True, vis=False):
    env.reset(seed=seed, options={'reconfigure': True})
    planner = FetchMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        # base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=True,
        verbose=False,
    )

    env = env.unwrapped
    target_actors = [
        env.actors['products']['[ENV#0]_food.ENERGY_DRINKS.MonsterEnergyDrink:0:0:3:0'],
        env.actors['products']['[ENV#0]_food.CRACKERS_COOKIES.OreoLemonCremeSandwichCookies:0:1:2:0'],
        env.actors['products']['[ENV#0]_food.CEREALS.NestleFitnessChocolateCereals:0:2:1:0'],
        # env.actors['products']['[ENV#0]_food.ENERGY_DRINKS.MonsterEnergyDrink:0:0:2:0'],
        env.actors['products']['[ENV#0]_food.BEER.DuffBeerCan:0:0:1:0'],
        env.actors['products']['[ENV#0]_food.dairy_products.milkCarton:0:3:2:0'],
        # env.actors['products']['[ENV#0]_food.HOUSEHOLD.AceDetergent:0:4:2:0']
    ]
    # target_actor = env.actors['products']['[ENV#0]_food.dairy_products.milkCarton:0:2:4:0']
    for target_actor in target_actors:
        res = align_to_target_product(env, planner, target_actor)
        if res == -1:
            return res
        
        res = fetch_object_from_shelf(env, planner, target_actor, n_grasps=6, num_tries=5)
        if res == -1:
            return res

        res = drop_to_basket(env, planner)
        if res == -1:
            return res
            
    return res
    
def _main(proc_id: int = 0, start_seed: int = 0) -> str:
    env_id = 'PickToBasketContEnv'
    env = gym.make(
        env_id,
        obs_mode='state',
        robot_uids='ds_fetch_basket',
        config_dir_path = 'generated_envs/ds_small_scene2/',
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
    num_traj = 1
    vis = True
    only_count_success = False
    pbar = tqdm(range(num_traj), desc=f"proc_id: {proc_id}")
    seed = start_seed
    successes = []
    solution_episode_lengths = []
    failed_motion_plans = 0
    passed = 0
    while True:
        #try:
        res = solve(env, seed=seed, debug=False, vis=True if vis else False)
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