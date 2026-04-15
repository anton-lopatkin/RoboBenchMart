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
import sapien
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


from dsynth.robots.ds_fetch import DSFetchBasket2

class MyMotionPlanningSolver:
    OPEN = 1
    CLOSED = -1
    # MOVE_GROUP = "panda_hand_tcp"
    MOVE_GROUP = "gripper_link"

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        # base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        #==========================================#
        visualize_target_grasp_pose: bool = True,
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        # self.base_pose = to_sapien_pose(base_pose)

        self.planner = self.setup_planner()
        self.update_base_pose()
        self.control_mode = self.base_env.control_mode

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        
        self.elapsed_steps = 0

        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None

        #==========================================#
        self.gripper_state = self.OPEN
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            if "grasp_pose_visual" not in self.base_env.scene.actors:
                self.grasp_pose_visual = build_two_finger_gripper_grasp_pose_visual(
                    self.base_env.scene
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
            self.grasp_pose_visual.set_pose(self.base_env.agent.tcp_pose)

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.base_env.render_human()
        while True:
            if viewer.window.key_down("c"):
                break
            self.base_env.render_human()

    def setup_planner(self):
        move_group = self.MOVE_GROUP if hasattr(self, "MOVE_GROUP") else "eef"
        # link_names = [link.get_name() for link in self.robot.get_links()]
        link_names = [
            'torso_lift_link',
            'head_pan_link',
            'shoulder_pan_link',
            'bellows_link',
            'bellows_link2',
            'head_tilt_link',
            'shoulder_lift_link',
            'head_camera_link',
            'upperarm_roll_link',
            'head_camera_rgb_frame',
            'head_camera_depth_frame',
            'elbow_flex_link',
            'head_camera_rgb_optical_frame',
            'head_camera_depth_optical_frame',
            'forearm_roll_link',
            'wrist_flex_link',
            'wrist_roll_link',
            'gripper_link',
            'r_gripper_finger_link',
            'l_gripper_finger_link'
            ]
        # joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        joint_names = [
            'head_pan_joint',
            'shoulder_pan_joint',
            'head_tilt_joint',
            'shoulder_lift_joint',
            'upperarm_roll_joint',
            'elbow_flex_joint',
            'forearm_roll_joint',
            'wrist_flex_joint',
            'wrist_roll_joint',
            'r_gripper_finger_joint',
            'l_gripper_finger_joint',
        ]
        urdf_path = self.env_agent.urdf_arm_ik_path
        planner = mplib.Planner(
            urdf=urdf_path,
            srdf=urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=move_group,
        )
        # planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        planner.joint_vel_limits = np.asarray(planner.joint_vel_limits) * self.joint_vel_limits
        planner.joint_acc_limits = np.asarray(planner.joint_acc_limits) * self.joint_acc_limits
        return planner

    def update_base_pose(self):
        base_pose = self.base_env.agent.torso_lift_link.pose.sp
        self.planner.set_base_pose(np.hstack([base_pose.p, base_pose.q]))

    def _update_grasp_visual(self, target: sapien.Pose) -> None:
        #==========================================#
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(target)

    def _transform_pose_for_planning(self, target: sapien.Pose) -> sapien.Pose:
        return target

    def follow_path(self, result, refine_steps: int = 0):
        # n_step = result["position"].shape[0]
        # for i in range(n_step + refine_steps):
        #     qpos = result["position"][min(i, n_step - 1)]
        #     if self.control_mode == "pd_joint_pos_vel":
        #         qvel = result["velocity"][min(i, n_step - 1)]
        #         action = np.hstack([qpos, qvel])
        #     else:
        #         action = np.hstack([qpos])
        #     obs, reward, terminated, truncated, info = self.env.step(action)
        #     self.elapsed_steps += 1
        #     if self.print_env_info:
        #         print(
        #             f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
        #         )
        #     if self.vis:
        #         self.base_env.render_human()
        # return obs, reward, terminated, truncated, info
        #==========================================#
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            body_action = self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()
            if self.control_mode == "pd_joint_pos_vel":
                # qvel = result["velocity"][min(i, n_step - 1)]
                # action = np.hstack([qpos, qvel, self.gripper_state])
                raise NotImplementedError
            else:
                action = np.hstack([qpos, self.gripper_state, body_action, [0, 0]])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

        
    def move_to_pose_with_RRTStar(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        self.update_base_pose()
        pose = to_sapien_pose(pose)
        self._update_grasp_visual(pose)
        pose = self._transform_pose_for_planning(pose)
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            rrt_range=0.0,
            planning_time=1,
            planner_name="RRTstar",
            wrt_world=True,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        self.update_base_pose()
        pose = to_sapien_pose(pose)
        self._update_grasp_visual(pose)
        pose = self._transform_pose_for_planning(pose)
        current_qpos = self.robot.get_qpos().cpu().numpy()[0, 4:]
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            current_qpos,
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            wrt_world=True,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        # try screw two times before giving up
        self._update_grasp_visual(pose)
        pose = self._transform_pose_for_planning(pose)
        result = self.planner.plan_screw(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def add_box_collision(self, extents: np.ndarray, pose: sapien.Pose):
        self.use_point_cloud = True
        box = trimesh.creation.box(extents, transform=pose.to_transformation_matrix())
        pts, _ = trimesh.sample.sample_surface(box, 256)
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

        pc = trimesh.points.PointCloud(pts)
        scene = trimesh.Scene([pc])
        scene.show()


    def add_collision_pts(self, pts: np.ndarray):
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def clear_collisions(self):
        self.all_collision_pts = None
        self.use_point_cloud = False

    def close(self):
        pass

    #==========================================#
    def open_gripper(self, t=6, gripper_state=None):
        if gripper_state is None:
            gripper_state = self.OPEN
        self.gripper_state = gripper_state
        qpos = self.robot.get_qpos()[0, : len(self.planner.joint_vel_limits)].cpu().numpy()
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6, gripper_state=None):
        if gripper_state is None:
            gripper_state = self.CLOSED
        self.gripper_state = gripper_state
        qpos = self.robot.get_qpos()[0, : len(self.planner.joint_vel_limits)].cpu().numpy()
        qpos = self.env_agent.controller.controllers['arm'].qpos[0].cpu().numpy()
        body_action = self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state, body_action, [0, 0]])
            else:
                raise NotImplementedError
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

@register_env("PickCube-v2", max_episode_steps=50)
class PickCube2Env(PickCubeEnv):
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[5, 0, self.cube_half_size]),
        )

        self.wall = actors.build_box(
            self.scene,
            half_sizes=[0.02, 0.35, 0.15],
            color=[0, 0, 1, 1],
            name="wall",
            initial_pose=sapien.Pose(p=[0.06, 0, 0.15]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[10, 10, 10]),
        )
        self._hidden_objects.append(self.goal_site)


    def _load_agent(self, options: dict):
        super(PickCubeEnv, self)._load_agent(options, sapien.Pose(p=[-0.915, 0, 0]))
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        height = 0.9196429 / 2
        with torch.device(self.device):
            b = len(env_idx)
            
            if self.robot_uids == "ds_fetch_basket2":
                qpos = np.array(
                    [
                        0,
                        0,
                        0,
                        0.0,#386,
                        0,
                        0,
                        0,
                        -np.pi / 4,
                        0,
                        np.pi / 4,
                        0,
                        np.pi / 3,
                        0,
                        0.015,
                        0.015,
                    ]
                )
                self.agent.reset(qpos)
                self.agent.robot.set_pose(sapien.Pose([-0.715, 0, 0]))

                self.table_scene.ground.set_collision_group_bit(
                    group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
                )
            
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.zeros((b, 2)) * self.cube_spawn_half_size * 2 + 0.01
                - self.cube_spawn_half_size
            )
            xyz[:, 0] += self.cube_spawn_center[0] - 0.2
            xyz[:, 1] += self.cube_spawn_center[1]

            xyz[:, 2] = self.cube_half_size +  height
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2] + height
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
            self.wall.set_pose(Pose.create_from_pq(p=[-0.1, 0, 0.15 + height]))


def solve(env: PickCube2Env, seed=None, debug=True, vis=False):
    env.reset(seed=seed)
    planner = MyMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        # base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    planner.add_box_collision(env.wall.get_collision_meshes()[0].extents, env.wall.pose.sp)
    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(env.cube)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.cube.pose.sp.p)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    # planner.move_to_pose_with_screw(reach_pose)
    planner.move_to_pose_with_RRTConnect(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    # planner.move_to_pose_with_screw(grasp_pose)
    planner.move_to_pose_with_RRTConnect(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p, grasp_pose.q)
    res = planner.move_to_pose_with_RRTConnect(goal_pose)

    planner.close()
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
    env_id = 'PickCube-v2'
    env = gym.make(
        env_id,
        obs_mode='none',
        robot_uids='ds_fetch_basket2',
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