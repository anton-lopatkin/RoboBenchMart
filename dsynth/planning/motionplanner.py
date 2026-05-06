import mplib
import numpy as np
from collections import deque
import sapien
import trimesh
import sapien.physx as physx
from transforms3d.euler import euler2quat, euler2mat
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

from mani_skill.examples.motionplanning.two_finger_gripper.motionplanner import build_two_finger_gripper_grasp_pose_visual

from mplib.collision_detection.fcl import FCLObject
from mplib.sapien_utils.conversion import convert_object_name
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
from mplib.pymp import ArticulatedModel

from dsynth.planning.utils import SapienPlanningWorldV2#, SapienPlannerV2

OPEN = 1
CLOSED = -1

TORSO_LINK_NAMES = [
    'torso_lift_link', 'head_pan_link', 'shoulder_pan_link', 'bellows_link', 
    'bellows_link2', 'head_tilt_link', 'shoulder_lift_link', 'head_camera_link', 
    'upperarm_roll_link', 'head_camera_rgb_frame', 'head_camera_depth_frame', 
    'elbow_flex_link', 'head_camera_rgb_optical_frame', 'head_camera_depth_optical_frame', 
    'forearm_roll_link', 'wrist_flex_link', 'wrist_roll_link', 'gripper_link', 
    'r_gripper_finger_link', 'l_gripper_finger_link'
]
TORSO_JOINT_NAMES = [
    'head_pan_joint', 'shoulder_pan_joint', 'head_tilt_joint', 'shoulder_lift_joint', 
    'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 
    'wrist_roll_joint', 'r_gripper_finger_joint', 'l_gripper_finger_joint'
]

class FetchMotionPlanningSapienSolver:
    OPEN = 1
    CLOSED = -1
    # MOVE_GROUP = "panda_hand_tcp"
    MOVE_GROUP = "gripper_link"

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        #==========================================#
        visualize_target_grasp_pose: bool = True,
        verbose: bool = False,
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        self._sim_scene: sapien.Scene = self.base_env.scene.sub_scenes[0]

        self.planner = self.setup_planner()
        self.update_torso_pose()
        self.control_mode = self.base_env.control_mode

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.verbose = verbose
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

    def setup_planner(self, objects = []):
        # raise NotImplementedError
        link_names = TORSO_LINK_NAMES
        joint_names = TORSO_JOINT_NAMES

        planned_articulation = self._sim_scene.get_all_articulations()[0]
        planning_world = SapienPlanningWorldV2(
            self._sim_scene, 
            user_link_names=link_names,
            user_joint_names=joint_names,
            planned_articulations=[planned_articulation], 
            planned_urdf_paths=[self.env_agent.urdf_arm_ik_path],
            disable_actors_collision=False,
            new_package_keyword="",
            use_convex=False,
            verbose=False,
            )
        planner = SapienPlanner(
            planning_world,
            "gripper_link",
            joint_vel_limits=np.ones(7) * self.joint_vel_limits,
            joint_acc_limits=np.ones(7) * self.joint_acc_limits
        )
        return planner

    def update_torso_pose(self):
        base_pose = self.base_env.agent.torso_lift_link.pose.sp
        self.planner.set_base_pose(mplib.Pose(base_pose.p, base_pose.q))

    def _update_grasp_visual(self, target: sapien.Pose) -> None:
        #==========================================#
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(target)

    def _transform_pose_for_planning(self, target: sapien.Pose) -> sapien.Pose:
        return target

    def lift_body(self, 
        delta_h = 0.1, 
        abort_when_collision: bool = True, 
        k_p=1.0, 
        k_d=0.2,
        k_i=0.,
        max_abs_delta_control=0.2, 
        tol=1e-2,
        max_stuck_steps: int = 20,
        stuck_tol: float = 1e-3,
        steps_to_abortion: int = 200,
    ):
        LIFT_JOINT_INDEX = 10

        current_q_lift_joint = self.robot.get_qpos().cpu().numpy()[0, 3]
        qlimits = self.robot.qlimits[0, 3].cpu().numpy()

        target_q_lift_joint = current_q_lift_joint + delta_h
        target_q_lift_joint = np.clip(target_q_lift_joint, qlimits[0], qlimits[1])

        true_delta_h = target_q_lift_joint - current_q_lift_joint

        last_lift_heights = deque(maxlen=max_stuck_steps)

        if self.grasp_pose_visual is not None:
            tcp_pose = self.base_env.agent.tcp.pose.sp
            target_p = tcp_pose.p
            target_p[2] += true_delta_h
            target_tcp_pose = sapien.Pose(p=target_p, q=tcp_pose.q)
            self._update_grasp_visual(target_tcp_pose)
        self.render_wait()

        last_error = 0
        error_integral = 0
        dt = 1 / self.base_env.control_freq

        arm_action = self.env_agent.controller.controllers['arm'].qpos[0].cpu().numpy()
        body_action = np.zeros(3)
        action = np.hstack([arm_action, self.gripper_state, body_action, [0, 0]])

        n_steps = 0
        while True:
            current_q_lift_joint = self.robot.get_qpos().cpu().numpy()[0, 3]
            last_lift_heights.append(current_q_lift_joint)


            if len(last_lift_heights) >= max_stuck_steps and np.std(last_lift_heights) < stuck_tol:
                # robot is stuck
                print("Robot is stuck")
                self.planner.update_from_simulation()
                return self.idle_steps(t=1)

            current_error = target_q_lift_joint - current_q_lift_joint
            error_diff = (current_error - last_error) / dt
            error_integral += current_error * dt
            # print(current_error, last_error, error_diff)

            if np.abs(current_error) < tol and np.abs(error_diff) < 0.1 * tol and \
                    np.abs(self.robot.get_qvel().cpu().numpy()[0, 3]) < tol:
                self.planner.update_from_simulation()
                return self.idle_steps(t=1)

            last_error = current_error.copy()
            control_delta = k_p * current_error + k_d * error_diff + k_i * error_integral
            control_delta = np.clip(control_delta, -max_abs_delta_control, max_abs_delta_control)
            action[LIFT_JOINT_INDEX] = current_q_lift_joint + control_delta

            obs, reward, terminated, truncated, info = self.env.step(action)
            if self.verbose:
                print(f"n_steps: {n_steps} body Action:", np.round(action[LIFT_JOINT_INDEX], 4))
                print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))

            n_steps += 1
            self.update_torso_pose()

            self.elapsed_steps += 1
            if self.print_env_info:
                print(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
            if self.vis:
                self.base_env.render_human()

            if abort_when_collision:
                if len(collisions := self.planner.planning_world.check_collision()) > 0:
                    print("Collision detected while lifting body")
                    for collision in collisions:
                            print(
                                f"Collision between {collision.link_name1} of entity "
                                f"{collision.object_name1} with {collision.link_name2} "
                                f"of entity {collision.object_name2}"
                            )
                    return -1
            if n_steps > steps_to_abortion:
                print("Reached max steps. Something went wrong.")
                return -1

    def base_x_direction(self):
        base_link_pose = self.base_env.agent.base_link.pose.sp
        return base_link_pose.to_transformation_matrix()[:3, 0]

    def rotate_base_z(self, 
        new_direction,
        arm_actions={"position": [], "velocity": []},
        tol=1e-2, 
        k_p=1.7, 
        k_d=0.01, 
        max_vel=0.8,
        max_stuck_steps: int = 20,
        stuck_tol: float = 1e-3,
        abort_when_collision: bool = True,
        steps_to_abortion: int = 300,
    ):
        def _calc_current_error(target_direction):
            cur_x_direction = self.base_x_direction()
            current_error = np.arccos(np.clip(np.dot(target_direction, cur_x_direction) / \
                                    np.linalg.norm(cur_x_direction) / \
                                        np.linalg.norm(target_direction),
                                    -1, 1
                            ))
            if np.cross(cur_x_direction, target_direction)[2] < 0:
                current_error = -current_error
            return current_error

        arm_action = self.env_agent.controller.controllers['arm'].qpos[0].cpu().numpy()
        body_action = self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()
        action = np.hstack([arm_action, self.gripper_state, body_action, [0., 0.]])

        last_error = 0
        dt = 1 / self.base_env.control_freq

        if self.grasp_pose_visual is not None:
            current_error = _calc_current_error(new_direction)
            base_link_pose = self.base_env.agent.base_link.pose.sp
            tcp_pose = self.env.unwrapped.agent.tcp.pose.sp
            rotation_wrt_base_link = sapien.Pose(q=euler2quat(0, 0, current_error))
            target_tcp_pose = base_link_pose * rotation_wrt_base_link * base_link_pose.inv() * tcp_pose
            self.grasp_pose_visual.set_pose(target_tcp_pose)

        self.render_wait()

        n_steps = 0
        rotations = deque(maxlen=max_stuck_steps)
        while True:
            cur_x_direction = self.base_x_direction()
            current_error = np.arccos(np.clip(np.dot(new_direction, cur_x_direction) / \
                                  np.linalg.norm(cur_x_direction) / \
                                    np.linalg.norm(new_direction),
                                  -1, 1
                        ))

            rotations.append(current_error)

            if len(rotations) >= max_stuck_steps and np.std(rotations) < stuck_tol:
                # robot is stuck
                print("Robot is stuck")
                self.planner.update_from_simulation()
                return self.idle_steps(t=1), arm_actions
            
            
            if np.cross(cur_x_direction, new_direction)[2] < 0:
                current_error = -current_error
            
            error_diff = (current_error - last_error) / dt
            last_error = current_error
            
            if np.abs(current_error) < tol and np.abs(error_diff) < tol ** 2:
                self.planner.update_from_simulation()
                return self.idle_steps(t=1), arm_actions


            control_omega = k_p * current_error + k_d * error_diff
            control_omega = np.clip(control_omega, -max_vel, max_vel)

            if len(arm_actions["position"]) > 0:
                arm_action = arm_actions["position"][0]
                arm_actions["position"] = np.delete(arm_actions["position"], 0, axis=0)
                arm_actions["velocity"] = np.delete(arm_actions["velocity"], 0, axis=0)
            
            action[:7] = arm_action
            action[-1] = control_omega

            if self.verbose:
                print(f"n_steps: {n_steps} base Action:", np.round(action[-2:], 4))
                print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self.env.step(action)
            n_steps += 1
            self.update_torso_pose()

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()

            if abort_when_collision:
                if len(collisions := self.planner.planning_world.check_collision()) > 0:
                    print("Collision detected while rotating base")
                    for collision in collisions:
                            print(
                                f"Collision between {collision.link_name1} of entity "
                                f"{collision.object_name1} with {collision.link_name2} "
                                f"of entity {collision.object_name2}"
                            )
                    return -1, None
            if n_steps > steps_to_abortion:
                print("Reached max steps. Something went wrong.")
                return -1, None

    
    def drive_base(self, target_pos=None, target_view_vec=None, arm_actions=None):
        if arm_actions is None:
            arm_actions = {"position": [], "velocity": []}
        
        if not target_pos is None:
            moving_direction = target_pos - self.base_env.agent.base_link.pose.sp.p
            moving_direction[2] = 0.

            if np.linalg.norm(moving_direction) < 1e-2:
                res = self.idle_steps(t=1)
                if res == -1:
                    return res
                self.planner.update_from_simulation()

            else:
                res, arm_actions = self.rotate_base_z(moving_direction, 
                    arm_actions=arm_actions)
                if res == -1:
                    return res
                self.planner.update_from_simulation()

                delta = np.linalg.norm(moving_direction)

                res, arm_actions = self.move_base_forward_delta(delta, 
                    arm_actions=arm_actions)
                if res == -1:
                    return res
                self.planner.update_from_simulation()
        
        # view_direction = target_view_pos.p - self.base_env.agent.base_link.pose.sp.p
        if not target_view_vec is None:
            res, arm_actions = self.rotate_base_z(target_view_vec,
                arm_actions=arm_actions)
            if res == -1:
                return res

        if len(arm_actions["position"]) > 0:
            res = self.follow_path(arm_actions)
            if res == -1:
                return res
        self.planner.update_from_simulation()

        return res

    def plan_reset_arm(self):
        self.planner.update_from_simulation()
        self.update_torso_pose()
        current_qpos = self.robot.get_qpos().cpu().numpy()[0, 4:]
        goal_qpos = [self.env_agent.keyframes['rest'].qpos[4:]]
        if np.all(np.isclose(goal_qpos, current_qpos)):
            return {"position": [], "velocity": []}
        result = self.planner.plan_qpos(
            goal_qpos,  # type: ignore
            current_qpos,
            time_step=0.1,
            rrt_range=0.1,
            planning_time=1,
            fix_joint_limits=True,
            simplify=True,
            constraint_function=None,
            constraint_jacobian=None,
            constraint_tolerance=1e-3,
            verbose=self.verbose,
        )
        if result["status"] != "Success":
            return -1
        return result

    
    def base_x_pos(self):
        base_link_pose = self.base_env.agent.base_link.pose.sp
        return base_link_pose.to_transformation_matrix()[:3, 3]

    def move_base_forward_delta(
        self, 
        delta, 
        arm_actions={"position": [], "velocity": []},
        tol=1e-2, 
        k_p=1.7, 
        k_d=0.02, 
        max_vel=0.8,
        max_stuck_steps: int = 20,
        stuck_tol: float = 1e-3,
        abort_when_collision: bool = True,
        steps_to_abortion: int = 300,
    ):
        arm_action = self.env_agent.controller.controllers['arm'].qpos[0].cpu().numpy()
        body_action = self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()
        action = np.hstack([arm_action, self.gripper_state, body_action, [0., 0.]])

        last_errors = deque(maxlen=max_stuck_steps)
        base_direction = self.base_x_direction()

        dst_pos = self.base_x_pos() + delta * base_direction
        
        if self.grasp_pose_visual is not None:
            tcp_pose = self.base_env.agent.tcp.pose.sp
            tcp_pos_dst = tcp_pose.p + delta * base_direction
            self.grasp_pose_visual.set_pose(sapien.Pose(p=tcp_pos_dst, q=tcp_pose.q))

        last_error = 0
        dt = 0.01

        self.render_wait()

        n_steps = 0
        while True:
            cur_pos = self.base_x_pos()
            current_error = np.dot(base_direction, dst_pos - cur_pos)
            
            last_errors.append(current_error)
            error_diff = (current_error - last_error) / dt
            last_error = current_error

            if len(last_errors) >= max_stuck_steps and np.std(last_errors) < stuck_tol:
                # robot is stuck
                print("Robot is stuck")
                self.planner.update_from_simulation()
                return self.idle_steps(t=1), arm_actions

            if np.abs(current_error) < tol:# and np.abs(error_diff) < tol:
                self.planner.update_from_simulation()
                return self.idle_steps(t=1), arm_actions

            control_vel = k_p * current_error + k_d * error_diff
            control_vel = np.clip(control_vel, -max_vel, max_vel)

            if len(arm_actions["position"]) > 0:
                arm_action = arm_actions["position"][0]
                arm_actions["position"] = np.delete(arm_actions["position"], 0, axis=0)
                arm_actions["velocity"] = np.delete(arm_actions["velocity"], 0, axis=0)
            action[:7] = arm_action
            action[-2] = control_vel

            if self.verbose:
                print(f"n_steps: {n_steps} base Action:", np.round(action[-2:], 4))
                print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self.env.step(action)
            n_steps += 1
            self.update_torso_pose()

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()


            if abort_when_collision:
                if len(collisions := self.planner.planning_world.check_collision()) > 0:
                    print("Collision detected while moving base")
                    for collision in collisions:
                            print(
                                f"Collision between {collision.link_name1} of entity "
                                f"{collision.object_name1} with {collision.link_name2} "
                                f"of entity {collision.object_name2}"
                            )
                    return -1, None
            if n_steps > steps_to_abortion:
                print("Reached max steps. Something went wrong.")
                return -1, None
    
                
    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        body_action = self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()
        # body_action = [0, 0, 0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                # qvel = result["velocity"][min(i, n_step - 1)]
                # action = np.hstack([qpos, qvel, self.gripper_state])
                raise NotImplementedError
            else:
                action = np.hstack([qpos, self.gripper_state, body_action, [0, 0]])
                if self.verbose:
                    print("arm action:", np.round(qpos, 4))
                    print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self.env.step(action)

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        self.planner.update_from_simulation()
        return obs, reward, terminated, truncated, info

    def static_manipulation(self, target_tcp_pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0):
        res = self.move_to_pose_with_screw(target_tcp_pose, dry_run=dry_run, refine_steps=refine_steps)
        if res == -1:
            res = self.move_to_pose_with_RRTConnect(target_tcp_pose, dry_run=dry_run, refine_steps=refine_steps)
            if res == -1:
                return res
        self.planner.update_from_simulation()
        return res
    
    def check_IK(self, pose: sapien.Pose):
        self.update_torso_pose()
        pose = to_sapien_pose(pose)
        pose = self._transform_pose_for_planning(pose)
        current_qpos = self.robot.get_qpos().cpu().numpy()[0, 4:]
        pose = mplib.Pose(p=pose.p, q=pose.q)
        current_qpos = np.clip(
            current_qpos, self.planner.joint_limits[:, 0], self.planner.joint_limits[:, 1]
        )
        current_qpos = self.planner.pad_move_group_qpos(current_qpos)
        pose = self.planner._transform_goal_to_wrt_base(pose)
        ik_status, goal_qpos = self.planner.IK(pose, current_qpos, [])
        return ik_status == "Success"

    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        self.update_torso_pose()
        pose = to_sapien_pose(pose)
        self._update_grasp_visual(pose)
        pose = self._transform_pose_for_planning(pose)
        current_qpos = self.robot.get_qpos().cpu().numpy()[0, 4:]
        result = self.planner.plan_pose(
            mplib.Pose(p=pose.p, q=pose.q),
            current_qpos,
            time_step=self.base_env.control_timestep,
            wrt_world=True,
            verbose=True,
            planning_time=2,
            rrt_range=0.1,
            simplify=True,
            # n_init_qpos=20

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
        self.update_torso_pose()
        pose = to_sapien_pose(pose)
        # try screw two times before giving up
        self._update_grasp_visual(pose)
        pose = self._transform_pose_for_planning(pose)
        current_qpos = self.robot.get_qpos().cpu().numpy()[0, 4:]
        result = self.planner.plan_screw(
            mplib.Pose(p=pose.p, q=pose.q),
            current_qpos,
            time_step=self.base_env.control_timestep,
            # use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                mplib.Pose(p=pose.p, q=pose.q),
                current_qpos,
                time_step=self.base_env.control_timestep,
                # use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def open_gripper(self, t=6, gripper_state=None):
        return self.change_gripper_state(t=t, gripper_state=self.OPEN)

    def close_gripper(self, t=6, gripper_state=None):
        return self.change_gripper_state(t=t, gripper_state=self.CLOSED)

    def change_gripper_state(self, t=6, gripper_state=None):
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

    def idle_steps(self, t=20):
        arm_action = self.env_agent.controller.controllers['arm'].qpos[0].cpu().numpy()
        # body_action = np.zeros_like(self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy())
        body_action = self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()
        base_action = np.array([0, 0])
        action = np.hstack([arm_action, self.gripper_state, body_action, base_action])
        for i in range(t):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info
