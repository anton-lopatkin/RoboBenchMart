import argparse
import time
from pathlib import Path

import gymnasium as gym
from mani_skill.utils.wrappers import RecordEpisode

import sys 
sys.path.append('.')
from dsynth.envs import *
from dsynth.robots import *

from planning.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from planning.controller import Controller
from planning.task_planner import TaskPlanner
from planning.utils import prepare_observations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", help="Path to dir with scene JSON-config")
    parser.add_argument("-e", "--env-id", type=str, default="CollectOrderItemsContLatteEnv", help="Environment to run")
    parser.add_argument("-r", "--robot-uids", type=str, default="ds_fetch_basket", help="Robot id")
    parser.add_argument("-n", "--num-envs", type=int, default=1, help="Number of scenes")
    parser.add_argument("-s", "--seed", type=int, nargs='+', default=13)
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument("--execute", action='store_true', default=False)
    parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--save-traj", action="store_true", help="whether or not to save trajectory locally")
    parser.add_argument("--history", action="store_true", default=False)
    parser.add_argument("-m", "--model", type=str, default=False)

    args = parser.parse_args()
    return args


def execute_with_replanning(env, planner, controller):
    language_instruction = 'take one milk and one beer' # env.language_instructions[0]
    observations = prepare_observations(env)

    plan = planner.plan(language_instruction, observations)

    history = []
    i = 0

    while i < len(plan.steps):
        step = plan.steps[i]
        fn = getattr(controller, step.name, None)

        if not callable(fn):
            raise KeyError(f"Unknown skill '{step.name}' in plan step {i + 1}")

        line = f"{i + 1}. {step.name}{f' {step.params}' if step.params else ''}"
        print(line, end=" ")

        result = fn(**step.params)
        if result == -1:
            print(f"[motion planning failed] \n{controller.last_stdout}")
            history.append(
                f"{line} [motion planning failed] \n{controller.last_stdout}"
            )
            observations = prepare_observations(env)
            replanned_steps = planner.plan(
                language_instruction, observations, "\n".join(history)
            )
            if not replanned_steps:
                break
            plan = plan[: i + 1] + replanned_steps
            i += 1
            continue

        prev_observations = observations
        observations = prepare_observations(env)

        result = planner.assess(step, prev_observations, observations)

        if result.success:
            print("[success]")
            history.append(f"{line} [success]")
            i += 1
            continue

        print(f"[failure] {result.reason}")
        history.append(f"{line} [failure] {result.reason}")

        new_plan = planner.plan(
            language_instruction, observations, "\n".join(history)
        )
        if not new_plan:
            break
        plan = plan.steps[: i + 1] + new_plan.steps

        i += 1

    return "\n".join(history)


def main(args):
    model_name = args.model.split('/')[-1]
    output_dir = Path(f"{args.scene_dir}/artifacts_model={model_name}/{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        args.env_id, 
        robot_uids=args.robot_uids, 
        config_dir_path = args.scene_dir,
        num_envs=args.num_envs, 
        render_mode="rgb_array", 
        control_mode='pd_joint_pos',
        enable_shadow=True,
        sim_config={'spacing': 20},
        obs_mode="rgb+segmentation",
    )

    if args.save_traj:
        env = RecordEpisode(
            env,
            output_dir=output_dir,
            save_video=args.save_video,
            video_fps=30,
            avoid_overwriting_video=True
        )

    env.reset(seed=args.seed, options={"reconfigure": True})

    planner = TaskPlanner(args.model)
    controller = Controller(env, debug=args.debug, vis=args.vis)

    history = execute_with_replanning(
        env=env,
        planner=planner,
        controller=controller,
    )

    if args.vis:
        viewer = env.render_human()
        while True:
            if viewer.closed:
                exit()
            if viewer.window.key_down("c"):
                break
            env.render_human()

    env.close()

    if args.history:
        history_file = output_dir / "history.txt"
        with open(history_file, "w") as f:
            f.write(history)

if __name__ == '__main__':
    main(parse_args())