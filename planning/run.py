import argparse
import gymnasium as gym
from mani_skill.utils.wrappers import RecordEpisode
import time

import sys 
sys.path.append('.')
from dsynth.envs import *
from dsynth.robots import *

from planning.task_planner import TaskPlanner
from planning.controller import Controller
from planning.utils import prepare_observations
from planning.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL


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
    parser.add_argument('--video', action='store_true', default=False)

    args = parser.parse_args()
    return args

def replan():
    print("Enter new steps (one per line), empty line to finish.")
    print("Format: skill_name key=value key=value")

    new_steps = []

    while True:
        line = input(">>> ").strip()
        if line == "":
            break

        parts = line.split()
        skill_name = parts[0]

        params = {}
        for p in parts[1:]:
            k, v = p.split("=")
            params[k] = int(v) if v.isdigit() else float(v)

        new_steps.append({
            "name": skill_name,
            "params": params
        })

    return new_steps


def main(args):
    scene_dir = Path(args.scene_dir)
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

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

    model = "x-ai/grok-4.1-fast"

    new_traj_name = time.strftime("%Y%m%d_%H%M%S")
    video_path = scene_dir / f"./videos_seed={args.seed}_model={model.split('/')[1]}"
    env = RecordEpisode(
        env,
        output_dir=video_path,
        trajectory_name=new_traj_name,
        save_video=args.video,
        video_fps=30,
        avoid_overwriting_video=True
    )
    print("Video path:", video_path)
    print("Trajectoty name:", new_traj_name)

    obs, _ = env.reset(seed=args.seed, options={'reconfigure': True}) 

    planner = TaskPlanner(model, OPENROUTER_API_KEY, OPENROUTER_BASE_URL)
    controller = Controller(env, debug=args.debug, vis=args.vis)


    language_instruction = 'take one milk and one beer' # env.language_instructions[0]
    observations = prepare_observations(env, obs)
    plan = planner.plan(language_instruction, observations)
    print(plan)

    skills_map = {
        'move_base_forward': controller.move_base_forward,
        'rotate_base': controller.rotate_base,

        'drive_to_shelf': controller.drive_to_shelf,
        'drive_to_product': controller.drive_to_product,
        'align_to_product': controller.align_to_product,
        'move_base_towards_product': controller.move_base_towards_product,

        'lift_ee': controller.lift_ee,

        'move_ee_to_neutral_pose': controller.move_ee_to_neutral_pose,
        'move_ee_to_product_height': controller.move_ee_to_product_height,

        'grasp_product': controller.grasp_product,
        'drop_to_basket': controller.drop_to_basket,
    }

    history = ""

    if args.execute:
        i = 0
        while i < len(plan):
            step = plan[i]
            i += 1

            skill_name = step.get("name")
            skill_params = step.get("params", {})

            step_desription = f"{i}. {skill_name} {skill_params}"

            print(step_desription, end=' ')
            history += step_desription

            skill_fn = skills_map.get(skill_name)
            result = skill_fn(**skill_params)

            if result == -1:
                status = "[failure]"
                history += f" {status}\n"
                new_steps = planner.plan(language_instruction, observations, history)
                if not new_steps:
                    break
                plan = plan[:i] + new_steps
                continue
            else:
                status = "[success]"
                history += f" {status}\n"

            print(status)

    if args.vis:
        viewer = env.render_human()
        while True:
            if viewer.closed:
                exit()
            if viewer.window.key_down("c"):
                break
            env.render_human()

    env.close()

    history_file = video_path / f"{new_traj_name}_history.txt"
    with open(history_file, "w") as f:
        f.write(history)

    print("History saved to:", history_file)

if __name__ == '__main__':
    main(parse_args())