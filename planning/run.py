import argparse
import gymnasium as gym

import sys 
sys.path.append('.')
from dsynth.envs import *
from dsynth.robots import *

from planning.task_planner import TaskPlanner
from planning.utils import prepare_observations
from planning.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", help="Path to dir with scene JSON-config")
    parser.add_argument("-e", "--env-id", type=str, default="CollectOrderItemsContEnv", help="Environment to run")
    parser.add_argument("-r", "--robot-uids", type=str, default="ds_fetch_basket", help="Robot id")
    parser.add_argument("-n", "--num-envs", type=int, default=1, help="Number of scenes")
    parser.add_argument("-s", "--seed", type=int, nargs='+', default=42)
    parser.add_argument('--gui', action='store_true', default=False)
    parser.add_argument("--execute", action='store_true', default=False)

    args = parser.parse_args()
    return args


def main(args):
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        args.env_id, 
        robot_uids=args.robot_uids, 
        config_dir_path = args.scene_dir,
        num_envs=args.num_envs, 
        render_mode="rgb_array", 
        control_mode=None,
        enable_shadow=True,
        sim_config={'spacing': 20},
        obs_mode="rgb+segmentation",
    )

    obs, _ = env.reset(seed=args.seed, options={'reconfigure': True})    

    planner = TaskPlanner("google/gemma-3-12b-it", OPENROUTER_API_KEY, OPENROUTER_BASE_URL)

    language_instruction = env.language_instructions[0]
    observations = prepare_observations(env, obs)
    plan = planner.plan(language_instruction, observations)
    print(plan)

    if args.execute:
        SKILLS = {
            "pick_item": env.pick_item,
            "place_to_basket": env.place_to_basket,
        }

        plan = re.search(r'(\[.*\])', plan, re.DOTALL).group(1)
        plan = json.loads(plan)
        for step in plan:
            if args.gui:
                env.render_human()

                import time
                time.sleep(2)

            skill_name = step.get("name")
            skill_params = step.get("params", {})
            skill = SKILLS.get(skill_name)
            if skill:
                skill(**skill_params)

        if args.gui:
            viewer = env.render_human()
            while True:
                if viewer.closed:
                    exit()
                if viewer.window.key_down("c"):
                    break
                env.render_human()


    env.close()


if __name__ == '__main__':
    main(parse_args())