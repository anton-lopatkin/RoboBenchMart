import argparse
import time
from pathlib import Path

import gymnasium as gym
from mani_skill.utils.wrappers import RecordEpisode

import sys 
sys.path.append('.')
from dsynth.envs import *
from dsynth.robots import *

from planning.evaluator import Evaluator
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
    parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--save-traj", action="store_true", help="whether or not to save trajectory locally")
    parser.add_argument("--history", action="store_true", default=False)
    parser.add_argument("-m", "--model", type=str, default=False)

    args = parser.parse_args()
    return args


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

    evaluator = Evaluator(args.debug, args.vis)
    history = evaluator.run_episode(args.model, env)

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