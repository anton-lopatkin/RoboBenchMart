import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym

sys.path.append(".")
from dsynth.envs import *
from dsynth.robots import *
from planning.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", help="Path to dir with scene JSON-config")
    parser.add_argument("-e", "--env-id", type=str, help="Environment to run")
    parser.add_argument(
        "-r", "--robot-uids", type=str, default="ds_fetch_basket", help="Robot id"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=13, help="start seed for scene randomization"
    )
    parser.add_argument(
        "--robot-seed", type=int, default=None, help="start seed for robot init pose"
    )
    parser.add_argument(
        "-n", "--num-episodes", type=int, default=1, help="number of episodes to run"
    )
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--enable-reflection", action="store_true", help="enable agent self-reflection"
    )
    parser.add_argument(
        "--save-images", action="store_true", help="save per-step camera images"
    )
    parser.add_argument(
        "--save-video", action="store_true", help="save video of the episode"
    )
    parser.add_argument(
        "--save-traj", action="store_true", help="save trajectory to h5"
    )
    args = parser.parse_args()
    return args


def main(args):
    model_name = args.model.split("/")[-1]
    output_dir = Path(
        f"{args.scene_dir}/artifacts_model={model_name}/{time.strftime('%Y%m%d_%H%M%S')}"
    )

    env = gym.make(
        args.env_id,
        robot_uids=args.robot_uids,
        config_dir_path=args.scene_dir,
        render_mode="rgb_array",
        control_mode="pd_joint_pos",
        enable_shadow=True,
        sim_config={"spacing": 20},
        obs_mode="rgb+segmentation",
    )

    evaluator = Evaluator(
        output_dir,
        debug=args.debug,
        vis=args.vis,
        enable_reflection=args.enable_reflection,
        save_images=args.save_images,
        save_traj=args.save_traj,
        save_video=args.save_video,
    )
    results = evaluator.run_episodes(
        args.model,
        env,
        n=args.num_episodes,
        start_seed=args.seed,
        robot_init_pose_start_seed=args.robot_seed,
    )

    print(f"success_rate={sum(results)}/{len(results)}")

    if args.vis:
        viewer = env.render_human()
        while True:
            if viewer.closed:
                exit()
            if viewer.window.key_down("c"):
                break
            env.render_human()

    env.close()


if __name__ == "__main__":
    main(parse_args())
