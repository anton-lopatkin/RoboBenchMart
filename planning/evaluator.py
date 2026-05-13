from datetime import datetime
from pathlib import Path

import cv2
from mani_skill.utils.wrappers import RecordEpisode

from planning.controller import Controller
from planning.task_planner import DarkstoreAgent
from planning.utils import prepare_observations

CAMERAS = ["left_base_camera_link", "fetch_hand", "right_base_camera_link", "combined"]


class Evaluator:
    MAX_STEPS = 20

    def __init__(
        self,
        output_dir: str,
        debug: bool = False,
        vis: bool = False,
        enable_reflection: bool = False,
        save_images: bool = False,
        save_traj: bool = False,
        save_video: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.debug = debug
        self.vis = vis
        self.enable_reflection = enable_reflection
        self.save_images = save_images
        self.save_traj = save_traj
        self.save_video = save_video

    def run_episodes(
        self,
        model: str,
        env,
        n: int,
        start_seed: int = None,
        robot_init_pose_start_seed: int = None,
    ) -> list[dict]:
        if self.save_traj or self.save_video:
            env = RecordEpisode(
                env,
                output_dir=str(self.output_dir),
                save_trajectory=self.save_traj,
                save_video=self.save_video,
                video_fps=30,
                avoid_overwriting_video=True,
                save_on_reset=False,
            )

        results = []
        for episode in range(n):
            print(f"\n[evaluator] running episode {episode}/{n}\n")

            seed = None if start_seed is None else start_seed + episode

            reset_options = {"reconfigure": True}
            if robot_init_pose_start_seed is not None:
                reset_options["robot_init_pose_seed"] = (
                    robot_init_pose_start_seed + episode
                )

            try:
                started_at = datetime.now().isoformat(timespec="seconds")
                env.reset(seed=seed, options=reset_options)

                instruction = env.unwrapped.language_instructions[0]
                controller = Controller(env, debug=self.debug, vis=self.vis)
                agent = DarkstoreAgent(
                    model,
                    controller,
                    instruction,
                    enable_reflection=self.enable_reflection,
                )

                for step in range(self.MAX_STEPS):
                    obs = prepare_observations(env)

                    if self.save_images:
                        self._save_obs(obs, episode, step)

                    status = agent.next_action(obs)

                    if self.save_images and agent.last_grounder_image is not None:
                        self._save_image(
                            self._step_dir(episode, step) / "grounder_bbox.png",
                            agent.last_grounder_image,
                        )

                    if status in ("done", "fail"):
                        break

                results.append(
                    {
                        "timestamp": started_at,
                        "seed": seed, 
                        "success": bool(env.unwrapped.evaluate()["success"]), 
                        "crash": None,
                    }
                )

                if self.save_traj:
                    env.flush_trajectory()
                if self.save_video:
                    suffix = str(seed) if seed is not None else str(episode)
                    env.flush_video(suffix)

            except Exception as e:
                print(f"[evaluator] episode {episode} crashed: {e}")
                results.append(
                    {
                        "timestamp": started_at,
                        "seed": seed, 
                        "success": bool(env.unwrapped.evaluate()["success"]), 
                        "crash": str(e),
                    }
                )
                
        return results

    def _step_dir(self, episode: int, step: int) -> Path:
        return self.output_dir / f"episode_{episode:03d}" / f"step_{step:03d}"

    def _save_obs(self, obs, episode: int, step: int):
        step_dir = self._step_dir(episode, step)
        step_dir.mkdir(parents=True, exist_ok=True)
        for camera in CAMERAS:
            for kind in ["image", "annotated_image"]:
                self._save_image(step_dir / f"{camera}_{kind}.png", obs[camera][kind])

    def _save_image(self, path: Path, image):
        path.parent.mkdir(parents=True, exist_ok=True)
        _, encoded = cv2.imencode(".png", image)
        path.write_bytes(encoded.tobytes())
