from pathlib import Path

import cv2

from planning.task_planner import DarkstoreAgent
from planning.controller import Controller
from planning.utils import prepare_observations


CAMERAS = ["left_base_camera_link", "fetch_hand", "right_base_camera_link", "combined"]


class Evaluator:
    def __init__(self, output_dir, debug=False, vis=False):
        self.debug = debug
        self.vis = vis
        self.output_dir = Path(output_dir)

    def run_episode(self, model, env):
        instruction = "move beer 98 on a shelf with a milk"# "take one milk and one beer"  # env.language_instructions[0]

        controller = Controller(env, debug=self.debug, vis=self.vis)
        agent = DarkstoreAgent(model, controller, instruction, enable_reflection=False)

        i = 0
        while True:
            obs = prepare_observations(env)
            self._save_obs(obs, i)
            agent.last_grounder_image = None
            agent.next_action(obs)

            if agent.last_grounder_image is not None:
                step_dir = self.output_dir / f"step_{i:03d}"
                self._save_image(step_dir / "grounder_bbox.png", agent.last_grounder_image)
                
            i += 1

    def _save_obs(self, obs, step: int):
        step_dir = self.output_dir / f"step_{step:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for camera in CAMERAS:
            for kind in ["image", "annotated_image"]:
                self._save_image(step_dir / f"{camera}_{kind}.png", obs[camera][kind])

    def _save_image(self, path, image):
        _, encoded = cv2.imencode(".png", image)
        path.write_bytes(encoded.tobytes())
