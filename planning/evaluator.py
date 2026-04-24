from planning.task_planner import DarkstoreAgent
from planning.controller import Controller
from planning.utils import prepare_observations


class Evaluator:
    def __init__(self, output_dir, save_conv=False, debug=False, vis=False):
        self.debug = debug
        self.vis = vis
        self.output_dir = output_dir
        self.save_conv = save_conv

    def run_episode(self, model, env):
        instruction = "take one milk and one beer"  # env.language_instructions[0]

        controller = Controller(env, debug=self.debug, vis=self.vis)
        agent = DarkstoreAgent(model, controller, instruction)

        i = 0
        while True:
            obs = prepare_observations(env)
            agent.next_action(obs)
            i += 1
