from planning.task_planner import TaskPlanner
from planning.controller import Controller
from planning.utils import prepare_observations


class Evaluator:
    def __init__(self, output_dir, save_conv=False, debug=False, vis=False):
        self.debug = debug
        self.vis = vis
        self.output_dir = output_dir
        self.save_conv = save_conv

    def run_episode(self, model, env):
        task_planner = TaskPlanner(model)
        controller = Controller(env, debug=self.debug, vis=self.vis)

        language_instruction = "take one milk and one beer"  # env.language_instructions[0]
        observations = prepare_observations(env)

        plan = task_planner.plan(language_instruction, observations)
        if self.save_conv:
            task_planner.save_conversation(self.output_dir)

        history = []
        i = 0

        while i < len(plan):
            step = plan[i]
            name = step["name"]
            params = step.get("params") or {}
            fn = getattr(controller, name, None)

            if not callable(fn):
                raise KeyError(f"Unknown skill '{name}' in plan step {i + 1}")

            line = f"{i + 1}. {name}{f' {params}' if params else ''}"
            print(line)

            result = fn(**params)
            if result == -1:
                history.append(
                    f"{line} [motion planning failed] \n{controller.last_stdout}"
                )
            else:
                history.append(f"{line} [motion planning succeed]")

            observations = prepare_observations(env)
            replanned_steps = task_planner.replan(
                language_instruction, observations, "\n".join(history)
            )
            if self.save_conv:
                task_planner.save_conversation(self.output_dir)
            if not replanned_steps:
                break
            plan = plan[: i + 1] + replanned_steps
            i += 1
    
        return "\n".join(history)
