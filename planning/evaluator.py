from planning.task_planner import TaskPlanner
from planning.controller import Controller
from planning.utils import prepare_observations


class Evaluator:
    def __init__(self, debug=False, vis=False):
        self.debug = debug
        self.vis = vis

    def run_episode(self, model, env):
        task_planner = TaskPlanner(model)
        controller = Controller(env, debug=self.debug, vis=self.vis)

        language_instruction = 'take one milk and one beer' # env.language_instructions[0]
        observations = prepare_observations(env)

        plan = task_planner.plan(language_instruction, observations)

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
                replanned_steps = task_planner.plan(
                    language_instruction, observations, "\n".join(history)
                )
                if not replanned_steps:
                    break
                plan = plan[: i + 1] + replanned_steps
                i += 1
                continue

            prev_observations = observations
            observations = prepare_observations(env)

            result = task_planner.assess(step, prev_observations, observations)

            if result.success:
                print("[success]")
                history.append(f"{line} [success]")
                i += 1
                continue

            print(f"[failure] {result.reason}")
            history.append(f"{line} [failure] {result.reason}")

            new_plan = task_planner.plan(
                language_instruction, observations, "\n".join(history)
            )
            if not new_plan:
                break
            plan = plan.steps[: i + 1] + new_plan.steps

            i += 1

        return "\n".join(history)   