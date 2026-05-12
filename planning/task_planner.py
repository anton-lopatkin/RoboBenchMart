import json
import re
import time
from typing import Any

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.messages import trim_messages
from langchain_openrouter import ChatOpenRouter

from planning.config import OPENROUTER_API_KEY
from planning.controller import Controller
from planning.prompts import (
    GROUNDER_SYSTEM_PROMPT,
    GROUNDER_USER_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    REFLECTOR_USER_PROMPT,
)
from planning.utils import (
    build_skills_description,
    draw_normalized_bbox,
    image_to_base64,
)


class DarkstoreAgent:
    def __init__(
        self,
        model: str,
        controller: Controller,
        instruction: str,
        enable_reflection: bool = True,
        max_history_messages: int = 4,
    ):
        self.model = ChatOpenRouter(model=model, api_key=OPENROUTER_API_KEY)
        self.controller = controller
        self.instruction = instruction
        self.enable_reflection = enable_reflection
        self.max_history_messages = max_history_messages
        self.conversation = {
            "planner": [self._build_planner_system_message()],
            "reflector": [self._build_reflector_system_message()],
        }
        self.history = []
        self.last_grounder_image = None

    def next_action(
        self,
        obs: dict[str, Any],
    ) -> str | None:
        self.last_grounder_image = None

        reflection = None
        if self.enable_reflection and self.history:
            msg = self._build_reflector_human_message(obs, self.history[-1])
            reflection = self._call_agent("reflector", msg)

        msg = self._build_planner_human_message(obs, reflection)
        answer = self._call_agent("planner", msg)

        skill, params = self._parse_skill(answer)
        fn = getattr(self.controller, skill, None)

        if not callable(fn):
            raise KeyError(f"Unknown skill '{skill}'")

        print(f"\n{len(self.history) + 1}. {skill}{f' {params}' if params else ''}")

        if skill in ("done", "fail"):
            self.history.append(
                {"skill": skill, "params": params, "status": "terminal"}
            )
            return skill

        if skill == "place_to_shelf":
            bbox = self._call_grounder(obs, params["camera"], params["description"])
            fn = self.controller._place_to_shelf
            params = {"bbox": bbox, "camera": params["camera"]}

        result = fn(**params)
        step = {"skill": skill, "params": params, "motion-planning-status": "success"}
        if result == -1:
            step["motion-planning-status"] = "failed"
            step["stdout"] = self.controller.last_stdout
        self.history.append(step)

        return None

    def _call_grounder(
        self, obs: dict[str, Any], camera: str, description: str
    ) -> dict:
        start = time.time()
        print("[grounder] thinking...")

        messages = [
            SystemMessage(GROUNDER_SYSTEM_PROMPT),
            self._build_grounder_human_message(obs[camera]["image"], description),
        ]
        answer = self.model.invoke(messages)

        elapsed = time.time() - start
        print(f"[grounder] thought for {elapsed:.1f}s")
        self._print_usage("grounder", answer)
        print(f"[grounder] response:\n{answer.content}")

        match = re.search(r"<bbox>(.*?)</bbox>", answer.content, re.DOTALL)
        if not match:
            raise ValueError("No <bbox> tag found in grounder response")
        bbox = json.loads(match.group(1).strip())

        self.last_grounder_image = draw_normalized_bbox(obs[camera]["image"], bbox)
        return bbox

    def _call_agent(self, agent, msg):
        start = time.time()
        print(f"[{agent}] thinking...")

        self.conversation[agent].append(msg)
        trimmed = trim_messages(
            self.conversation[agent],
            max_tokens=self.max_history_messages,
            token_counter=len,
            strategy="last",
            include_system=True,
            start_on="human",
        )
        answer = self.model.invoke(trimmed)
        self.conversation[agent].append(answer)

        elapsed = time.time() - start
        print(f"[{agent}] thought for {elapsed:.1f}s")
        self._print_usage(agent, answer)
        print(f"[{agent}] response:\n{answer.content}")

        return answer.content

    def _parse_skill(self, answer):
        match = re.search(r"<skill>(.*?)</skill>", answer, re.DOTALL)
        if not match:
            raise ValueError("No <skill> tag found in response")
        skill = json.loads(match.group(1).strip())
        name = skill.get("name")
        if not name:
            raise ValueError(f"Skill JSON missing 'name' field: {skill}")
        return name, skill.get("params", {})

    def _build_planner_system_message(self) -> SystemMessage:
        skills = build_skills_description(Controller)
        system_prompt = PLANNER_SYSTEM_PROMPT.format(
            instruction=self.instruction,
            skills=skills,
        )
        return SystemMessage(system_prompt)

    def _build_reflector_system_message(self) -> SystemMessage:
        skills = build_skills_description(Controller)
        system_prompt = REFLECTOR_SYSTEM_PROMPT.format(
            instruction=self.instruction,
            skills=skills,
        )
        return SystemMessage(system_prompt)

    def _build_planner_human_message(
        self,
        obs: dict[str, Any],
        reflection: str | None,
    ) -> HumanMessage:
        reflection = (
            f"REFLECTION: You may use this reflection on the previous action and overall trajectory:\n{reflection}\n\n"
            if reflection
            else ""
        )
        user_prompt = PLANNER_USER_PROMPT.format(
            scene_description=obs["scene_description"],
            reflection_prefix=reflection,
            history=self.history,
        )
        print(f"[human] {user_prompt}")
        return HumanMessage(
            content=[
                {
                    "type": "image",
                    "base64": image_to_base64(obs["combined"]["image"]),
                    "mime_type": "image/png",
                },
                {
                    "type": "image",
                    "base64": image_to_base64(obs["combined"]["annotated_image"]),
                    "mime_type": "image/png",
                },
                {"type": "text", "text": user_prompt},
            ]
        )

    def _build_reflector_human_message(
        self, obs: dict[str, Any], last_step: str
    ) -> HumanMessage:
        user_prompt = REFLECTOR_USER_PROMPT.format(
            last_step=last_step,
            scene_description=obs["scene_description"],
        )
        return HumanMessage(
            content=[
                {
                    "type": "image",
                    "base64": image_to_base64(obs["combined"]["image"]),
                    "mime_type": "image/png",
                },
                {
                    "type": "image",
                    "base64": image_to_base64(obs["combined"]["annotated_image"]),
                    "mime_type": "image/png",
                },
                {"type": "text", "text": user_prompt},
            ]
        )

    def _build_grounder_human_message(self, image, description) -> HumanMessage:
        return HumanMessage(
            content=[
                {
                    "type": "image",
                    "base64": image_to_base64(image),
                    "mime_type": "image/png",
                },
                {
                    "type": "text",
                    "text": GROUNDER_USER_PROMPT.format(description=description),
                },
            ]
        )

    def _print_usage(self, agent: str, answer):
        usage = answer.usage_metadata
        if not usage:
            return
        in_tok = usage.get("input_tokens", "?")
        out_tok = usage.get("output_tokens", "?")
        total = usage.get("total_tokens", "?")
        cache_read = usage.get("input_token_details", {}).get("cache_read", 0)
        cache_write = usage.get("input_token_details", {}).get("cache_creation", 0)
        reasoning = usage.get("output_token_details", {}).get("reasoning", 0)

        in_str = (
            f"in={in_tok}"
            + (f" (cache_read={cache_read})" if cache_read else "")
            + (f" (cache_write={cache_write})" if cache_write else "")
        )
        out_str = f"out={out_tok}" + (f" (reasoning={reasoning})" if reasoning else "")
        print(f"[{agent}] tokens: {in_str} {out_str} total={total}")
