import base64
import json
import os
import re
from typing import Any, Dict, List, Optional
import time

from langchain_openrouter import ChatOpenRouter
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import trim_messages

from planning.controller import Controller
from planning.prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    REFLECTOR_USER_PROMPT,
)
from planning.utils import build_skills_description


class DarkstoreAgent:
    def __init__(
        self,
        model: str,
        controller: Controller,
        instruction: str,
        enable_reflection: bool = True,
        max_history_messages: int = 16,
    ):
        self.model = ChatOpenRouter(model=model)
        self.controller = controller
        self.instruction = instruction
        self.enable_reflection = enable_reflection
        self.max_history_messages = max_history_messages
        self.conversation = {
            "planner": [self._build_planner_system_message()],
            "reflector": [self._build_reflector_system_message()],
        }
        self.history = []

    def next_action(
        self,
        obs: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
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

        line = f"{len(self.history) + 1}. {skill}{f' {params}' if params else ''}"
        print(line)

        result = fn(**params)
        if result == -1:
            self.history.append(
                f"{line} [motion planning failed] \n{self.controller.last_stdout}"
            )
        else:
            self.history.append(f"{line} [motion planning succeed]")

        return answer
    
    def _call_agent(self, agent, msg):
        start = time.time()
        print(f"[{agent}] thinking...")

        self.conversation[agent].append(msg)
        trimmed = trim_messages(
            self.conversation[agent],
            max_tokens=self.max_history_messages,
            token_counter=len,
            strategy='last',
            include_system=True,
            start_on='human',
        )
        answer = self.model.invoke(trimmed)
        self.conversation[agent].append(answer)

        elapsed = time.time() - start
        print(f"[{agent}] thought for {elapsed:.1f}s")
        self._print_usage(agent, answer)
        print(f"[{agent}] response:\n{answer.content}")

        return answer
    
    def _parse_skill(self, answer):
        match = re.search(r"\{.*\}", answer.content, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON array found in response")
        skill = json.loads(match.group(0))
        return skill.get("name"), skill.get("params", {})

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
        obs: Dict[str, Any],
        reflection: str,
    ) -> HumanMessage:
        user_prompt = PLANNER_USER_PROMPT.format(
            scene_description=obs["scene_description"],
            reflection=reflection,
            history=self.history,
        )
        return HumanMessage(
            content=[
                {
                    "type": "image",
                    "base64": obs["image"],
                    "mime_type": "image/png",
                },
                {
                    "type": "image",
                    "base64": obs["annotated_image"],
                    "mime_type": "image/png",
                },
                {"type": "text", "text": user_prompt},
            ]
        )

    def _build_reflector_human_message(
        self, obs: Dict[str, Any], last_step: str
    ) -> HumanMessage:
        user_prompt = REFLECTOR_USER_PROMPT.format(
            last_step=last_step,
            scene_description=obs["scene_description"],
        )
        return HumanMessage(
            content=[
                {
                    "type": "image", 
                    "base64": obs["image"], 
                    "mime_type": "image/png"
                },
                {
                    "type": "image", 
                    "base64": obs["annotated_image"], 
                    "mime_type": "image/png"
                },
                {"type": "text", "text": user_prompt},
            ]
        )

    def _print_usage(self, agent: str, answer):
        usage = answer.usage_metadata
        if not usage:
            return
        in_tok = usage.get('input_tokens', '?')
        out_tok = usage.get('output_tokens', '?')
        total = usage.get('total_tokens', '?')
        cache_read = usage.get('input_token_details', {}).get('cache_read', 0)
        cache_write = usage.get('input_token_details', {}).get('cache_creation', 0)
        reasoning = usage.get('output_token_details', {}).get('reasoning', 0)

        in_str = f"in={in_tok}" + (f" (cache_read={cache_read})" if cache_read else "") + (f" (cache_write={cache_write})" if cache_write else "")
        out_str = f"out={out_tok}" + (f" (reasoning={reasoning})" if reasoning else "")
        print(f"[{agent}] tokens: {in_str} {out_str} total={total}")
