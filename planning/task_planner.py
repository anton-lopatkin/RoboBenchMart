import json
import re
from typing import Any, Dict, List, Optional
import time

from langchain_openrouter import ChatOpenRouter
from langchain.messages import SystemMessage, HumanMessage

from planning.prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    ASSESSOR_SYSTEM_PROMPT,
    ASSESSOR_USER_PROMPT,
    REPLANNER_SYSTEM_PROMPT,
    REPLANNER_USER_PROMPT,
)
from planning.utils import build_skills_description, get_function_description
from planning.controller import Controller


class TaskPlanner:
    def __init__(self, model: str):
        self.model = ChatOpenRouter(model=model)
        self.conversation = []

    def plan(
        self,
        instruction: str,
        obs: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        start = time.time()
        print("[planner] thinking...")

        messages = [
            self._build_planner_system_message(),
            self._build_planner_human_message(instruction, obs),
        ]

        for attempt in range(2):
            answer = self.model.invoke(messages)
            try:
                plan = self._parse_plan(answer)
                self.conversation.append([messages, answer])
                elapsed = time.time() - start
                print(f"[planner] thought for {elapsed:.1f}s")
                return plan
            except (ValueError, json.JSONDecodeError) as e:
                print(f"[planner] parse attempt {attempt + 1} failed: {e}")

        print("[planner] all attempts failed, returning None")
        return None

    def assess(
        self,
        step: Dict[str, Any],
        before_obs: Dict[str, Any],
        after_obs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        start = time.time()
        print("[assessor] thinking...")

        messages = [
            SystemMessage(ASSESSOR_SYSTEM_PROMPT),
            self._build_assessor_human_message(step, before_obs, after_obs),
        ]

        for attempt in range(2):
            answer = self.model.invoke(messages)
            try:
                result = self._parse_assessment_result(answer)
                self.conversation.append([messages, answer])
                elapsed = time.time() - start
                print(f"[assessor] thought for {elapsed:.1f}s")
                if result["success"]:
                    print(f"[assessor] step succeed")
                else:
                    print(f"[assessor] step failed (reason: {result.get('reason')})")
                return result
            except (ValueError, json.JSONDecodeError) as e:
                print(f"[assessor] parse attempt {attempt + 1} failed: {e}")

        print("[assessor] all attempts failed, returning None")
        return None

    def replan(
        self,
        instruction: str,
        obs: Dict[str, Any],
        history: str,
    ) -> Optional[List[Dict[str, Any]]]:
        start = time.time()
        print("[replanner] thinking...")

        messages = [
            self._build_replanner_system_message(),
            self._build_replanner_human_message(instruction, obs, history),
        ]

        for attempt in range(2):
            answer = self.model.invoke(messages)
            try:
                plan = self._parse_plan(answer)
                self.conversation.append([messages, answer])
                elapsed = time.time() - start
                print(f"[replanner] thought for {elapsed:.1f}s")
                return plan
            except (ValueError, json.JSONDecodeError) as e:
                print(f"[replanner] parse attempt {attempt + 1} failed: {e}")

        print("[replanner] all attempts failed, returning None")
        return None

    def _build_planner_system_message(self) -> SystemMessage:
        skills_description = build_skills_description(Controller)
        system_prompt = PLANNER_SYSTEM_PROMPT.format(
            skills_description=skills_description
        )
        return SystemMessage(system_prompt)

    def _build_planner_human_message(
        self, instruction: str, obs: Dict[str, Any]
    ) -> HumanMessage:
        user_prompt = PLANNER_USER_PROMPT.format(
            task_instruction=instruction,
            scene_description=obs["scene_description"],
        )
        return HumanMessage(
            content=[
                {"type": "text", "text": user_prompt},
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
            ]
        )

    def _build_assessor_human_message(
        self,
        step: Dict[str, Any],
        before_obs: Dict[str, Any],
        after_obs: Dict[str, Any],
    ) -> HumanMessage:
        name = step["name"]
        params = step.get("params") or {}
        skill_fn = getattr(Controller, name, None)
        skill_description = get_function_description(name, skill_fn)
        user_prompt = ASSESSOR_USER_PROMPT.format(
            skill_name=name,
            skill_params=str(params),
            skill_description=skill_description,
            scene_before=before_obs["scene_description"],
            scene_after=after_obs["scene_description"],
        )
        return HumanMessage(
            content=[
                {"type": "text", "text": "BEFORE execution:"},
                {
                    "type": "image",
                    "base64": before_obs["image"],
                    "mime_type": "image/png",
                },
                {
                    "type": "image",
                    "base64": before_obs["annotated_image"],
                    "mime_type": "image/png",
                },
                {"type": "text", "text": "AFTER execution:"},
                {
                    "type": "image",
                    "base64": after_obs["image"],
                    "mime_type": "image/png",
                },
                {
                    "type": "image",
                    "base64": after_obs["annotated_image"],
                    "mime_type": "image/png",
                },
                {"type": "text", "text": user_prompt},
            ]
        )

    def _build_replanner_system_message(self) -> SystemMessage:
        skills_description = build_skills_description(Controller)
        system_prompt = REPLANNER_SYSTEM_PROMPT.format(
            skills_description=skills_description
        )
        return SystemMessage(system_prompt)

    def _build_replanner_human_message(
        self, instruction: str, obs: Dict[str, Any], history: str
    ) -> HumanMessage:
        user_prompt = REPLANNER_USER_PROMPT.format(
            task_instruction=instruction,
            scene_description=obs["scene_description"],
            history=history,
        )
        return HumanMessage(
            content=[
                {"type": "text", "text": user_prompt},
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
            ]
        )

    def _parse_plan(self, answer):
        match = re.search(r"\[.*\]", answer.content, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON array found in response")
        plan = json.loads(match.group())
        return plan

    def _parse_assessment_result(self, answer):
        match = re.search(r"\{.*\}", answer.content, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in response")
        result = json.loads(match.group())
        return result
