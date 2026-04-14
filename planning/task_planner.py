from typing import Any, Dict, List, Optional
import time

from langchain_openrouter import ChatOpenRouter
from langchain.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from planning.prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    ASSESSOR_SYSTEM_PROMPT,
    ASSESSOR_USER_PROMPT,
)
from planning.utils import build_skills_description, get_function_description
from planning.controller import Controller


class Step(BaseModel):
    name: str = Field(description="Name of the skill to call")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the skill"
    )


class Plan(BaseModel):
    steps: List[Step] = Field(description="Sequence of skills to call")

    
class AssessmentResult(BaseModel):
    success: bool = Field(
        description="Whether the skill execution achieved its intended goal"
    )
    reason: str = Field(
        description="Brief explanation of why the step succeeded or failed"
    )


class TaskPlanner:
    def __init__(self, model: str):
        self.model = ChatOpenRouter(model=model, reasoning={"enable": False})
        self.planner = self.model.with_structured_output(Plan, method="json_schema", strict=True)
        self.assessor = self.model.with_structured_output(AssessmentResult, method="json_schema", strict=True)

    def plan(
        self,
        instruction: str,
        obs: Dict[str, Any],
        history: Optional[str] = None,
    ) -> Plan:
        start = time.time()
        print("[planner] thinking...")
        result = self.planner.invoke(
            [
                self._build_planner_system_message(),
                self._build_planner_human_message(instruction, obs, history or ""),
            ]
        )
        elapsed = time.time() - start
        print(f"[planner] thought for {elapsed:.1f}s")
        return result


    def assess(
        self,
        step: Dict[str, Any],
        before_obs: Dict[str, Any],
        after_obs: Dict[str, Any],
    ) -> AssessmentResult:
        start = time.time()
        print("[assessor] thinking...")
        result = self.assessor.invoke(
            [
                SystemMessage(ASSESSOR_SYSTEM_PROMPT),
                self._build_assessor_human_message(step, before_obs, after_obs),
            ]
        )
        elapsed = time.time() - start
        print(f"[assessor] thought for {elapsed:.1f}s")
        if result.success:
            print(f"[assessor] step succeed")
        else:
            print(f"[assessor] step failed (reason: {result.reason})")

        return result

    def _build_planner_system_message(self) -> SystemMessage:
        skills_description = build_skills_description(Controller)
        system_prompt = PLANNER_SYSTEM_PROMPT.format(
            skills_description=skills_description
        )
        return SystemMessage(system_prompt)

    def _build_planner_human_message(
        self, instruction: str, obs: Dict[str, Any], history: str
    ) -> HumanMessage:
        user_prompt = PLANNER_USER_PROMPT.format(
            task_description=instruction,
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

    def _build_assessor_human_message(
        self,
        step: Dict[str, Any],
        before_obs: Dict[str, Any],
        after_obs: Dict[str, Any],
    ) -> HumanMessage:
        skill_fn = getattr(Controller, step.name, None)
        skill_description = get_function_description(step.name, skill_fn)

        user_prompt = ASSESSOR_USER_PROMPT.format(
            skill_name=step.name,
            skill_params=str(step.params),
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
