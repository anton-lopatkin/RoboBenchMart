from typing import Any, Dict, List, Optional

from langchain_openrouter import ChatOpenRouter
from langchain.messages import SystemMessage, HumanMessage
import re
import json

from planning.prompts import SYSTEM_PROMPT, USER_PROMPT
from planning.utils import build_skills_description
from planning.controller import Controller


class TaskPlanner:
    def __init__(self, model: str):
        self.model = ChatOpenRouter(model=model)
        self.system_prompt = self.build_system_message()
        self.messages = [self.system_prompt]

    def build_system_message(self) -> SystemMessage:
        skills_description = build_skills_description(Controller)
        system_prompt = SYSTEM_PROMPT.format(
            skills_description=skills_description
        )
        return SystemMessage(system_prompt)

    def build_human_message(
        self, language_instruction: str, observations: Dict[str, Any], history: str
    ) -> HumanMessage:
        user_prompt = USER_PROMPT.format(
            task_description=language_instruction,
            scene_description=observations["scene_description"],
            history=history,
        )

        return HumanMessage(
            content=[
                {"type": "text", "text": user_prompt},
                {
                    "type": "image",
                    "base64": observations["image"],
                    "mime_type": "image/png",
                },
                {
                    "type": "image",
                    "base64": observations["annotated_image"],
                    "mime_type": "image/png",
                },
            ]
        )

    def plan(
        self,
        language_instruction: str,
        observations: Dict[str, Any],
        history: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        user_prompt = self.build_human_message(
            language_instruction, observations, history
        )
        self.messages.append(user_prompt)
        response = self.model.invoke(self.messages)
        self.messages.append(response)
        
        answer = response.content

        plan = re.search(r"(\[.*\])", answer, re.DOTALL).group(1)
        plan = json.loads(plan)

        return plan
