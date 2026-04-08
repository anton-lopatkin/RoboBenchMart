from typing import Any, Dict, List, Optional

from openai import OpenAI
import re
import json

from planning.prompts import SYSTEM_PROMPT, USER_PROMPT
from planning.utils import build_skills_description
from planning.controller import Controller

class TaskPlanner:
    def __init__(self, model: str, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_prompt = self.build_system_prompt()
        self.messages = [{"role": "user", "content": self.system_prompt}]

    def build_system_prompt(self) -> str:
        skills_description = build_skills_description(Controller)
        system_prompt = SYSTEM_PROMPT.format(
            skills_description=skills_description
        )
        return system_prompt

    def build_user_prompt(
        self, language_instruction: str, observations: Dict[str, Any], history: str
    ) -> List[Dict[str, str]]:
        user_prompt = USER_PROMPT.format(
            task_description=language_instruction,
            scene_description=observations["scene_description"],
            history=history,
        )

        return [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": f"data:image/png;base64,{observations['image']}",
            },
            {
                "type": "image_url",
                "image_url": f"data:image/png;base64,{observations['annotated_image']}",
            },
        ]

    def plan(
        self,
        language_instruction: str,
        observations: Dict[str, Any],
        history: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        user_prompt = self.build_user_prompt(
            language_instruction, observations, history
        )
        self.messages.append(
            {"role": "user", "content": user_prompt}
        )
        retries = 0
        max_retries = 5
        while retries < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    extra_body={"reasoning": {"enabled": True}},
                )
                break
            except Exception as e:
                retries += 1
                print(f"openrouter error, {e}")
        else:
            print("max retries reached")
            return None

        response = response.choices[0].message

        self.messages.append(
            {
                "role": "assistant",
                "content": response.content,
                "reasoning_details": response.reasoning_details
            },
        )

        answer = response.content

        plan = re.search(r"(\[.*\])", answer, re.DOTALL).group(1)
        plan = json.loads(plan)

        return plan
