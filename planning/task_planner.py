from typing import Any, Dict, List, Optional

from openai import OpenAI

from planning.prompts import SYSTEM_PROMPT, USER_PROMPT


class TaskPlanner():
    def __init__(self, model: str, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def build_user_prompt(self, language_instruction, observations: Dict[str, Any]) -> List[Dict[str, str]]:
        user_prompt = USER_PROMPT.format(
            task_description=language_instruction,
            scene_description=observations['scene_objects'],
        )

        print(user_prompt)
        
        return [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{observations['image']}"},
            {"type": "image_url", "image_url": f"data:image/png;base64,{observations['annotated_image']}"},
        ]

    def plan(self, language_instruction: str, observations: Dict[str, Any]) -> Optional[str]:
        user_prompt = self.build_user_prompt(language_instruction, observations)
        retries = 0
        max_retries = 5
        while retries < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    extra_body={"reasoning": {"enabled": True}}
                )
                break
            except Exception as e:
                retries += 1
                print(f"openrouter error, {e}")
        else:
            print("max retries reached")
            return None

        return response.choices[0].message.content
