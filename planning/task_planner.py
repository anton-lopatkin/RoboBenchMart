import base64
import json
import os
import re
from typing import Any, Dict, List, Optional
import time

from langchain_openrouter import ChatOpenRouter
from langchain.messages import AIMessage, HumanMessage, SystemMessage

from planning.prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
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
                self.conversation += messages + [answer]
                elapsed = time.time() - start
                print(f"[planner] thought for {elapsed:.1f}s")
                return plan
            except (ValueError, json.JSONDecodeError) as e:
                print(f"[planner] parse attempt {attempt + 1} failed: {e}")

        print("[planner] all attempts failed, returning None")
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
                self.conversation += messages + [answer]
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

    def save_conversation(self, output_dir: str):
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        saved = []
        img_counter = 0

        for msg in self.conversation:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "human"
            elif isinstance(msg, AIMessage):
                role = "ai"
            else:
                role = "unknown"

            saved.append({"role": role, "text": msg.text})

            content = msg.content

            if isinstance(msg, AIMessage):
                reasoning_content = msg.additional_kwargs.get("reasoning_content")
                if reasoning_content:
                    saved[-1]["reasoning"] = reasoning_content

            if isinstance(content, list):
                images = []
                for part in content:
                    if part.get("type") == "image":
                        img_data = base64.b64decode(part.get("base64"))
                        img_name = f"{img_counter}.png"
                        img_path = os.path.join(images_dir, img_name)
                        with open(img_path, "wb") as f:
                            f.write(img_data)
                        images.append(f"images/{img_name}")
                        img_counter += 1

                saved[-1]["images"] = images

        json_path = os.path.join(output_dir, "conversation.json")
        with open(json_path, "w") as f:
            json.dump({"messages": saved}, f, indent=2, ensure_ascii=False)

        txt_lines = []
        for i, msg in enumerate(saved):
            role = msg.get("role", "unknown")
            txt_lines.append(f"[{i}] {role.upper()}")
            if msg.get("reasoning"):
                txt_lines.append(f"Reasoning: {msg['reasoning']}")
            if role == "ai":
                txt_lines.append(f"Answer: {msg.get('text', '')}")
            else:
                txt_lines.append(msg.get("text", ""))
            if msg.get("images"):
                txt_lines.append(f"Images: {', '.join(msg['images'])}")
            txt_lines.append("")

        txt_path = os.path.join(output_dir, "conversation.txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(txt_lines))
