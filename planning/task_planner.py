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
    REFLECTOR_SYSTEM_PROMPT,
    REFLECTOR_USER_PROMPT,
)
from planning.utils import build_skills_description, get_function_description
from planning.controller import Controller


class TaskPlanner:
    def __init__(self, model: str):
        self.model = ChatOpenRouter(model=model)
        self.conversation = []
        self.planner_history = [self._build_planner_system_message()]
        self.reflector_history = []
        self._replanner_initialized = False

    def plan(
        self,
        instruction: str,
        obs: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        start = time.time()
        print("[planner] thinking...")

        human_msg = self._build_planner_human_message(instruction, obs)
        self.planner_history.append(human_msg)

        for attempt in range(2):
            answer = self.model.invoke(self.planner_history)
            try:
                plan = self._parse_plan(answer)
                self.planner_history.append(answer)
                elapsed = time.time() - start
                print(f"[planner] thought for {elapsed:.1f}s")
                self._print_usage("planner", answer)
                print(f"[planner] response:\n{answer.content}")
                return plan
            except (ValueError, json.JSONDecodeError) as e:
                print(f"[planner] parse attempt {attempt + 1} failed: {e}")

        print("[planner] all attempts failed, returning None")
        return None

    def reflect(
        self,
        instruction: str,
        obs: Dict[str, Any],
        last_step: str,
    ) -> str:
        start = time.time()
        print("[reflector] thinking...")

        if not self.reflector_history:
            skills_description = build_skills_description(Controller)
            system_prompt = (
                REFLECTOR_SYSTEM_PROMPT.format(skills_description=skills_description)
                + f"\n\nTask Instruction: {instruction}\nCurrent Trajectory below:"
            )
            self.reflector_history = [SystemMessage(system_prompt)]

        human_msg = self._build_reflector_human_message(obs, last_step)
        self.reflector_history.append(human_msg)

        answer = self.model.invoke(self.reflector_history)
        self.reflector_history.append(answer)

        elapsed = time.time() - start
        print(f"[reflector] thought for {elapsed:.1f}s")
        self._print_usage("reflector", answer)
        print(f"[reflector] response:\n{answer.content}")
        return answer.content

    def replan(
        self,
        instruction: str,
        obs: Dict[str, Any],
        history: str,
        reflection: str,
    ) -> Optional[List[Dict[str, Any]]]:
        start = time.time()
        print("[replanner] thinking...")

        if not self._replanner_initialized:
            self.planner_history.append(self._build_replanner_system_message())
            self._replanner_initialized = True
        human_msg = self._build_replanner_human_message(instruction, obs, history, reflection)
        self.planner_history.append(human_msg)

        for attempt in range(2):
            answer = self.model.invoke(self.planner_history)
            try:
                plan = self._parse_plan(answer)
                self.planner_history.append(answer)
                elapsed = time.time() - start
                print(f"[replanner] thought for {elapsed:.1f}s")
                self._print_usage("replanner", answer)
                print(f"[replanner] response:\n{answer.content}")
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
        return SystemMessage(REPLANNER_SYSTEM_PROMPT)

    def _build_reflector_human_message(
        self, obs: Dict[str, Any], last_step: str
    ) -> HumanMessage:
        user_prompt = REFLECTOR_USER_PROMPT.format(
            last_step=last_step,
            scene_description=obs["scene_description"],
        )
        return HumanMessage(
            content=[
                {"type": "text", "text": user_prompt},
                {"type": "image", "base64": obs["image"], "mime_type": "image/png"},
                {"type": "image", "base64": obs["annotated_image"], "mime_type": "image/png"},
            ]
        )

    def _build_replanner_human_message(
        self, instruction: str, obs: Dict[str, Any], history: str, reflection: str
    ) -> HumanMessage:
        user_prompt = REPLANNER_USER_PROMPT.format(
            task_instruction=instruction,
            scene_description=obs["scene_description"],
            history=history,
            reflection=reflection,
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

    def _print_usage(self, tag: str, answer):
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
        print(f"[{tag}] tokens: {in_str} {out_str} total={total}")

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
