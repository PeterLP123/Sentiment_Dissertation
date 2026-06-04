from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

from .constants import DEFAULT_PROMPTS_PATH
from .models import BlindExample, PromptConfig


def prompt_hash(prompt_id: str, system_prompt: str, user_template: str, output_mode: str) -> str:
    payload = {
        "prompt_id": prompt_id,
        "system_prompt": system_prompt.strip(),
        "user_template": user_template.strip(),
        "output_mode": output_mode,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def make_prompt(prompt_id: str, system_prompt: str, user_template: str, output_mode: str) -> PromptConfig:
    if output_mode not in {"label_only", "explanation"}:
        raise ValueError("output_mode must be label_only or explanation")
    if "{sentence}" not in user_template:
        raise ValueError("user_template must include {sentence}")
    return PromptConfig(
        prompt_id=prompt_id,
        system_prompt=system_prompt.strip(),
        user_template=user_template.strip(),
        output_mode=output_mode,  # type: ignore[arg-type]
        prompt_hash=prompt_hash(prompt_id, system_prompt, user_template, output_mode),
    )


def load_prompts(path: str | Path = DEFAULT_PROMPTS_PATH) -> dict[str, PromptConfig]:
    prompt_path = Path(path)
    with prompt_path.open("rb") as file:
        data = tomllib.load(file)
    prompts: dict[str, PromptConfig] = {}
    for item in data.get("prompts", []):
        prompt = make_prompt(
            prompt_id=item["id"],
            system_prompt=item["system_prompt"],
            user_template=item["user_template"],
            output_mode=item["output_mode"],
        )
        prompts[prompt.prompt_id] = prompt
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_path}")
    return prompts


def render_messages(prompt: PromptConfig, example: BlindExample) -> list[dict[str, str]]:
    if not isinstance(example, BlindExample):
        raise TypeError("render_messages only accepts BlindExample to avoid leaking hidden labels")
    user_content = prompt.user_template.format(sentence=example.sentence)
    return [
        {"role": "system", "content": prompt.system_prompt},
        {"role": "user", "content": user_content},
    ]

