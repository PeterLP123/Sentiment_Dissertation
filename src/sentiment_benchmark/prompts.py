from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Sequence
from pathlib import Path

from .constants import ALLOWED_LABELS, DEFAULT_PROMPTS_PATH
from .models import BlindExample, PromptConfig

OUTPUT_MODES = ("label_only", "explanation", "cot")


def prompt_hash(
    prompt_id: str,
    system_prompt: str,
    user_template: str,
    output_mode: str,
    demonstrations: Sequence[tuple[str, str]] = (),
    few_shot_seed: int | None = None,
) -> str:
    payload = {
        "prompt_id": prompt_id,
        "system_prompt": system_prompt.strip(),
        "user_template": user_template.strip(),
        "output_mode": output_mode,
        "demonstrations": [[sentence, label] for sentence, label in demonstrations],
        "few_shot_seed": few_shot_seed,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def make_prompt(
    prompt_id: str,
    system_prompt: str,
    user_template: str,
    output_mode: str,
    demonstrations: Sequence[tuple[str, str]] = (),
    few_shot_seed: int | None = None,
) -> PromptConfig:
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"output_mode must be one of {', '.join(OUTPUT_MODES)}")
    if "{sentence}" not in user_template:
        raise ValueError("user_template must include {sentence}")
    demos = tuple((sentence.strip(), label.strip().lower()) for sentence, label in demonstrations)
    for _, label in demos:
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Demonstration label {label!r} must be one of {', '.join(ALLOWED_LABELS)}")
    return PromptConfig(
        prompt_id=prompt_id,
        system_prompt=system_prompt.strip(),
        user_template=user_template.strip(),
        output_mode=output_mode,  # type: ignore[arg-type]
        prompt_hash=prompt_hash(prompt_id, system_prompt, user_template, output_mode, demos, few_shot_seed),
        demonstrations=demos,
        few_shot_seed=few_shot_seed,
    )


def with_demonstrations(
    base: PromptConfig,
    demonstrations: Sequence[tuple[str, str]],
    few_shot_seed: int | None = None,
) -> PromptConfig:
    """Return a copy of ``base`` carrying few-shot demonstrations.

    The demonstration content changes the prompt hash, so a k-shot prompt is a
    distinct, reproducible prompt that can be compared against its zero-shot
    sibling with the existing ``compare`` tooling.
    """
    return make_prompt(
        prompt_id=base.prompt_id,
        system_prompt=base.system_prompt,
        user_template=base.user_template,
        output_mode=base.output_mode,
        demonstrations=demonstrations,
        few_shot_seed=few_shot_seed,
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
    messages = [{"role": "system", "content": prompt.system_prompt}]
    # Few-shot demonstrations are prior, already-answered turns. Their labels are
    # training signal, not the hidden label of the example under test.
    for demo_sentence, demo_label in prompt.demonstrations:
        messages.append({"role": "user", "content": prompt.user_template.format(sentence=demo_sentence)})
        messages.append({"role": "assistant", "content": demo_label})
    messages.append({"role": "user", "content": prompt.user_template.format(sentence=example.sentence)})
    return messages

