from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import permutations

from .constants import ALLOWED_LABELS
from .models import PromptConfig
from .prompts import make_prompt

_LABEL_PREFIX = {"positive": "pos", "negative": "neg", "neutral": "neu"}

# Base instruction whose label list is reordered to probe label-position sensitivity.
_ORDER_TEMPLATE = "Classify the sentiment of the sentence. Respond with exactly one lowercase label: {labels}. Output only the label."

# Distinct, natural-sounding system-prompt paraphrasings. Each names all three
# canonical labels and requests a single lowercase label, keeping the output
# vocabulary compatible with the downstream parser.
_DEFAULT_PHRASINGS: tuple[str, ...] = (
    "Classify the sentiment. Reply with one lowercase word: positive, negative, or neutral.",
    "You are a financial analyst. Read the sentence and decide whether its sentiment is positive, negative, or neutral. "
    "Answer with a single lowercase label and nothing else.",
    "Your task is sentiment analysis. Determine the overall sentiment of the given sentence and output exactly one of the "
    "following lowercase labels: positive, negative, neutral.",
    "positive, negative, or neutral? Answer with one lowercase label only.",
)


def _order_suffix(order: Sequence[str]) -> str:
    return "-".join(_LABEL_PREFIX[label] for label in order)


def label_order_variants(base: PromptConfig, *, orders: Sequence[Sequence[str]] | None = None) -> list[PromptConfig]:
    """One variant per ordering of the three labels in the instruction text.

    Default: all 6 permutations of ALLOWED_LABELS. Each variant keeps base.user_template
    and base.output_mode; prompt_id = f'{base.prompt_id}__order-{a}-{b}-{c}' using 3-letter
    prefixes (pos/neg/neu).
    """
    if orders is None:
        orders = [list(order) for order in permutations(ALLOWED_LABELS)]
    variants: list[PromptConfig] = []
    for order in orders:
        system_prompt = _ORDER_TEMPLATE.format(labels=", ".join(order))
        variants.append(
            make_prompt(
                prompt_id=f"{base.prompt_id}__order-{_order_suffix(order)}",
                system_prompt=system_prompt,
                user_template=base.user_template,
                output_mode=base.output_mode,
            )
        )
    return variants


def paraphrase_variants(base: PromptConfig, *, phrasings: Sequence[str] | None = None) -> list[PromptConfig]:
    """One variant per instruction paraphrasing.

    Provides a built-in default list of >=4 distinct system-prompt phrasings. Each names all
    three canonical labels and requests a single lowercase label, consistent with base.output_mode.
    prompt_id = f'{base.prompt_id}__paraphrase-{i}'.
    """
    if phrasings is None:
        phrasings = _DEFAULT_PHRASINGS
    variants: list[PromptConfig] = []
    for index, phrasing in enumerate(phrasings):
        variants.append(
            make_prompt(
                prompt_id=f"{base.prompt_id}__paraphrase-{index}",
                system_prompt=phrasing,
                user_template=base.user_template,
                output_mode=base.output_mode,
            )
        )
    return variants


def generate_prompt_suite(
    base: PromptConfig,
    *,
    include: Sequence[str] = ("label_order", "paraphrase"),
) -> list[PromptConfig]:
    """Concatenate the requested variant families, de-duplicated by prompt_hash.

    'include' may contain 'label_order' and/or 'paraphrase'. Order is preserved and duplicates
    (by prompt_hash) are dropped at their later encounter. Raises ValueError on unknown family names.
    """
    builders: dict[str, Callable[[PromptConfig], list[PromptConfig]]] = {
        "label_order": label_order_variants,
        "paraphrase": paraphrase_variants,
    }
    unknown = [name for name in include if name not in builders]
    if unknown:
        raise ValueError(f"Unknown family name(s): {', '.join(unknown)}")

    suite: list[PromptConfig] = []
    seen: set[str] = set()
    for name in include:
        for variant in builders[name](base):
            if variant.prompt_hash in seen:
                continue
            seen.add(variant.prompt_hash)
            suite.append(variant)
    return suite
