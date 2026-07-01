"""Self-consistency analysis for LLM sentiment ambiguity measurement.

This module implements the core experimental apparatus for the dissertation:
measuring *within-model* self-consistency by sampling the same LLM multiple
times at temperature > 0 on the same rows. Per-row label entropy serves as
a proxy for sentiment ambiguity.

The central hypothesis tested here:
    Rows with conflicting human annotations exhibit higher LLM self-consistency
    entropy than non-conflicting rows, validating LLM disagreement as a lens on
    ground-truth uncertainty.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from statistics import median

from .constants import ALLOWED_LABELS

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def entropy_from_counts(counts: dict[str, int], *, normalize: bool = True) -> float:
    """Shannon entropy (in bits) of a label-count distribution.

    Parameters
    ----------
    counts:
        Mapping from label to observed count (e.g. {"positive": 3, "neutral": 1}).
    normalize:
        If True (default), entropy is normalised by log₂(K) where K is the
        *fixed* number of possible labels (len(ALLOWED_LABELS)), yielding a
        [0, 1] scale. Because K is constant, this is a linear rescale of the raw
        entropy: it preserves the ordering of rows by uncertainty, so a 3-way
        even split (max ambiguity) scores 1.0 while a 2-way even split scores
        ~0.63. Normalising by the number of *observed* categories instead would
        collapse both to 1.0, conflating uncertainty with label richness — the
        wrong construct for an ambiguity measure.

    Returns
    -------
    Entropy in bits (or its [0, 1] normalisation when ``normalize`` is True).
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0

    result = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / total
        result -= p * math.log2(p)

    if normalize:
        k = len(ALLOWED_LABELS)
        if k > 1:
            result /= math.log2(k)

    return result


def majority_fraction(counts: dict[str, int]) -> float:
    """Fraction of samples that landed on the majority label."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return max(counts.values()) / total


def majority_label(counts: dict[str, int]) -> str:
    """Label with the highest count (ties broken by lexical order)."""
    if not counts:
        return ""
    return max(sorted(counts), key=counts.__getitem__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowSelfConsistency:
    """Per-row self-consistency results from multiple LLM samples."""

    row_number: int
    sentence: str
    hidden_label: str
    is_conflicting_duplicate: bool
    n_samples: int
    label_counts: dict[str, int]
    majority_label: str
    majority_fraction: float
    entropy: float
    correct_majority: bool  # does the majority vote match the hidden label?
    n_valid: int  # how many samples produced parseable labels


@dataclass(frozen=True)
class SelfConsistencyResult:
    """Aggregate self-consistency metrics for one model at one temperature."""

    model_id: str
    temperature: float
    num_samples: int
    scope: str
    n_rows: int
    rows: list[RowSelfConsistency]

    # --- Aggregate ---
    mean_entropy: float
    median_entropy: float
    mean_majority_fraction: float
    consistency_rate: float  # fraction of rows with entropy == 0 (perfect consistency)
    majority_vote_accuracy: float  # accuracy of majority-vote label vs hidden label
    majority_vote_balanced_accuracy: float

    # --- Conflicting vs non-conflicting breakdown ---
    conflicting_entropy: float | None = None
    non_conflicting_entropy: float | None = None
    conflicting_rows: int = 0
    non_conflicting_rows: int = 0

    # --- Cost ---
    total_cost: float | None = None

    @property
    def entropy_gap(self) -> float | None:
        """Non-conflicting minus conflicting entropy.

        A positive gap means non-conflicting rows have higher entropy (unexpected).
        A negative gap means conflicting rows have higher entropy (supports the hypothesis).
        """
        if self.conflicting_entropy is not None and self.non_conflicting_entropy is not None:
            return self.non_conflicting_entropy - self.conflicting_entropy
        return None


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_row_consistency(
    samples: list[str],
    *,
    row_number: int,
    sentence: str,
    hidden_label: str,
    is_conflicting_duplicate: bool,
) -> RowSelfConsistency:
    """Compute self-consistency for a single row from its N sample labels.

    Samples with parse errors (labels outside ALLOWED_LABELS) are excluded.
    """
    valid = [s for s in samples if s in ALLOWED_LABELS]
    counts = dict(Counter(valid))
    maj_label = majority_label(counts)
    maj_frac = majority_fraction(counts) if counts else 0.0
    ent = entropy_from_counts(counts) if counts else 0.0

    return RowSelfConsistency(
        row_number=row_number,
        sentence=sentence,
        hidden_label=hidden_label,
        is_conflicting_duplicate=is_conflicting_duplicate,
        n_samples=len(samples),
        label_counts=counts,
        majority_label=maj_label,
        majority_fraction=maj_frac,
        entropy=ent,
        correct_majority=maj_label == hidden_label,
        n_valid=len(valid),
    )


def compute_self_consistency(
    model_id: str,
    temperature: float,
    num_samples: int,
    scope: str,
    rows: list[RowSelfConsistency],
    *,
    total_cost: float | None = None,
) -> SelfConsistencyResult:
    """Aggregate per-row self-consistency results into a summary."""
    if not rows:
        return SelfConsistencyResult(
            model_id=model_id,
            temperature=temperature,
            num_samples=num_samples,
            scope=scope,
            n_rows=0,
            rows=[],
            mean_entropy=0.0,
            median_entropy=0.0,
            mean_majority_fraction=0.0,
            consistency_rate=0.0,
            majority_vote_accuracy=0.0,
            majority_vote_balanced_accuracy=0.0,
            total_cost=total_cost,
        )

    entropies = [r.entropy for r in rows]
    maj_fracs = [r.majority_fraction for r in rows]
    zero_entropy = sum(1 for e in entropies if e == 0.0)
    correct_maj = sum(1 for r in rows if r.correct_majority)

    # Balanced accuracy for majority vote
    from sklearn.metrics import balanced_accuracy_score

    y_true = [r.hidden_label for r in rows]
    y_pred_maj = [r.majority_label for r in rows]
    bal_acc = (
        1.0
        if len(set(y_true) | set(y_pred_maj)) == 1
        else float(balanced_accuracy_score(y_true, y_pred_maj))
    )

    # Conflicting vs non-conflicting breakdown
    conflicting = [r for r in rows if r.is_conflicting_duplicate]
    non_conflicting = [r for r in rows if not r.is_conflicting_duplicate]

    conflicting_entropy = (
        sum(r.entropy for r in conflicting) / len(conflicting) if conflicting else None
    )
    non_conflicting_entropy = (
        sum(r.entropy for r in non_conflicting) / len(non_conflicting) if non_conflicting else None
    )

    return SelfConsistencyResult(
        model_id=model_id,
        temperature=temperature,
        num_samples=num_samples,
        scope=scope,
        n_rows=len(rows),
        rows=rows,
        mean_entropy=sum(entropies) / len(entropies),
        median_entropy=float(median(entropies)),
        mean_majority_fraction=sum(maj_fracs) / len(maj_fracs),
        consistency_rate=zero_entropy / len(rows),
        majority_vote_accuracy=correct_maj / len(rows),
        majority_vote_balanced_accuracy=bal_acc,
        conflicting_entropy=conflicting_entropy,
        non_conflicting_entropy=non_conflicting_entropy,
        conflicting_rows=len(conflicting),
        non_conflicting_rows=len(non_conflicting),
        total_cost=total_cost,
    )
