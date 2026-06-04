from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations


def cohen_kappa(y1: Sequence[str], y2: Sequence[str]) -> float:
    """Cohen's kappa for two raters over paired nominal labels.

    ``y1[i]`` and ``y2[i]`` are the two raters' labels for item ``i``.
    kappa = (p_o - p_e) / (1 - p_e). Returns 0.0 for empty input and 1.0 for the
    degenerate single-category case where expected agreement is 1.0.
    """
    n = len(y1)
    if n == 0:
        return 0.0

    categories = sorted(set(y1) | set(y2))
    index = {label: position for position, label in enumerate(categories)}
    size = len(categories)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for a, b in zip(y1, y2, strict=True):
        matrix[index[a]][index[b]] += 1

    p_o = sum(matrix[k][k] for k in range(size)) / n
    rows = [sum(matrix[r]) / n for r in range(size)]
    cols = [sum(matrix[r][c] for r in range(size)) / n for c in range(size)]
    p_e = sum(rows[k] * cols[k] for k in range(size))

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def fleiss_kappa(rating_counts: Sequence[Sequence[int]]) -> float:
    """Fleiss' kappa for a fixed number of raters per item.

    ``rating_counts`` is an ``N_items x N_categories`` matrix where entry
    ``[i][j]`` is the number of raters who assigned category ``j`` to item ``i``.
    Every row must sum to the same number of raters ``n``. Returns 0.0 if there
    are no items and 1.0 in the degenerate ``P_e == 1.0`` case.
    """
    n_items = len(rating_counts)
    if n_items == 0:
        return 0.0

    row_sums = {sum(row) for row in rating_counts}
    if len(row_sums) != 1:
        raise ValueError("every item must be rated by the same number of raters")
    n = row_sums.pop()
    if n < 2:
        raise ValueError("Fleiss' kappa requires at least 2 raters per item")

    n_categories = len(rating_counts[0])
    # Agreement per item: proportion of agreeing rater pairs (Fleiss 1971).
    p_i = [(sum(value * value for value in row) - n) / (n * (n - 1)) for row in rating_counts]
    p_bar = sum(p_i) / n_items
    # Category marginals across all ratings.
    p_j = [sum(rating_counts[i][j] for i in range(n_items)) / (n_items * n) for j in range(n_categories)]
    p_e = sum(value * value for value in p_j)

    if p_e == 1.0:
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)


def krippendorff_alpha_nominal(reliability_data: Sequence[Sequence[str | None]]) -> float:
    """Krippendorff's alpha for nominal data via the coincidence matrix.

    ``reliability_data`` is an ``N_raters x N_units`` matrix; entry ``[r][u]`` is
    the label rater ``r`` gave unit ``u`` (or ``None`` if missing). Units with
    fewer than two non-missing ratings are excluded. Returns 1.0 when there is no
    observed disagreement and 0.0 when there are no pairable values.
    """
    if not reliability_data:
        return 0.0

    n_units = len(reliability_data[0])
    columns: list[list[str]] = []
    for unit in range(n_units):
        values = [reliability_data[rater][unit] for rater in range(len(reliability_data))]
        present = [value for value in values if value is not None]
        if len(present) >= 2:
            columns.append(present)

    categories = sorted({value for column in columns for value in column})
    index = {label: position for position, label in enumerate(categories)}
    size = len(categories)
    # Coincidence matrix: every ordered within-unit pair contributes 1/(m_u - 1).
    coincidences = [[0.0 for _ in range(size)] for _ in range(size)]
    for column in columns:
        weight = 1.0 / (len(column) - 1)
        for position_a, value_a in enumerate(column):
            for position_b, value_b in enumerate(column):
                if position_a != position_b:
                    coincidences[index[value_a]][index[value_b]] += weight

    n_c = [sum(coincidences[c]) for c in range(size)]
    total = sum(n_c)
    if total == 0:
        return 0.0

    observed_agreement = sum(coincidences[k][k] for k in range(size))
    if observed_agreement == total:
        return 1.0

    # Nominal alpha = ((n - 1) sum_c o_cc - sum_c n_c (n_c - 1)) / (n (n - 1) - sum_c n_c (n_c - 1)).
    expected_pairs = sum(value * (value - 1) for value in n_c)
    numerator = (total - 1) * observed_agreement - expected_pairs
    denominator = total * (total - 1) - expected_pairs
    if denominator == 0:
        return 0.0
    return numerator / denominator


@dataclass(frozen=True)
class AgreementResult:
    """Inter-model agreement summary.

    ``observed_agreement`` is the mean pairwise exact-match rate across model pairs,
    not a single multi-rater percent-agreement statistic.
    """

    n_units: int
    n_units_all_raters: int
    n_raters: int
    raters: list[str]
    observed_agreement: float
    fleiss_kappa: float | None
    krippendorff_alpha: float | None
    pairwise_cohen_kappa: list[dict]


def compute_agreement(predictions: Mapping[str, Mapping[int, str]]) -> AgreementResult:
    """Compute inter-rater agreement statistics over nominal sentiment labels.

    ``predictions`` maps rater id (e.g. model id) to ``{item_id: label}``. The
    category set is derived internally from the observed labels.
    """
    raters = sorted(predictions)
    n_raters = len(raters)

    pairwise: list[dict] = []
    agreement_fractions: list[float] = []
    for rater_a, rater_b in combinations(raters, 2):
        labels_a = predictions[rater_a]
        labels_b = predictions[rater_b]
        shared = sorted(set(labels_a) & set(labels_b))
        if not shared:
            continue
        y_a = [labels_a[item] for item in shared]
        y_b = [labels_b[item] for item in shared]
        matches = sum(1 for a, b in zip(y_a, y_b, strict=True) if a == b)
        agreement_fractions.append(matches / len(shared))
        pairwise.append(
            {
                "rater_a": rater_a,
                "rater_b": rater_b,
                "kappa": cohen_kappa(y_a, y_b),
                "n": len(shared),
            }
        )

    observed_agreement = sum(agreement_fractions) / len(agreement_fractions) if agreement_fractions else 0.0

    all_items = sorted({item for labels in predictions.values() for item in labels})
    complete_items = [item for item in all_items if all(item in predictions[rater] for rater in raters)]

    fleiss_value: float | None = None
    if n_raters >= 2 and complete_items:
        categories = sorted({predictions[rater][item] for item in complete_items for rater in raters})
        index = {label: position for position, label in enumerate(categories)}
        counts = [[0 for _ in categories] for _ in complete_items]
        for row, item in enumerate(complete_items):
            for rater in raters:
                counts[row][index[predictions[rater][item]]] += 1
        fleiss_value = fleiss_kappa(counts)

    alpha_value: float | None = None
    usable_units = [item for item in all_items if sum(1 for rater in raters if item in predictions[rater]) >= 2]
    if n_raters >= 2 and usable_units:
        reliability_data: list[list[str | None]] = [
            [predictions[rater].get(item) for item in all_items] for rater in raters
        ]
        alpha_value = krippendorff_alpha_nominal(reliability_data)

    return AgreementResult(
        n_units=len(usable_units),
        n_units_all_raters=len(complete_items),
        n_raters=n_raters,
        raters=raters,
        observed_agreement=observed_agreement,
        fleiss_kappa=fleiss_value,
        krippendorff_alpha=alpha_value,
        pairwise_cohen_kappa=pairwise,
    )
