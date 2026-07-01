import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from sentiment_benchmark.l3_reliability import aggregate_daily_reliability, fit_crossed_gstudy, item_reliability


def _measurements() -> pd.DataFrame:
    generator = np.random.default_rng(42)
    rows = []
    item_effects = generator.normal(0, 0.5, 12)
    model_effects = {"m1": -0.2, "m2": 0.2}
    prompt_effects = {"p1": -0.1, "p2": 0.1}
    for item_index, item_effect in enumerate(item_effects):
        for model, model_effect in model_effects.items():
            for prompt, prompt_effect in prompt_effects.items():
                for sample in range(3):
                    rows.append({
                        "item_id": f"i{item_index}",
                        "model_id": model,
                        "prompt_id": prompt,
                        "sample_index": sample,
                        "score": item_effect + model_effect + prompt_effect + generator.normal(0, 0.05),
                        "symbol": "AAA",
                        "news_date": f"2026-01-{1 + item_index % 3:02d}",
                        "chronological_split": "development",
                    })
    return pd.DataFrame(rows)


def test_gstudy_and_reliability_are_bounded() -> None:
    measurements = _measurements()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = fit_crossed_gstudy(measurements, n_models=2, n_prompts=2, n_samples=3)
    assert result.converged is True
    assert result.components["item"] > 0
    assert result.variance_shares["item"] == max(result.variance_shares.values())
    assert 0 <= result.g_coefficient <= 1
    assert 0 <= result.dependability_coefficient <= 1

    items = item_reliability(measurements, result.components["item"], "p1")
    daily = aggregate_daily_reliability(items, result.components["item"])
    assert items["reliability"].between(0, 1).all()
    assert daily["reliability"].between(0, 1).all()
