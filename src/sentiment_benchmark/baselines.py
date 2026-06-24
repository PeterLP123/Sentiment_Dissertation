"""Non-LLM sentiment baselines for contextualising benchmark results.

Two families are provided:

* Fitted baselines (``majority``, ``tfidf_logreg``) are trained and evaluated
  with stratified k-fold cross-validation. Every row receives an out-of-fold
  prediction, so no row is ever scored by a model that was trained on it. This
  keeps them directly comparable to the zero-shot LLMs, which never see any of
  the rows during "training".
* Pretrained baselines (``vader``, ``finbert``) require no fitting and predict
  on every selected row directly. Their optional dependencies are imported
  lazily so the core package stays lightweight.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from .constants import ALLOWED_LABELS, DEFAULT_SEED
from .models import DatasetRow

# VADER's conventional decision threshold on the normalised compound score.
VADER_THRESHOLD = 0.05


@dataclass(frozen=True)
class VaderSentiment:
    label: str
    compound: float


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    kind: str  # "fitted" or "pretrained"
    requires: tuple[str, ...]  # optional import dependencies
    description: str


BASELINE_SPECS: dict[str, BaselineSpec] = {
    "majority": BaselineSpec(
        name="majority",
        kind="fitted",
        requires=(),
        description="Always predicts the most frequent training label (out-of-fold).",
    ),
    "tfidf_logreg": BaselineSpec(
        name="tfidf_logreg",
        kind="fitted",
        requires=(),
        description="TF-IDF features with multinomial logistic regression (out-of-fold).",
    ),
    "vader": BaselineSpec(
        name="vader",
        kind="pretrained",
        requires=("nltk",),
        description="NLTK VADER lexicon scores mapped to labels via the compound threshold.",
    ),
    "finbert": BaselineSpec(
        name="finbert",
        kind="pretrained",
        requires=("transformers", "torch"),
        description="ProsusAI/finbert transformer fine-tuned on financial sentiment.",
    ),
}

# Baselines that have no extra dependencies and are safe to run by default.
DEFAULT_BASELINES: tuple[str, ...] = ("majority", "tfidf_logreg")


FitPredict = Callable[[list[DatasetRow], list[DatasetRow]], list[str]]


def _stratified_oof(
    rows: list[DatasetRow],
    fit_predict: FitPredict,
    folds: int,
    seed: int,
) -> list[str | None]:
    """Run stratified k-fold cross-validation and collect out-of-fold predictions.

    ``fit_predict`` is a callable ``(train_rows, test_rows) -> list[label]`` that
    fits on the training fold and predicts the test fold in order.
    """
    from sklearn.model_selection import StratifiedKFold

    labels = [row.hidden_label for row in rows]
    min_class = min(Counter(labels).values())
    effective_folds = max(2, min(folds, min_class))
    if min_class < 2:
        raise ValueError(
            f"Cannot cross-validate: smallest class has {min_class} row(s); need at least 2"
        )

    predictions: list[str | None] = [None] * len(rows)
    indices = list(range(len(rows)))
    splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(indices, labels):
        train_rows = [rows[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]
        fold_predictions = fit_predict(train_rows, test_rows)
        for position, row_index in enumerate(test_idx):
            predictions[row_index] = fold_predictions[position]
    return predictions


def _predict_majority(rows: list[DatasetRow], folds: int, seed: int) -> list[str | None]:
    def fit_predict(train_rows: list[DatasetRow], test_rows: list[DatasetRow]) -> list[str]:
        majority_label = Counter(row.hidden_label for row in train_rows).most_common(1)[0][0]
        return [majority_label] * len(test_rows)

    return _stratified_oof(rows, fit_predict, folds, seed)


def _predict_tfidf_logreg(rows: list[DatasetRow], folds: int, seed: int) -> list[str | None]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    def fit_predict(train_rows: list[DatasetRow], test_rows: list[DatasetRow]) -> list[str]:
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        )
        pipeline.fit([row.sentence for row in train_rows], [row.hidden_label for row in train_rows])
        return list(pipeline.predict([row.sentence for row in test_rows]))

    return _stratified_oof(rows, fit_predict, folds, seed)


@lru_cache(maxsize=1)
def _vader_analyzer():
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except ImportError as exc:  # pragma: no cover - exercised only without nltk
        raise RuntimeError("VADER baseline requires nltk. Install with: pip install '.[baselines]'") from exc

    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        import nltk

        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def classify_vader_text(text: str) -> VaderSentiment:
    """Classify arbitrary text with VADER's conventional compound thresholds."""
    compound = float(_vader_analyzer().polarity_scores(text)["compound"])
    if compound >= VADER_THRESHOLD:
        label = "positive"
    elif compound <= -VADER_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"
    return VaderSentiment(label=label, compound=compound)


def _predict_vader(rows: list[DatasetRow]) -> list[str | None]:
    analyzer = _vader_analyzer()

    predictions: list[str | None] = []
    for row in rows:
        compound = analyzer.polarity_scores(row.sentence)["compound"]
        if compound >= VADER_THRESHOLD:
            predictions.append("positive")
        elif compound <= -VADER_THRESHOLD:
            predictions.append("negative")
        else:
            predictions.append("neutral")
    return predictions


def _disable_hf_progress_bars() -> None:
    """Silence Hugging Face / tqdm progress bars before loading FinBERT.

    The TUI runs baselines in a worker thread. On first use tqdm tries to build a
    *multiprocessing* lock (spawning the resource tracker via fork_exec), which
    fails from a non-main thread on Python 3.13 with "bad value(s) in fds_to_keep".
    Disabling the bars avoids creating any tqdm instance, and pre-seeding tqdm with
    a plain thread lock keeps it off the multiprocessing path even if something
    else instantiates it.
    """
    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import threading

        from tqdm import tqdm

        tqdm.set_lock(threading.RLock())
    except Exception:  # pragma: no cover - tqdm internals are best-effort
        pass
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
    except Exception:  # pragma: no cover - depends on transformers version
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:  # pragma: no cover - depends on hub version
        pass


def classify_finbert_texts(texts: list[str], batch_size: int = 32) -> list[str | None]:
    """Classify arbitrary financial texts with the pretrained FinBERT baseline."""
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:  # pragma: no cover - exercised only without transformers
        raise RuntimeError(
            "FinBERT baseline requires transformers and torch. Install with: pip install '.[finbert]'"
        ) from exc

    _disable_hf_progress_bars()
    classifier = hf_pipeline("text-classification", model="ProsusAI/finbert", truncation=True)
    predictions: list[str | None] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        for output in classifier(batch):
            label = str(output.get("label", "")).strip().lower()
            predictions.append(label if label in ALLOWED_LABELS else None)
    return predictions


def _predict_finbert(rows: list[DatasetRow], batch_size: int = 32) -> list[str | None]:
    return classify_finbert_texts([row.sentence for row in rows], batch_size=batch_size)


def predict_baseline(
    name: str,
    rows: list[DatasetRow],
    folds: int = 5,
    seed: int = DEFAULT_SEED,
) -> list[str | None]:
    """Return a per-row predicted label (or ``None`` if no valid label) for ``name``.

    The returned list is aligned with ``rows`` order.
    """
    if name not in BASELINE_SPECS:
        available = ", ".join(sorted(BASELINE_SPECS))
        raise ValueError(f"Unknown baseline {name!r}. Available: {available}")
    if name == "majority":
        return _predict_majority(rows, folds, seed)
    if name == "tfidf_logreg":
        return _predict_tfidf_logreg(rows, folds, seed)
    if name == "vader":
        return _predict_vader(rows)
    if name == "finbert":
        return _predict_finbert(rows)
    raise ValueError(f"Baseline {name!r} has no predictor implementation")  # pragma: no cover
