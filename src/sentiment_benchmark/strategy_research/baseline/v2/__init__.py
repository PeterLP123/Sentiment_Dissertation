"""Frozen v2 sentiment trading baseline: canonical VADER, two-name legs, event-level inference."""

from .config import BaselineV2ConfigurationError, LiteratureBaselineV2Config, load_baseline_v2_config
from .pipeline import BaselineV2RunError, inspect_baseline_v2, run_baseline_v2

__all__ = [
    "BaselineV2ConfigurationError",
    "BaselineV2RunError",
    "LiteratureBaselineV2Config",
    "inspect_baseline_v2",
    "load_baseline_v2_config",
    "run_baseline_v2",
]
