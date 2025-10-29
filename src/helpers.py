"""
helpers.py
Utility helpers for WellCo churn assignment.
Each function has a single responsibility and a short docstring.
"""

from __future__ import annotations
import re
import pandas as pd
from typing import Iterable, List, Dict, Optional, Set

CATEGORY_PATTERNS: Dict[str, str] = {
    # nutrition / diet topics
    "nutrition": r"\b(nutrition|diet|mediterranean|fiber|cholesterol|lipid|high[- ]fiber)\b",
    # physical activity
    "activity":  r"\b(exercise|workout|cardio|aerobic|strength|resistance|fitness)\b",
    # sleep health
    "sleep":     r"\b(sleep|apnea|sleep hygiene|restorative sleep|sleep quality)\b",
    # stress & mindfulness
    "stress":    r"\b(stress|mindfulness|meditation|resilience|wellbeing)\b",
}

def categorize_visit(
    url: str = "",
    title: str = "",
    description: str = "",
    category_patterns: Dict[str, str] = CATEGORY_PATTERNS,
    default: str = "other"
) -> List[str]:
    """
    Map a single web/app content record to zero or more high-level categories.
    Returns a list of matched categories (or [default] if none matched).

    Args:
        url, title, description: Text fields to search.
        category_patterns: dict of {category: regex} patterns.
        default: category to use if nothing matches.

    Notes:
        - Multiple categories can match; ordering is deterministic (dict insertion order).
        - Adjust CATEGORY_PATTERNS to align with business taxonomy.
    """
    text = f"{url} {title} {description}".lower()
    hits: List[str] = []
    for cat, pat in category_patterns.items():
        if re.search(pat, text):
            hits.append(cat)
    return hits or [default]

def compute_obs_end(app_ts: pd.Series, web_ts: pd.Series, claims_dt: pd.Series) -> pd.Timestamp:
    """
    Return a single observation window end (OBS_END) as the normalized max across sources.
    This anchor prevents temporal leakage when computing recency/tenure features.
    """
    return max(app_ts.max(), web_ts.max(), claims_dt.max()).normalize()

def add_recency_from_last(se: pd.Series, obs_end: pd.Timestamp) -> pd.Series:
    """
    Compute days since the last event in a per-member 'last_seen' timestamp series.
    Missing timestamps become NaN here; impute later with a large value.
    """
    return (obs_end - se).dt.days

def assert_integrity(features: pd.DataFrame, obs_end: pd.Timestamp, required: Optional[Set[str]] = None) -> None:
    """
    Basic integrity checks after merge: required columns present, one row per member_id,
    and (optional) no future dates beyond OBS_END in any date/timestamp-like columns.
    """
    required = required or {"member_id", "churn", "outreach"}
    missing = required - set(features.columns)
    assert not missing, f"Missing required columns: {missing}"
    assert features["member_id"].is_unique, "Duplicate member_id after merge"

    # Optional future-date sentinel
    date_like = [c for c in features.columns if ("date" in c.lower() or "timestamp" in c.lower())]
    for c in date_like:
        if pd.api.types.is_datetime64_any_dtype(features[c]):
            assert not features[c].gt(obs_end).any(), f"Future dates detected in {c}"

def impute_with_flags(
    df: pd.DataFrame,
    count_cols: Iterable[str],
    recency_cols: Iterable[str],
    binary_cols: Iterable[str],
    tenure_cols: Iterable[str],
    missing_flag_threshold: float = 0.05,
    large_recency: int = 3650
) -> pd.DataFrame:
    """
    Impute dataframe in-place with sensible defaults and add missingness flags for columns
    with null rate above threshold. Returns the modified dataframe.
    """
    # Missingness flags
    null_rate = df.isnull().mean()
    missing_targets = [c for c, r in null_rate.items() if r > missing_flag_threshold]
    for c in missing_targets:
        df[f"{c}__was_missing"] = df[c].isna().astype(int)

    # Imputations
    df[list(count_cols)] = df[list(count_cols)].fillna(0)
    df[list(recency_cols)] = df[list(recency_cols)].fillna(large_recency)
    df[list(binary_cols)] = df[list(binary_cols)].fillna(0)
    for c in tenure_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    return df