from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

def churn_by_bins_table(
    df: pd.DataFrame,
    feature: str,
    target: str = "churn",
    bins: int = 8,
    strategy: str = "quantile",   # "quantile" or "uniform"
    min_bin_size: int = 30,
) -> pd.DataFrame:
    """
    Build a small table with churn rate per bin for a given feature.

    Parameters
    ----------
    df : DataFrame containing the feature and target.
    feature : column to bin.
    target : binary target column (0/1).
    bins : number of bins.
    strategy : "quantile" -> pd.qcut ; "uniform" -> pd.cut
    min_bin_size : drop bins with < min_bin_size rows.

    Returns
    -------
    DataFrame with columns: bin, count, churn, mean_feature.
    """
    s = df[feature]
    y = df[target]

    # drop rows with missing feature or target
    m = s.notna() & y.notna()
    s, y = s[m], y[m]

    if strategy == "quantile":
        # guard against ties / too few uniques
        try:
            binned = pd.qcut(s, q=min(bins, s.nunique()), duplicates="drop")
        except ValueError:
            # fallback: everything into one bin
            binned = pd.Series(["all"] * len(s), index=s.index)
    else:
        binned = pd.cut(s, bins=bins)

    out = (
        pd.DataFrame({feature: s, target: y, "_bin": binned})
        .groupby("_bin", observed=True)
        .agg(
            count=(target, "size"),
            churn=(target, "mean"),
            mean_feature=(feature, "mean"),
        )
        .reset_index()
        .rename(columns={"_bin": "bin"})
        .sort_values("mean_feature")
    )

    # drop tiny bins (usually from heavy ties at edges)
    out = out[out["count"] >= min_bin_size]
    return out

def plot_churn_by_quantile(
    df: pd.DataFrame,
    feature: str,
    target: str = "churn",
    bins: int = 8,
    strategy: str = "quantile",
    min_bin_size: int = 30,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Bar chart of churn rate by binned feature (quantile or uniform).

    Returns (fig, ax) so you can further style or save.
    """
    tbl = churn_by_bins_table(
        df, feature, target=target, bins=bins, strategy=strategy, min_bin_size=min_bin_size
    )
    if tbl.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"No valid bins for '{feature}'", ha="center", va="center")
        ax.axis("off")
        return fig, ax

    # Plot
    fig, ax = plt.subplots(figsize=(7.5, 4)) if ax is None else (ax.figure, ax)
    ax.bar(tbl["bin"].astype(str), tbl["churn"].values)
    ax.set_title(f"Churn Rate by {feature} ({strategy} bins)")
    ax.set_ylabel("Churn rate")
    ax.set_xlabel("bin")
    ax.set_ylim(0, max(0.001, tbl["churn"].max()) * 1.15)
    ax.tick_params(axis="x", labelrotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")

    fig.tight_layout()
    return fig, ax

def plot_top_corrs(
    df: pd.DataFrame,
    numeric_cols: List[str],
    target: str = "churn",
    method: str = "pearson",  # "pearson" or "spearman"
    top_n: int = 12,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, pd.Series]:
    """
    Horizontal bar chart of top |correlation| features with the target.

    Returns (fig, ax, corrs) where corrs is the full (sorted) correlation Series.
    """
    # keep only columns present and numeric
    cols = [c for c in numeric_cols if c in df.columns]
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in DataFrame.")

    corrs = df[cols].corrwith(df[target], method=method).dropna()
    corrs = corrs.sort_values(key=np.abs, ascending=False)

    top = corrs.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6)) if ax is None else (ax.figure, ax)
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_title(f"Top {len(top)} {method.title()} Correlations with '{target}'")
    ax.set_xlabel("Correlation")
    ax.axvline(0, lw=1, color="k")
    fig.tight_layout()
    return fig, ax, corrs