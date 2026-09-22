"""Matplotlib charts used by the Streamlit app.

Every function returns a matplotlib Figure (no pyplot state), so it is safe to
call from Streamlit and easy to test on its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

NAVY = "#16324F"
TEAL = "#2A9D8F"
AMBER = "#E9A23B"
CORAL = "#E76F51"
GREY = "#8A9BA8"
LIGHT = "#E3EAF0"

_money = FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k")


def _new(width: float = 7.5, height: float = 4.2):
    fig = Figure(figsize=(width, height), dpi=110)
    fig.patch.set_facecolor("white")
    ax = fig.subplots()
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=NAVY, labelsize=9)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)
    ax.set_axisbelow(True)
    return fig, ax


def _title(ax, text: str):
    ax.set_title(text, loc="left", fontsize=11, fontweight="bold", color=NAVY, pad=10)


def price_histogram(df: pd.DataFrame) -> Figure:
    fig, ax = _new()
    ax.hist(df["Price"], bins=30, color=TEAL, edgecolor="white")
    mean, median = df["Price"].mean(), df["Price"].median()
    ax.axvline(mean, color=AMBER, linewidth=2, label=f"Mean  ${mean:,.0f}")
    ax.axvline(median, color=CORAL, linewidth=2, linestyle="--", label=f"Median  ${median:,.0f}")
    ax.xaxis.set_major_formatter(_money)
    ax.set_xlabel("Price")
    ax.set_ylabel("Number of houses")
    ax.grid(axis="y", color=LIGHT)
    ax.legend(frameon=False, fontsize=9)
    _title(ax, "How house prices are spread")
    fig.tight_layout()
    return fig


def price_boxplot(df: pd.DataFrame, col: str) -> Figure:
    order = df.groupby(col)["Price"].median().sort_values().index.tolist()
    data = [df.loc[df[col] == k, "Price"].values for k in order]
    fig, ax = _new()
    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": NAVY, "linewidth": 2},
        whiskerprops={"color": GREY},
        capprops={"color": GREY},
        flierprops={"markeredgecolor": GREY, "markersize": 3},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(TEAL)
        patch.set_alpha(0.55)
        patch.set_edgecolor(TEAL)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([str(k) for k in order])
    ax.yaxis.set_major_formatter(_money)
    ax.set_xlabel(col)
    ax.set_ylabel("Price")
    ax.grid(axis="y", color=LIGHT)
    _title(ax, f"Price range by {col}")
    fig.tight_layout()
    return fig


def _bar_with_labels(ax, labels, values, overall_mean, color=TEAL):
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=color, width=0.62)
    ax.axhline(overall_mean, color=AMBER, linewidth=1.8, linestyle="--", label="Overall average")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in labels])
    top = max(values) if len(values) else 1
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + top * 0.015,
            f"${val / 1000:,.0f}k",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=NAVY,
        )
    ax.set_ylim(0, top * 1.15)
    ax.yaxis.set_major_formatter(_money)
    ax.grid(axis="y", color=LIGHT)
    ax.legend(frameon=False, fontsize=9, loc="lower right")


def mean_price_bar(df: pd.DataFrame, col: str) -> Figure:
    """Average price for each value of a categorical / discrete column."""
    grouped = df.groupby(col)["Price"].mean()
    if col == "Condition":
        wanted = [v for v in ["Poor", "Fair", "Good", "Excellent"] if v in grouped.index]
        grouped = grouped.loc[wanted + [v for v in grouped.index if v not in wanted]]
    elif pd.api.types.is_numeric_dtype(df[col]):
        grouped = grouped.sort_index()
    else:
        grouped = grouped.sort_values(ascending=False)
    fig, ax = _new()
    _bar_with_labels(ax, list(grouped.index), grouped.values, df["Price"].mean())
    ax.set_xlabel(col)
    ax.set_ylabel("Average price")
    _title(ax, f"Average price by {col}")
    fig.tight_layout()
    return fig


def mean_price_by_bucket(df: pd.DataFrame, col: str) -> Figure:
    """Average price per value (few values) or per equal-sized bin (many values)."""
    if df[col].nunique() <= 10:
        return mean_price_bar(df, col)
    bins = pd.qcut(df[col], q=8, duplicates="drop")
    grouped = df.groupby(bins, observed=True)["Price"].mean()
    labels = [f"{int(iv.left):,}\u2013{int(iv.right):,}" for iv in grouped.index]
    fig, ax = _new()
    _bar_with_labels(ax, labels, grouped.values, df["Price"].mean())
    ax.tick_params(axis="x", labelsize=7.5)
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha("right")
    ax.set_xlabel(col)
    ax.set_ylabel("Average price")
    _title(ax, f"Average price by {col} range")
    fig.tight_layout()
    return fig


def scatter_vs_price(df: pd.DataFrame, col: str) -> Figure:
    x = df[col].to_numpy(dtype=float)
    y = df["Price"].to_numpy(dtype=float)
    fig, ax = _new()
    ax.scatter(x, y, s=10, color=TEAL, alpha=0.35, edgecolors="none")
    if len(np.unique(x)) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, color=CORAL, linewidth=2.2, label="Trend line")
        r = float(np.corrcoef(x, y)[0, 1])
        ax.legend(frameon=False, fontsize=9, title=f"correlation r = {r:.3f}", title_fontsize=9)
    ax.yaxis.set_major_formatter(_money)
    ax.set_xlabel(col)
    ax.set_ylabel("Price")
    ax.grid(color=LIGHT)
    _title(ax, f"{col} vs price")
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame, columns: list[str]) -> Figure:
    corr = df[columns].corr()
    fig, ax = _new(6.6, 5.0)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=40, ha="right")
    ax.set_yticklabels(columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(val) > 0.6 else NAVY)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _title(ax, "Correlation between numeric columns (scale -1 to +1)")
    fig.tight_layout()
    return fig


def decade_line(df: pd.DataFrame) -> Figure:
    decade = (df["YearBuilt"] // 10 * 10).astype(int)
    grouped = df.groupby(decade)["Price"].mean()
    fig, ax = _new()
    ax.plot(grouped.index, grouped.values, color=NAVY, linewidth=2, marker="o", markersize=5)
    ax.axhline(df["Price"].mean(), color=AMBER, linewidth=1.8, linestyle="--", label="Overall average")
    ax.yaxis.set_major_formatter(_money)
    ax.set_xlabel("Decade built")
    ax.set_ylabel("Average price")
    ax.grid(axis="y", color=LIGHT)
    ax.legend(frameon=False, fontsize=9)
    _title(ax, "Average price by decade built")
    fig.tight_layout()
    return fig


def model_comparison(results: pd.DataFrame) -> Figure:
    names = results["Model"].tolist()
    short = [n.replace("Baseline (always predict the mean)", "Baseline") for n in names]
    y = np.arange(len(names))
    fig, ax = _new(7.5, 4.4)
    h = 0.36
    ax.barh(y - h / 2, results["CV R2 (train)"], height=h, color=GREY, label="Cross-validation R\u00b2 (training data)")
    ax.barh(y + h / 2, results["Test R2"], height=h, color=TEAL, label="Test R\u00b2 (unseen data)")
    ax.axvline(0, color=NAVY, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(short)
    ax.invert_yaxis()
    ax.set_xlabel("R\u00b2 score (1 = perfect, 0 = no better than the average)")
    ax.grid(axis="x", color=LIGHT)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    _title(ax, "How well does each model predict unseen houses?")
    fig.tight_layout()
    return fig


def actual_vs_predicted(y_true, y_pred, name: str) -> Figure:
    fig, ax = _new(5.6, 4.6)
    ax.scatter(y_true, y_pred, s=12, color=TEAL, alpha=0.45, edgecolors="none")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color=CORAL, linewidth=2, linestyle="--", label="Perfect prediction")
    ax.xaxis.set_major_formatter(_money)
    ax.yaxis.set_major_formatter(_money)
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.grid(color=LIGHT)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _title(ax, f"Actual vs predicted ({name})")
    fig.tight_layout()
    return fig


def residual_histogram(y_true, y_pred) -> Figure:
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = _new(5.6, 4.6)
    ax.hist(resid, bins=30, color=NAVY, alpha=0.85, edgecolor="white")
    ax.axvline(0, color=CORAL, linewidth=2)
    ax.xaxis.set_major_formatter(_money)
    ax.set_xlabel("Error = actual \u2212 predicted")
    ax.set_ylabel("Number of houses")
    ax.grid(axis="y", color=LIGHT)
    _title(ax, "How far off are the predictions?")
    fig.tight_layout()
    return fig


def importance_bar(importance: pd.Series) -> Figure:
    imp = importance.sort_values()
    fig, ax = _new(7.5, 4.2)
    colors = [TEAL if v > 0 else GREY for v in imp.values]
    ax.barh(imp.index, imp.values, color=colors)
    ax.axvline(0, color=NAVY, linewidth=1)
    ax.set_xlabel("Drop in test R\u00b2 when this column is shuffled (higher = more useful)")
    ax.grid(axis="x", color=LIGHT)
    _title(ax, "Which inputs matter to the model?")
    fig.tight_layout()
    return fig


def prediction_comparison(predicted: float, location: str, location_avg: float, overall_avg: float) -> Figure:
    labels = ["Your estimate", f"Average in {location}", "Average of all houses"]
    values = [predicted, location_avg, overall_avg]
    colors = [AMBER, TEAL, NAVY]
    fig, ax = _new(6.2, 3.4)
    ax.barh(labels, values, color=colors, height=0.55)
    ax.invert_yaxis()
    top = max(values)
    for i, v in enumerate(values):
        ax.text(v + top * 0.015, i, f"${v:,.0f}", va="center", fontsize=9, color=NAVY)
    ax.set_xlim(0, top * 1.22)
    ax.xaxis.set_major_formatter(_money)
    ax.grid(axis="x", color=LIGHT)
    _title(ax, "Estimate compared with typical prices")
    fig.tight_layout()
    return fig
