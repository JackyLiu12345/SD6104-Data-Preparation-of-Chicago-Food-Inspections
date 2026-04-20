"""
visualization.py
================
Data visualization step for the Chicago Food Inspections pipeline.

Generates white-themed charts for:
  • Missing value percentages
  • Category distributions (Results, Risk, Facility Type)
  • FD confidence ranking
  • FD violation counts

All plots use a consistent white background + black text theme and are
saved as PNG files to the specified output directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Global matplotlib theme (white background, black text)
# ---------------------------------------------------------------------------
plt.style.use("default")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["savefig.edgecolor"] = "white"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["axes.titlecolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["savefig.transparent"] = False


def apply_white_theme(ax):
    """Apply the white background / black text theme to an Axes object."""
    ax.set_facecolor("white")
    ax.title.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.grid(True, linestyle="--", alpha=0.4, color="gray")


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_missing_value_percentage(df, output_dir="output"):
    """Bar chart of missing-value percentage for every column."""
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    ax.barh(missing_pct.index, missing_pct.values)
    ax.set_xlabel("Missing Percentage (%)")
    ax.set_ylabel("Columns")
    ax.set_title("Missing Value Percentage by Column")
    apply_white_theme(ax)
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="gray")

    output_path = os.path.join(output_dir, "missing_value_percentage.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="white", transparent=False)
    plt.close()
    print(f"  Saved: {output_path}")
    return missing_pct


def plot_category_distribution(df, column, top_n=15, output_dir="output"):
    """Bar chart of the top-N most frequent values in *column*."""
    if column not in df.columns:
        print(f"  Column not found: {column}")
        return None

    counts = df[column].fillna("MISSING").value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(f"Top {top_n} Categories in {column}")
    apply_white_theme(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="gray")
    plt.xticks(rotation=45, ha="right")

    safe_name = column.lower().replace(" ", "_").replace("#", "num")
    output_path = os.path.join(output_dir, f"category_distribution_{safe_name}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="white", transparent=False)
    plt.close()
    print(f"  Saved: {output_path}")
    return counts


def _compute_fd_confidence_for_viz(df, lhs, rhs):
    """Compute FD confidence (local helper to avoid circular imports)."""
    group_nunique = df.groupby(lhs)[rhs].nunique(dropna=True)
    total_groups = len(group_nunique)
    violating_groups = (group_nunique > 1).sum()
    if total_groups == 0:
        return 0.0
    return round(1 - violating_groups / total_groups, 4)


def _compute_fd_violation_counts(df, lhs, rhs):
    """Return (violating_groups, total_groups) for the FD lhs -> rhs."""
    group_nunique = df.groupby(lhs)[rhs].nunique(dropna=True)
    violating_groups = (group_nunique > 1).sum()
    total_groups = len(group_nunique)
    return violating_groups, total_groups


def plot_fd_confidence_ranking(df, fd_list, output_dir="output"):
    """Bar chart ranking FD confidence for a list of (LHS, RHS) pairs."""
    rows = []
    for lhs, rhs in fd_list:
        if lhs not in df.columns or rhs not in df.columns:
            continue
        conf = _compute_fd_confidence_for_viz(df, lhs, rhs)
        rows.append({"FD": f"{lhs} -> {rhs}", "Confidence": conf})

    fd_conf_df = pd.DataFrame(rows).sort_values(by="Confidence", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.bar(fd_conf_df["FD"], fd_conf_df["Confidence"])
    ax.set_ylabel("FD Confidence")
    ax.set_xlabel("Functional Dependency")
    ax.set_title("Functional Dependency Confidence Ranking")
    ax.set_ylim(0.85, 1.01)
    apply_white_theme(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="gray")
    plt.xticks(rotation=30, ha="right")

    output_path = os.path.join(output_dir, "fd_confidence_ranking.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="white", transparent=False)
    plt.close()
    print(f"  Saved: {output_path}")
    return fd_conf_df


def plot_fd_violation_counts(df, fd_list, output_dir="output"):
    """Bar chart of FD violation group counts for a list of (LHS, RHS) pairs."""
    rows = []
    for lhs, rhs in fd_list:
        if lhs not in df.columns or rhs not in df.columns:
            continue
        violating_groups, total_groups = _compute_fd_violation_counts(df, lhs, rhs)
        rows.append({
            "FD": f"{lhs} -> {rhs}",
            "Violating Groups": violating_groups,
            "Total Groups": total_groups,
        })

    fd_violation_df = pd.DataFrame(rows).sort_values(by="Violating Groups", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.bar(fd_violation_df["FD"], fd_violation_df["Violating Groups"])
    ax.set_ylabel("Number of Violating Groups")
    ax.set_xlabel("Functional Dependency")
    ax.set_title("FD Violation Counts")
    apply_white_theme(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="gray")
    plt.xticks(rotation=30, ha="right")

    output_path = os.path.join(output_dir, "fd_violation_counts.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="white", transparent=False)
    plt.close()
    print(f"  Saved: {output_path}")
    return fd_violation_df


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------

_VIZ_FD_LIST = [
    ("Address", "Zip"),
    ("License #", "Address"),
    ("License #", "Zip"),
    ("License #", "Risk"),
    ("License #", "Facility Type"),
]


def run_visualization(df, output_dir="output"):
    """
    Generate all standard visualizations for the cleaned DataFrame.

    Produces:
      • missing_value_percentage.png
      • category_distribution_results.png
      • category_distribution_risk.png
      • category_distribution_facility_type.png
      • fd_confidence_ranking.png
      • fd_violation_counts.png

    Parameters
    ----------
    df         : Cleaned inspection DataFrame.
    output_dir : Directory to save PNG files into.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Normalise text columns for consistent labels
    text_cols = ["DBA Name", "AKA Name", "Address", "Facility Type",
                 "Risk", "Results", "Inspection Type"]
    viz_df = df.copy()
    for col in text_cols:
        if col in viz_df.columns:
            viz_df[col] = viz_df[col].astype(str).str.strip().str.upper()
    if "Zip" in viz_df.columns:
        viz_df["Zip"] = pd.to_numeric(viz_df["Zip"], errors="coerce")

    plot_missing_value_percentage(viz_df, output_dir)
    plot_category_distribution(viz_df, "Results", top_n=10, output_dir=output_dir)
    plot_category_distribution(viz_df, "Risk", top_n=10, output_dir=output_dir)
    plot_category_distribution(viz_df, "Facility Type", top_n=15, output_dir=output_dir)
    plot_fd_confidence_ranking(viz_df, _VIZ_FD_LIST, output_dir=output_dir)
    plot_fd_violation_counts(viz_df, _VIZ_FD_LIST, output_dir=output_dir)

    print("  All visualization charts generated.")
