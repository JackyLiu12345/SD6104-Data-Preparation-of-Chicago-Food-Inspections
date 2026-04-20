"""
entity_aggregation.py
=====================
Aggregate food inspection data by Entity_ID to produce:

  • A full entity-level summary table with pass/fail rates, violation
    counts, and a composite risk index.
  • A high-risk ranking table (top-N entities by risk index).

Functions are adapted from the standalone entity aggregation script
provided by the user.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_violation_terms(value: str) -> list[str]:
    """Parse a comma-separated string of violation term numbers."""
    text = (value or "").strip()
    if not text:
        return []

    parsed = []
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        try:
            n = int(float(part))
        except Exception:
            continue
        if n != 0:
            parsed.append(str(n))
    return parsed


def mode_or_first(series: pd.Series):
    """Return the mode (most frequent value) or the first non-null value."""
    counts = series.value_counts(dropna=True)
    if len(counts) == 0:
        return np.nan
    return counts.index[0]


def safe_divide(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Element-wise division that returns NaN when the denominator is zero."""
    return np.where(b > 0, a / b, np.nan)


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def build_entity_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an entity-level aggregation table from an inspection DataFrame
    that contains an ``Entity_ID`` column.

    The output includes inspection counts, pass/fail rates, violation
    statistics, and a composite **risk_index**.
    """
    data = df.copy()
    data["Entity_ID"] = data["Entity_ID"].astype(str).str.strip()
    data = data[data["Entity_ID"].ne("")].copy()

    # Normalise text columns that are present
    for col in ["Results", "Risk", "Violation Terms", "DBA Name", "Facility Type", "City"]:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()

    # --- Boolean flags ---
    evaluated = ["Pass", "Pass w/ Conditions", "Fail"]
    data["is_evaluated"] = data["Results"].isin(evaluated)
    data["is_pass"] = data["Results"].eq("Pass")
    data["is_pass_cond"] = data["Results"].eq("Pass w/ Conditions")
    data["is_fail"] = data["Results"].eq("Fail")
    data["is_out_of_business"] = data["Results"].eq("Out of Business")
    data["is_no_entry"] = data["Results"].eq("No Entry")
    data["is_not_ready"] = data["Results"].eq("Not Ready")
    data["is_business_not_located"] = data["Results"].eq("Business Not Located")

    data["is_risk1_high"] = data["Risk"].eq("Risk 1 (High)")
    data["is_risk2_medium"] = data["Risk"].eq("Risk 2 (Medium)")
    data["is_risk3_low"] = data["Risk"].eq("Risk 3 (Low)")

    terms_series = data["Violation Terms"].map(parse_violation_terms)
    data["violation_count_this_inspection"] = terms_series.map(len)
    data["has_violation"] = data["violation_count_this_inspection"] > 0

    # --- Base aggregation ---
    base = data.groupby("Entity_ID").agg(
        total_inspections=("Entity_ID", "size"),
        evaluated_inspections=("is_evaluated", "sum"),
        pass_count=("is_pass", "sum"),
        pass_w_conditions_count=("is_pass_cond", "sum"),
        fail_count=("is_fail", "sum"),
        out_of_business_count=("is_out_of_business", "sum"),
        no_entry_count=("is_no_entry", "sum"),
        not_ready_count=("is_not_ready", "sum"),
        business_not_located_count=("is_business_not_located", "sum"),
        inspections_with_violations=("has_violation", "sum"),
        total_violations=("violation_count_this_inspection", "sum"),
        avg_violations_per_inspection=("violation_count_this_inspection", "mean"),
        risk1_high_count=("is_risk1_high", "sum"),
        risk2_medium_count=("is_risk2_medium", "sum"),
        risk3_low_count=("is_risk3_low", "sum"),
        inspection_years=("Inspection Year", "nunique"),
    ).reset_index()

    # --- Attribute aggregation (mode per entity) ---
    attr_agg = {"dba_name": ("DBA Name", mode_or_first), "facility_type": ("Facility Type", mode_or_first)}
    if "City" in data.columns:
        attr_agg["city"] = ("City", mode_or_first)

    attrs = data.groupby("Entity_ID").agg(**attr_agg).reset_index()

    # --- Violation term analysis ---
    term_counts_by_entity: dict[str, Counter] = {}
    for eid, terms in zip(data["Entity_ID"].values, terms_series.values):
        if eid not in term_counts_by_entity:
            term_counts_by_entity[eid] = Counter()
        term_counts_by_entity[eid].update(terms)

    unique_terms = {}
    top3_terms = {}
    for eid, counter in term_counts_by_entity.items():
        unique_terms[eid] = len(counter)
        top = counter.most_common(3)
        top3_terms[eid] = "; ".join([f"{k}({v})" for k, v in top]) if top else ""

    base["unique_violation_terms"] = base["Entity_ID"].map(unique_terms).fillna(0).astype(int)
    base["top3_violation_terms"] = base["Entity_ID"].map(top3_terms).fillna("")

    # --- Derived rates ---
    base["pass_rate_evaluated"] = safe_divide(
        base["pass_count"] + base["pass_w_conditions_count"], base["evaluated_inspections"]
    )
    base["strict_pass_rate"] = safe_divide(base["pass_count"], base["evaluated_inspections"])
    base["fail_rate_evaluated"] = safe_divide(base["fail_count"], base["evaluated_inspections"])
    base["violation_inspection_rate"] = safe_divide(
        base["inspections_with_violations"], base["total_inspections"]
    )
    base["high_risk_inspection_rate"] = safe_divide(base["risk1_high_count"], base["total_inspections"])
    base["non_operational_result_rate"] = safe_divide(
        base["out_of_business_count"]
        + base["no_entry_count"]
        + base["not_ready_count"]
        + base["business_not_located_count"],
        base["total_inspections"],
    )

    max_avg_vio = base["avg_violations_per_inspection"].max()
    if pd.isna(max_avg_vio) or max_avg_vio == 0:
        max_avg_vio = 1

    base["risk_index"] = (
        base["fail_rate_evaluated"].fillna(0) * 0.45
        + base["violation_inspection_rate"].fillna(0) * 0.30
        + (base["avg_violations_per_inspection"] / max_avg_vio).fillna(0) * 0.15
        + base["non_operational_result_rate"].fillna(0) * 0.10
    )

    result = attrs.merge(base, on="Entity_ID", how="right")

    # --- Rounding ---
    rate_cols = [
        "pass_rate_evaluated",
        "strict_pass_rate",
        "fail_rate_evaluated",
        "violation_inspection_rate",
        "high_risk_inspection_rate",
        "non_operational_result_rate",
        "risk_index",
    ]
    for col in rate_cols:
        result[col] = result[col].round(4)
    result["avg_violations_per_inspection"] = result["avg_violations_per_inspection"].round(3)

    result = result.sort_values(["risk_index", "total_inspections"], ascending=[False, False])
    return result


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------

def run_entity_aggregation(
    df: pd.DataFrame,
    entity_output: str = "output/entity_inspection_analysis.csv",
    risk_output: str = "output/entity_high_risk_rank.csv",
    min_inspections: int = 5,
    top_n: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate inspection data by Entity_ID and produce a high-risk ranking.

    Parameters
    ----------
    df               : Cleaned DataFrame with ``Entity_ID`` column (from
                       the structuring step).
    entity_output    : Path for the full entity aggregation CSV.
    risk_output      : Path for the high-risk ranking CSV.
    min_inspections  : Minimum inspections to qualify for the ranking.
    top_n            : Number of top-risk entities to include in the ranking.

    Returns
    -------
    (entity_df, high_risk_df) as DataFrames.
    """
    print("  Building entity-level aggregation ...")
    entity_df = build_entity_aggregation(df)
    print(f"  Entity summary rows: {len(entity_df):,}")

    os.makedirs(os.path.dirname(entity_output), exist_ok=True)
    entity_df.to_csv(entity_output, index=False, encoding="utf-8-sig")
    print(f"  Entity aggregation saved to: {entity_output}")

    high_risk_df = (
        entity_df[entity_df["total_inspections"] >= min_inspections]
        .sort_values(
            ["risk_index", "fail_rate_evaluated", "total_violations"],
            ascending=[False, False, False],
        )
        .head(top_n)
    )

    os.makedirs(os.path.dirname(risk_output), exist_ok=True)
    high_risk_df.to_csv(risk_output, index=False, encoding="utf-8-sig")
    print(f"  High-risk ranking ({len(high_risk_df)} entities) saved to: {risk_output}")

    return entity_df, high_risk_df
