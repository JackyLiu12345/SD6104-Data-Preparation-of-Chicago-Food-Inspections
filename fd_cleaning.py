"""
fd_cleaning.py
==============
FD-based data cleaning step.

Combines:
  • Inspection-level cleaning  (formerly inspection_cleaning.py)
  • FD-driven imputation       (use discovered FDs to fix conflicts)
  • Final imputation & cleanup (formerly final_cleaning.py)
"""

import re
import numpy as np
import pandas as pd
from collections import Counter


# ---------------------------------------------------------------------------
# Part 1 – Inspection-level cleaning  (from inspection_cleaning.py)
# ---------------------------------------------------------------------------

def _extract_violation_terms(violation_text):
    """Parse the pipe-delimited Violations field and return a comma-separated
    string of violation term numbers."""
    if pd.isna(violation_text):
        return np.nan

    text = str(violation_text)
    clauses = [clause.strip() for clause in text.split("|")]
    all_matches = []

    for clause in clauses:
        main_part = clause
        comments_markers = [" - Comments:", "Comments:", " - ", "COMMENTS:"]
        for marker in comments_markers:
            if marker in main_part:
                parts = main_part.split(marker, 1)
                if len(parts) > 1:
                    main_part = parts[0]
                    break

        pattern = r"^\s*(\d+)\.\s"
        match = re.search(pattern, main_part)
        if match:
            term = match.group(1)
            try:
                term_clean = str(int(term))
            except ValueError:
                term_clean = term
            all_matches.append(term_clean)
        else:
            backup_pattern = r"\b(\d+)\.\s+[A-Z]"
            backup_matches = re.findall(backup_pattern, main_part)
            for term in backup_matches:
                try:
                    term_clean = str(int(term))
                except ValueError:
                    term_clean = term
                all_matches.append(term_clean)

    if all_matches:
        unique_matches = sorted(set(all_matches), key=lambda x: int(x) if x.isdigit() else 0)
        return ",".join(unique_matches)
    return np.nan


def _clean_inspection_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop nulls / invalid values and parse inspection-specific columns."""
    # 1. Inspection Type
    print("  1. Cleaning {Inspection Type}")
    if "Inspection Type" in df.columns:
        before = len(df)
        df = df.dropna(subset=["Inspection Type"]).copy()
        print(f"     Removed {before - len(df)} rows with null Inspection Type")

    # 2. Inspection Date → Year / Month / Day
    print("  2. Cleaning {Inspection Date}")
    if "Inspection Date" in df.columns:
        df["Inspection Date"] = pd.to_datetime(df["Inspection Date"], format="%m/%d/%Y", errors="coerce")
        failures = int(df["Inspection Date"].isna().sum())
        if failures:
            print(f"     {failures} date conversion failures (set to NaT)")
        df["Inspection Year"] = df["Inspection Date"].dt.year
        df["Inspection Month"] = df["Inspection Date"].dt.month
        df["Inspection Day"] = df["Inspection Date"].dt.day
        df = df.drop(columns=["Inspection Date"])
        print(
            f"     Date range: {int(df['Inspection Year'].min())} – {int(df['Inspection Year'].max())}"
        )

    # 3. Risk
    print("  3. Cleaning {Risk}")
    if "Risk" in df.columns:
        before = len(df)
        df = df.dropna(subset=["Risk"])
        all_mask = df["Risk"] == "All"
        df = df[~all_mask].copy()
        print(f"     Removed {before - len(df)} rows (null or 'All')")
        print(f"     Risk unique values: {df['Risk'].unique().tolist()}")

    # 4. Violations → Violation Terms
    print("  4. Cleaning {Violations}")
    if "Violations" in df.columns:
        df["Violation Terms"] = df["Violations"].apply(_extract_violation_terms)
        extracted = int(df["Violation Terms"].notna().sum())
        no_viol = int(df["Violations"].isna().sum())
        print(f"     Extracted terms from {extracted:,} records; {no_viol:,} have no violations")

    return df


# ---------------------------------------------------------------------------
# Part 2 – FD-driven imputation
# ---------------------------------------------------------------------------

def _fd_impute(df: pd.DataFrame, lhs: str, rhs: str) -> pd.DataFrame:
    """
    Use the functional dependency  lhs → rhs  to fill missing *rhs* values.

    For each group of rows sharing the same *lhs* value, if there is a single
    clear majority *rhs* value among the non-null rows it is used to fill the
    nulls in that group.
    """
    if lhs not in df.columns or rhs not in df.columns:
        return df

    def _fill_group(grp):
        if grp[rhs].isna().all():
            return grp
        majority = grp[rhs].dropna().mode()
        if majority.empty:
            return grp
        fill_val = majority.iloc[0]
        grp[rhs] = grp[rhs].fillna(fill_val)
        return grp

    df = df.groupby(lhs, group_keys=False).apply(_fill_group)
    return df


def _apply_fd_cleaning(df: pd.DataFrame, fd_table: pd.DataFrame) -> pd.DataFrame:
    """
    Apply FD-driven imputation for all high-accuracy FDs in *fd_table*
    (accuracy ≥ 90 %).
    """
    if fd_table is None or fd_table.empty:
        return df

    high_acc = fd_table[fd_table["Accuracy (%)"] >= 90].copy()
    high_acc = high_acc.sort_values("Accuracy (%)", ascending=False)

    filled_total = 0
    for _, fd_row in high_acc.iterrows():
        lhs = fd_row["LHS"]
        rhs = fd_row["RHS"]
        if lhs not in df.columns or rhs not in df.columns:
            continue
        before = int(df[rhs].isna().sum())
        df = _fd_impute(df, lhs, rhs)
        after = int(df[rhs].isna().sum())
        filled = before - after
        if filled:
            filled_total += filled
            print(f"     FD {lhs} → {rhs}: filled {filled} missing values")

    print(f"  FD imputation filled {filled_total} values in total.")
    return df


# ---------------------------------------------------------------------------
# Part 3 – Final imputation & cleanup  (from final_cleaning.py)
# ---------------------------------------------------------------------------

def _safe_mode(series: pd.Series):
    """Return the single unambiguous mode of *series* if it appears more than
    once; otherwise return None.  Used for reliable group-based imputation."""
    mode_vals = series.mode()
    if len(mode_vals) == 1 and (series == mode_vals[0]).sum() > 1:
        return mode_vals[0]
    return None


def _fill_by_location_fast(df: pd.DataFrame, group_cols: list, target_col: str) -> pd.DataFrame:
    """Fill NaN in *target_col* using the reliable mode within each location group."""
    mode_series = df.groupby(group_cols)[target_col].transform(_safe_mode)
    mask = df[target_col].isna() & mode_series.notna()
    df.loc[mask, target_col] = mode_series[mask]
    return df


def _fill_geo_by_zip_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Latitude/Longitude using the most common pair per ZIP."""
    def safe_mode_two(series1, series2):
        combined = list(zip(series1, series2))
        if not combined:
            return None, None
        counter = Counter(combined)
        most_common = counter.most_common(1)[0]
        if most_common[1] > 1:
            return most_common[0][0], most_common[0][1]
        return None, None

    zip_geo = (
        df.groupby("Zip", group_keys=False)
        .apply(
            lambda g: pd.Series(
                safe_mode_two(g["Latitude"], g["Longitude"]),
                index=["Lat_mode", "Lon_mode"],
            ),
            include_groups=False,
        )
        .dropna()
    )

    for zip_val, row in zip_geo.iterrows():
        mask = (df["Zip"] == zip_val) & (df["Latitude"].isna() | df["Longitude"].isna())
        df.loc[mask, "Latitude"] = df.loc[mask, "Latitude"].fillna(row["Lat_mode"])
        df.loc[mask, "Longitude"] = df.loc[mask, "Longitude"].fillna(row["Lon_mode"])
    return df


def _final_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback imputation and cleanup after FD-driven step."""
    if "Location" in df.columns:
        df = df.drop("Location", axis=1)

    # Address / Zip by lat-lon
    df = _fill_by_location_fast(df, ["Latitude", "Longitude"], "Address")
    df = _fill_by_location_fast(df, ["Latitude", "Longitude"], "Zip")

    # Lat / Lon by Zip
    df = _fill_geo_by_zip_fast(df)

    # Facility Type by License # — reuse _safe_mode for consistency
    license_mode = df.groupby("License #")["Facility Type"].transform(_safe_mode)
    mask = df["Facility Type"].isna() & license_mode.notna()
    df.loc[mask, "Facility Type"] = license_mode[mask]
    df["Facility Type"] = df["Facility Type"].fillna("Others")

    # AKA Name from DBA Name
    df["AKA Name"] = df["AKA Name"].fillna(df["DBA Name"])

    # City / State defaults
    df["City"] = df["City"].fillna("Chicago")
    df["State"] = df["State"].fillna("IL")

    # Drop rows still missing geo / zip
    before = len(df)
    df = df.dropna(subset=["Latitude", "Longitude", "Zip"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows still missing Latitude/Longitude/Zip")

    # Fill empty Violations / Violation Terms
    df["Violations"] = df["Violations"].fillna("0")
    df["Violation Terms"] = df["Violation Terms"].fillna("0")

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_fd_cleaning(df: pd.DataFrame, fd_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Full FD-based cleaning pipeline:
      1. Inspection-level cleaning (types, dates, risk, violations)
      2. FD-driven imputation using discovered FDs
      3. Fallback imputation + final cleanup

    Parameters
    ----------
    df       : Raw / lightly pre-processed DataFrame.
    fd_table : FD table from fd_detection.run_fd_detection() (may be None).

    Returns
    -------
    Cleaned DataFrame.
    """
    print("  --- Inspection-column cleaning ---")
    df = _clean_inspection_columns(df)

    print("  --- FD-driven imputation ---")
    df = _apply_fd_cleaning(df, fd_table)

    print("  --- Final fallback imputation ---")
    df = _final_imputation(df)

    return df
