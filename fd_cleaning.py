"""
fd_cleaning.py
==============
FD-based data cleaning step.

Delegates to the original main-branch modules and uses ``repair_fd``
from the FD discovery notebook:

  • Inspection-level cleaning   → inspection_cleaning.clean_inspection
  • FD-driven repair            → repair_fd (from FD discovery notebook)
  • Final imputation & cleanup  → final_cleaning.final_cleaning
"""

import pandas as pd

from inspection_cleaning import clean_inspection
from final_cleaning import final_cleaning
from fd_detection import compute_fd_confidence


# ---------------------------------------------------------------------------
# FD repair — from FD discovery notebook (FD discovery.ipynb)
# ---------------------------------------------------------------------------

# FDs selected for repair, ordered to handle upstream dependencies first
# (from the FD discovery notebook on main branch)
REPAIR_FDS = [
    ('Address',    'Zip'),
    ('License #',  'Address'),
    ('License #',  'Zip'),
    ('License #',  'Risk'),
    ('License #',  'Facility Type'),
]

MAX_RHS_CARDINALITY = 3


def repair_fd(df, lhs, rhs, max_rhs_cardinality=3):
    """Repair violations of the FD  lhs → rhs  by standardising minority
    RHS values to the majority value within each LHS group.

    Only groups where the RHS cardinality is between 2 and
    *max_rhs_cardinality* (inclusive) are repaired — groups with very
    high cardinality are skipped to avoid misrepair.

    Taken directly from the FD discovery notebook on main branch.
    """
    if isinstance(lhs, str):
        lhs_cols = [lhs]
    else:
        lhs_cols = list(lhs)

    rhs_cardinality = df.groupby(lhs_cols)[rhs].nunique()
    repair_targets = rhs_cardinality[
        (rhs_cardinality > 1) & (rhs_cardinality <= max_rhs_cardinality)
    ].index

    df_out = df.copy()
    change_log = []

    for lhs_val in repair_targets:
        if len(lhs_cols) == 1:
            mask = df_out[lhs_cols[0]] == lhs_val
        else:
            mask = (df_out[lhs_cols] == pd.Series(dict(zip(lhs_cols, lhs_val)))).all(axis=1)

        subset_rhs = df_out.loc[mask, rhs]
        majority_val = subset_rhs.value_counts().idxmax()
        minority_mask = mask & (df_out[rhs] != majority_val)
        n_changed = minority_mask.sum()

        if n_changed > 0:
            old_vals = df_out.loc[minority_mask, rhs].value_counts().to_dict()
            df_out.loc[minority_mask, rhs] = majority_val
            change_log.append({
                'FD':              f"{'+'.join(lhs_cols)} -> {rhs}",
                'LHS Value':       lhs_val if len(lhs_cols) == 1 else str(lhs_val),
                'RHS Column':      rhs,
                'Standardised To': majority_val,
                'Replaced Values': str(old_vals),
                'Rows Repaired':   n_changed,
            })

    return df_out, pd.DataFrame(change_log)


def _apply_fd_repair(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply FD repair for the pre-defined list of FDs using ``repair_fd``
    from the FD discovery notebook.

    Normalises text fields first (as in the notebook), then repairs each FD
    in order, printing a summary.
    """
    # Normalise text fields before FD repair so groupings are case-consistent
    # (from the FD discovery notebook)
    text_cols = ['DBA Name', 'AKA Name', 'Address', 'Facility Type', 'Risk']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()

    if 'Zip' in df.columns:
        df['Zip'] = df['Zip'].astype('Int64').astype(str)

    summary_rows = []
    all_change_logs = []

    for lhs, rhs in REPAIR_FDS:
        if lhs not in df.columns or rhs not in df.columns:
            continue
        conf_before = compute_fd_confidence(df, lhs, rhs)
        df, changes = repair_fd(df, lhs, rhs, MAX_RHS_CARDINALITY)
        conf_after = compute_fd_confidence(df, lhs, rhs)

        n_lhs_repaired = len(changes)
        n_rows_repaired = changes['Rows Repaired'].sum() if n_lhs_repaired > 0 else 0

        summary_rows.append({
            'FD':                  f"{lhs} -> {rhs}",
            'Confidence Before':   conf_before,
            'Confidence After':    conf_after,
            'Delta':               round(conf_after - conf_before, 4),
            'LHS Values Repaired': n_lhs_repaired,
            'Rows Repaired':       n_rows_repaired,
        })

        if n_lhs_repaired > 0:
            all_change_logs.append(changes)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))

    total_rows_changed = summary_df['Rows Repaired'].sum() if not summary_df.empty else 0
    print(f"\n  Total rows modified by FD repair: {int(total_rows_changed):,}")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_fd_cleaning(df: pd.DataFrame, fd_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Full FD-based cleaning pipeline using functions from the main branch:

      1. Inspection-level cleaning  (clean_inspection from inspection_cleaning.py)
      2. FD-driven repair           (repair_fd from FD discovery notebook)
      3. Fallback imputation        (final_cleaning from final_cleaning.py)

    Parameters
    ----------
    df       : Raw / lightly pre-processed DataFrame.
    fd_table : FD table from fd_detection.run_fd_detection() (informational,
               not used for repair — the repair FDs are predefined in the
               notebook).

    Returns
    -------
    Cleaned DataFrame.
    """
    print("  --- Inspection-column cleaning (clean_inspection) ---")
    df = clean_inspection(df)

    print("  --- FD-driven repair (repair_fd from FD discovery notebook) ---")
    df = _apply_fd_repair(df)

    print("  --- Final fallback imputation (final_cleaning) ---")
    df = final_cleaning(df)

    return df
