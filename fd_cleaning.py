"""
fd_cleaning.py
==============
FD-based data cleaning step.

Uses ``repair_fd`` from the FD discovery notebook for FD-driven repair,
followed by fallback imputation (``final_cleaning``).

Note: Inspection-level cleaning (Inspection Type, Date, Risk, Violations,
Facility Type, Zip, License #) is handled upstream by ``clean_data()`` in
``profiling.py`` — taken from Cell 1 of the Single-column profiling notebook.
"""

import pandas as pd

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
# Final fallback imputation (formerly final_cleaning.py)
# ---------------------------------------------------------------------------

def final_cleaning(df):
    """Fallback imputation & cleanup applied after FD repair.

    Fills remaining nulls (Address/Zip by location, geo by Zip, Facility Type
    by License, AKA Name by DBA Name, City/State defaults) and drops rows
    still missing Latitude/Longitude/Zip.
    """
    df = df.copy()
    if 'Location' in df.columns:
        df = df.drop('Location', axis=1)

    def fill_by_location_fast(df, group_cols, target_col):
        def safe_mode(series):
            mode_vals = series.mode()
            if len(mode_vals) == 1 and (series == mode_vals[0]).sum() > 1:
                return mode_vals[0]
            return None

        mode_series = df.groupby(group_cols)[target_col].transform(safe_mode)
        mask = df[target_col].isna() & mode_series.notna()
        df.loc[mask, target_col] = mode_series[mask]
        return df

    df = fill_by_location_fast(df, ['Latitude', 'Longitude'], 'Address')
    df = fill_by_location_fast(df, ['Latitude', 'Longitude'], 'Zip')

    def fill_geo_by_zip_fast(df):
        from collections import Counter

        def safe_mode_two(series1, series2):
            combined = list(zip(series1, series2))
            if not combined:
                return None, None
            counter = Counter(combined)
            most_common = counter.most_common(1)[0]
            if most_common[1] > 1:
                return most_common[0][0], most_common[0][1]
            return None, None

        zip_geo = df.groupby('Zip', group_keys=False).apply(
            lambda g: pd.Series(safe_mode_two(g['Latitude'], g['Longitude']), index=['Lat_mode', 'Lon_mode']),
            include_groups=False
        ).dropna()

        for zip_val, row in zip_geo.iterrows():
            mask = (df['Zip'] == zip_val) & (df['Latitude'].isna() | df['Longitude'].isna())
            df.loc[mask, 'Latitude'] = df.loc[mask, 'Latitude'].fillna(row['Lat_mode'])
            df.loc[mask, 'Longitude'] = df.loc[mask, 'Longitude'].fillna(row['Lon_mode'])
        return df

    df = fill_geo_by_zip_fast(df)

    # Fill Facility Type by License and then fill Others
    license_mode = df.groupby('License #')['Facility Type'].transform(
        lambda x: x.mode().iloc[0] if len(x.mode()) == 1 and (x == x.mode().iloc[0]).sum() > 1 else None
    )
    mask = df['Facility Type'].isna() & license_mode.notna()
    df.loc[mask, 'Facility Type'] = license_mode[mask]
    df['Facility Type'] = df['Facility Type'].fillna('Others')

    # Fill AKA Name by DBA Name
    df['AKA Name'] = df['AKA Name'].fillna(df['DBA Name'])

    # Fill City / State (if columns still exist — they may have been
    # dropped during single-column profiling cleaning)
    if 'City' in df.columns:
        df['City'] = df['City'].fillna('Chicago')
    if 'State' in df.columns:
        df['State'] = df['State'].fillna('IL')

    # Delete records still with missing values on Longitude, Latitude, Zip
    initial_rows = len(df)
    df = df.dropna(subset=['Latitude', 'Longitude', 'Zip'])
    print(f"Delete records still with missing values on Longitude, Latitude, Zip: {initial_rows - len(df)} rows")

    # Fill missing Violations and Violation Terms by '0'
    df['Violations'] = df['Violations'].fillna('0')
    df['Violation Terms'] = df['Violation Terms'].fillna('0')
    if 'Entity_ID' in df.columns:
        df.drop(['Entity_ID'], axis=1, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_fd_cleaning(df: pd.DataFrame, fd_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    FD-based cleaning pipeline:

      1. FD-driven repair           (repair_fd from FD discovery notebook)
      2. Fallback imputation        (final_cleaning)

    Note: Inspection-level cleaning (Inspection Type, Date, Risk, Violations,
    Facility Type, Zip, License #) is handled upstream by ``clean_data()``
    in ``profiling.py`` — taken from Cell 1 of the Single-column profiling
    notebook on the main branch.

    Parameters
    ----------
    df       : Pre-cleaned DataFrame (already processed by profiling.clean_data).
    fd_table : FD table from fd_detection.run_fd_detection() (informational,
               not used for repair — the repair FDs are predefined in the
               notebook).

    Returns
    -------
    Cleaned DataFrame.
    """
    print("  --- FD-driven repair (repair_fd from FD discovery notebook) ---")
    df = _apply_fd_repair(df)

    print("  --- Final fallback imputation (final_cleaning) ---")
    df = final_cleaning(df)

    return df
