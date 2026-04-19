"""
structuring.py
==============
Data structuring step: extract a normalised Restaurant table (with a unique
Restaurant_ID) and an Inspections table that references Restaurant_ID.

Entity resolution is delegated to restaurant_construction.py (from main branch).
The merge function ``join_infection`` is taken from the original main.py on
the main branch.
"""

import os
import pandas as pd
from restaurant_construction import restaurant_cleaning


# ---------------------------------------------------------------------------
# Function from the original main.py on the main branch
# ---------------------------------------------------------------------------

def join_infection(restaurant_std, infection_df, join_cols):
    """Merge standardised restaurant attributes and Entity_ID back into the
    inspection-level DataFrame.

    Taken directly from the original main.py on the main branch.
    """
    right_unique = restaurant_std.sort_values('cnt', ascending=False).drop_duplicates(subset=join_cols, keep='first')
    df = infection_df.merge(right_unique, on=join_cols, how='left')
    mask = df['cnt'].notna()
    for col in ['DBA Name', 'Zip', 'Latitude', 'Longitude', 'Facility Type']:
        std_col = col + '_std'
        if std_col in df.columns:
            df.loc[mask, col] = df.loc[mask, std_col]
            df = df.drop(std_col, axis=1)
    drop_cols = ['cnt', 'total_records', 'diff_info_cnt']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESTAURANT_COLS = [
    "License #",
    "DBA Name",
    "AKA Name",
    "Facility Type",
    "Address",
    "City",
    "State",
    "Zip",
    "Latitude",
    "Longitude",
]

_INSPECTION_COLS = [
    "Inspection Type",
    "Risk",
    "Results",
    "Violations",
    "Violation Terms",
    "Inspection Year",
    "Inspection Month",
    "Inspection Day",
]


_JOIN_COLS = ["License #", "DBA Name", "Zip", "Latitude", "Longitude", "Facility Type"]


def _join_standardised(restaurant_std: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the standardised restaurant attributes and Entity_ID back into the
    inspection-level DataFrame.  Delegates to ``join_infection`` from the
    original main.py on the main branch.
    """
    return join_infection(restaurant_std, df, _JOIN_COLS)


def _build_restaurant_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Restaurant table from the merged DataFrame.

    Each unique Entity_ID becomes one row.  The canonical attribute values
    (most frequent row per entity) are used.  A sequential Restaurant_ID is
    assigned.
    """
    avail = [c for c in _RESTAURANT_COLS if c in df.columns] + ["Entity_ID"]
    rest = df[avail].copy()

    # Keep one representative row per entity (highest-occurrence row is
    # already selected via the standardised columns; just deduplicate here).
    rest = rest.drop_duplicates(subset=["Entity_ID"], keep="first")
    rest = rest.sort_values("Entity_ID").reset_index(drop=True)
    rest.insert(0, "Restaurant_ID", range(1, len(rest) + 1))
    return rest


def _build_inspections_table(df: pd.DataFrame, restaurant_table: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Inspections table with Restaurant_ID as a foreign key.
    """
    # Map Entity_ID → Restaurant_ID
    entity_to_rid = restaurant_table.set_index("Entity_ID")["Restaurant_ID"].to_dict()
    df = df.copy()
    df["Restaurant_ID"] = df["Entity_ID"].map(entity_to_rid)

    avail_insp = [c for c in _INSPECTION_COLS if c in df.columns]
    inspections = df[["Restaurant_ID"] + avail_insp].copy()
    inspections.insert(0, "Inspection_ID", range(1, len(inspections) + 1))
    return inspections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_structuring(
    df: pd.DataFrame,
    restaurant_output: str = "output/restaurant_table.csv",
    inspections_output: str = "output/inspections_table.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract the Restaurant and Inspections tables from the cleaned DataFrame.

    Steps
    -----
    1. Run entity resolution (restaurant_construction.restaurant_cleaning).
    2. Merge standardised attributes back into the inspection rows.
    3. Build Restaurant table (one row per unique entity, with Restaurant_ID).
    4. Build Inspections table (one row per inspection, foreign key Restaurant_ID).
    5. Save both tables and return them.

    Returns
    -------
    (restaurant_table, inspections_table) as DataFrames.
    """
    print("  Running entity resolution ...")
    restaurant_std = restaurant_cleaning(df)
    print(f"  Entity table shape: {restaurant_std.shape}")

    print("  Merging standardised attributes ...")
    df_merged = _join_standardised(restaurant_std, df)

    print("  Building Restaurant table ...")
    restaurant_table = _build_restaurant_table(df_merged)
    print(f"  Restaurant table: {len(restaurant_table):,} unique restaurants")

    print("  Building Inspections table ...")
    inspections_table = _build_inspections_table(df_merged, restaurant_table)
    print(f"  Inspections table: {len(inspections_table):,} inspection records")

    # Drop Entity_ID from Restaurant table (internal key, not needed in output)
    restaurant_table = restaurant_table.drop(columns=["Entity_ID"], errors="ignore")

    os.makedirs(os.path.dirname(restaurant_output), exist_ok=True)
    restaurant_table.to_csv(restaurant_output, index=False)
    print(f"  Restaurant table saved to: {restaurant_output}")

    os.makedirs(os.path.dirname(inspections_output), exist_ok=True)
    inspections_table.to_csv(inspections_output, index=False)
    print(f"  Inspections table saved to: {inspections_output}")

    return restaurant_table, inspections_table
