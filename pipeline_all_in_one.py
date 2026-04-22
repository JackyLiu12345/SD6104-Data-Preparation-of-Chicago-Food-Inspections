"""
pipeline_all_in_one.py
======================
Self-contained, all-in-one version of the Chicago Food Inspections
data-preparation pipeline.

This file merges ALL functions and workflow from every module in the
repository into a single script with no local-module imports.  It can be
executed standalone:

    python pipeline_all_in_one.py

The 7-step pipeline:
  Raw CSV
    → Step 1: Single-column profiling + data cleaning
    → Step 2: Association rule mining
    → Step 3: FD detection
    → Step 4: FD repair + final cleaning
    → Step 5: Data structuring (Restaurant + Inspections tables)
    → Step 6: Entity aggregation + high-risk ranking
    → Step 7: Data visualization

Modules merged (in pipeline order):
  profiling.py, association_rules.py, fd_detection.py, fd_cleaning.py,
  restaurant_construction.py, structuring.py, entity_aggregation.py,
  visualization.py, main.py
"""

# ===================================================================
# Consolidated imports (stdlib → third-party)
# ===================================================================
import os
import sys
import re
import ast
from math import radians, sin, cos, sqrt, atan2
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from fuzzywuzzy import fuzz
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt

# ===================================================================
# Constants
# ===================================================================
INPUT_FILE = "Food_Inspections_20240215.csv"


# ###################################################################
#  SECTION 1 — PROFILING  (from profiling.py)
# ###################################################################

# ---------------------------------------------------------------------------
# Functions from Single-column profiling notebook
# ---------------------------------------------------------------------------

def format_value_counts(value_counts, max_items=5):
    """Format value counts for display"""
    if len(value_counts) == 0:
        return "No data"

    result = []
    for i, (value, count) in enumerate(value_counts.items()):
        if i >= max_items:
            break

        # Handle NaN values
        if pd.isna(value):
            value_str = "NaN"
        else:
            value_str = str(value)

        # Truncate long values
        if len(value_str) > 30:
            value_str = value_str[:27] + "..."

        result.append(f"{value_str}: {count:,}")

    return "\n".join(result)


def print_data_overview(df):
    """Print basic overview of the dataset"""
    print("\n" + "=" * 80)
    print("1. DATASET OVERVIEW")
    print("=" * 80)

    num_rows, num_cols = df.shape
    print(f"Dataset Dimensions: {num_rows:,} rows × {num_cols:,} columns")

    return num_rows, num_cols


def print_column_summary(df):
    """Print summary of all columns and their data types"""
    print("\n" + "-" * 80)
    print("2. COLUMN SUMMARY")
    print("-" * 80)

    # Create a formatted table of columns
    print(f"{'No.':<4} {'Column Name':<25} {'Data Type':<15} {'Unique Values':<15} {'Null Values':<12} {'Null %':<8}")
    print("-" * 80)

    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100

        print(f"{i:<4} {col:<25} {dtype:<15} {unique_count:<15,} {null_count:<12,} {null_pct:<8.2f}")


def analyze_single_columns(df, num_rows):
    """Perform detailed analysis of each column"""
    print("\n" + "-" * 80)
    print("3. DETAILED COLUMN ANALYSIS")
    print("-" * 80)

    analysis_results = []

    for column in df.columns:
        col_info = {
            'Column': column,
            'Data_Type': str(df[column].dtype),
            'Unique_Values': df[column].nunique(),
            'Null_Values': df[column].isnull().sum(),
            'Null_Percentage': round((df[column].isnull().sum() / num_rows) * 100, 2)
        }

        # Check if column is numeric
        is_numeric = pd.api.types.is_numeric_dtype(df[column])

        # Handle top 5 values and their counts
        if df[column].nunique() <= 20:
            # 20 or fewer unique values, show all values
            value_counts = df[column].value_counts(dropna=False)
            col_info['Value_Distribution'] = format_value_counts(value_counts)
        else:
            # More than 20 unique values
            if is_numeric:
                # High cardinality numeric column, show statistics
                if df[column].notna().any():
                    stats = f"Range: [{df[column].min():.2f}, {df[column].max():.2f}], "
                    stats += f"Mean: {df[column].mean():.2f}, "
                    stats += f"Std: {df[column].std():.2f}"
                    col_info['Value_Distribution'] = stats
                else:
                    col_info['Value_Distribution'] = 'All Null Values'
            else:
                # High cardinality non-numeric column, show top 5
                top_vals = df[column].value_counts(dropna=False).head(5)
                col_info['Value_Distribution'] = format_value_counts(top_vals)

        analysis_results.append(col_info)

    # Print analysis results in a clean format
    print(f"{'Column':<25} {'Type':<10} {'Unique':<8} {'Null':<6} {'Null%':<7} {'Top Values / Statistics'}")
    print("-" * 80)

    for result in analysis_results:
        col_name = result['Column']
        if len(col_name) > 24:
            col_name = col_name[:21] + "..."

        # Print main row
        print(f"{col_name:<25} {result['Data_Type']:<10} {result['Unique_Values']:<8,} "
              f"{result['Null_Values']:<6,} {result['Null_Percentage']:<6.2f} ", end="")

        # Print value distribution (first line)
        dist_lines = str(result['Value_Distribution']).split('\n')
        if dist_lines:
            print(dist_lines[0][:50] + ("..." if len(dist_lines[0]) > 50 else ""))

        # Print additional lines if needed
        for line in dist_lines[1:]:
            print(" " * 56 + line[:50] + ("..." if len(line) > 50 else ""))

    return analysis_results


# ---------------------------------------------------------------------------
# Data cleaning functions from Cell 1 of Single-column profiling notebook
# ---------------------------------------------------------------------------

def extract_violation_terms(violation_text):
    """
    Extract clause numbers from violation description.
    Improved extraction function that correctly handles multiple violation clauses.

    Taken directly from Cell 1 of the Single-column profiling notebook on
    the main branch.

    Args:
        violation_text (str): Original violation description text

    Returns:
        str: Extracted clause numbers, comma-separated, e.g., "23,37,38"
        np.nan: If no clause numbers are extracted
    """
    if pd.isna(violation_text):
        return np.nan

    text = str(violation_text)

    # 1. First split multiple violation clauses by "|"
    clauses = [clause.strip() for clause in text.split('|')]

    all_matches = []

    for clause in clauses:
        # 2. For each clause, separate main description and Comments part
        main_part = clause

        # Comments markers can have various forms
        comments_markers = [" - Comments:", "Comments:", " - ", "COMMENTS:"]
        for marker in comments_markers:
            if marker in main_part:
                parts = main_part.split(marker, 1)
                if len(parts) > 1:
                    main_part = parts[0]  # Keep only the main description part
                    break

        # 3. Extract clause numbers from the main description part
        # Pattern: Digits at the beginning, followed by a dot, then a space
        pattern = r'^\s*(\d+)\.\s'
        match = re.search(pattern, main_part)
        if match:
            # Extract clause number
            term = match.group(1)
            try:
                # Remove leading zeros
                term_clean = str(int(term))
                all_matches.append(term_clean)
            except ValueError:
                all_matches.append(term)
        else:
            # Backup pattern: Try to match other clause numbers in the main description
            backup_pattern = r'\b(\d+)\.\s+[A-Z]'
            backup_matches = re.findall(backup_pattern, main_part)
            for term in backup_matches:
                try:
                    term_clean = str(int(term))
                    all_matches.append(term_clean)
                except ValueError:
                    all_matches.append(term)

    if all_matches:
        # Remove duplicates and sort
        unique_matches = sorted(set(all_matches), key=lambda x: int(x) if x.isdigit() else 0)
        return ','.join(unique_matches)

    return np.nan


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full data cleaning and enrichment from Cell 1 of the Single-column
    profiling notebook on the main branch.

    Steps (matching the notebook exactly):
      1. Process Facility Type — fill nulls with 'Others'
      2. Delete Location, City, State columns
      3. Process Zip — delete nulls, convert to numeric
      4. Process License # — delete nulls and zeros
      5. Process Inspection Type — delete nulls
      6. Process Risk — delete nulls and 'All'
      7. Process Inspection Date — convert to Year/Month/Day columns
      8. Process Violations — extract Violation Terms

    Parameters
    ----------
    df : Raw DataFrame loaded from the CSV.

    Returns
    -------
    Cleaned DataFrame.
    """
    df = df.copy()

    print("=" * 60)
    print("Starting Data Cleaning")
    print("=" * 60)
    print(f"Original data shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # 1. Process Facility Type column
    print("\n1. Processing Facility Type Column")
    print("-" * 40)
    facility_null_count = df['Facility Type'].isnull().sum()
    df['Facility Type'] = df['Facility Type'].fillna('Others')
    print(f"  Replaced {facility_null_count} null values with 'Others'")
    print(f"  'Others' now has {df['Facility Type'].value_counts().get('Others', 0)} records")

    # 2. Delete Location, City, State columns
    print("\n2. Deleting Location, City, State Columns")
    print("-" * 40)
    cols_to_drop = ['Location', 'City', 'State']
    cols_exist = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_exist)
    print(f"  Deleted columns: {cols_exist}")
    print(f"  Data shape after deletion: {df.shape[0]} rows × {df.shape[1]} columns")

    # 3. Process Zip column
    print("\n3. Processing Zip Column")
    print("-" * 40)

    # Delete null values
    zip_null_count = df['Zip'].isnull().sum()
    df = df.dropna(subset=['Zip'])
    print(f"  a) Deleted {zip_null_count} null values")

    # Convert Zip column to numeric type
    df['Zip'] = pd.to_numeric(df['Zip'], errors='coerce')

    print(f"  b) Unique zip code count: {df['Zip'].nunique()}")
    print(f"  c) Zip code range: {df['Zip'].min()} to {df['Zip'].max()}")

    # 4. Process License # column
    print("\n4. Processing License # Column")
    print("-" * 40)
    original_rows = df.shape[0]

    # Delete null values
    license_null_count = df['License #'].isnull().sum()
    df = df.dropna(subset=['License #'])
    print(f"  a) Deleted {license_null_count} null values")

    # Delete rows with value 0
    df['License #'] = pd.to_numeric(df['License #'], errors='coerce')
    zeros_mask = (df['License #'] == 0)
    zero_count = zeros_mask.sum()
    df = df[~zeros_mask].copy()

    rows_after_license = df.shape[0]
    rows_removed_license = original_rows - rows_after_license
    print(f"  b) Deleted {zero_count} rows with value 0")
    print(f"  c) Total rows deleted: {rows_removed_license}")
    print(f"  d) License # range: {df['License #'].min()} to {df['License #'].max()}")

    # 5. Process Inspection Type column
    print("\n5. Processing Inspection Type Column")
    print("-" * 40)
    original_rows = df.shape[0]

    inspection_type_null_count = df['Inspection Type'].isnull().sum()
    df = df.dropna(subset=['Inspection Type'])

    rows_after_inspection = df.shape[0]
    rows_removed_inspection = original_rows - rows_after_inspection
    print(f"  Deleted {inspection_type_null_count} null values")
    print(f"  Total rows deleted: {rows_removed_inspection}")

    # 6. Process Risk column
    print("\n6. Processing Risk Column")
    print("-" * 40)
    original_rows = df.shape[0]

    risk_null_count = df['Risk'].isnull().sum()
    df = df.dropna(subset=['Risk'])
    print(f"  a) Deleted {risk_null_count} null values")

    # Delete rows with value "All"
    all_mask = (df['Risk'] == 'All')
    all_count = all_mask.sum()
    df = df[~all_mask].copy()

    rows_after_risk = df.shape[0]
    rows_removed_risk = original_rows - rows_after_risk
    print(f"  b) Deleted {all_count} rows with value 'All'")
    print(f"  c) Total rows deleted: {rows_removed_risk}")
    print(f"  d) Risk unique values: {df['Risk'].unique().tolist()}")

    # 7. Process Inspection Date column
    print("\n7. Processing Inspection Date Column")
    print("-" * 40)
    df['Inspection Date'] = pd.to_datetime(df['Inspection Date'], format='%m/%d/%Y', errors='coerce')

    conversion_failures = df['Inspection Date'].isnull().sum()
    if conversion_failures > 0:
        print(f"  Warning: {conversion_failures} date conversion failures")

    # Extract year, month, day
    df['Inspection Year'] = df['Inspection Date'].dt.year
    df['Inspection Month'] = df['Inspection Date'].dt.month
    df['Inspection Day'] = df['Inspection Date'].dt.day

    # Delete original column
    df = df.drop(columns=['Inspection Date'])

    print(f"  Added three columns: Inspection Year, Inspection Month, Inspection Day")
    print(f"  Date range: {df['Inspection Year'].min()} to {df['Inspection Year'].max()}")

    # 8. Process Violations column
    print("\n8. Processing Violations Column")
    print("-" * 40)

    df['Violation Terms'] = df['Violations'].apply(extract_violation_terms)

    extracted_count = df['Violation Terms'].notna().sum()
    no_violation_count = df['Violations'].isna().sum()
    total_rows = df.shape[0]

    print(f"  a) Successfully extracted clause numbers from {extracted_count} records")
    print(f"  b) {no_violation_count} records have no violations")

    # Show cleaned data overview
    print("\n" + "=" * 60)
    print("Cleaning Complete! Data Overview")
    print("=" * 60)
    print(f"Total records: {df.shape[0]:,}")
    print(f"Total columns: {df.shape[1]}")
    print(f"Data type distribution:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"  {dtype}: {count} columns")

    print(f"\nColumn information:")
    for i, col in enumerate(df.columns, 1):
        null_count = df[col].isnull().sum()
        null_pct = null_count / len(df) * 100
        unique_count = df[col].nunique()
        print(f"  {i:2d}. {col:25s} Null: {null_count:6,d} ({null_pct:5.2f}%) Unique: {unique_count:6,d}")

    print("\n" + "=" * 60)
    print("Data Cleaning Process Complete!")
    print("=" * 60)

    return df


# ---------------------------------------------------------------------------
# Profiling pipeline wrapper
# ---------------------------------------------------------------------------

def run_profiling(df: pd.DataFrame, output_path: str = "output/profiling_report.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Profile every column in *df* using the notebook profiling functions,
    then clean the data using the notebook cleaning functions.

    This matches the full workflow of the Single-column profiling notebook
    on the main branch (Cell 0 = profiling, Cell 1 = data cleaning).

    Parameters
    ----------
    df          : Raw DataFrame loaded from the CSV.
    output_path : Where to save the profiling report CSV.

    Returns
    -------
    (profiling_report, cleaned_df) — the profiling report DataFrame and the
    cleaned DataFrame ready for downstream steps.
    """
    # --- Cell 0: Profiling ---
    num_rows, num_cols = print_data_overview(df)
    print_column_summary(df)
    results = analyze_single_columns(df, num_rows)

    # Build a DataFrame report for CSV export
    report = pd.DataFrame(results)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report.to_csv(output_path, index=False)
    print(f"\n  Profiling report saved to: {output_path}")

    high_null = report[report["Null_Percentage"] > 50]
    if not high_null.empty:
        cols_str = ", ".join(high_null["Column"].tolist())
        print(f"  Columns with >50 % nulls: {cols_str}")

    # --- Cell 1: Data cleaning ---
    df_clean = clean_data(df)

    return report, df_clean


# ###################################################################
#  SECTION 2 — ASSOCIATION RULES  (from association_rules.py)
# ###################################################################

# Columns whose values (or tokenised versions) are treated as "items"
_ITEM_COLUMNS = ["Facility Type", "Risk", "Results"]
_VIOLATION_TERMS_COL = "Violation Terms"


def _build_transactions(df: pd.DataFrame) -> list:
    """
    Convert each inspection row into a list of items.
    Items are sourced from Facility Type, Risk, Results, and individual
    violation term numbers extracted from the 'Violation Terms' column.
    """
    transactions = []
    for _, row in df.iterrows():
        items = set()
        for col in _ITEM_COLUMNS:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                items.add(f"{col}={str(val).strip()}")

        vt = row.get(_VIOLATION_TERMS_COL)
        if pd.notna(vt) and str(vt).strip() not in ("", "0"):
            # pd.notna handles actual NaN; the string "nan" can appear when
            # missing values were serialised to text and then read back.
            raw = str(vt).strip()
            if raw.lower() == "nan":
                continue
            # Stored as "1,3,14" or "['1','3','14']"
            if raw.startswith("["):
                try:
                    terms = ast.literal_eval(raw)
                except Exception:
                    terms = [t.strip() for t in raw.strip("[]").split(",")]
            else:
                terms = [t.strip() for t in raw.split(",")]
            for t in terms:
                t = t.strip("' ")
                if t:
                    items.add(f"Violation={t}")

        transactions.append(list(items))
    return transactions


def run_association_rules(
    df: pd.DataFrame,
    min_support: float = 0.02,
    min_confidence: float = 0.3,
    output_path: str = "output/association_rules.csv",
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Mine association rules from *df* using the Apriori algorithm.

    Parameters
    ----------
    df            : Cleaned inspection DataFrame.
    min_support   : Minimum support threshold (fraction of transactions).
    min_confidence: Minimum confidence threshold.
    output_path   : Where to save the discovered rules CSV.
    top_n         : Number of top rules (by lift) to keep in the output.
                    Set to 0 or None to keep all rules.

    Returns
    -------
    DataFrame of association rules sorted by lift (descending).
    """
    print("  Building transaction list ...")
    transactions = _build_transactions(df)

    print(f"  Encoding {len(transactions):,} transactions ...")
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    te_df = pd.DataFrame(te_array, columns=te.columns_)

    print(f"  Running Apriori (min_support={min_support}) ...")
    frequent_itemsets = apriori(te_df, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        print("  No frequent itemsets found — try lowering min_support.")
        rules = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    else:
        print(f"  Found {len(frequent_itemsets)} frequent itemsets. Mining rules ...")
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        # Convert frozensets to readable strings
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
        rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
        cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        rules = rules[cols]
        print(f"  Discovered {len(rules)} association rules.")

    # Keep only the top N rules (by lift) if requested
    if top_n and len(rules) > top_n:
        rules = rules.head(top_n).reset_index(drop=True)
        print(f"  Kept top {top_n} rules (by lift).")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rules.to_csv(output_path, index=False)
    print(f"  Association rules saved to: {output_path}")
    return rules


# ###################################################################
#  SECTION 3 — FD DETECTION  (from fd_detection.py)
# ###################################################################

# ---------------------------------------------------------------------------
# Function from FD discovery notebook
# ---------------------------------------------------------------------------

def compute_fd_confidence(df, lhs, rhs):
    """Compute the confidence of the FD  lhs → rhs.

    Confidence = 1 − (number of LHS groups with >1 distinct RHS value) / total groups.

    Taken directly from the FD discovery notebook on main branch.
    """
    if isinstance(lhs, str):
        lhs = [lhs]
    rhs_counts = df.groupby(lhs)[rhs].nunique()
    violations = (rhs_counts > 1).sum()
    total = len(rhs_counts)
    return round(1 - violations / total, 4)


# ---------------------------------------------------------------------------
# FD detection pipeline helpers
# ---------------------------------------------------------------------------

def discover_fds(df: pd.DataFrame, lhs_cols: list, rhs_cols: list) -> pd.DataFrame:
    """
    Discover functional dependencies of the form  LHS → RHS using
    ``compute_fd_confidence`` from the FD discovery notebook.

    For every (lhs, rhs) pair the function computes the FD confidence
    (= accuracy).  If confidence is 1.0 the FD holds exactly; otherwise
    the violation rate and approximate accuracy are reported.

    Parameters
    ----------
    df       : DataFrame to analyse.
    lhs_cols : List of column names to use as LHS (determinant).
    rhs_cols : List of column names to use as RHS (dependent).

    Returns
    -------
    DataFrame with columns:
        LHS, RHS, Total Groups, Violating Groups, Violation Rate (%),
        Accuracy (%), Holds Exactly
    """
    rows = []
    for lhs in lhs_cols:
        if lhs not in df.columns:
            continue
        for rhs in rhs_cols:
            if rhs not in df.columns or rhs == lhs:
                continue
            sub = df[[lhs, rhs]].dropna()
            if sub.empty:
                continue

            confidence = compute_fd_confidence(sub, lhs, rhs)
            accuracy_pct = round(confidence * 100, 2)

            grouped = sub.groupby(lhs)[rhs].nunique()
            total_groups = len(grouped)
            violating = int((grouped > 1).sum())
            violation_rate = round(violating / total_groups * 100, 2) if total_groups > 0 else 0.0

            rows.append(
                {
                    "LHS": lhs,
                    "RHS": rhs,
                    "Total Groups": total_groups,
                    "Violating Groups": violating,
                    "Violation Rate (%)": violation_rate,
                    "Accuracy (%)": accuracy_pct,
                    "Holds Exactly": violating == 0,
                }
            )

    fd_df = pd.DataFrame(rows).sort_values(["Holds Exactly", "Accuracy (%)"], ascending=[False, False])
    return fd_df.reset_index(drop=True)


_LHS_CANDIDATES = ["License #", "Inspection ID", "Address", "Zip", "Latitude", "Longitude"]
_RHS_CANDIDATES = [
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


def run_fd_detection(
    df: pd.DataFrame, output_path: str = "output/fd_table.csv"
) -> pd.DataFrame:
    """
    Run FD detection on the key column pairs for the Chicago Food Inspections
    dataset and save the result to *output_path*.

    Returns the FD table DataFrame.
    """
    lhs_candidates = _LHS_CANDIDATES
    rhs_candidates = _RHS_CANDIDATES

    print(f"  Testing {len(lhs_candidates)} × {len(rhs_candidates)} LHS/RHS pairs ...")
    fd_table = discover_fds(df, lhs_candidates, rhs_candidates)

    exact = fd_table[fd_table["Holds Exactly"]]
    approx = fd_table[~fd_table["Holds Exactly"] & (fd_table["Accuracy (%)"] >= 90)]
    print(f"  Exact FDs found   : {len(exact)}")
    print(f"  Approx FDs (≥90%) : {len(approx)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fd_table.to_csv(output_path, index=False)
    print(f"  FD table saved to: {output_path}")
    return fd_table


# ###################################################################
#  SECTION 4 — FD CLEANING  (from fd_cleaning.py, including final_cleaning)
# ###################################################################

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
# FD cleaning public API
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


# ###################################################################
#  SECTION 5 — RESTAURANT CONSTRUCTION  (from restaurant_construction.py)
# ###################################################################

def generate_union_keys_df(df: pd.DataFrame) -> None:
    agg_df = df.groupby(
        ['License #', 'DBA Name', 'Zip', 'Latitude', 'Longitude', 'Facility Type'],
        as_index=False
    ).size()
    agg_df.rename(columns={'size': 'cnt'}, inplace=True)

    license_total = agg_df.groupby('License #')['cnt'].sum().reset_index()
    license_total.rename(columns={'cnt': 'total_records'}, inplace=True)
    agg_df = agg_df.merge(license_total, on='License #', how='left')
    del license_total

    license_info_diff_cnt = agg_df.groupby('License #')['cnt'].count().reset_index()
    license_info_diff_cnt.rename(columns={'cnt': 'diff_info_cnt'}, inplace=True)
    agg_df = agg_df.merge(license_info_diff_cnt, on='License #', how='left')
    del license_info_diff_cnt

    agg_df = agg_df.sort_values(['total_records', 'cnt'], ascending=False)
    prepared_agg_df = agg_df[agg_df['diff_info_cnt'] > 1].copy()
    prepared_agg_df0 = agg_df[agg_df['diff_info_cnt'] == 1].copy()

    return prepared_agg_df, prepared_agg_df0


def normalize_name(name):
    if pd.isna(name):
        return ''
    normalized = re.sub(r'[^A-Z0-9]', '', str(name).upper())
    return normalized


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def distance_similarity(d):
    if d == 0:
        return 1.0
    elif d <= 100:
        return 0.9
    elif d <= 500:
        return 0.9 - 0.3 * (d - 100) / 400
    elif d <= 5000:
        return 0.6 * ((5000 - d) / 4500) ** 2
    else:
        return 0.0


def compute_pair_similarity(r1, r2):
    # DBA Name
    norm1 = normalize_name(r1['DBA Name'])
    norm2 = normalize_name(r2['DBA Name'])
    if norm1 and norm2:
        sim_dba = fuzz.token_set_ratio(norm1, norm2) / 100.0
    else:
        sim_dba = 0.0

    # Zip exact match
    zip1 = str(r1['Zip']) if pd.notna(r1['Zip']) else None
    zip2 = str(r2['Zip']) if pd.notna(r2['Zip']) else None
    sim_zip = 1.0 if (zip1 is not None and zip2 is not None and zip1 == zip2) else 0.0

    # Geographic similarity
    lat1, lon1 = r1.get('Latitude'), r1.get('Longitude')
    lat2, lon2 = r2.get('Latitude'), r2.get('Longitude')
    valid1 = pd.notna(lat1) and pd.notna(lon1)
    valid2 = pd.notna(lat2) and pd.notna(lon2)
    if valid1 and valid2:
        try:
            lat1_f, lon1_f = float(lat1), float(lon1)
            lat2_f, lon2_f = float(lat2), float(lon2)
            dist = haversine(lat1_f, lon1_f, lat2_f, lon2_f)
            if dist < 0.1:
                dist = 0.0
            sim_geo = distance_similarity(dist)
        except (ValueError, TypeError):
            sim_geo = 0.0
    else:
        sim_geo = 0.0

    # Facility Type exact match (case-insensitive)
    ft1 = str(r1['Facility Type']).strip().lower() if pd.notna(r1['Facility Type']) else ''
    ft2 = str(r2['Facility Type']).strip().lower() if pd.notna(r2['Facility Type']) else ''
    sim_fac = 1.0 if (ft1 and ft2 and ft1 == ft2) else 0.0

    # Weighted sum
    sim_total = 0.55 * sim_dba + 0.2 * sim_zip + 0.2 * sim_geo + 0.05 * sim_fac
    return sim_total


def build_similarity_matrices(prepared_agg_df):
    similarity_dict = {}
    for license_id, group in prepared_agg_df.groupby('License #'):
        records = group.to_dict('records')
        n = len(records)
        if n < 2:
            similarity_dict[license_id] = np.array([[1.0]])
            continue

        mat = np.eye(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                sim = compute_pair_similarity(records[i], records[j])
                mat[i, j] = sim
                mat[j, i] = sim

        similarity_dict[license_id] = mat

    return similarity_dict


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1


def cluster_by_similarity_matrix(similarity_matrix, license_id, threshold=0.5):
    n = similarity_matrix.shape[0]
    if n == 0:
        return {}
    try:
        lic_int = int(license_id)
    except (ValueError, TypeError):
        lic_int = str(license_id)

    if n == 1:
        return {0: f"{lic_int}_1"}

    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                uf.union(i, j)

    comp_map = defaultdict(list)
    for idx in range(n):
        root = uf.find(idx)
        comp_map[root].append(idx)

    sorted_comps = sorted(comp_map.values(), key=lambda x: -len(x))
    node_to_entity = {}
    for seq, comp_nodes in enumerate(sorted_comps, start=1):
        entity_id = f"{lic_int}_{seq}"
        for node_idx in comp_nodes:
            node_to_entity[node_idx] = entity_id
    return node_to_entity


def assign_entity_ids_to_prepared_df(prepared_agg_df, sim_matrices, threshold=0.5):
    df_out = prepared_agg_df.copy().reset_index(drop=True)
    df_out['_temp_idx'] = df_out.index
    entity_id_list = [None] * len(df_out)

    for license_id, group in df_out.groupby('License #'):
        if license_id not in sim_matrices:
            n_nodes = len(group)
            if n_nodes == 1:
                try:
                    lic_int = int(license_id)
                except Exception:
                    lic_int = license_id
                entity_id = f"{lic_int}_1"
                for orig_idx in group['_temp_idx']:
                    entity_id_list[orig_idx] = entity_id
            else:
                print(f"Warning: License {license_id} has no similarity matrix but has {n_nodes} nodes; skipping.")
            continue

        mat = sim_matrices[license_id]
        if mat.shape[0] != len(group):
            print(f"Warning: License {license_id} matrix size {mat.shape[0]} does not match record count {len(group)}; skipping.")
            continue

        node_to_entity = cluster_by_similarity_matrix(mat, license_id, threshold)

        for local_idx, (_, row) in enumerate(group.iterrows()):
            orig_idx = row['_temp_idx']
            entity_id = node_to_entity.get(local_idx)
            if entity_id:
                entity_id_list[orig_idx] = entity_id
            else:
                try:
                    lic_int = int(license_id)
                except Exception:
                    lic_int = license_id
                entity_id_list[orig_idx] = f"{lic_int}_UNKNOWN"

    df_out['Entity_ID'] = entity_id_list
    df_out.drop(columns=['_temp_idx'], inplace=True)
    return df_out


def add_standardized_columns(df_with_entity,
                             entity_col='Entity_ID',
                             weight_col='cnt',
                             attr_cols=None,
                             suffix='_std'):
    if attr_cols is None:
        attr_cols = ['DBA Name', 'Facility Type', 'Zip', 'Latitude', 'Longitude']
    df_out = df_with_entity.copy()
    idx_max = df_out.groupby(entity_col)[weight_col].idxmax()
    std_values = df_out.loc[idx_max, [entity_col] + attr_cols].copy()
    std_values = std_values.drop_duplicates(subset=entity_col, keep='first')
    rename_dict = {col: col + suffix for col in attr_cols}
    std_values = std_values.rename(columns=rename_dict)
    df_out = df_out.merge(std_values, on=entity_col, how='left')
    return df_out


def unify_zip_by_location(df,
                          lon_col='Longitude',
                          lat_col='Latitude',
                          cleaning_col='Zip'):
    df = df.copy()
    df[lon_col] = df[lon_col].round(6)
    df[lat_col] = df[lat_col].round(6)
    df[cleaning_col] = df.groupby([lon_col, lat_col])[cleaning_col].transform(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    )
    return df


def restaurant_cleaning(df):
    print(f'input shape:{df.shape}')
    prepared_agg_df, prepared_agg_df0 = generate_union_keys_df(df)
    print(f'Union Keys:{prepared_agg_df.shape[0] + prepared_agg_df0.shape[0]}')
    print(f'{prepared_agg_df0.shape[0]} Union Keys is unique for one License')
    print(f'{prepared_agg_df.shape[0]} Union Keys is not unique for one License')
    sim_matrices = build_similarity_matrices(prepared_agg_df)
    print(f"Generate {len(sim_matrices)} Similarity Matrices for Licenses")

    prepared_with_entity = assign_entity_ids_to_prepared_df(prepared_agg_df, sim_matrices, threshold=0.5)
    print(f'After add Entity_ID, the shape of Union_Key Table is {prepared_with_entity.shape}')
    restaurant = pd.concat([prepared_agg_df0, prepared_with_entity], axis=0)
    mask = restaurant['Entity_ID'].isna()

    def _safe_license_entity_id(x):
        try:
            return f"{int(x)}_0"
        except (ValueError, TypeError):
            print(f"Warning: could not cast License # {x!r} to int; using '{x}_0' as Entity_ID.")
            return f"{x}_0"

    restaurant.loc[mask, 'Entity_ID'] = restaurant.loc[mask, 'License #'].apply(_safe_license_entity_id)
    del mask
    restaurant_with_std = add_standardized_columns(restaurant,
                                                   entity_col='Entity_ID',
                                                   weight_col='cnt',
                                                   attr_cols=['DBA Name', 'Facility Type', 'Zip', 'Latitude',
                                                              'Longitude'],
                                                   suffix='_std')
    restaurant_with_std = unify_zip_by_location(restaurant_with_std, lon_col='Longitude_std', lat_col='Latitude_std',
                                                cleaning_col='Zip_std')
    return restaurant_with_std


# ###################################################################
#  SECTION 6 — STRUCTURING  (from structuring.py)
# ###################################################################

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
# Structuring helpers
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
# Structuring public API
# ---------------------------------------------------------------------------

def run_structuring(
    df: pd.DataFrame,
    restaurant_output: str = "output/restaurant_table.csv",
    inspections_output: str = "output/inspections_table.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    (restaurant_table, inspections_table, df_merged) as DataFrames.
    ``df_merged`` is the inspection-level DataFrame with Entity_ID attached,
    useful for downstream entity-level aggregation.
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

    return restaurant_table, inspections_table, df_merged


# ###################################################################
#  SECTION 7 — ENTITY AGGREGATION  (from entity_aggregation.py)
# ###################################################################

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
    # City may have been dropped during single-column profiling (Step 1),
    # so it is conditionally included here.
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
# Entity aggregation pipeline wrapper
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


# ###################################################################
#  SECTION 8 — VISUALIZATION  (from visualization.py)
# ###################################################################

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
    if df is None or df.empty:
        print("  plot_missing_value_percentage: DataFrame is empty; skipping.")
        return pd.Series(dtype=float)
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    ax.barh(missing_pct.index, missing_pct.values)
    ax.set_xlabel("Missing Percentage (%)")
    ax.set_ylabel("Columns")
    ax.set_title("Missing Value Percentage by Column")
    apply_white_theme(ax)
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="gray")

    output_path = os.path.join(output_dir, "missing_value_percentage.png")
    os.makedirs(output_dir, exist_ok=True)
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
    if counts.empty:
        print(f"  plot_category_distribution: no data for column '{column}'; skipping.")
        return counts

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
    os.makedirs(output_dir, exist_ok=True)
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

    if not rows:
        print("  plot_fd_confidence_ranking: no applicable FD pairs found in DataFrame; skipping.")
        return pd.DataFrame(columns=["FD", "Confidence"])

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
    os.makedirs(output_dir, exist_ok=True)
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

    if not rows:
        print("  plot_fd_violation_counts: no applicable FD pairs found in DataFrame; skipping.")
        return pd.DataFrame(columns=["FD", "Violating Groups", "Total Groups"])

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
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="white", transparent=False)
    plt.close()
    print(f"  Saved: {output_path}")
    return fd_violation_df


# ---------------------------------------------------------------------------
# Visualization pipeline wrapper
# ---------------------------------------------------------------------------

_VIZ_FD_LIST = [
    ("Address", "Zip"),
    ("License #", "Address"),
    ("License #", "Zip"),
    ("License #", "Risk"),
    ("License #", "Facility Type"),
]


def run_visualization(df, output_dir="output", pic_subdir="pic"):
    """
    Generate all standard visualizations for the cleaned DataFrame.

    All PNGs are written into ``<output_dir>/<pic_subdir>/`` (default
    ``output/pic/``) so picture artefacts are kept separate from CSVs.

    Produces:
      • missing_value_percentage.png
      • category_distribution_results.png
      • category_distribution_risk.png
      • category_distribution_facility_type.png
      • fd_confidence_ranking.png
      • fd_violation_counts.png

    Parameters
    ----------
    df          : Cleaned inspection DataFrame.
    output_dir  : Root output directory.
    pic_subdir  : Sub-directory (under ``output_dir``) for PNG files.
    """
    pic_dir = os.path.join(output_dir, pic_subdir) if pic_subdir else output_dir
    os.makedirs(pic_dir, exist_ok=True)

    # Normalise text columns for consistent labels
    text_cols = ["DBA Name", "AKA Name", "Address", "Facility Type",
                 "Risk", "Results", "Inspection Type"]
    viz_df = df.copy()
    for col in text_cols:
        if col in viz_df.columns:
            viz_df[col] = viz_df[col].astype(str).str.strip().str.upper()
    if "Zip" in viz_df.columns:
        viz_df["Zip"] = pd.to_numeric(viz_df["Zip"], errors="coerce")

    plot_missing_value_percentage(viz_df, pic_dir)
    plot_category_distribution(viz_df, "Results", top_n=10, output_dir=pic_dir)
    plot_category_distribution(viz_df, "Risk", top_n=10, output_dir=pic_dir)
    plot_category_distribution(viz_df, "Facility Type", top_n=15, output_dir=pic_dir)
    plot_fd_confidence_ranking(viz_df, _VIZ_FD_LIST, output_dir=pic_dir)
    plot_fd_violation_counts(viz_df, _VIZ_FD_LIST, output_dir=pic_dir)

    print(f"  All visualization charts generated in {pic_dir}/")


# ###################################################################
#  SECTION 9 — MAIN PIPELINE  (from main.py)
# ###################################################################

def _step_header(n: int, title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  Step {n}: {title}")
    print(bar)


def main():
    # ------------------------------------------------------------------
    # 0. Load raw data
    # ------------------------------------------------------------------
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        print("Please place the raw CSV in the repository root and re-run.")
        sys.exit(1)

    print("Loading raw data ...")
    try:
        df_raw = pd.read_csv(INPUT_FILE, low_memory=False)
    except (OSError, pd.errors.ParserError) as exc:
        print(f"[ERROR] Failed to read input CSV '{INPUT_FILE}': {exc}")
        sys.exit(1)
    if "License #" in df_raw.columns:
        df_raw["License #"] = df_raw["License #"].fillna(0)
    print(f"Raw shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

    # ------------------------------------------------------------------
    # Step 0b: Data visualization — BEFORE cleaning
    #
    # Plot the raw DataFrame so we can compare against the post-pipeline
    # state and see the effect of the cleaning pipeline visually.
    # ------------------------------------------------------------------
    _step_header(0, "Data visualization — BEFORE cleaning")
    run_visualization(df_raw, output_dir="output", pic_subdir="pic/before")

    # ------------------------------------------------------------------
    # Step 1: Single-column profiling + data cleaning
    #
    # Both profiling and data cleaning come from the Single-column
    # profiling notebook on the main branch (Cell 0 and Cell 1).
    # ------------------------------------------------------------------
    _step_header(1, "Single-column profiling + data cleaning")
    _profiling_report, df_clean = run_profiling(df_raw, output_path="output/profiling_report.csv")
    print(f"  Cleaned shape: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")

    # ------------------------------------------------------------------
    # Step 2: Association rule mining
    # ------------------------------------------------------------------
    _step_header(2, "Association rule mining")
    run_association_rules(
        df_clean,
        min_support=0.02,
        min_confidence=0.3,
        output_path="output/association_rules.csv",
        top_n=50,
    )

    # ------------------------------------------------------------------
    # Step 3: FD detection
    # ------------------------------------------------------------------
    _step_header(3, "Functional Dependency (FD) detection")
    fd_table = run_fd_detection(df_clean, output_path="output/fd_table.csv")

    # ------------------------------------------------------------------
    # Step 4: Data cleaning (FD repair + final cleaning)
    #
    # Note: Inspection-level cleaning (Inspection Type, Date, Risk,
    # Violations, Facility Type, Zip, License #) was already done in
    # Step 1 by clean_data() from the Single-column profiling notebook.
    # This step only applies FD repair and final imputation.
    # ------------------------------------------------------------------
    _step_header(4, "FD repair + final cleaning")
    df_clean = run_fd_cleaning(df_clean.copy(), fd_table=fd_table)
    print(f"  Cleaned shape: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
    remaining_nulls = df_clean.isna().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    if not remaining_nulls.empty:
        print("  Remaining nulls after cleaning:")
        print(remaining_nulls.to_string(header=False))

    # ------------------------------------------------------------------
    # Step 5: Data structuring (Restaurant + Inspections tables)
    # ------------------------------------------------------------------
    _step_header(5, "Data structuring (Restaurant + Inspections tables)")
    restaurant_table, inspections_table, df_merged = run_structuring(
        df_clean,
        restaurant_output="output/restaurant_table.csv",
        inspections_output="output/inspections_table.csv",
    )

    # ------------------------------------------------------------------
    # Step 6: Entity-level aggregation + high-risk ranking
    # ------------------------------------------------------------------
    _step_header(6, "Entity aggregation + high-risk ranking")
    entity_df, high_risk_df = run_entity_aggregation(
        df_merged,
        entity_output="output/entity_inspection_analysis.csv",
        risk_output="output/entity_high_risk_rank.csv",
        min_inspections=5,
        top_n=100,
    )

    # ------------------------------------------------------------------
    # Step 7: Data visualization — AFTER cleaning
    # ------------------------------------------------------------------
    _step_header(7, "Data visualization — AFTER cleaning")
    run_visualization(df_clean, output_dir="output", pic_subdir="pic/after")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Pipeline complete — outputs written to output/")
    print("=" * 60)
    print(f"  Profiling report  : output/profiling_report.csv")
    print(f"  Association rules : output/association_rules.csv")
    print(f"  FD table          : output/fd_table.csv")
    print(f"  Restaurant table  : output/restaurant_table.csv  ({len(restaurant_table):,} rows)")
    print(f"  Inspections table : output/inspections_table.csv ({len(inspections_table):,} rows)")
    print(f"  Entity analysis   : output/entity_inspection_analysis.csv ({len(entity_df):,} rows)")
    print(f"  High-risk ranking : output/entity_high_risk_rank.csv ({len(high_risk_df):,} rows)")
    print(f"  Visualizations    : output/pic/before/*.png, output/pic/after/*.png")


if __name__ == "__main__":
    main()
