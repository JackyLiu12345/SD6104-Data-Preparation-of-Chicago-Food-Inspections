"""
profiling.py
============
Single-column profiling and data cleaning step.

All functions are taken from the Single-column profiling notebook
(Single-column profiling.ipynb) on the main branch.

Cell 0 of the notebook contains the profiling functions
(print_data_overview, print_column_summary, analyze_single_columns, etc.).

Cell 1 of the notebook contains the data cleaning / enrichment functions
(clean_data) which handle Facility Type, Location/City/State deletion,
Zip filtering, License # cleaning, Inspection Type, Risk, Inspection Date
conversion, and Violation Terms extraction.
"""

import os
import re
import pandas as pd
import numpy as np


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
# Pipeline wrapper
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
