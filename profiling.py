"""
profiling.py
============
Single-column profiling step.

Functions are taken from the Single-column profiling notebook
(Single-column profiling.ipynb) on the main branch.
"""

import os
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
# Pipeline wrapper
# ---------------------------------------------------------------------------

def run_profiling(df: pd.DataFrame, output_path: str = "output/profiling_report.csv") -> pd.DataFrame:
    """
    Profile every column in *df* using the notebook profiling functions,
    print a summary, and save the report to *output_path*.

    Returns the profiling report DataFrame.
    """
    num_rows, num_cols = print_data_overview(df)
    print_column_summary(df)
    results = analyze_single_columns(df, num_rows)

    # Also build a DataFrame report for CSV export
    report = pd.DataFrame(results)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report.to_csv(output_path, index=False)
    print(f"\n  Profiling report saved to: {output_path}")

    high_null = report[report["Null_Percentage"] > 50]
    if not high_null.empty:
        cols_str = ", ".join(high_null["Column"].tolist())
        print(f"  Columns with >50 % nulls: {cols_str}")

    return report
