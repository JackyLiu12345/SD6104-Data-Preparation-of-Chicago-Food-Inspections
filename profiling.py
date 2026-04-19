import pandas as pd
import numpy as np
import os


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-column profiling statistics:
      - dtype, null count/%, unique count, top-5 values (with frequencies),
        and basic numeric stats (mean, std, min, 25%, 50%, 75%, max).
    Returns a DataFrame with one row per column.
    """
    rows = []
    n = len(df)
    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = round(null_count / n * 100, 2) if n > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))
        dtype = str(series.dtype)

        # Top-5 most frequent values
        vc = series.dropna().value_counts().head(5)
        top_values = "; ".join(f"{v}({c})" for v, c in vc.items())

        row = {
            "Column": col,
            "DType": dtype,
            "Null Count": null_count,
            "Null %": null_pct,
            "Unique Count": unique_count,
            "Top 5 Values (count)": top_values,
            "Mean": np.nan,
            "Std": np.nan,
            "Min": np.nan,
            "25%": np.nan,
            "50%": np.nan,
            "75%": np.nan,
            "Max": np.nan,
        }

        # Numeric stats
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            row["Mean"] = round(float(desc.get("mean", np.nan)), 4)
            row["Std"] = round(float(desc.get("std", np.nan)), 4)
            row["Min"] = round(float(desc.get("min", np.nan)), 4)
            row["25%"] = round(float(desc.get("25%", np.nan)), 4)
            row["50%"] = round(float(desc.get("50%", np.nan)), 4)
            row["75%"] = round(float(desc.get("75%", np.nan)), 4)
            row["Max"] = round(float(desc.get("max", np.nan)), 4)

        rows.append(row)

    return pd.DataFrame(rows)


def run_profiling(df: pd.DataFrame, output_path: str = "output/profiling_report.csv") -> pd.DataFrame:
    """
    Profile every column in *df*, print a summary, and save the report to
    *output_path*.  Returns the profiling report DataFrame.
    """
    print("  Computing per-column statistics ...")
    report = profile_dataframe(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report.to_csv(output_path, index=False)
    print(f"  Profiling report saved to: {output_path}")

    # Brief console summary
    print(f"\n  Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    high_null = report[report["Null %"] > 50]
    if not high_null.empty:
        cols_str = ", ".join(high_null["Column"].tolist())
        print(f"  Columns with >50 % nulls: {cols_str}")

    return report
