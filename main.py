"""
main.py
=======
Orchestrator for the Chicago Food Inspections data-preparation pipeline.

The pipeline follows the same flow as the original main.py on the main
branch, augmented with profiling, association-rule mining, and FD
detection/repair steps.

Pipeline
--------
  Raw CSV
    → Step 1: Single-column profiling     → output/profiling_report.csv
    → Step 2: Association rule mining     → output/association_rules.csv
    → Step 3: FD detection                → output/fd_table.csv
    → Step 4: Data cleaning
        4a. Inspection cleaning            (clean_inspection)
        4b. FD repair                      (repair_fd from FD discovery notebook)
    → Step 5: Restaurant entity construction + merge
        5a. Restaurant construction        (restaurant_cleaning)
        5b. Merge standardised columns     (join_infection)
    → Step 6: Final cleaning               (final_cleaning)
    → Step 7: Data structuring            → output/restaurant_table.csv
                                          → output/inspections_table.csv

Usage
-----
    python main.py
"""

import os
import sys
import pandas as pd

from profiling import run_profiling
from association_rules import run_association_rules
from fd_detection import run_fd_detection
from fd_cleaning import run_fd_cleaning
from structuring import run_structuring

INPUT_FILE = "Food_Inspections_20240215.csv"


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
    df_raw = pd.read_csv(INPUT_FILE, low_memory=False)
    df_raw["License #"] = df_raw["License #"].fillna(0)
    print(f"Raw shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

    # ------------------------------------------------------------------
    # Step 1: Single-column profiling
    # ------------------------------------------------------------------
    _step_header(1, "Single-column profiling")
    run_profiling(df_raw, output_path="output/profiling_report.csv")

    # ------------------------------------------------------------------
    # Step 2: Association rule mining
    # ------------------------------------------------------------------
    _step_header(2, "Association rule mining")
    run_association_rules(
        df_raw,
        min_support=0.02,
        min_confidence=0.3,
        output_path="output/association_rules.csv",
    )

    # ------------------------------------------------------------------
    # Step 3: FD detection
    # ------------------------------------------------------------------
    _step_header(3, "Functional Dependency (FD) detection")
    fd_table = run_fd_detection(df_raw, output_path="output/fd_table.csv")

    # ------------------------------------------------------------------
    # Step 4: Data cleaning (inspection cleaning + FD repair + final cleaning)
    #
    # Internally delegates to:
    #   • clean_inspection()  from inspection_cleaning.py
    #   • repair_fd()         from FD discovery notebook
    #   • final_cleaning()    from final_cleaning.py
    # ------------------------------------------------------------------
    _step_header(4, "Data cleaning")
    df_clean = run_fd_cleaning(df_raw.copy(), fd_table=fd_table)
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
    restaurant_table, inspections_table = run_structuring(
        df_clean,
        restaurant_output="output/restaurant_table.csv",
        inspections_output="output/inspections_table.csv",
    )

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


if __name__ == "__main__":
    main()
