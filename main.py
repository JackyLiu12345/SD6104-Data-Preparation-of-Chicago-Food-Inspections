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
    → Step 1: Single-column profiling + data cleaning
              (profiling from Cell 0, cleaning from Cell 1 of the
               Single-column profiling notebook)
              → output/profiling_report.csv
    → Step 2: Association rule mining     → output/association_rules.csv
    → Step 3: FD detection                → output/fd_table.csv
    → Step 4: FD repair + final cleaning
        4a. FD repair                      (repair_fd from FD discovery notebook)
        4b. Final cleaning                 (final_cleaning)
    → Step 5: Data structuring            → output/restaurant_table.csv
                                          → output/inspections_table.csv
    → Step 6: Entity aggregation          → output/entity_inspection_analysis.csv
              + high-risk ranking          → output/entity_high_risk_rank.csv
    → Step 7: Data visualization          → output/*.png

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
from entity_aggregation import run_entity_aggregation
from visualization import run_visualization

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
    # Step 7: Data visualization
    # ------------------------------------------------------------------
    _step_header(7, "Data visualization")
    run_visualization(df_clean, output_dir="output")

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
    print(f"  Visualizations    : output/*.png")


if __name__ == "__main__":
    main()
