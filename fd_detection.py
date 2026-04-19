"""
fd_detection.py
===============
Functional dependency detection step.

The core ``compute_fd_confidence`` function is taken from the
FD discovery notebook (FD discovery.ipynb) on the main branch.
``discover_fds`` wraps it to scan all candidate LHS/RHS pairs.
"""

import os
import pandas as pd


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
# Pipeline helpers
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
