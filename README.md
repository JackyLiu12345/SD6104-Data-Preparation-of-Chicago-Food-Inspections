# SD6104 — Data Preparation of Chicago Food Inspections

A course project (SD6104) that implements a complete, sequential **data-preparation pipeline** for the [Chicago Food Inspections](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5) public dataset.

---

## Pipeline Overview

```
Raw CSV (Food_Inspections_20240215.csv)
  │
  ▼
Step 1 ── Single-column profiling
  │         per-column null %, unique count, dtype, top values, numeric stats
  │         → output/profiling_report.csv
  │
  ▼
Step 2 ── Association rule mining
  │         Apriori on Facility Type, Risk, Results, Violation Terms
  │         → output/association_rules.csv
  │
  ▼
Step 3 ── Functional Dependency (FD) detection
  │         check LHS → RHS by counting distinct RHS values per group
  │         → output/fd_table.csv
  │
  ▼
Step 4 ── FD-based data cleaning
  │         • parse Inspection Type / Date / Risk / Violations
  │         • FD-driven imputation (use discovered FDs to fill missing values)
  │         • fallback imputation (mode by location/zip, default City/State)
  │
  ▼
Step 5 ── Data structuring
            entity resolution (fuzzy matching + Union-Find clustering)
            → output/restaurant_table.csv   (one row per unique restaurant)
            → output/inspections_table.csv  (one row per inspection, FK: Restaurant_ID)
```

---

## Repository Structure

```
.
├── main.py                        # Pipeline orchestrator — run this
├── profiling.py                   # Step 1: single-column profiling
├── association_rules.py           # Step 2: association rule mining (Apriori)
├── fd_detection.py                # Step 3: functional dependency detection
├── fd_cleaning.py                 # Step 4: FD-based data cleaning
├── structuring.py                 # Step 5: data structuring (Restaurant + Inspections tables)
├── restaurant_construction.py     # Utility: entity resolution (Union-Find, fuzzy matching, Haversine)
├── requirements.txt               # Python dependencies
├── .gitignore
├── notebooks/                     # Exploratory Jupyter notebooks
│   ├── Single-column profiling.ipynb
│   ├── Association rule mining.ipynb
│   ├── FD discovery.ipynb
│   └── INDs&structuring.ipynb
└── output/                        # Generated outputs (git-ignored except .gitkeep)
    ├── profiling_report.csv
    ├── association_rules.csv
    ├── fd_table.csv
    ├── restaurant_table.csv
    └── inspections_table.csv
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the raw data file

Put `Food_Inspections_20240215.csv` in the repository root (it is git-ignored).

### 3. Run the pipeline

```bash
python main.py
```

All outputs are written to the `output/` directory.

---

## Step Descriptions

| Step | Module | Description |
|------|--------|-------------|
| 1 | `profiling.py` | Computes per-column statistics: null count/%, unique count, dtype, top-5 values, and basic numeric stats (mean, std, quartiles). Saves `output/profiling_report.csv`. |
| 2 | `association_rules.py` | Mines association rules using the Apriori algorithm (`mlxtend`). Items are built from Facility Type, Risk, Results, and Violation Terms. Saves `output/association_rules.csv` (antecedents, consequents, support, confidence, lift). |
| 3 | `fd_detection.py` | Discovers functional dependencies by grouping on candidate LHS columns and counting distinct RHS values per group. Reports exact FDs and approximate FDs (≥ 90 % accuracy). Saves `output/fd_table.csv`. |
| 4 | `fd_cleaning.py` | Cleans inspection-specific columns (type, date, risk, violations), applies FD-driven imputation to fill missing values, then runs fallback mode-based imputation for geographic fields. |
| 5 | `structuring.py` | Runs entity resolution (fuzzy name matching + Haversine distance + Union-Find clustering) to assign a unique `Restaurant_ID` to each real-world restaurant. Produces a normalised `Restaurant` table and an `Inspections` table with `Restaurant_ID` as a foreign key. |

---

## Input / Output

| Item | Description |
|------|-------------|
| **Input** | `Food_Inspections_20240215.csv` — raw Chicago Food Inspections data (~250 k rows) |
| **output/profiling_report.csv** | Per-column statistics for the raw dataset |
| **output/association_rules.csv** | Discovered association rules with support / confidence / lift |
| **output/fd_table.csv** | Functional dependency table (LHS, RHS, accuracy, violation rate) |
| **output/restaurant_table.csv** | Normalised restaurant dimension table with `Restaurant_ID` |
| **output/inspections_table.csv** | Fact table of inspections referencing `Restaurant_ID` |

---

## Dependencies

See `requirements.txt`.  Key packages:

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `fuzzywuzzy` | Fuzzy string matching for entity resolution |
| `python-Levenshtein` | Fast edit-distance computation (speeds up fuzzywuzzy) |
| `mlxtend` | Apriori algorithm for association rule mining |
