# SD6104 — Data Preparation of Chicago Food Inspections

A course project (SD6104) that implements a complete, sequential **data-preparation pipeline** for the [Chicago Food Inspections](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5) public dataset.

---

## Pipeline Overview

```
Raw CSV (Food_Inspections_20240215.csv)
  │
  ▼
Step 1 ── Single-column profiling + data cleaning
  │         Cell 0: per-column null %, unique count, dtype, top values, stats
  │         Cell 1: Facility Type fill, drop Location/City/State, Zip filter,
  │                 License # cleaning, Inspection Type/Risk/Date/Violations
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
Step 4 ── FD repair + final cleaning
  │         • FD-driven repair (standardise minority RHS to majority)
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
├── profiling.py                   # Step 1: profiling + data cleaning (both from notebook)
├── association_rules.py           # Step 2: association rule mining (Apriori)
├── fd_detection.py                # Step 3: FD detection (compute_fd_confidence from notebook)
├── fd_cleaning.py                 # Step 4: FD repair + final cleaning orchestrator
├── inspection_cleaning.py         # Inspection-level cleaning (clean_inspection, from main branch)
├── final_cleaning.py              # Final imputation & cleanup (final_cleaning, from main branch)
├── structuring.py                 # Step 5: data structuring (Restaurant + Inspections tables)
├── restaurant_construction.py     # Entity resolution (Union-Find, fuzzy matching, Haversine)
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
| 1 | `profiling.py` | Uses `print_data_overview`, `print_column_summary`, `analyze_single_columns` from Cell 0 of the Single-column profiling notebook to profile each column. Then applies `clean_data` from Cell 1 of the same notebook: fills Facility Type nulls, drops Location/City/State, filters Zip codes, cleans License #, processes Inspection Type/Risk/Date, and extracts Violation Terms. Saves `output/profiling_report.csv`. |
| 2 | `association_rules.py` | Mines association rules using the Apriori algorithm (`mlxtend`). Items are built from Facility Type, Risk, Results, and Violation Terms. Saves `output/association_rules.csv` (antecedents, consequents, support, confidence, lift). |
| 3 | `fd_detection.py` | Uses `compute_fd_confidence` from the FD discovery notebook to detect functional dependencies across candidate LHS/RHS pairs. Reports exact and approximate FDs. Saves `output/fd_table.csv`. |
| 4 | `fd_cleaning.py` | Applies FD-driven repair using `repair_fd()` from the FD discovery notebook, then runs `final_cleaning()` from `final_cleaning.py` for fallback imputation. |
| 5 | `structuring.py` | Runs entity resolution (`restaurant_cleaning` from `restaurant_construction.py`), merges standardised attributes (`join_infection` from the original `main.py`), then splits into normalised `Restaurant` and `Inspections` tables with `Restaurant_ID` as a foreign key. |

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
