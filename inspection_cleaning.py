import numpy as np
import pandas as pd
import re


def clean_inspection(df: pd.DataFrame) -> None:
    # ---------- 1. Inspection Type ----------
    print('1. Cleaning {Inspection Type}')
    if 'Inspection Type' in df.columns:
        original_rows = df.shape[0]
        inspection_type_null_count = df['Inspection Type'].isnull().sum()
        df = df.dropna(subset=['Inspection Type']).copy()
        rows_removed = original_rows - df.shape[0]
        print(f"Inspection Type: deleted {inspection_type_null_count} null values (total rows removed: {rows_removed})")
    else:
        print("Warning: {Inspection Type} column not found")
    print("-" * 40)

    # ---------- 2. Inspection Date ----------
    print('2. Cleaning {Inspection Date}')
    if 'Inspection Date' in df.columns:
        df['Inspection Date'] = pd.to_datetime(df['Inspection Date'], format='%m/%d/%Y', errors='coerce')
        conversion_failures = df['Inspection Date'].isnull().sum()
        if conversion_failures > 0:
            print(f"Inspection Date: {conversion_failures} date conversion failures (set to NaT)")

        df['Inspection Year'] = df['Inspection Date'].dt.year
        df['Inspection Month'] = df['Inspection Date'].dt.month
        df['Inspection Day'] = df['Inspection Date'].dt.day
        df = df.drop(columns=['Inspection Date'])
        print(
            f"Inspection Date: converted to three columns (Year, Month, Day). Date range: {df['Inspection Year'].min()} to {df['Inspection Year'].max()}")
    else:
        print("Warning: {Inspection Date} column not found")
    print("-" * 40)

    # ---------- 3. Risk ----------
    print('3. Cleaning {Risk}')
    if 'Risk' in df.columns:
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
    else:
        print("Warning: {Risk} column not found")
    print("-" * 40)

    # ---------- 4. Violations ----------
    def extract_violation_terms(violation_text):
        if pd.isna(violation_text):
            return np.nan

        text = str(violation_text)
        clauses = [clause.strip() for clause in text.split('|')]
        all_matches = []

        for clause in clauses:
            main_part = clause
            comments_markers = [" - Comments:", "Comments:", " - ", "COMMENTS:"]
            for marker in comments_markers:
                if marker in main_part:
                    parts = main_part.split(marker, 1)
                    if len(parts) > 1:
                        main_part = parts[0]
                        break

            pattern = r'^\s*(\d+)\.\s'
            match = re.search(pattern, main_part)
            if match:
                term = match.group(1)
                try:
                    term_clean = str(int(term))
                except ValueError:
                    term_clean = term
                all_matches.append(term_clean)
            else:
                backup_pattern = r'\b(\d+)\.\s+[A-Z]'
                backup_matches = re.findall(backup_pattern, main_part)
                for term in backup_matches:
                    try:
                        term_clean = str(int(term))
                    except ValueError:
                        term_clean = term
                    all_matches.append(term_clean)

        if all_matches:
            unique_matches = sorted(set(all_matches), key=lambda x: int(x) if x.isdigit() else 0)
            return ','.join(unique_matches)
        return np.nan

    print('4. Cleaning {Violations}')
    if 'Violations' in df.columns:
        df['Violation Terms'] = df['Violations'].apply(extract_violation_terms)
        extracted_count = df['Violation Terms'].notna().sum()
        no_violation_count = df['Violations'].isna().sum()
        print(
            f"Violations: successfully extracted clause numbers from {extracted_count} records; {no_violation_count} records have no violations")
    else:
        print("Warning: {Violations} column not found")

    return df
