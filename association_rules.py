import os
import ast
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


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
            raw = str(vt).strip()
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
) -> pd.DataFrame:
    """
    Mine association rules from *df* using the Apriori algorithm.

    Parameters
    ----------
    df            : Cleaned inspection DataFrame.
    min_support   : Minimum support threshold (fraction of transactions).
    min_confidence: Minimum confidence threshold.
    output_path   : Where to save the discovered rules CSV.

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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rules.to_csv(output_path, index=False)
    print(f"  Association rules saved to: {output_path}")
    return rules
