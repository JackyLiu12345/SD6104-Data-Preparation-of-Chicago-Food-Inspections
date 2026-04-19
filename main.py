import pandas as pd
from inspection_cleaning import clean_inspection
from restaurant_construction import restaurant_cleaning
from final_cleaning import final_cleaning


def join_infection(restaurant_std, infection_df, join_cols):
    right_unique = restaurant_std.sort_values('cnt', ascending=False).drop_duplicates(subset=join_cols, keep='first')
    df = infection_df.merge(right_unique, on=join_cols, how='left')
    mask = df['cnt'].notna()
    for col in ['DBA Name', 'Zip', 'Latitude', 'Longitude', 'Facility Type']:
        std_col = col + '_std'
        if std_col in df.columns:
            df.loc[mask, col] = df.loc[mask, std_col]
            df = df.drop(std_col, axis=1)
    drop_cols = ['cnt', 'total_records', 'diff_info_cnt']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    return df


def main():
    input_file = 'Food_Inspections_20240215.csv'   # 原始数据
    output_file = 'Food_Inspections_Final_Cleaned.csv'

    print("Loading raw data...")
    df_raw = pd.read_csv(input_file, low_memory=False)
    df_raw['License #'] = df_raw['License #'].fillna(0)
    print(f"Raw shape: {df_raw.shape}")

    # Inspection
    print("\n=== Step 1: Inspection cleaning ===")
    df_step1 = clean_inspection(df_raw)
    print(f"After step 1 shape: {df_step1.shape}")

    # Restaurant Construction
    print("\n=== Step 2: Restaurant entity construction ===")
    restaurant_std = restaurant_cleaning(df_step1)
    print(f"Restaurant table shape: {restaurant_std.shape}")

    # Merge to Original Table
    print("\n=== Step 3: Merge restaurant standardized columns ===")
    join_cols = ['License #', 'DBA Name', 'Zip', 'Latitude', 'Longitude', 'Facility Type']
    df_step2 = join_infection(restaurant_std, df_step1, join_cols)
    print(f"After merge shape: {df_step2.shape}")

    # Final_cleaning
    print("\n=== Step 4: Final cleaning (fill missing & drop invalid) ===")
    df_final = final_cleaning(df_step2)
    print(f"Final shape: {df_final.shape}")

    # Store result
    df_final.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to: {output_file}")
    print('The missing values for final table:')
    print(df_final.isna().sum())
if __name__ == "__main__":
    main()