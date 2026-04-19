import pandas as pd

def final_cleaning(df):
    df = df.copy()
    #  Fill missing Address, Zip by Longitude, Latitude
    if 'Location' in df.columns:
        df = df.drop('Location', axis=1)

    def fill_by_location_fast(df, group_cols, target_col):
        # 计算每个组的众数（返回None若不可靠）
        def safe_mode(series):
            mode_vals = series.mode()
            if len(mode_vals) == 1 and (series == mode_vals[0]).sum() > 1:
                return mode_vals[0]
            return None

        mode_series = df.groupby(group_cols)[target_col].transform(safe_mode)
        mask = df[target_col].isna() & mode_series.notna()
        df.loc[mask, target_col] = mode_series[mask]
        return df

    df = fill_by_location_fast(df, ['Latitude', 'Longitude'], 'Address')
    df = fill_by_location_fast(df, ['Latitude', 'Longitude'], 'Zip')

    #  Fill missing Longitude, Latitude by Zip
    def fill_geo_by_zip_fast(df):

        def safe_mode_two(series1, series2):
            combined = list(zip(series1, series2))
            if not combined:
                return None, None
            from collections import Counter
            counter = Counter(combined)
            most_common = counter.most_common(1)[0]
            if most_common[1] > 1:
                return most_common[0][0], most_common[0][1]
            return None, None

        zip_geo = df.groupby('Zip', group_keys=False).apply(
            lambda g: pd.Series(safe_mode_two(g['Latitude'], g['Longitude']), index=['Lat_mode', 'Lon_mode']),
            include_groups=False
        ).dropna()

        for zip_val, row in zip_geo.iterrows():
            mask = (df['Zip'] == zip_val) & (df['Latitude'].isna() | df['Longitude'].isna())
            df.loc[mask, 'Latitude'] = df.loc[mask, 'Latitude'].fillna(row['Lat_mode'])
            df.loc[mask, 'Longitude'] = df.loc[mask, 'Longitude'].fillna(row['Lon_mode'])
        return df

    df = fill_geo_by_zip_fast(df)

    # Fill Facility Type by License and then fill Others
    license_mode = df.groupby('License #')['Facility Type'].transform(
        lambda x: x.mode().iloc[0] if len(x.mode()) == 1 and (x == x.mode().iloc[0]).sum() > 1 else None
    )
    mask = df['Facility Type'].isna() & license_mode.notna()
    df.loc[mask, 'Facility Type'] = license_mode[mask]
    df['Facility Type'] = df['Facility Type'].fillna('Others')

    #  Fill AKA Name by DBA Name
    df['AKA Name'] = df['AKA Name'].fillna(df['DBA Name'])

    #  Fill City / State
    df['City'] = df['City'].fillna('Chicago')
    df['State'] = df['State'].fillna('IL')

    #  Delete records still with missing values on Longitude, Latitude, Zip
    initial_rows = len(df)
    df = df.dropna(subset=['Latitude', 'Longitude', 'Zip'])
    print(f"Delete records still with missing values on Longitude, Latitude, Zip: {initial_rows - len(df)} 行")
    #  Fill missing Violations 和 Violation Terms by '0'
    df['Violations'] = df['Violations'].fillna('0')
    df['Violation Terms'] = df['Violation Terms'].fillna('0')
    df.drop(['Entity_ID'],axis=1,inplace=True)
    return df