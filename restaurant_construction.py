import pandas as pd
import numpy as np
import re
from math import radians, sin, cos, sqrt, atan2
from fuzzywuzzy import fuzz
from collections import defaultdict


def generate_union_keys_df(df: pd.DataFrame) -> None:
    agg_df = df.groupby(
        ['License #', 'DBA Name', 'Zip', 'Latitude', 'Longitude', 'Facility Type'],
        as_index=False
    ).size()
    agg_df.rename(columns={'size': 'cnt'}, inplace=True)

    license_total = agg_df.groupby('License #')['cnt'].sum().reset_index()
    license_total.rename(columns={'cnt': 'total_records'}, inplace=True)
    agg_df = agg_df.merge(license_total, on='License #', how='left')
    del license_total

    license_info_diff_cnt = agg_df.groupby('License #')['cnt'].count().reset_index()
    license_info_diff_cnt.rename(columns={'cnt': 'diff_info_cnt'}, inplace=True)
    agg_df = agg_df.merge(license_info_diff_cnt, on='License #', how='left')
    del license_info_diff_cnt

    agg_df = agg_df.sort_values(['total_records', 'cnt'], ascending=False)
    prepared_agg_df = agg_df[agg_df['diff_info_cnt'] > 1].copy()
    prepared_agg_df0 = agg_df[agg_df['diff_info_cnt'] == 1].copy()

    return prepared_agg_df, prepared_agg_df0


def normalize_name(name):
    if pd.isna(name):
        return ''
    normalized = re.sub(r'[^A-Z0-9]', '', str(name).upper())
    return normalized


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def distance_similarity(d):
    if d == 0:
        return 1.0
    elif d <= 100:
        return 0.9
    elif d <= 500:
        # 线性从 0.9 降到 0.6
        return 0.9 - 0.3 * (d - 100) / 400
    elif d <= 5000:
        # 平方衰减：0.6 降到 0
        return 0.6 * ((5000 - d) / 4500) ** 2
    else:
        return 0.0


def compute_pair_similarity(r1, r2):
    # DBA Name
    norm1 = normalize_name(r1['DBA Name'])
    norm2 = normalize_name(r2['DBA Name'])
    if norm1 and norm2:
        sim_dba = fuzz.token_set_ratio(norm1, norm2) / 100.0
    else:
        sim_dba = 0.0

    # Zip exact match
    zip1 = str(r1['Zip']) if pd.notna(r1['Zip']) else None
    zip2 = str(r2['Zip']) if pd.notna(r2['Zip']) else None
    sim_zip = 1.0 if (zip1 is not None and zip2 is not None and zip1 == zip2) else 0.0

    # Geographic similarity
    lat1, lon1 = r1.get('Latitude'), r1.get('Longitude')
    lat2, lon2 = r2.get('Latitude'), r2.get('Longitude')
    valid1 = pd.notna(lat1) and pd.notna(lon1)
    valid2 = pd.notna(lat2) and pd.notna(lon2)
    if valid1 and valid2:
        try:
            lat1_f, lon1_f = float(lat1), float(lon1)
            lat2_f, lon2_f = float(lat2), float(lon2)
            dist = haversine(lat1_f, lon1_f, lat2_f, lon2_f)
            if dist < 0.1:
                dist = 0.0
            sim_geo = distance_similarity(dist)
        except (ValueError, TypeError):
            sim_geo = 0.0
    else:
        sim_geo = 0.0

    # Facility Type exact match (case-insensitive)
    ft1 = str(r1['Facility Type']).strip().lower() if pd.notna(r1['Facility Type']) else ''
    ft2 = str(r2['Facility Type']).strip().lower() if pd.notna(r2['Facility Type']) else ''
    sim_fac = 1.0 if (ft1 and ft2 and ft1 == ft2) else 0.0

    # Weighted sum
    sim_total = 0.55 * sim_dba + 0.2 * sim_zip + 0.2 * sim_geo + 0.05 * sim_fac
    return sim_total


def build_similarity_matrices(prepared_agg_df):
    similarity_dict = {}
    for license_id, group in prepared_agg_df.groupby('License #'):
        records = group.to_dict('records')
        n = len(records)
        if n < 2:
            similarity_dict[license_id] = np.array([[1.0]])
            continue

        mat = np.eye(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                sim = compute_pair_similarity(records[i], records[j])
                mat[i, j] = sim
                mat[j, i] = sim

        similarity_dict[license_id] = mat

    return similarity_dict


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1


def cluster_by_similarity_matrix(similarity_matrix, license_id, threshold=0.5):
    n = similarity_matrix.shape[0]
    if n == 0:
        return {}
    try:
        lic_int = int(license_id)
    except (ValueError, TypeError):
        lic_int = str(license_id)

    if n == 1:
        return {0: f"{lic_int}_1"}

    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                uf.union(i, j)

    comp_map = defaultdict(list)
    for idx in range(n):
        root = uf.find(idx)
        comp_map[root].append(idx)

    sorted_comps = sorted(comp_map.values(), key=lambda x: -len(x))
    node_to_entity = {}
    for seq, comp_nodes in enumerate(sorted_comps, start=1):
        entity_id = f"{lic_int}_{seq}"
        for node_idx in comp_nodes:
            node_to_entity[node_idx] = entity_id
    return node_to_entity


def assign_entity_ids_to_prepared_df(prepared_agg_df, sim_matrices, threshold=0.5):
    df_out = prepared_agg_df.copy().reset_index(drop=True)
    df_out['_temp_idx'] = df_out.index
    entity_id_list = [None] * len(df_out)

    for license_id, group in df_out.groupby('License #'):
        if license_id not in sim_matrices:
            n_nodes = len(group)
            if n_nodes == 1:
                try:
                    lic_int = int(license_id)
                except:
                    lic_int = license_id
                entity_id = f"{lic_int}_1"
                for orig_idx in group['_temp_idx']:
                    entity_id_list[orig_idx] = entity_id
            else:
                print(f"警告: License {license_id} 没有相似度矩阵，但有 {n_nodes} 个节点，跳过")
            continue

        mat = sim_matrices[license_id]
        if mat.shape[0] != len(group):
            print(f"警告: License {license_id} 矩阵大小 {mat.shape[0]} 与记录数 {len(group)} 不一致，跳过")
            continue

        node_to_entity = cluster_by_similarity_matrix(mat, license_id, threshold)

        for local_idx, (_, row) in enumerate(group.iterrows()):
            orig_idx = row['_temp_idx']
            entity_id = node_to_entity.get(local_idx)
            if entity_id:
                entity_id_list[orig_idx] = entity_id
            else:
                try:
                    lic_int = int(license_id)
                except:
                    lic_int = license_id
                entity_id_list[orig_idx] = f"{lic_int}_UNKNOWN"

    df_out['Entity_ID'] = entity_id_list
    df_out.drop(columns=['_temp_idx'], inplace=True)
    return df_out


def add_standardized_columns(df_with_entity,
                             entity_col='Entity_ID',
                             weight_col='cnt',
                             attr_cols=['DBA Name', 'Facility Type', 'Zip', 'Latitude', 'Longitude'],
                             suffix='_std'):
    df_out = df_with_entity.copy()
    idx_max = df_out.groupby(entity_col)[weight_col].idxmax()
    std_values = df_out.loc[idx_max, [entity_col] + attr_cols].copy()
    std_values = std_values.drop_duplicates(subset=entity_col, keep='first')
    rename_dict = {col: col + suffix for col in attr_cols}
    std_values = std_values.rename(columns=rename_dict)
    df_out = df_out.merge(std_values, on=entity_col, how='left')
    return df_out


def unify_zip_by_location(df,
                          lon_col='Longitude',
                          lat_col='Latitude',
                          cleaning_col='Zip'):
    df = df.copy()
    df[lon_col] = df[lon_col].round(6)
    df[lat_col] = df[lat_col].round(6)
    df[cleaning_col] = df.groupby([lon_col, lat_col])[cleaning_col].transform(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    )
    return df


def restaurant_cleaning(df):
    print(f'input shape:{df.shape}')
    prepared_agg_df, prepared_agg_df0 = generate_union_keys_df(df)
    print(f'Union Keys:{prepared_agg_df.shape[0] + prepared_agg_df0.shape[0]}')
    print(f'{prepared_agg_df0.shape[0]} Union Keys is unique for one License')
    print(f'{prepared_agg_df.shape[0]} Union Keys is not unique for one License')
    sim_matrices = build_similarity_matrices(prepared_agg_df)
    print(f"Generate {len(sim_matrices)} Similarity Matrices for Licenses")

    prepared_with_entity = assign_entity_ids_to_prepared_df(prepared_agg_df, sim_matrices, threshold=0.5)
    print(f'After add Entity_ID, the shape of Union_Key Table is {prepared_with_entity.shape}')
    restaurant = pd.concat([prepared_agg_df0, prepared_with_entity], axis=0)
    mask = restaurant['Entity_ID'].isna()
    restaurant.loc[mask, 'Entity_ID'] = restaurant.loc[mask, 'License #'].apply(lambda x: f"{int(x)}_0")
    del mask
    restaurant_with_std = add_standardized_columns(restaurant,
                                                   entity_col='Entity_ID',
                                                   weight_col='cnt',
                                                   attr_cols=['DBA Name', 'Facility Type', 'Zip', 'Latitude',
                                                              'Longitude'],
                                                   suffix='_std')
    restaurant_with_std = unify_zip_by_location(restaurant_with_std, lon_col='Longitude_std', lat_col='Latitude_std',
                                                cleaning_col='Zip_std')
    return restaurant_with_std
