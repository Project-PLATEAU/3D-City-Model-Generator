import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import MultiPolygon
import xgboost as xgb

def count_vertices(geom):
    if isinstance(geom, MultiPolygon):
        return sum(len(poly.exterior.coords) for poly in geom.geoms)
    return len(geom.exterior.coords)


def get_length_width(row):
    # Calculate length and width by comparing the distances of the sides
    length = max(row['maxx'] - row['minx'], row['maxy'] - row['miny'])
    width = min(row['maxx'] - row['minx'], row['maxy'] - row['miny'])
    return pd.Series([length, width], index=['mbr_length', 'mbr_width'])


def para_calc(df):
    df['area'] = df.geometry.area
    df['perimeter'] = df.geometry.length
    df['compactness'] = df['perimeter'] / np.sqrt(4 * np.pi * df['area'])
    df['complexity'] = df['perimeter'] / df['area']

    df['simplified_geometry'] = df['geometry'].simplify(0.1, preserve_topology=True)
    df['vertices'] = df['simplified_geometry'].apply(count_vertices)
    df = df.drop(['simplified_geometry'], axis=1)

    df[['minx', 'miny', 'maxx', 'maxy']] = df.geometry.bounds
    df[['mbr_length', 'mbr_width']] = df.apply(get_length_width, axis=1)
    df['slimness'] = df['mbr_length'] / df['mbr_width']
    df = df.drop(['minx', 'miny', 'maxx', 'maxy'], axis=1)

    df.loc[:, 'buffer'] = df['geometry'].buffer(1)
    sindex = df['buffer'].sindex
    def count_adjacent(row):
        possible_matches_index = list(sindex.intersection(row['buffer'].bounds))
        possible_matches = df.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches['buffer'].intersects(row['buffer'])]
        # Subtract 1 to exclude the building itself from its own neighbor count
        return len(precise_matches) - 1

    df['num_adjacent'] = df.apply(count_adjacent, axis=1)
    df = df.drop(['buffer'], axis=1)

    df['centroid'] = df.geometry.centroid

    centroid_df = gpd.GeoDataFrame(df, geometry='centroid')
    spatial_index = centroid_df.sindex

    df['num_neighbours'] = 0
    dist = 25
    for idx, row in centroid_df.iterrows():
        current_centroid = row['centroid']
        possible_matches_index = list(spatial_index.intersection(current_centroid.buffer(dist).bounds))
        possible_matches = centroid_df.iloc[possible_matches_index]

        neighbours_within_distance = possible_matches[possible_matches.centroid.distance(current_centroid) <= dist]

        neighbours_count = len(neighbours_within_distance[neighbours_within_distance.index != idx])
        df.at[idx, 'num_neighbours'] = neighbours_count
    df = df.drop(columns=['centroid'])

    df['category'] = 0

    return df


def height_calc(geojson_path):
    df = gpd.read_file(geojson_path)

    df = para_calc(df)

    x_data = df[
        ['area', 'compactness', "num_neighbours", "num_adjacent", "vertices", "mbr_length", "mbr_width", "slimness",
         "complexity", 'category']].values

    regressor = xgb.XGBRegressor()
    regressor.load_model('./Para_calc/xgb_model_20250109-113703.json')

    pred_data = regressor.predict(x_data)
    df['predHeight'] = pred_data

    df.to_file(geojson_path, index=False)
    return df

