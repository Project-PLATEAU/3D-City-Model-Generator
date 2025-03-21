import geopandas as gpd
import networkx as nx
import numpy as np
from shapely import Polygon

def detect_overlapping(generated_polygon: Polygon, 
                       footprints: gpd.GeoDataFrame):
    intersected_idx = []
    intersected_areas = []
    print(f'source: {generated_polygon}')
    
    for idx in range(len(footprints)):
        if generated_polygon.intersects(footprints.iloc[idx].geometry):
            intersected_idx.append(idx)
            intersected_areas.append(generated_polygon.intersection(footprints.iloc[idx].geometry).area)
    
    isMatched = intersected_areas != []
    if isMatched:
        intersected_areas = np.array(intersected_areas)
        shared_idx = np.argmax(intersected_areas)
        bldg_idx = intersected_idx[shared_idx]
        print(f'matched: {bldg_idx}')
    else:
        bldg_idx = None
    
    return isMatched, bldg_idx
    

def remove_overlapping(file_path, output_path):
    gdf = gpd.read_file(file_path)
    
    G = nx.Graph()
    G.add_nodes_from(range(len(gdf)))
    
    for i in range(len(gdf)):
        for j in range(i + 1, len(gdf)):
            if gdf.iloc[i].geometry.intersects(gdf.iloc[j].geometry):
                G.add_edge(i, j)
                
    groups = list(nx.connected_components(G))
    
    remained = []
    for group in groups:
        group_list = list(group)
        areas = [gdf.iloc[i].geometry.area for i in group_list]
        largest_idx = group_list[np.argmax(areas)]
        remained.append(largest_idx)
        
    non_group_polygons = set(range(len(gdf))) - set([i for group in groups for i in group])
    remained.extend(non_group_polygons)
    
    updated_gdf = gdf.iloc[remained]
    
    updated_gdf.to_file(output_path, driver='GeoJSON')
    

if __name__ == '__main__':
    remove_overlapping('route01_1.geojson', 'route01_1_upd.geojson')
    
    