import pandas as pd
from shapely import Polygon

from .geoinfo_load import polygon_to_mesh
from .generation import generate
from .coordinate_util import convert_mercator_to_Japan_Rect_30169

def building_generation(footprint_path, route_path, extracted=True, route_exists=False):
    vertices, faces, params, polygons = polygon_to_mesh(footprint_path)
    if extracted:
        reproj_vertices = []
        for polygon in vertices:
            reproj_polygon = []
            for x, y, z in polygon:
                x_rep, y_rep = convert_mercator_to_Japan_Rect_30169(x, -y)
                reproj_polygon.append([x_rep, -y_rep, z])
            reproj_vertices.append(reproj_polygon)
        
        vertices = reproj_vertices
            
        # vertices = [[[*convert_mercator_to_Japan_Rect_30169(x, -y), z] for x, y, z in polygon] for polygon in vertices]
        
        reproj_polygons = []
        for polygon in polygons:
            coords = list(polygon.exterior.coords)
            transformed_coords = [convert_mercator_to_Japan_Rect_30169(x, y) for x, y in coords]
            reproj_polygons.append(Polygon(transformed_coords))
         
        polygons = reproj_polygons
        
    
    params = pd.DataFrame(params)
    
    sod = [2 for _ in range(len(params))]
    
    params_upd = pd.DataFrame({
        'Bldg_ID': params['id'], 
        'SoD': sod, 
        'Height': params['predHeight'], 
        'Roof_Type': params['roof_type']
    })
    p = params_upd.values.tolist()
    
    # generation
    bldg_mesh = generate(vertices, faces, p, polygons, opening=route_path, route_exists=route_exists)
    bldg_mesh.export('buildings.obj')
    
    
if __name__ == '__main__':
    building_generation('route01_1.geojson', 'Building_Generation_Opening/route_elem/bldg-elem_route1.json')