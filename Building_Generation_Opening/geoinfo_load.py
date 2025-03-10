import trimesh
import numpy as np
import json

from earcut import earcut

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import triangulate


def simple_geojson_loader(file_path):
    with open(file_path, 'r') as geojson:
        polygon_data = json.load(geojson)
        
    return polygon_data


def polygon_triangulation(polygon: Polygon):
    coords = np.array(polygon.exterior.coords)[:-1]
    
    holes = []
    hole_indices = []
    if len(polygon.interiors) > 0:
        start = 0
        for interior in polygon.interiors:
            hole_coords = np.array(interior.coords)[:-1]
            holes.append(hole_coords)
            start += len(coords)
            hole_indices.append(start)
            coords = np.vstack((coords, hole_coords))
            
    flattened_vertices = coords.flatten()
    
    triangles = earcut.earcut(flattened_vertices, hole_indices, dim=2)
    
    triangles = np.array(triangles).reshape(-1, 3)
    triangles = triangles[:, ::-1]
    
    return coords, triangles


def footprint_align(vertices, faces):
    # TODO: align and put the footprint to bottom (-0.95)
    pass


def polygon_to_mesh(file_path):
    polygon_data = simple_geojson_loader(file_path)
    
    meshes = []
    
    polygons = []
    vertices = []
    faces = []
    properties = []
    vertex_index = {}
    current_index = 0
    
    # print(len(polygon_data))

    if 'features' in polygon_data.keys():
        for feature in polygon_data['features']:
            try:
                geometry = feature['geometry']
                properties.append(feature['properties'])
                
                if geometry['type'] == 'MultiPolygon' or geometry['type'] == 'Polygon':
                    coords = np.array(geometry['coordinates'][0][0]) if geometry['type'] == 'MultiPolygon' else np.array(geometry['coordinates'][0])
                    
                    polygon = Polygon(coords)
                    polygons.append(polygon)

                    v, tri = polygon_triangulation(polygon)

                    # triangulate(polygon)
                    
                    vertices.append([[float(vertex[0]), float(-vertex[1]), 0.0] for vertex in v])
                    faces.append(tri)

                    # tri = triangulate(polygon)
                    
                    # for triangle in tri:
                    #     if not triangle.within(polygon):
                    #         continue
                        
                    #     tri_coords = np.array(triangle.exterior.coords)[:-1]
                        
                    #     tri_indices = []
                    #     for vertex in tri_coords:
                    #         vertex_tuple = tuple(vertex)
                    #         if vertex_tuple not in vertex_index:
                    #             vertices.append([float(vertex[0]), float(vertex[1]), 0.0])
                    #             vertex_index[vertex_tuple] = current_index
                    #             current_index += 1
                                
                    #         tri_indices.append(vertex_index[vertex_tuple])
                            
                    #     faces.append(tri_indices)
                        
                    # mesh = {
                    #     'vertices' : np.array(vertices), 
                    #     'faces' : np.array(faces)
                    # }
                    # print(mesh)
                    
                    # meshes.append(mesh)
                    
            except Exception as e:
                print(f'{e} exception encountered. \n')
    
    else:
        try:
            geometry = polygon_data['geometry']
            properties.append(feature['properties'])
            
            if geometry['type'] == 'MultiPolygon' or geometry['type'] == 'Polygon':
                coords = np.array(geometry['coordinates'][0][0]) if geometry['type'] == 'MultiPolygon' else np.array(geometry['coordinates'][0])
                
                polygon = Polygon(coords)
                polygons.append(polygon)
                
                v, tri = polygon_triangulation(polygon)
                                
                # triangulate(polygon)
                
                vertices = [[[float(vertex[0]), float(-vertex[1]), 0.0] for vertex in v]]
                faces = [tri]
                
                # fp = trimesh.Trimesh(vertices=vertices[0], faces=faces[0])
                # fp.export('fp_test_01_1.obj')
                # aa
                
        except Exception as e:
            print(f'{e} exception encountered. \n')
    
    # print(vertices, faces)
    return vertices, faces, properties, polygons

    
if __name__ == '__main__':
    file_path = "../mxl/test_geojson/test.geojson"
    polygon_to_mesh(file_path)