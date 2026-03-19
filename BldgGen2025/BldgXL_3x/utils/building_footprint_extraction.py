import numpy as np
from collections import Counter

import json

from .building_edge_polygon import edges_to_polygons


def to_vector_polygon(bottom_vertices, 
                      bottom_faces, 
                      logging=True):
    """
    Convert 3D mesh bottom face to vector polygon format.
    
    Returns:
        List of 2D coordinates ready for vector graphics: [(x1, y1), (x2, y2), ...]
    """
    # Extract bottom polygon edges
    bottom_edges = []
    for face in bottom_faces:
        v1, v2, v3 = face
        bottom_edges.append([int(v1), int(v2)])
        bottom_edges.append([int(v2), int(v3)])
        bottom_edges.append([int(v1), int(v3)])
        
    bottom_edges = [tuple(sorted(edge)) for edge in bottom_edges]
    # print(bottom_vertices, bottom_faces)
    
    bottom_edge_count = Counter(bottom_edges)
    bottom_edges_outer = [edge for edge, count in bottom_edge_count.items() if count == 1]
    # print(bottom_edges_outer)
    
    bottom_edge_polygons = edges_to_polygons(bottom_edges_outer, 
                                             bottom_vertices, 
                                             logging=logging)
    bottom_edge_polygon_areas = [polygon.area for polygon in bottom_edge_polygons]
    bottom_edge_polygon_areas = np.array(bottom_edge_polygon_areas)
        
    if not len(bottom_edge_polygon_areas):
        return None
    
    vector_polygon = bottom_edge_polygons[np.argmax(bottom_edge_polygon_areas)]
    # print(vector_polygon)
    # abc
    
    return vector_polygon


