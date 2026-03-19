import os, json
import trimesh
import numpy as np
import networkx as nx
from collections import defaultdict, Counter

import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon, MultiPolygon
from shapely import to_geojson

from .building_edge_polygon import edges_to_polygons
from .building_footprint_extraction import to_vector_polygon

import shutil


def load_building_mesh(mesh_path):
    # mesh = trimesh.load(mesh_path, 
    #                     force='mesh', 
    #                     merge_norm=True, 
    #                     merge_tex=True)
    vertices, faces = [], []
    with open(mesh_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            values = line.split()
            if not values:
                continue
            
            if values[0] == 'v':
                v = [float(x) for x in values[1:4]]
                vertices.append(v)
                
            elif values[0] == 'f':
                face = []
                for v in values[1:]:
                    vertex_idx = int(v.split('/')[0]) - 1
                    face.append(vertex_idx)
                    
                faces.append(face)
                
    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64)
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    return mesh

@DeprecationWarning
def extract_building_footprint(floor_mesh: trimesh.Trimesh):
    vertices = floor_mesh.vertices
    faces = floor_mesh.faces
    
    # Project the vertices onto the xz-plane (ignore y-coordinate)
    # We'll use x and z as our 2D coordinates
    vertices_2d = np.column_stack((vertices[:, 0], vertices[:, 2]))
    
    # Get the boundary edges of the mesh
    edges = floor_mesh.edges_unique
    edge_face_count = np.zeros(len(edges), dtype=np.int64)
    
    # Count how many faces use each edge
    for i, edge in enumerate(edges):
        v1, v2 = edge
        count = 0
        for face in faces:
            # Check if this edge is in this face
            if (v1 in face and v2 in face):
                # Check if they are adjacent in the face
                face_v_list = list(face) + [face[0]]  # Close the loop
                for j in range(len(face)):
                    if (face_v_list[j] == v1 and face_v_list[j+1] == v2) or \
                       (face_v_list[j] == v2 and face_v_list[j+1] == v1):
                        count += 1
                        break
        edge_face_count[i] = count
    
    # Boundary edges are used by exactly one face
    boundary_edges = edges[edge_face_count == 1]
    # print(boundary_edges)
    
    # Extract boundary vertices
    boundary_vertices = set()
    for edge in boundary_edges:
        boundary_vertices.add(edge[0])
        boundary_vertices.add(edge[1])
    
    boundary_vertices = list(boundary_vertices)
    
    # If we couldn't find the boundary using the method above,
    # try using trimesh's convex hull as a fallback
    # if len(boundary_vertices) == 0:
    #     print("Using convex hull as fallback method")
    #     hull = floor_mesh.convex_hull
    #     boundary_vertices = hull.vertices
    
    # Sort the boundary vertices to form a proper polygon
    # This is a simplified approach - for complex polygons, 
    # we would need a more sophisticated method
    # But since you guaranteed a single polygon, this should work
    # print(boundary_vertices)
    boundary_points_2d = vertices_2d[boundary_vertices]
    
    # Calculate centroid
    centroid = np.mean(boundary_points_2d, axis=0)
    
    # Sort vertices by angle from centroid
    angles = np.arctan2(
        boundary_points_2d[:, 1] - centroid[1],
        boundary_points_2d[:, 0] - centroid[0]
    )
    sorted_indices = np.argsort(angles)
    sorted_boundary_points = boundary_points_2d[sorted_indices]
    
    # Create a Shapely polygon
    floor_polygon = Polygon(sorted_boundary_points)
    
    return floor_polygon

def delete_building_duplicated_edges(edges: np.array):
    sorted_edges = np.sort(edges, axis=1)
    edge_tuples = [tuple(edge) for edge in sorted_edges]
    
    edge_tuples_unique = set(edge_tuples)
    edge_tuples_unique = {edge for edge in edge_tuples_unique if edge[0] != edge[1]}
    edge_unique = np.array(list(edge_tuples_unique))
    
    return edge_unique, edge_tuples_unique, sorted_edges

def delete_building_duplicated_and_invalid_faces(mesh: trimesh.Trimesh):
    valid_face_mask = np.ones(len(mesh.faces), dtype=bool)
    for idx, face in enumerate(mesh.faces):
        if len(np.unique(face)) < 3:
            valid_face_mask[idx] = False
    
    sorted_faces = np.sort(mesh.faces[valid_face_mask], axis=1)
    _, unique_idx = np.unique(sorted_faces, axis=0, return_index=True)
    
    valid_idx = np.arange(len(mesh.faces))[valid_face_mask][unique_idx]
    
    updated_faces = mesh.faces[valid_idx]
    updated_mesh = trimesh.Trimesh(vertices=mesh.vertices, 
                                   faces=updated_faces)
    
    return updated_mesh

def exponential_window(x, span_min=-0.1, span_max=0.1, decay_rate=50):
    """
    Exponential decay function outside the span.
    
    Args:
        x: Input value(s)
        span_min, span_max: The anticipated span where function should be 1
        decay_rate: Controls how fast the function decays (higher = faster decay)
    """
    # Distance from the span
    distance = np.maximum(0, np.maximum(span_min - x, x - span_max))
    
    return np.exp(-decay_rate * distance)


def analyze_building_roof_conn(roof_mesh: trimesh.Trimesh, 
                               facade_mesh: trimesh.Trimesh, 
                               footprint_polygon: Polygon):
    def footprint_longest_edge(footprint_polygon: Polygon):
        coords = np.array(footprint_polygon.exterior.coords)
        
        points1 = coords[:-1]
        points2 = coords[1:]
        
        distances = np.sqrt(np.sum((points2 - points1) ** 2, axis=1))
        
        max_idx = np.argmax(distances)
        
        return distances[max_idx], points1[max_idx], points2[max_idx]
    
        
    def extract_edges_with_frequency(faces):
        """Extract edges from faces and count their frequency."""
        edge_count = Counter()
        
        for face in faces:
            # Skip invalid faces
            if len(set(face)) != 3:  # Should have 3 unique vertices
                continue
                
            # Create edges from face vertices
            edges = [
                (min(face[0], face[1]), max(face[0], face[1])),
                (min(face[1], face[2]), max(face[1], face[2])),
                (min(face[0], face[2]), max(face[0], face[2]))
            ]
            
            for edge in edges:
                edge_count[edge] += 1
        
        return edge_count
    
    def map_roof_vertices_to_facade(roof_vertices, facade_vertices, eps=1e-5):
        """Map roof vertices to facade vertices based on spatial proximity."""
        mapping = []
        
        for roof_vertex in roof_vertices:
            distances = np.linalg.norm(facade_vertices - roof_vertex, axis=1)
            matches = np.where(distances < eps)[0]
            
            if len(matches) == 1:
                mapping.append(int(matches[0]))
            elif len(matches) == 0:
                mapping.append(-1)  # No match found
            else:
                # Multiple matches - choose the closest one
                closest_idx = matches[np.argmin(distances[matches])]
                mapping.append(int(closest_idx))
        
        return mapping
    
    # Clean meshes by removing duplicated and invalid faces
    roof_mesh = delete_building_duplicated_and_invalid_faces(roof_mesh)
    facade_mesh = delete_building_duplicated_and_invalid_faces(facade_mesh)
    
    if not len(roof_mesh.faces):
        return 0.0
    
    # Map roof vertices to facade vertices
    vertex_mapping = map_roof_vertices_to_facade(roof_mesh.vertices, facade_mesh.vertices)
    
    # Extract edges with frequency from roof
    roof_edge_counts = extract_edges_with_frequency(roof_mesh.faces)
    
    # Extract unique edges from facade
    facade_edge_counts = extract_edges_with_frequency(facade_mesh.faces)
    facade_edges = set(facade_edge_counts.keys())
    
    # Map roof edges to facade coordinate system
    mapped_roof_edges = {}
    for (v1, v2), count in roof_edge_counts.items():
        mapped_v1 = vertex_mapping[v1]
        mapped_v2 = vertex_mapping[v2]
        
        # Skip edges that can't be mapped (vertices not in facade)
        if mapped_v1 == -1 or mapped_v2 == -1:
            mapped_edge = (v1, v2)  # Keep original indices
        else:
            mapped_edge = (min(mapped_v1, mapped_v2), max(mapped_v1, mapped_v2))
        
        mapped_roof_edges[mapped_edge] = count
    
    # Find edges that appear only once in roof and don't appear in facade
    sole_roof_edges = []
    
    for mapped_edge, count in mapped_roof_edges.items():
        # Check if edge appears only once in roof
        if count == 1:
            # Check if this mapped edge exists in facade
            if mapped_edge not in facade_edges:
                # Find the original roof edge indices
                for (orig_v1, orig_v2), orig_count in roof_edge_counts.items():
                    if orig_count == 1:
                        orig_mapped_v1 = vertex_mapping[orig_v1] if vertex_mapping[orig_v1] != -1 else orig_v1
                        orig_mapped_v2 = vertex_mapping[orig_v2] if vertex_mapping[orig_v2] != -1 else orig_v2
                        orig_mapped_edge = (min(orig_mapped_v1, orig_mapped_v2), max(orig_mapped_v1, orig_mapped_v2))
                        
                        if orig_mapped_edge == mapped_edge:
                            sole_roof_edges.append((orig_v1, orig_v2))
                            break
    
    # Remove duplicates and sort
    sole_roof_edges = list(set(sole_roof_edges))
    sole_roof_edges.sort()
    
    # Get vertices for these edges
    sole_roof_edge_vertices = []
    for v1, v2 in sole_roof_edges:
        vertex1 = roof_mesh.vertices[v1]
        vertex2 = roof_mesh.vertices[v2]
        sole_roof_edge_vertices.append([vertex1, vertex2])
    
    # Find the longest edge among sole roof edges for potential connection analysis
    longest_sole_edge_length = 0
    if sole_roof_edge_vertices:
        for i, (vertex1, vertex2) in enumerate(sole_roof_edge_vertices):
            edge_length = np.linalg.norm(vertex2 - vertex1)
            if edge_length > longest_sole_edge_length:
                longest_sole_edge_length = edge_length

    
    # length punishment
    footprint_longest_edge_length, _, _ = footprint_longest_edge(footprint_polygon)
    # print(footprint_longest_edge_length)
    length_punishment_index = 1.0 - longest_sole_edge_length / footprint_longest_edge_length
    if length_punishment_index < 0:
        length_punishment_index = 0.0
    
    # EVAL 3: Edge ratio
    edge_ratio = 1.0 - len(sole_roof_edges) / len(roof_mesh.faces)
    edge_ratio *= length_punishment_index
    # print(roof_mesh.vertices)
    # print(facade_mesh.vertices)
    # print(roof_edges_sorted_unique)
    
    return edge_ratio
    

def calculate_horizontal_projected_area(vertices, faces):
    """
    Calculate the sum of horizontally-projected areas of all faces in a mesh.
    
    Parameters:
    vertices: numpy array of shape (N, 3) containing vertex coordinates
    faces: numpy array of shape (M, 3) containing face indices
    
    Returns:
    float: Total horizontally-projected area
    """
    total_area = 0.0
    
    for face in faces:
        # Get the three vertices of the current face
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        # Project vertices onto XZ plane (ignore Y coordinate - height)
        # Create polygon using [x, z] coordinates
        triangle_coords = [
            [v0[0], v0[2]],  # [x, z]
            [v1[0], v1[2]],  # [x, z]
            [v2[0], v2[2]]   # [x, z]
        ]
        
        # Create Shapely polygon and calculate area
        triangle_polygon = Polygon(triangle_coords)
        area = triangle_polygon.area
        
        total_area += area
    
    return total_area


def analyze_building_conn(building_mesh: trimesh.Trimesh):
    """
    Analyze building mesh connectivity by finding sole edges.

    A sole edge is an edge that appears only once in the mesh (boundary edge),
    which indicates a non-manifold or incomplete mesh structure.

    Args:
        building_mesh: The complete building mesh

    Returns:
        tuple: (sole_edge_ratio, sole_edge_lengths)
            - sole_edge_ratio: Ratio of sole edges to total edges (lower is better)
            - sole_edge_lengths: List of lengths for each sole edge
    """
    # Extract edges with their frequency from the mesh faces
    edge_count = Counter()

    for face in building_mesh.faces:
        # Skip invalid faces
        if len(set(face)) != 3:  # Should have 3 unique vertices
            continue

        # Create edges from face vertices (normalized to have smaller index first)
        edges = [
            (min(face[0], face[1]), max(face[0], face[1])),
            (min(face[1], face[2]), max(face[1], face[2])),
            (min(face[0], face[2]), max(face[0], face[2]))
        ]

        for edge in edges:
            edge_count[edge] += 1

    # Count sole edges (edges that appear only once)
    sole_edges = [edge for edge, count in edge_count.items() if count == 1]

    total_edges = len(edge_count)
    num_sole_edges = len(sole_edges)

    # Calculate length of each sole edge
    sole_edge_lengths = []
    for edge in sole_edges:
        v1_idx, v2_idx = edge
        v1 = building_mesh.vertices[v1_idx]
        v2 = building_mesh.vertices[v2_idx]
        edge_length = np.linalg.norm(v2 - v1)
        sole_edge_lengths.append(edge_length)

    # Return the ratio of sole edges to total edges
    # A perfectly manifold closed mesh should have ratio 0.0
    # Higher ratio indicates more boundary/disconnected edges
    if total_edges == 0:
        print('Warning: no valid edges. ')
        return 0.0, []  # Worst case: no valid edges

    sole_edge_ratio = num_sole_edges / total_edges

    return sole_edge_ratio, sole_edge_lengths


def divide_building_mesh(mesh: trimesh.Trimesh, 
                         building_footprint: Polygon = None, 
                         footprint_tolerance = 0.1, 
                         logging=True):    
    building = mesh.copy()
    y_values = building.vertices[:, 1]
    
    # split building mesh
    y_min = np.min(y_values)
    y_max = np.max(y_values)
    y_mid = y_min + footprint_tolerance
    
    face_vertices = building.vertices[building.faces]
    face_y_values = face_vertices[:, :, 1]
    
    floor_mask = np.all(face_y_values < y_mid, axis=1)
    roof_mask = np.all(face_y_values > y_mid, axis=1)
    facade_mask = ~(floor_mask | roof_mask)
    
    
    # floor & roof preprocessing
    building_floor = building.submesh([floor_mask])
    building_roof = building.submesh([roof_mask])
    
    if not len(building_floor) or not len(building_roof):
        return None, -100.0
    
    building_floor = building_floor[0]
    building_roof = building_roof[0]
    
    building_floor.update_faces(building_floor.unique_faces())
    building_roof.update_faces(building_roof.unique_faces())
    
    building_footprint = to_vector_polygon(building_floor.vertices, building_floor.faces, logging=logging)
    if not building_footprint:
        return None, 0.0
    
    # facade preprocessing
    building_facade_face_indices = np.where(facade_mask)[0]
    building_facade_faces = building.faces[building_facade_face_indices]
    building_facade = trimesh.Trimesh(
        vertices=building.vertices.copy(), 
        faces=building_facade_faces, 
        process=False
    )
    building_facade.remove_unreferenced_vertices()

    # building_facade.export('facades.obj')
    
    # 1. footprint
    if building_footprint is None:
        raise Exception('No valid footprint provided. ')
        # building_footprint = extract_building_footprint(floor_mesh=building_floor)
    
    if building_footprint.geom_type == 'MultiPolygon':
        building_footprint = list(building_footprint.geoms)[0]
    
    # create a small buffer to avoid precision issues
    building_footprint_buffer = building_footprint.buffer(0.5)
    
    building_vertices_2d = np.column_stack((building.vertices[:, 0], building.vertices[:, 2]))
    
    enclosed_num = 0
    for point in building_vertices_2d:
        if building_footprint_buffer.contains(Point(point)):
            enclosed_num += 1
    
    enclosed_ratio = enclosed_num / len(building_vertices_2d)
    
    
    # 2. facade
    ## vertical alignment
    tolerance = 1e-2
    
    building_facade_normals = building_facade.face_normals
    building_facade_normal_y = np.abs(building_facade_normals[:, 1])
    building_facade_horizontal_mask = building_facade_normal_y <= tolerance
    building_facade_face_horizontal_indices = np.where(building_facade_horizontal_mask)[0]
    
    building_facade_horizontal_count = np.sum(building_facade_horizontal_mask)
    
    building_facade_horizontal_ratio = building_facade_horizontal_count / len(building_facade_normals)
    
    # print(building_facade_normal_y)
    # print(building_facade_horizontal_count, len(building_facade_normals))
    # print(f'initial_facade: {building_facade_horizontal_ratio}')
    
    ## bottom edge enclosing area
    building_facade_edges = []
    for face in building_facade.faces:
        v1, v2, v3 = face
        v1_h = building_facade.vertices[v1][1]
        v2_h = building_facade.vertices[v2][1]
        v3_h = building_facade.vertices[v3][1]
        
        if v1_h <= y_mid:
            if v2_h <= y_mid:
                building_facade_edges.append([v1, v2])
            elif v3_h <= y_mid:
                building_facade_edges.append([v1, v3])
        elif v2_h <= y_mid:
            if v3_h <= y_mid:
                building_facade_edges.append([v2, v3])
        
    building_facade_edges = np.array(building_facade_edges)
    if not building_facade_edges.shape[0]:
        return None, 0.0
    
    building_facade_edges = np.sort(building_facade_edges, axis=1)
    
    building_facade_edges_unique = np.array(list(set(map(tuple, building_facade_edges))))
    building_facade_edges_unique = building_facade_edges_unique[np.lexsort((building_facade_edges_unique[:, 1], building_facade_edges_unique[:, 0]))]
    
    building_facade_edges_polygons = edges_to_polygons(building_facade_edges_unique, 
                                                       building_facade.vertices, 
                                                       logging=logging)
    
    building_facade_edges_polygons_areas = [polygon.area for polygon in building_facade_edges_polygons]
    if not len(building_facade_edges_polygons_areas):
        return None, 0.0
    building_facade_edges_polygons_areas = np.array(building_facade_edges_polygons_areas)
    building_facade_edges_bottom_polygon = building_facade_edges_polygons[np.argmax(building_facade_edges_polygons_areas)]
    
    ##
    building_facade_edges_bottom_polygon_area = building_facade_edges_bottom_polygon.area
    building_footprint_area = building_footprint.area
    building_facade_subtracted_area = building_facade_edges_bottom_polygon_area - building_footprint_area
    building_facade_subtracted_ratio = building_facade_subtracted_area / building_footprint_area
    
    building_facade_anticipated_span = [-0.1, 0.1]
    building_facade_inout_punishment = exponential_window(building_facade_subtracted_ratio, 
                                                          building_facade_anticipated_span[0], building_facade_anticipated_span[1])
    
    building_facade_horizontal_ratio *= building_facade_inout_punishment
    
    # print(f'punishment_facade: {building_facade_inout_punishment}')
    
    # 3. roof
    ## roof connectivity
    connected_edge_ratio = analyze_building_roof_conn(building_roof, building_facade, building_footprint)
    sole_edge_ratio, sole_edge_lengths = analyze_building_conn(building)
    
    if len(sole_edge_lengths):
        sole_edge_max_length = max(sole_edge_lengths)
        sole_edge_anticipated_span = [-0.5, 1.0]
        sole_edge_punishment_index = exponential_window(sole_edge_max_length, 
                                                        sole_edge_anticipated_span[0], 
                                                        sole_edge_anticipated_span[1])
    else:
        sole_edge_punishment_index = 1.0
    
    
    # print(f'sole puni. {sole_edge_punishment_index}')
    connected_edge_ratio = min(connected_edge_ratio, 1 - sole_edge_ratio) * sole_edge_punishment_index
    
    ## roof floor area ratio
    building_roof_area = calculate_horizontal_projected_area(building_roof.vertices, building_roof.faces)
    building_roof_subtracted_area = building_roof_area - building_footprint_area
    building_roof_subtracted_area_ratio = building_roof_subtracted_area / building_footprint_area
    
    building_roof_anticipated_span = [-0.1, 0.5]
    building_roof_area_punishment = exponential_window(building_roof_subtracted_area_ratio, 
                                                       building_roof_anticipated_span[0], building_roof_anticipated_span[1])
    
    connected_edge_ratio *= building_roof_area_punishment
    
    # 4. Overall connection
    # sole_edge_ratio, sole_edge_lengths = analyze_building_conn(building)
    # print(sole_edge_ratio)
    # print(f'Sole edge lengths: {sole_edge_lengths}')
    # print(f'Total sole edge length: {sum(sole_edge_lengths) if sole_edge_lengths else 0.0}')
    
    # AVG: BMQI
    bmqi = enclosed_ratio * 0.1 + building_facade_horizontal_ratio * 0.3 + connected_edge_ratio * 0.6
    
    building_parts = [building_roof, building_facade, building_floor]
    
    if logging:
        print(f'EVAL 1: Floor score: {enclosed_ratio * 100}%')
        print(f'EVAL 2: Facade score: {building_facade_horizontal_ratio * 100}%')
        print(f'EVAL 3: Connection score: {connected_edge_ratio * 100}%')
        print(f'BMQI: {bmqi}')
    
    return building_parts, bmqi
    

if __name__ == '__main__':
    # plateau_obj_dir = '/home/sekilab-liao/Documents/gen3d_2_0/BldgGen2024/test_bmqi_obj/52385628'
    # # fp_geojson_path = '/home/sekilab-liao/Documents/gen3d_2_0/BldgGen2024/52353680.geojson'
        
    # bmqi_sum = []
    # for r, ds, fs in os.walk(plateau_obj_dir):
    #     for f in fs:
    #         if f.lower().endswith('.obj'):
    #             building_mesh_path = os.path.join(r, f)
    #             mesh = load_building_mesh(building_mesh_path)

    #             print(f)
    #             mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]
    #             _, bmqi = divide_building_mesh(mesh)
    #             bmqi_sum.append(bmqi)
                
    #             if bmqi < 1.0:
    #                 mesh.fix_normals()
    #                 mesh.export(f'../test_bmqi_obj/outliers_52385628/bmqi_{bmqi:.3f}_{f}')
    
    # bmqi_sum_array = np.array(bmqi_sum)
    # bmqi_avg = np.average(bmqi_sum_array)
    # print(f'avg bmqi: {bmqi_avg}')
    
    # # Create the histogram
    # plt.figure(figsize=(10, 6))
    # plt.hist(bmqi_sum_array, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    # plt.title('Histogram of Values in [0, 1]')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig('bmqi.png', dpi=300)
    
    
    
    building_mesh_path = '../../test_bmqi_obj/outliers_52385628/bmqi_0.49630011496840926_bldg_4614b8f0-07c2-4423-84a3-41d3bacfa808.obj'
    
    building = divide_building_mesh(load_building_mesh(building_mesh_path))
    
    