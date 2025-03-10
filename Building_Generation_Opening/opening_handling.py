import open3d as o3d
import numpy as np

import os, time
import trimesh

from earcut import earcut
import triangle

from scipy.spatial import cKDTree

from .json_handler import plateau_route
from .coordinate_util import convert_ECEF_to_WGS84, convert_WGS84_to_Japan_Rect_30169


def triangulate_polygon_with_holes(exterior_points, holes_points):
    """
    Triangulate a polygon with holes
    
    Parameters:
    exterior_points: list of points defining the exterior boundary [(x1,y1), (x2,y2),...]
    holes_points: list of list of points defining each hole [[(x1,y1), (x2,y2),...], [...]]
    """
    # Create the dictionary for triangle
    polygon = {
        'vertices': np.array(exterior_points),
        'segments': np.array([(i, (i + 1) % len(exterior_points)) 
                            for i in range(len(exterior_points))]),
        'holes': []
    }
    
    # Add holes
    vertex_offset = len(exterior_points)
    for hole in holes_points:
        # Add hole vertices
        polygon['vertices'] = np.vstack((
            polygon['vertices'],
            np.array(hole)
        ))
        
        # Add hole segments
        hole_segments = np.array([
            (vertex_offset + i, vertex_offset + (i + 1) % len(hole))
            for i in range(len(hole))
        ])
        polygon['segments'] = np.vstack((
            polygon['segments'],
            hole_segments
        ))
        
        # Add a point inside the hole for triangle to recognize it
        hole_center = np.mean(hole, axis=0)
        polygon['holes'].append(hole_center)
        
        # Update offset for next hole
        vertex_offset += len(hole)
    
    # Triangulate
    triangulation = triangle.triangulate(polygon, 'p')
    
    return triangulation


def o3d_to_obj(mesh: o3d.t.geometry.TriangleMesh, 
               path: str = "test_obj/test_opening.obj"):
    vertices = mesh.vertex.positions.numpy().astype(np.float32)
    faces = mesh.triangle.indices.numpy().astype(np.int8)
    
    output_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    output_mesh.export(path)

def wall_opening(wall_width=3.0, wall_height=2.0, wall_thickness=0.5,
                 window_width=0.5, window_height=0.4, window_depth=0.1, window_x=1.0, window_y=0.6):
    device = o3d.core.Device("CPU:0")
    
    wall_vertices = np.array([
        [0, 0, 0],
        [wall_width, 0, 0],
        [wall_width, wall_height, 0],
        [0, wall_height, 0],
        [0, 0, wall_thickness],
        [wall_width, 0, wall_thickness],
        [wall_width, wall_height, wall_thickness],
        [0, wall_height, wall_thickness]
    ])
    window_vertices = np.array([
        [window_x, window_y, -window_depth],  # front bottom-left
        [window_x + window_width, window_y, -window_depth],  # front bottom-right
        [window_x + window_width, window_y + window_height, -window_depth],  # front top-right
        [window_x, window_y + window_height, -window_depth],  # front top-left
        [window_x, window_y, window_depth],  # back bottom-left
        [window_x + window_width, window_y, window_depth],  # back bottom-right
        [window_x + window_width, window_y + window_height, window_depth],  # back top-right
        [window_x, window_y + window_height, window_depth]  # back top-left
    ])
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # front face
        [4, 6, 5], [4, 7, 6],  # back face
        [0, 4, 5], [0, 5, 1],  # bottom face
        [1, 5, 6], [1, 6, 2],  # right face
        [2, 6, 7], [2, 7, 3],  # top face
        [3, 7, 4], [3, 4, 0]   # left face
    ])
    
    wall = o3d.t.geometry.TriangleMesh(device)
    window = o3d.t.geometry.TriangleMesh(device)
    
    wall.vertex.positions = o3d.core.Tensor(wall_vertices, 
                                            o3d.core.float32, 
                                            device)
    wall.triangle.indices = o3d.core.Tensor(faces, 
                                            o3d.core.int32, 
                                            device)
    
    window.vertex.positions = o3d.core.Tensor(window_vertices, 
                                              o3d.core.float32, 
                                              device)
    window.triangle.indices = o3d.core.Tensor(faces, 
                                              o3d.core.int32, 
                                              device)
    
    o3d_to_obj(wall, "test_obj/wall.obj")
    o3d_to_obj(window, "test_obj/window.obj")
    
    # wall = o3d.t.geometry.TriangleMesh.create_box(
    #     height=wall_height,
    #     depth=wall_thickness, 
    #     device=device
    # )
    # wall.translate([-wall_thickness / 2, 0, 0])
    
    # window = o3d.t.geometry.TriangleMesh.create_box(
    #     height=window_height,
    #     depth=window_thickness,
    #     device=device
    # )
    
    start_time = time.time()
    wall_with_opening = window.boolean_difference(wall)
    o3d_to_obj(wall_with_opening)
    print(f'time elapsed: {time.time() - start_time}')
    
    return wall_with_opening
    
    
def local_to_global(local_original: np.array, 
                    local_point: np.array, 
                    transformation_matrix: np.array):
    return local_original + np.dot(transformation_matrix, local_point)

def xyz_footprint_conversion(footprint):
    xyz_footprint = []
    for point in footprint:
        x, y, z = convert_ECEF_to_WGS84(point[0], point[1], point[2])
        x, y = convert_WGS84_to_Japan_Rect_30169(x, y)
        
        xyz_footprint.append([x, y, z])
        
    xyz_footprint = np.array(xyz_footprint)
    return xyz_footprint
    

def vertices_match(xyz_footprint, 
                   mesh, 
                   lod2_vertices):
    threshold = 0.1
    distances = np.sqrt(np.sum((xyz_footprint[:, None, :2] - lod2_vertices[None, :, :2]) ** 2, axis=2))
    within_indices = np.where(distances < threshold)
    
    # print(xyz_footprint, lod2_vertices)
    # print(within_indices)
    # aa
    
    matching_lists = [within_indices[1][within_indices[0] == i].tolist() for i in range(len(xyz_footprint))]
    
    # for pair in matching_lists:
    #     if pair == []:
    
    matching_z_lists = [lod2_vertices[pair, 2] for pair in matching_lists]
    matching_mean_delta_z = np.mean(np.array([abs(pair[1] - pair[0]) for pair in matching_lists if not pair == []]))
    
    matching_z_lists = np.array(matching_z_lists)
    
    matching_z_lists_min_idx = np.argmin(matching_z_lists, axis=1)
    matching_z_lists_max_idx = np.argmax(matching_z_lists, axis=1)
    
    lod2_footprint_idx = [pair[idx] for pair, idx in zip(matching_lists, matching_z_lists_min_idx)]
    lod2_top_idx = [pair[idx] for pair, idx in zip(matching_lists, matching_z_lists_max_idx)]

    return lod2_footprint_idx, lod2_top_idx


def facade_retriangulation(obj_path = None, 
                           obj: trimesh.Trimesh = None, 
                           footprint: np.array = None):
    if not obj:
        mesh = trimesh.load(obj_path)
    else:
        mesh = obj
    mesh.fix_normals(multibody=True)
    
    footprint_idx, top_idx = vertices_match(footprint, mesh, mesh.vertices)
    footprint_top_idx = dict(zip(footprint_idx, top_idx))
    
    def triangle_normal(mesh, face_index):
        v0, v1, v2 = mesh.vertices[mesh.faces[face_index]]
        
        vec1 = v1 - v0
        vec2 = v2 - v0
        
        normal = np.cross(vec1, vec2)
        return normal / np.linalg.norm(normal)
    
    normals = []
    for face in range(len(mesh.faces)):
        normals.append(triangle_normal(mesh, face))
    
    normals = np.array(normals)
    normals_vertical_mask = np.abs(normals[:, 2]) < 0.05
    normals_vertical_indices = np.where(normals_vertical_mask)[0]
    normals_vertical = normals[normals_vertical_mask]
    
    # group by normal orientation
    groups = []
    group_ids = []
    processed = 1
    
    def normal_vector_angle(n1, n2):
        return np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))
    
    for normal, normal_idx in zip(normals_vertical, normals_vertical_indices):
        if not len(groups):
            groups.append([int(normal_idx)])
            group_ids.append(1)
            continue
        
        processed += 1
        
        for idx, n in enumerate(groups):
            if normal_vector_angle(normal, normals[n[0]]) <= 2.0:
                n.append(int(normal_idx))
                group_ids.append(idx + 1)
                break

        if not len(group_ids) == processed:
            groups.append([int(normal_idx)])
            group_ids.append(len(groups))
    
    # re-triangulation
    error_groups = []
    for group in groups:
        if len(group) <= 2:
            continue  # TODO: Handle two tri but not rect
        
        error_groups.extend(group)
        
        faces_extracted = mesh.faces[group]
        
        # save original connection
        edge_extracted = []
        for face in faces_extracted:
            edge_extracted.append(sorted([int(face[0]), int(face[1])]))
            edge_extracted.append(sorted([int(face[1]), int(face[2])]))
            edge_extracted.append(sorted([int(face[2]), int(face[0])]))
            
        edge_extracted = np.array(edge_extracted)
        edge_extracted_unique, count = np.unique(edge_extracted, axis=0, return_counts=True)
        edge_extracted_polygon = edge_extracted_unique[count == 1]
        
        def edges_to_polygon(edges):
            connections = {}
            for v1, v2 in edges:
                if v1 not in connections:
                    connections[v1] = []
                if v2 not in connections:
                    connections[v2] = []
                
                connections[v1].append(v2)
                connections[v2].append(v1)
                
                if len(connections[v1]) > 2 or len(connections[v2]) > 2:
                    raise Exception('Non-boundary vertex exists. ')
            
                
            # Start with vertex that appears only twice (if it's a boundary)
            vertex_count = {v: len(connections[v]) for v in connections}
            start_vertex = min(vertex_count, key=vertex_count.get)
            
            # Build polygon by following connections
            polygon = [start_vertex]
            current = start_vertex
            used_edges = set()
            
            while True:
                # Find next unused vertex
                for next_vertex in connections[current]:
                    edge = tuple(sorted([current, next_vertex]))
                    if edge not in used_edges:
                        polygon.append(next_vertex)
                        used_edges.add(edge)
                        current = next_vertex
                        break
                        
                if len(used_edges) == len(edges):
                    break
                    
            return np.array(polygon)
        
        polygon_extracted = edges_to_polygon(edge_extracted_polygon)
        
        fp_idx = set(polygon_extracted).intersection(footprint_idx)
        assert len(fp_idx) == 2, 'Invalid facade surface due for more than two footprint vertices. '
        left_bottom, right_bottom = fp_idx
        left_top, right_top = footprint_top_idx[left_bottom], footprint_top_idx[right_bottom]
        
        polygon_extracted_subtracted = polygon_extracted[~np.isin(polygon_extracted, [left_bottom, right_bottom])]
        # if polygon_extracted_subtracted[0] != polygon_extracted_subtracted[-1]:
        #     polygon_extracted_subtracted = np.append(polygon_extracted_subtracted, polygon_extracted_subtracted[0])
        
        polygon_extracted_subtracted_vertices = mesh.vertices[polygon_extracted_subtracted]
        
        def project_to_2d(vertices):
            mean = np.mean(vertices, axis=0)
            centered = vertices - mean
            _, _, vh = np.linalg.svd(centered)
            
            vertices_projected = centered @ vh[:2].T
            
            return vertices_projected
        
        vertices_projected = project_to_2d(polygon_extracted_subtracted_vertices)
        vertices_projected_flattened = vertices_projected.ravel()
        
        polygon_extracted_subtracted_vertices_projected_triangulated = earcut.earcut(vertices_projected_flattened)
        polygon_extracted_subtracted_vertices_projected_triangulated = np.array(polygon_extracted_subtracted_vertices_projected_triangulated).reshape(-1, 3)
        
        polygon_esvpt_matched = polygon_extracted_subtracted[polygon_extracted_subtracted_vertices_projected_triangulated]
        
        # re-triangulate
        new_facade_faces = np.array([
            [left_bottom, right_bottom, right_top], 
            [left_bottom, right_top, left_top]
        ])
        
        mesh.faces = np.vstack((mesh.faces, new_facade_faces, polygon_esvpt_matched))
              
    # drop error faces
    error_groups = np.array(error_groups)
    face_mask = np.ones(len(mesh.faces), dtype=bool)
    face_mask[error_groups] = False
    mesh.update_faces(face_mask)

    # final orientation
    mesh.fix_normals(multibody=True)
    
    return mesh


def available_mesh_creation(xyz_footprint, height):
    xyz_footprint_min_z = np.min(xyz_footprint[:, 2])
    for v in xyz_footprint:
        v[-1] -= xyz_footprint_min_z
    
    # top
    xyz_footprint_flattened = xyz_footprint[:, :2].ravel()
    xyz_footprint_flattened_triangulated = earcut.earcut(xyz_footprint_flattened)
    xyz_footprint_flattened_faces = np.array(xyz_footprint_flattened_triangulated).reshape(-1, 3)
    
    xyz_top = np.array([[v[0], v[1], v[2] + height] for v in xyz_footprint])
    xyz_top_faces = xyz_footprint_flattened_faces + len(xyz_footprint)
    
    mesh_vertices = np.vstack([xyz_footprint, xyz_top])
    mesh_faces = np.vstack([xyz_footprint_flattened_faces, xyz_top_faces])
    
    # facade
    for i in range(len(xyz_footprint)):
        left_bottom = i
        right_bottom = 0 if i == len(xyz_footprint) - 1 else i + 1
        right_top = len(xyz_footprint) if i == len(xyz_footprint) - 1 else len(xyz_footprint) + i + 1
        left_top = len(xyz_footprint) + i
        
        faces = np.array([
            [left_bottom, right_top, left_top], 
            [left_bottom, right_bottom, right_top]
        ])
        
        mesh_faces = np.vstack([mesh_faces, faces])
        
    mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
    # mesh.fix_normals(multibody=True)
    
    return mesh
    

def mesh_opening(building, 
                 route, 
                 generated_mesh, 
                 default_color, 
                 generated = True):
    start_time = time.time()
    
    demo_building = building
    translated_footprint = demo_building.footprint + route.geo_center
    
    xyz_footprint = xyz_footprint_conversion(translated_footprint)
    
    # lod2 model loading
    # lod2_mesh = trimesh.load('../citygml_io/obj/53393555/bldg_ca2b5b06-6bd8-4313-812f-67580049c368.obj')
    if generated:
        # lod2_mesh = generated_mesh
        lod2_mesh = available_mesh_creation(xyz_footprint, demo_building.height)
    # else:
    #     lod2_mesh = trimesh.load('test_obj/sample/01_type3.obj')
    
    lod2_mesh.fix_normals(multibody=True)
    
    # if generated:
    #     lod2_mesh.vertices = np.array([[x, -z, y] for x, y, z in lod2_mesh.vertices])
        # lod2_mesh = face_orientation(lod2_mesh)
        # lod2_mesh.export('test_obj/flip_test.obj')
    
    # lod2_mesh = facade_retriangulation(obj=lod2_mesh, 
    #                                    footprint=xyz_footprint)
    
    # retri_time = time.time() - start_time
    # print(f're-triangulation: {retri_time}')
    
    # match lod2 vertices
    lod2_footprint_idx, lod2_top_idx = vertices_match(xyz_footprint, lod2_mesh, lod2_mesh.vertices)

    # add color representation
    lod2_mesh.visual.face_colors = np.tile(default_color, (len(lod2_mesh.faces), 1))

    # print(lod2_mesh.visual.face_colors)
    # print(lod2_mesh.vertices, lod2_mesh.faces)
    # aa

    def local_to_global(left_bottom_point, 
                        vector_local_scale, 
                        local_point):
        return [left_bottom[0] + vector_local_scale[0] * local_point[0], 
                left_bottom[1] + vector_local_scale[1] * local_point[0],
                left_bottom[2] + vector_local_scale[2] * local_point[1]]

    # create openings
    for face in demo_building.opening:
        edge_id = face.edge_id
        left_bottom_idx = lod2_footprint_idx[edge_id]
        right_bottom_idx = lod2_footprint_idx[0] if edge_id == len(lod2_footprint_idx) - 1 else lod2_footprint_idx[edge_id + 1]
        right_top_idx = lod2_top_idx[0] if edge_id == len(lod2_footprint_idx) - 1 else lod2_top_idx[edge_id + 1]
        left_top_idx = lod2_top_idx[edge_id]
        
        bottom_possible_faces = np.array([sorted([left_bottom_idx, right_bottom_idx, right_top_idx]), sorted([left_bottom_idx, right_bottom_idx, left_top_idx])])
        top_possible_faces = np.array([sorted([left_top_idx, right_top_idx, left_bottom_idx]), sorted([left_top_idx, right_top_idx, right_bottom_idx])])
        
        sorted_faces = np.sort(lod2_mesh.faces)
        # print(bottom_possible_faces, top_possible_faces, sorted_faces)
        
        # remove the original wall faces
        mask = ~np.all(np.isin(sorted_faces, bottom_possible_faces), axis=1)
        lod2_mesh.faces = lod2_mesh.faces[mask]
        lod2_mesh.visual.face_colors = lod2_mesh.visual.face_colors[mask]
        
        left_bottom = lod2_mesh.vertices[left_bottom_idx]
        right_top = lod2_mesh.vertices[right_top_idx]
        
        vector_local_scale = right_top - left_bottom
        
        holes = []
        for opening in face.openings:
            opening_left_bottom = np.array([left_bottom[0] + vector_local_scale[0] * opening['x'], 
                                            left_bottom[1] + vector_local_scale[1] * opening['x'], 
                                            left_bottom[2] + vector_local_scale[2] * opening['y']])
            opening_left_top = np.array([left_bottom[0] + vector_local_scale[0] * opening['x'], 
                                         left_bottom[1] + vector_local_scale[1] * opening['x'], 
                                         left_bottom[2] + vector_local_scale[2] * (opening['y'] + opening['h'])])
            opening_right_top = np.array([left_bottom[0] + vector_local_scale[0] * (opening['x'] + opening['w']), 
                                          left_bottom[1] + vector_local_scale[1] * (opening['x'] + opening['w']), 
                                          left_bottom[2] + vector_local_scale[2] * (opening['y'] + opening['h'])])
            opening_right_bottom = np.array([left_bottom[0] + vector_local_scale[0] * (opening['x'] + opening['w']), 
                                             left_bottom[1] + vector_local_scale[1] * (opening['x'] + opening['w']), 
                                             left_bottom[2] + vector_local_scale[2] * opening['y']])

            opening_vertices = np.array([opening_left_bottom, opening_right_bottom, opening_right_top, opening_left_top])
            opening_faces = np.array([[len(lod2_mesh.vertices), 
                                       len(lod2_mesh.vertices) + 1, 
                                       len(lod2_mesh.vertices) + 2], 
                                      [len(lod2_mesh.vertices), 
                                       len(lod2_mesh.vertices) + 2, 
                                       len(lod2_mesh.vertices) + 3]])
            opening_colors = np.array([[245, 206, 66, 255], 
                                       [245, 206, 66, 255]])
            
            lod2_mesh.vertices = np.vstack((lod2_mesh.vertices, opening_vertices))
            lod2_mesh.faces = np.vstack((lod2_mesh.faces, opening_faces))
            lod2_mesh.visual.face_colors = np.vstack((lod2_mesh.visual.face_colors, opening_colors))
            
            holes.append([(opening['x'], opening['y']), 
                          (opening['x'] + opening['w'], opening['y']),
                          (opening['x'] + opening['w'], opening['y'] + opening['h']),
                          (opening['x'], opening['y'] + opening['h'])])            

        # opening holes
        face_with_holes = triangulate_polygon_with_holes([(0, 0), (1, 0), (1, 1), (0, 1)], 
                                                         holes)
        
        face_with_holes_vertices = np.array([local_to_global(left_bottom, vector_local_scale, point) for point in face_with_holes['vertices']])
        face_with_holes_faces = np.array([[face[0] + len(lod2_mesh.vertices), face[1] + len(lod2_mesh.vertices), face[2] + len(lod2_mesh.vertices)] for face in face_with_holes['triangles']])
        face_with_holes_colors = np.array([default_color for _ in range(len(face_with_holes_faces))])

        lod2_mesh.vertices = np.vstack((lod2_mesh.vertices, face_with_holes_vertices))
        lod2_mesh.faces = np.vstack((lod2_mesh.faces, face_with_holes_faces))
        lod2_mesh.visual.face_colors = np.vstack((lod2_mesh.visual.face_colors, face_with_holes_colors))

        # print(len(lod2_mesh.faces), len(lod2_mesh.visual.face_colors))
        # print(len(lod2_mesh.vertices))
        # print(lod2_mesh.faces)
        # print(lod2_mesh.visual.face_colors)
        # break
    
    opening_time = time.time() - start_time
    print(f'opening time: {opening_time}')
    
    if False:
        lod2_mesh.export('test_obj/test_color_opening_re.obj')
    
    if generated:
        lod2_mesh.vertices = np.array([[x, y, -z] for x, z, y in lod2_mesh.vertices])
    
    lod2_mesh.export(f'Building_Generation_Opening/temp/{building.bldg_id}.obj')
    
    return lod2_mesh
    
    
if __name__ == "__main__":
    mesh_opening('test_route/bldg-elem_route3.json')
    # wall_with_opening = wall_opening()
    # o3d.io.write_triangle_mesh("test_obj/wall_with_opening.obj", wall_with_opening)