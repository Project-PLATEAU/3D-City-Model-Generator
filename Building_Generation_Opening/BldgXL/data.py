import os
import torch
import trimesh
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
from pathlib import Path

from dataset import MeshDataset

def type_map_bdg(data):
    data[(data == 5) | (data == 6) | (data == 7) | (data == 9) | (data == 12) | (data == 13)] = 5
    data[(data == 8) | (data == 11)] = 6
    data[(data == 10)] = 7
    return data

def get_mesh(file_path):
    mesh = trimesh.load(file_path, force='mesh')
    vertices = mesh.vertices.tolist()
    if ".off" in file_path:  # ModelNet dataset
        mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]
        rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0])
        mesh.apply_transform(rotation_matrix)
        # Extract vertices and faces from the rotated mesh
        vertices = mesh.vertices.tolist()

    original_vertices = np.array(vertices)
    
    x_coords = original_vertices[:, 0]
    y_coords = original_vertices[:, 1]
    
    x_scale = np.max(x_coords) - np.min(x_coords)
    y_scale = np.max(y_coords) - np.min(y_coords)

    xy_scale_factor = np.sqrt(x_scale ** 2 + y_scale ** 2)

    faces = mesh.faces.tolist()

    centered_vertices = vertices - np.mean(vertices, axis=0)
    max_abs = np.max(np.abs(centered_vertices))
    vertices = centered_vertices / (max_abs / 0.95)  # Limit vertices to [-0.95, 0.95]

    vertices[:, [1, 2]] = vertices[:, [2, 1]]
    # min_y = np.min(vertices[:, 1])
    # difference = -0.95 - min_y
    # vertices[:, 1] += difference


    def sort_vertices(vertex):
        return vertex[1], vertex[2], vertex[0]


    seen = OrderedDict()
    for point in vertices:
        key = tuple(point)
        if key not in seen:
            seen[key] = point

    unique_vertices = list(seen.values())
    sorted_vertices = sorted(unique_vertices, key=sort_vertices)

    vertices_as_tuples = [tuple(v) for v in vertices]
    sorted_vertices_as_tuples = [tuple(v) for v in sorted_vertices]

    vertex_map = {old_index: new_index for old_index, vertex_tuple in enumerate(vertices_as_tuples) for
                  new_index, sorted_vertex_tuple in enumerate(sorted_vertices_as_tuples) if
                  vertex_tuple == sorted_vertex_tuple}
    reindexed_faces = [[vertex_map[face[0]], vertex_map[face[1]], vertex_map[face[2]]] for face in faces]
    sorted_faces = np.array([sorted(sub_arr) for sub_arr in reindexed_faces])


    face_centroids = np.mean(vertices[faces], axis=1)  # (num_faces, 3)
    sorted_indices = np.lexsort((face_centroids[:, 2], face_centroids[:, 0], face_centroids[:, 1]))
    sorted_faces = sorted_faces[sorted_indices]

    return np.array(sorted_vertices), sorted_faces, xy_scale_factor

def load_label(obj_folder, label, ratio=0.1, variation = 1, text_path = None):
    obj_datas = []
    sampled_label = label.sample(frac=ratio)
    
    text_df = None
    if text_path is not None:
        text_df = pd.read_csv(text_path)
        text_df = text_df[['id', 'text']]
        text_df.set_index('id', inplace=True)

    for i in tqdm(range(len(sampled_label))):
        obj_path = os.path.join(obj_folder, sampled_label['id'].values[i]+'.obj')
        if os.path.isfile(obj_path):
            vertices, faces, scale_factor = get_mesh(obj_path)
            if len(faces) > 200:
                continue

            def augment_mesh(vertices: np.array, scale_factor: float):
                jitter_factor = 0.01
                possible_values = np.arange(-jitter_factor, jitter_factor, 0.0005)
                offsets = np.random.choice(possible_values, size = vertices.shape)

                vertices += offsets
                vertices *= scale_factor

                min_y = np.min(vertices[:, 1])
                difference = -0.95 - min_y
                vertices[:, 1] += difference

                return vertices

            faces = torch.tensor(faces.tolist(), dtype=torch.long).to("cuda")
            
            if text_path is not None:
                if text_df.index.isin([sampled_label['id'].values[i]]).any():
                    texts = text_df.loc[sampled_label['id'].values[i]]['text']
                else:
                    continue
            else:
                texts = 'type ' + str(sampled_label['type'].values[i])
            
            # scale_possible_values = np.arange(0.75, 1.0, 0.005)
            # scale_factors = np.random.choice(scale_possible_values, size = variation)

            # for sf in scale_factors:
            # vertices = augment_mesh(vertices.copy(), scale_factor = 1.)

            obj_data = {"vertices": torch.tensor(vertices.tolist(), dtype=torch.float).to("cuda"),
                        "faces": faces, "texts": texts, "scale": scale_factor}
            obj_datas.append(obj_data)


    print(f"[create_mesh_dataset] Returning {len(obj_datas)} meshes")
    return obj_datas

obj_label = pd.read_csv('obj_label.csv')
# obj_label['type'] = type_map_bdg(obj_label['type'].values)
# obj_label = obj_label[obj_label['type'] == 5]

project_name = "plateau_lod2_type_mixed_full"

working_dir = f'./{project_name}'

working_dir = Path(working_dir)
working_dir.mkdir(exist_ok=True, parents=True)
dataset_path = working_dir / (project_name + ".npz")

if not os.path.isfile(dataset_path):
    data = load_label("obj_unlabeled_flattened", obj_label, 
                      ratio = 1., variation = 1, 
                      text_path = "obj_unlabeled_text_type_simple.csv")
    dataset = MeshDataset(data)
    dataset.save(dataset_path)

dataset = MeshDataset.load(dataset_path)
print(dataset.data[0].keys())
print(dataset[0])