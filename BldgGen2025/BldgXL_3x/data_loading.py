import os, random
import torch
import trimesh
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from utils.bmqi import divide_building_mesh

from dataset import MeshDataset

import clip

from PIL import Image

device = "cuda" if torch.cuda.is_available() else 'cpu'
# clipModel, clipPreprocess = clip.load("ViT-B/32", device=device)

def load_obj_file(obj_path):
    """Load OBJ file and extract vertices and faces, converting z-up to y-up."""
    vertices = []
    faces = []
    
    with open(obj_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('v '):
                # Vertex line: v x y z
                coords = line.split()[1:4]
                x, y, z = [float(coord) for coord in coords]
                # Convert from z-up to y-up: (x, y, z) -> (x, z, -y)
                vertex = [x, z, -y]
                vertices.append(vertex)
            elif line.startswith('f '):
                # Face line: f v1 v2 v3 (or f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3)
                face_data = line.split()[1:]
                face = []
                for vertex_data in face_data:
                    # Handle faces with texture/normal indices (v/vt/vn format)
                    vertex_index = int(vertex_data.split('/')[0]) - 1  # OBJ uses 1-based indexing
                    face.append(vertex_index)
                faces.append(face)
    
    return vertices, faces


def normalize_vertices(vertices, target_range=(-0.95, 0.95)):
    """Normalize vertices to target range and return scale factor."""
    vertices = np.array(vertices)
    
    # Find bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    
    # Calculate original size
    original_size = np.max(max_coords - min_coords)
    
    # Calculate center
    center = (min_coords + max_coords) / 2
    
    # Center the vertices
    centered_vertices = vertices - center
    
    # Calculate scale factor to fit in target range
    target_size = target_range[1] - target_range[0]
    scale_factor = target_size / original_size if original_size > 0 else 1.0
    
    # Apply scaling
    normalized_vertices = centered_vertices * scale_factor
    
    return normalized_vertices.tolist(), original_size / target_size


def load_text_mapping(csv_path):
    """Load text mapping from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        # Convert to dictionary mapping id to text
        return dict(zip(df['id'], df['text']))
    except Exception as e:
        print(f"Warning: Could not load text mapping from {csv_path}: {e}")
        return {}


def find_obj_files(root_folder):
    """Find all OBJ files in nested subfolders."""
    obj_files = []
    root_path = Path(root_folder)

    for obj_file in root_path.rglob("*.obj"):
        obj_files.append(obj_file)

    return obj_files


def read_certain_list_folder(certain_list_folder):
    """
    Recursively read the certain_list folder containing .obj files (with sub-folders).

    Args:
        certain_list_folder: Path to the certain_list folder

    Returns:
        List of Path objects pointing to all .obj files found
    """
    obj_files = []
    folder_path = Path(certain_list_folder)

    if not folder_path.exists():
        print(f"Warning: Folder {certain_list_folder} does not exist")
        return obj_files

    obj_paths = []
    for obj_file in folder_path.rglob("*.tiff"):
        obj_file_name = os.path.splitext(os.path.basename(obj_file))[0] + '_lod2'
        obj_files.append(obj_file_name)
        
        obj_paths.append(obj_file)

    return obj_files, obj_paths


def create_mesh_dataset(root_folder, csv_path=None, ratio=None, certain_list=None, certain_paths=None):
    """
    Create MeshDataset from OBJ files in nested folders.
    
    Args:
        root_folder: Path to root folder containing subfolders with OBJ files
        csv_path: Path to CSV file with id,text mapping (optional)
    
    Returns:
        MeshDataset instance
    """
    # Load text mapping if provided
    text_mapping = {}
    if csv_path and os.path.exists(csv_path):
        text_mapping = load_text_mapping(csv_path)
    
    # Find all OBJ files
    obj_files = find_obj_files(root_folder)
    print(f"Found {len(obj_files)} OBJ files")
    
    if ratio:
        obj_files = random.sample(obj_files, k=int(len(obj_files) * ratio))
    
    data = []
    
    for idx, obj_file in tqdm(enumerate(obj_files)):
        try:
            obj_img = None
            
            obj_file_name = os.path.splitext(os.path.basename(obj_file))[0]
            if certain_list and not obj_file_name in certain_list:
                continue
            elif certain_list:
                obj_img_path = certain_paths[certain_list.index(obj_file_name)]
                obj_img = Image.open(obj_img_path)
                
                # obj_img = clipPreprocess(obj_img)
            
            # print(obj_file_name, certain_paths[certain_list.index(obj_file_name)])
            # ancd
            
            # Load OBJ file
            vertices, faces = load_obj_file(obj_file)
            if vertices is None or len(faces) > 200:
                continue
            
            mesh_bmqi = trimesh.Trimesh(vertices, faces)
            
            _, mesh_bmqi = divide_building_mesh(mesh_bmqi, logging=False)
            # print(mesh_bmqi)
            if mesh_bmqi <= 0.9999:
                continue
            
            if not vertices:
                print(f"Warning: No vertices found in {obj_file}")
                continue
            
            # Normalize vertices
            normalized_vertices, scale = normalize_vertices(vertices)
            
            # Get text from mapping (use filename as fallback)
            file_id = obj_file.stem  # filename without extension
            text = text_mapping.get(file_id)
            if csv_path and text is None:
                continue
            
            # Create data entry
            entry = {
                'vertices': torch.from_numpy(np.array(normalized_vertices, dtype=np.float32)),
                'faces': torch.from_numpy(np.array(faces, dtype=np.long)),
                'texts': text,
                'scale': scale, 
                'img': obj_img, 
                'id': obj_file_name
            }
            
            data.append(entry)
            
        except Exception as e:
            print(f"Error processing {obj_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(data)} OBJ files")
    return MeshDataset(data)


if __name__ == "__main__":
    # Load dataset
    root_folder = "obj_kyoto_for_training_01"
    # csv_file = "obj_descriptions.csv"  # Optional
    dataset_dir = "plateau_lod2_withimg_buf2"
    certain_list = "tiff_for_training_buf1"
    
    cl_list, cl_paths = read_certain_list_folder(certain_list)
    # print(len(cl_list), cl_list[0])
    # aaaa
    
    dataset_path = Path(dataset_dir)
    dataset_path.mkdir(exist_ok=True, parents=True)
    dataset_path = dataset_path / (dataset_dir + ".npz")
    
    ratio = 1.0
    if not os.path.isfile(dataset_path):
        dataset = create_mesh_dataset(root_folder, csv_path=None, 
                                      ratio=ratio, certain_list=cl_list, certain_paths=cl_paths)
        dataset.save(dataset_path)
    
    dataset = MeshDataset.load(dataset_path)
    print(dataset[0])
    # dataset[0]['img'].save("data_check.tiff")
    # for idx in range(len(dataset.data)):
    #     if dataset[idx]['faces'].shape[0] >= 200:
    #         print(dataset[idx]['faces'].shape)
    
    mesh = trimesh.Trimesh(dataset[0]['vertices'], 
                           dataset[0]['faces'])
    mesh.export('data_test_buf2.obj')
    
    # Access data
    # print(f"Dataset length: {len(dataset)}")