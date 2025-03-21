import os, sys
from typing_extensions import deprecated

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
import trimesh
import numpy as np
import pandas as pd
import geopandas as gpd
import random
from shapely import Polygon

from .BldgXL.models import MeshXL, train_aug
from .BldgXL.dataset import MeshDataset
from .BldgXL.utils.param_gen import random_param

from .geoinfo_load import polygon_to_mesh
from .opening_handling import mesh_opening, xyz_footprint_conversion
from .json_handler import plateau_route

from utils.overlaps import detect_overlapping
from utils.gml_io import bldg_citygml, save_citygml

from tqdm import tqdm

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_model(args):
    model = MeshXL(args)
    return model


def post_process_mesh(mesh_coords):
    mesh_coords = mesh_coords[~torch.isnan(mesh_coords[:, 0, 0])]  # nvalid_face x 3 x 3
    vertices = mesh_coords.reshape(-1, 3).numpy()
    vertices_index = np.arange(len(vertices))  # 0, 1, ..., 3 x face
    faces = vertices_index.reshape(-1, 3)

    return vertices, faces


def mesh_scale(vertices: np.array):
    # if not isinstance(vertices, np.array):
    #     vertices = np.array(vertices)
        
    assert vertices.shape[1] == 3
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 2]
    z_coords = vertices[:, 1]

    x_scale = np.max(x_coords) - np.min(x_coords)
    y_scale = np.max(y_coords) - np.min(y_coords) 
    z_scale = np.max(z_coords) - np.min(z_coords) 

    xy_scale = np.sqrt(x_scale ** 2 + y_scale ** 2)
    
    return xy_scale, z_scale


def mesh_scaling(vertices: np.array, 
                 original_scale, 
                 heights = 10.):
    norm_scale, norm_height_scale = mesh_scale(vertices.copy())
    
    scaling_factor = original_scale / norm_scale
    height_scaling_factor = heights / norm_height_scale
    
    scaled_vertices = [[x * scaling_factor, 
                        y * height_scaling_factor, 
                        z * scaling_factor] for x, y, z in vertices]
    scaled_vertices = np.array(scaled_vertices)
    
    height_min_value = np.min(scaled_vertices[:, 1])
    print(norm_height_scale, height_min_value)
    scaled_vertices[:, 1] -= height_min_value

    return scaled_vertices.tolist()


def fixed_mesh_scaling(vertices: np.array, 
                       scale_factor = 10.):
    norm_scale, norm_height_scale = mesh_scale(vertices.copy())
    
    scaled_vertices = [[x * scale_factor, 
                        y * scale_factor, 
                        z * scale_factor] for x, y, z in vertices]
    scaled_vertices = np.array(scaled_vertices)
    
    height_min_value = np.min(scaled_vertices[:, 1])
    scaled_vertices[:, 1] -= height_min_value

    return scaled_vertices.tolist()


def load_model():
    args_dict = {
        "n_discrete_size": 128,
        "llm": 'BldgXL/config/mesh-xl-350m'
    }
    args = argparse.Namespace(**args_dict)
    model = get_model(args)
    model.to("cuda")

    checkpoint = torch.load('BldgXL/checkpoints/BldgGen2024/BldgXL/checkpoints/mesh-transformer.ckpt.epoch_130_avg_loss_0.021.pt', map_location=torch.device("cuda"))
    model.load_state_dict(checkpoint["model"], strict=False)

    return model

@deprecated("For test only. Use generate() instead. ")
def footprint_conditioned_generation():
    args_dict = {
        "n_discrete_size": 128,
        "llm": 'BldgXL/config/mesh-xl-350m'
    }
    args = argparse.Namespace(**args_dict)
    model = get_model(args)
    model.to("cuda")

    # print(model)
    # aa
    
    torch.manual_seed(8192)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(8192)

    checkpoint = torch.load('BldgXL/ckpt/plateau_lod2_type_mixed.pt', map_location=torch.device("cuda"))
    model.load_state_dict(checkpoint["model"], strict=False)

    dataset_path = 'BldgXL/plateau_lod2_type1/plateau_lod2_type1_simple_text.npz'
    dataset = MeshDataset.load(dataset_path)

    prompt_mes = dataset.data[159:160]
    out = []
    
    # texts = ["type 1, flat surface roof, small pitch, detailed design", 
    #          "type 2, flat stepped roof, flat surfaces, detailed design", 
    #          "type 3, hybrid roof, slopes, flat surfaces, detailed design", 
    #          "type 4, hipped roof, slopes, detailed design", 
    #          "type 5, gable roof, one-side long slopes, detailed design"]
    # texts = ["type 1, flat surface roof, small pitch, sophisticated design", 
    #          "type 1, flat surface roof, small pitch, sophisticated design", 
    #          "type 1, flat surface roof, small pitch, sophisticated design", 
    #          "type 1, flat surface roof, small pitch, sophisticated design", 
    #          "type 1, flat surface roof, small pitch, sophisticated design"]

    type_texts = ["type 1, flat surface roof, small pitch", 
                  "type 2, flat stepped roof, flat surfaces", 
                  "type 3, hybrid roof, slopes, flat surfaces", 
                  "type 4, hipped roof, slopes", 
                  "type 5, gable roof, one-side long slopes"]
    sods = ["simple", "exact", "detailed", "sophisticated"]

    # type_text = "type 1, flat surface roof, small pitch, "
    
    for j in tqdm(range(len(prompt_mes))):
        data_dict = {'vertices': prompt_mes[j]['vertices'][None, ...], 'faces': prompt_mes[j]['faces'][None, ...], 
                    #  'texts': texts[0], 
                     'scale': prompt_mes[j]['scale']}
        # print(data_dict)
        # aa
        
        # print(data_dict['vertices'][0])
        # aa
        
        data_dict['vertices'], data_dict['faces'], _ = train_aug(data_dict['vertices'].clone(), data_dict['faces'].clone())
        print(data_dict['vertices'], data_dict['faces'])
        
        ori = trimesh.Trimesh(vertices=prompt_mes[j]['vertices'].cpu().numpy(), faces=prompt_mes[j]['faces'].cpu().numpy())
        fp = trimesh.Trimesh(vertices=data_dict['vertices'][0].cpu().numpy(), faces=data_dict['faces'][0].cpu().numpy())
        
        ori.vertices = fixed_mesh_scaling(ori.vertices, 5.)
        fp.vertices = fixed_mesh_scaling(fp.vertices, 5.)
        
        # res = trimesh.util.concatenate([ori, fp.apply_translation([20, 0, 0])])
        res = trimesh.util.concatenate([fp])
        # res = None

        # decoder_output = model.generate(num_return_sequences=1, generation_config=dict(do_sample=True, top_k=50, top_p=0.95, ))
        n_samples = 1
        
        # for sod in sods:
        texts = [ttext + "simple design" for ttext in type_texts]
        
        for type_id, text in enumerate(texts):
            data_dict['texts'] = text
            # data_dict['texts'] = "type 1, flat surface roof, small pitch, simple design"

            os.makedirs(f'fs_demo/7/{text[:6]}', exist_ok=True)

            for i in range(n_samples):
                decoder_output = model.generate_partial(data_dict=data_dict, n_samples=1)
                
                v, f = post_process_mesh(decoder_output['recon_faces'][0].cpu())
                
                v = mesh_scaling(v, data_dict['scale'], heights = (i + 1) * 10.0)
                
                # norm_scale = mesh_xy_scale(np.array(v.copy()))
                # original_scale = data_dict['scale']
                # print(f'scale altering: {norm_scale} & {original_scale}')
                # aa
                
                recon = trimesh.Trimesh(vertices=v, faces=f)
                # if not recon.is_watertight:
                
                # recon.export(f'fs_demo/7/{text[:6]}/{sod}_{i}.obj')
                
                res = trimesh.util.concatenate([res, recon.apply_translation([35 * (2 + type_id), 0, 0])])

        out.append(res.apply_translation([0, 0, 35 * j]))

    out = trimesh.util.concatenate(out)
    out.export('BldgXL/plateau_lod2_type_mixed_full/type_mixed_full_test_text_aug.obj')


def pre_processing(vertices, faces):
    vertices = np.array(vertices)
    
    vertices[:, [1, 2]] = vertices[:, [2, 1]]
    
    # position
    original_centroid = np.mean(vertices, axis=0)
    
    # scaling and centering
    centered_vertices = vertices - original_centroid
    max_abs = np.max(np.abs(centered_vertices))
    vertices = centered_vertices / (max_abs / 0.95)  # Limit vertices to [-0.95, 0.95]
    scale_factor = max_abs / 0.95
    
    def sort_vertices(vertex):
        return vertex[1], vertex[2], vertex[0]
    
    sorted_vertices = sorted(vertices.tolist(), key=sort_vertices)
    
    # face indexing
    vertex_map = {}
    for new_index, vertex in enumerate(sorted_vertices):
        original_index = np.where((vertices == vertex).all(axis=1))[0][0]
        vertex_map[original_index] = new_index
    
    sorted_faces = [[vertex_map[v] for v in face] for face in faces]
    
    # print(vertices)
    
    # to bottom
    min_y = min(v[1] for v in sorted_vertices)
    difference = -0.95 - min_y
    sorted_vertices = [[v[0], v[1] + difference, v[2]] for v in sorted_vertices]
    
    # vertices[:, 1] += difference
    
    return sorted_vertices, sorted_faces, original_centroid, scale_factor


def post_processing_and_positioning(decoded_output, 
                                    scaling_factor, 
                                    original_centroid, 
                                    height, 
                                    height_assignment = True):
    v, f = post_process_mesh(decoded_output['recon_faces'][0].cpu())
    
    v = np.array(v)
    v = v * scaling_factor + original_centroid
    
    height_min = np.min(v[:, 1])
    v[:, 1] -= height_min
    
    # height assignment
    max_y = v[:, 1].max()
    min_y = v[:, 1].min()
    current_height = max_y - min_y
    
    delta_ratio_height = height / current_height
    
    if height_assignment:
        non_ground_mask = v[:, 1] != min_y
        v[non_ground_mask, 1] *= delta_ratio_height
    
    # double-sided rendering
    # reversed_faces = np.flip(f, axis=1)
    # f = np.vstack((f, reversed_faces))
    
    return v.tolist(), f


def opening_id_loading(json_path):
    route = plateau_route(json_path)
    bldg_id_list = [bldg.bldg_id for bldg in route.buildings]
    
    return route, bldg_id_list


def route_reference(route_path):
    route = plateau_route(route_path)
    
    footprints = []
    for building in route.buildings:
        footprint = building.footprint + route.geo_center
        xyz_footprint = xyz_footprint_conversion(footprint)
        xy_footprint = xyz_footprint[:, :2]
        footprints.append(xy_footprint)
    
    footprint_polygons = [Polygon(group) for group in footprints]
    
    footprint_gdf = gpd.GeoDataFrame(geometry=footprint_polygons, crs='EPSG:30169')
    
    return route, footprint_gdf
    

def generate(vertices: list, 
             faces: list, 
             params: list,  # sod, height, roof_type
             polygons: list, 
             seed: int = 1024, 
             model = None, 
             opening = None,
             route_exists = False,  
             gt_mode = False
             ):
    assert len(vertices) == len(params)
    
    if route_exists:
        route, footprint_reference = route_reference(opening)
    
    # Generate
    if model is None:
        args_dict = {
            "n_discrete_size": 128,
            "llm": 'Building_Generation_Opening/BldgXL/config/mesh-xl-350m'
        }
        args = argparse.Namespace(**args_dict)
        model = get_model(args)
        model.to("cuda")
        
        model.eval()

        checkpoint = torch.load('Building_Generation_Opening/BldgXL/ckpt/plateau_lod2_type_mixed.pt', map_location=torch.device("cuda"))
        model.load_state_dict(checkpoint["model"], strict=False)

    # random seed configuration
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    
    # prompt
    type_texts = ["type 1, flat surface roof, small pitch", 
                  "type 2, flat stepped roof, flat surfaces", 
                  "type 3, hybrid roof, slopes, flat surfaces", 
                  "type 4, hipped roof, slopes", 
                  "type 5, gable roof, slopes"]
    # type_texts = ["type 1", 
    #               "type 2", 
    #               "type 3", 
    #               "type 4", 
    #               "type 5"]
    sods = ["simple", "exact", "detailed", "sophisticated"]
    
    out = []
    s = 0
    default_color = np.array([129, 207, 242, 255])
    for v, f, p, polygon in tqdm(zip(vertices, faces, params, polygons)):
        v, f, centroid, scale_factor = pre_processing(v, f)
        isMatched = False
        # fp = trimesh.Trimesh(vertices=v, faces=f)
        # fp.export('fp_test_02.obj')
        # aa
        
        footprint_vertices = np.array([v])
        footprint_faces = np.array([f])
        # print(footprint_vertices, footprint_faces)
        # aa
        
        footprint_vertices = torch.tensor(footprint_vertices).cuda()
        footprint_faces = torch.tensor(footprint_faces).cuda()
        
        # print(footprint_vertices, footprint_faces)
        
        sod = int(p[1])
        height = float(p[2])
        roof_type = int(p[3])

        text = f"{type_texts[roof_type - 1]}, {sods[sod - 1]} design"

        data_dict = {
            'vertices' : footprint_vertices, 
            'faces' : footprint_faces, 
            'texts' : text
        }
        
        torch.manual_seed(random.randint(0, 65535))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random.randint(0, 65535))
        
        decoded_output = model.generate_partial(data_dict=data_dict, n_samples=1)
        
        gen_v, gen_f = post_processing_and_positioning(decoded_output, 
                                                       scale_factor, 
                                                       centroid, 
                                                       height)
        
        mesh = trimesh.Trimesh(vertices=gen_v, faces=gen_f)
        
        if route_exists:
            isMatched, matched_idx = detect_overlapping(polygon, footprint_reference)
        
        # if p[0] in valid_ids:
        if isMatched and route_exists:
            # idx = valid_ids.index(p[0])
            # building = route.buildings[idx]
            building = route.buildings[matched_idx]
            
            mesh = mesh_opening(building, route, mesh, default_color)
        else:
            mesh.fix_normals(multibody=True)
            mesh.visual.face_colors = np.tile(default_color, (len(mesh.faces), 1))
        
        out.append(mesh)
    
    bldg_vertices = [mesh.vertices for mesh in out]
    bldg_faces = [mesh.faces for mesh in out]
    save_citygml(bldg_citygml(bldg_vertices, bldg_faces, lod=2), 'building.gml')
    
    concat_mesh = trimesh.util.concatenate(out)
    return concat_mesh
    

if __name__ == '__main__':
    # footprint_conditioned_generation()
    vertices, faces, properties = polygon_to_mesh('fs_demo/footprint/group2.geojson')
    params = pd.read_csv('temp_param/params_group2.csv')
    params = params.values.tolist()
    generate(vertices, faces, params=params, seed=1024)
    