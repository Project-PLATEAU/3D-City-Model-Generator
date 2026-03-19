import os, sys
from typing_extensions import deprecated

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
import trimesh
import numpy as np
import pandas as pd
import random
from shapely.geometry import Polygon

from BldgXL_3x.models import MeshXL, train_aug
from BldgXL_3x.dataset import MeshDataset
from BldgXL_3x.utils.param_gen import random_param

from geoinfo_load import polygon_to_mesh
from json_handler import plateau_route

from matplotlib.colors import hex2color
import matplotlib.pyplot as plt

from tqdm import tqdm

from BldgXL_3x.utils.bmqi import divide_building_mesh

from gen_bg_lod3.all_run import generate_city_assets

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
    
    model.eval()

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

    # Remove duplicate vertices and update face indices
    unique_vertices, inverse_indices = np.unique(vertices, axis=0, return_inverse=True)
    vertices = unique_vertices
    faces = [[inverse_indices[int(v)] for v in face] for face in faces]

    # position
    original_centroid = np.mean(vertices, axis=0)
    
    # # scaling and centering
    # centered_vertices = vertices - original_centroid
    # max_abs = np.max(np.abs(centered_vertices[:, [0, 2]]))
    # vertices = centered_vertices / (max_abs / 0.95)  # Limit vertices to [-0.95, 0.95]
    # scale_factor = max_abs / 0.95
    
    # print(max_abs, scale_factor)
    
    def sort_vertices(vertex):
        return vertex[1], vertex[2], vertex[0]
    
    sorted_vertices = sorted(vertices.tolist(), key=sort_vertices)
    
    # face indexing
    vertex_map = {}
    for new_index, vertex in enumerate(sorted_vertices):
        original_index = np.where((vertices == vertex).all(axis=1))[0][0]
        vertex_map[original_index] = new_index
    
    sorted_faces = [[vertex_map[int(v)] for v in face] for face in faces]
    
    # print(vertices)
    
    # to bottom
    min_y = min(v[1] for v in sorted_vertices)
    difference = -0.95 - min_y
    sorted_vertices = [[v[0], v[1] + difference, v[2]] for v in sorted_vertices]
    
    # scale up to -0.95, 0.95
    sorted_vertices_array = np.array(sorted_vertices)
    sorted_vertices_array_xz = sorted_vertices_array[:, [0, 2]]
    
    min_coords = sorted_vertices_array_xz.min(axis=0)
    max_coords = sorted_vertices_array_xz.max(axis=0)
    
    bbox_x = max_coords[0] - min_coords[0]
    bbox_z = max_coords[1] - min_coords[1]
    longest_edge = max(bbox_x, bbox_z)
    
    rescale_factor = (2 * 0.95) / longest_edge
    
    re_centroid = (min_coords + max_coords) / 2
    sorted_vertices_array[:, 0] = (sorted_vertices_array[:, 0] - re_centroid[0]) * rescale_factor
    sorted_vertices_array[:, 2] = (sorted_vertices_array[:, 2] - re_centroid[1]) * rescale_factor
    # scale_factor = max_abs / 0.95
    
    sorted_vertices = sorted_vertices_array.tolist()
    
    scale_factor = 1.0 / rescale_factor
    
    # print(sorted_vertices, sorted_faces)
    # mesh_sample = trimesh.Trimesh(sorted_vertices, sorted_faces)
    # mesh_sample.export('gen_fp.obj')
    # # aa
    
    centroid_returned = np.array([re_centroid[0], 0., re_centroid[1]])
    
    return sorted_vertices, sorted_faces, centroid_returned, scale_factor


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


def generate(vertices: list,
             faces: list,
             params: list,  # sod, height, roof_type
             footprints: list,
             seed: int = 1024,
             model = None,
             opening = None,
             gt_mode = False,
             color_mode = True,
             front_mode = False,
             orthophoto_path: str = None,
             geojson_name: str = None,  # geojson filename without extension
             geojson_root_folder: str = None,  # parent folder name (e.g., 'kyoto')
             progress_callback = None  # callback function(current, total) for progress updates
             ):
    assert len(vertices) == len(params)
    
    # route, valid_ids = opening_id_loading('/home/sekilab-liao/Documents/gen3d_2_0/BldgGen2024/bldg-elem_route1.json')
    
    # spatial intersection -> original footprint -> lod1 -> lod3
    
    # if len(vertices.shape) == 2:
    #     vertices = [vertices]
    #     faces = [faces]
    
    # vertices = np.array(vertices)
    # faces = np.array(faces)
    
    bg_output_mesh = generate_city_assets(orthophoto_path, lod2_obj_path=None) \
        if orthophoto_path is not None else None
    
    # Generate
    if model is None:
        args_dict = {
            "n_discrete_size": 128,
            "llm": 'BldgXL/config/mesh-xl-350m'
        }
        args = argparse.Namespace(**args_dict)
        model = get_model(args)
        model.to("cuda")
        
        model.eval()

        checkpoint = torch.load('BldgXL/ckpt/plateau_lod2_type_mixed.pt', map_location=torch.device("cuda"))
        model.load_state_dict(checkpoint["model"], strict=False)

    # random seed configuration
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    
    # prompt
    # type_texts = ["type 1, flat surface roof, small pitch", 
    #               "type 2, flat stepped roof, flat surfaces", 
    #               "type 3, hybrid roof, slopes, flat surfaces", 
    #               "type 4, hipped roof, slopes", 
    #               "type 5, gable roof, slopes"]
    # type_texts = ["type 1", 
    #               "type 2", 
    #               "type 3", 
    #               "type 4", 
    #               "type 5"]
    # sods = ["simple", "exact", "detailed", "sophisticated"]
    low_color = hex2color("#FF0000")
    high_color = hex2color("#FFFFFF")
    
    low_color_a = [int(c * 255) for c in low_color]
    high_color_a = [int(c * 255) for c in high_color]
    
    front_out = []
    out = []
    out_bmqi = []
    out_bmqi_origin = []
    
    bldg_ids = []
    bldg_meshes = []
    
    bldg_mesh_lod1 = []
    # default_color = np.array([255, 255, 255])
    front_default_color = np.array([129, 207, 242, 255])

    # Track progress for callback
    total_buildings = len(vertices)
    current_building = 0

    for v, f, p, fp in tqdm(zip(vertices, faces, params, footprints)):
        v, f, centroid, scale_factor = pre_processing(v, f)
        
        # fp_test = trimesh.Trimesh(vertices=v, faces=f)
        # fp_test.export('fp_test.obj')
        # aa
        
        footprint_vertices = np.array([v])
        footprint_faces = np.array([f])
        # print(footprint_vertices, footprint_faces)
        # aa
        
        footprint_vertices = torch.tensor(footprint_vertices).cuda()
        footprint_faces = torch.tensor(footprint_faces).cuda()
        
        # print(footprint_vertices, footprint_faces)
        
        # sod = int(p[1])
        bldg_id = str(p[0])
        height = float(p[1])
        # roof_type = int(p[2])

        # text = f"{type_texts[roof_type - 1]}, {sods[sod - 1]} design"
        # text = f"{type_texts[roof_type - 1]}"

        data_dict = {
            'vertices' : footprint_vertices,
            'faces' : footprint_faces,
            'id': bldg_id,
            'geojson_name': geojson_name  # Pass geojson filename for image path construction
        }
        
        # torch.manual_seed(random.randint(0, 65535))
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(random.randint(0, 65535))
        
        # decoded_output = model.generate_partial(data_dict=data_dict, n_samples=1)
        
        # gen_v, gen_f = post_processing_and_positioning(decoded_output, 
        #                                                scale_factor, 
        #                                                centroid, 
        #                                                height)
        
        # mesh = trimesh.Trimesh(vertices=gen_v, faces=gen_f)
        # print(f'is_watertight: {mesh.is_watertight}')
        
        
        # if color_mode:
        #     footprint_coords = list(fp.exterior.coords)
        #     reversed_footprint_coords = [(x, -y) for x, y in footprint_coords]
        #     fp = Polygon(reversed_footprint_coords)
            
        #     _, bmqi = divide_building_mesh(mesh, fp)
        #     default_color = np.array([int(low_color_a[0] + bmqi * (high_color_a[0] - low_color_a[0])),
        #                             int(low_color_a[1] + bmqi * (high_color_a[1] - low_color_a[1])),
        #                             int(low_color_a[2] + bmqi * (high_color_a[2] - low_color_a[2])),
        #                             255])
        
        bmqi = 0.0
        round_num = 0
        mesh = None
        original = True
        data_dict_original = data_dict
        
        footprint_coords = list(fp.exterior.coords)
        reversed_footprint_coords = [(x, -y) for x, y in footprint_coords]
        fp = Polygon(reversed_footprint_coords)
        
        lod1_mesh = trimesh.creation.extrude_polygon(fp, height)
        lod1_mesh.vertices[:, [1, 2]] = lod1_mesh.vertices[:, [2, 1]]
        bldg_mesh_lod1.append(lod1_mesh)
        
        while bmqi < 0.95:
            if round_num > 2:
                mesh = lod1_mesh
                bmqi = 1.0
                
                default_color = np.array([int(low_color_a[0] + bmqi * (high_color_a[0] - low_color_a[0])),
                                         int(low_color_a[1] + bmqi * (high_color_a[1] - low_color_a[1])),
                                         int(low_color_a[2] + bmqi * (high_color_a[2] - low_color_a[2]))])
                
                break
            
            torch.manual_seed(random.randint(0, 65535))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random.randint(0, 65535))
            
            decoded_output = model.generate_partial(data_dict=data_dict, n_samples=1)
            
            gen_v, gen_f = post_processing_and_positioning(decoded_output, 
                                                            scale_factor, 
                                                            centroid, 
                                                            height)
        
            mesh = trimesh.Trimesh(vertices=gen_v, faces=gen_f)
            print(f'is_watertight: {mesh.is_watertight}')
            
            
            if color_mode:
                _, bmqi = divide_building_mesh(mesh, fp)
                default_color = np.array([int(low_color_a[0] + (bmqi - abs(1.0 - bmqi) * 8.0) * (high_color_a[0] - low_color_a[0])),
                                        int(low_color_a[1] + (bmqi - abs(1.0 - bmqi) * 8.0) * (high_color_a[1] - low_color_a[1])),
                                        int(low_color_a[2] + (bmqi - abs(1.0 - bmqi) * 8.0) * (high_color_a[2] - low_color_a[2]))])
                if original:
                    out_bmqi_origin.append(bmqi)
                    original = False
            else:
                bmqi = 1.0
                    
            data_dict = data_dict_original
            
            round_num += 1
        
        # if p[0] in valid_ids:
        #     idx = valid_ids.index(p[0])
        #     building = route.buildings[idx]
            
        #     mesh = mesh_opening(building, route, mesh)
        # else:
        mesh.fix_normals(multibody=True)
        
        front_mesh = None
        if front_mode:
            front_mesh = mesh.copy()
            front_mesh.visual.face_colors = np.tile(front_default_color, (len(front_mesh.faces), 1))
            front_out.append(front_mesh)
            
            bldg_meshes.append(front_mesh)
            bldg_ids.append(bldg_id)
            
            # default_color = front_default_color
        mesh.visual.face_colors = np.tile(default_color, (len(mesh.faces), 1))
        
        if gt_mode:
            mesh.export(f'test_gen_obj/{p[0]}.obj')

        out_bmqi.append(bmqi)
        out.append(mesh)

        # Update progress
        current_building += 1
        if progress_callback:
            progress_callback(current_building, total_buildings)

    bmqi_sum_array_origin = np.array(out_bmqi_origin)
    bmqi_sum_array = np.array(out_bmqi)
    bmqi_avg = np.average(bmqi_sum_array)
    print(f'avg bmqi: {bmqi_avg}')
    
    # Create the histogram
    # plt.figure(figsize=(10, 6))
    # plt.hist(bmqi_sum_array_origin, bins=10, color='orange', edgecolor='black', alpha=0.3)
    # plt.hist(bmqi_sum_array, bins=5, color='skyblue', edgecolor='black', alpha=1.0)
    # plt.title('Comparison of BMQI refinement Y/N')
    # plt.xlabel('BMQI')
    # plt.ylabel('Frequency')
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig('bmqi.png', dpi=300)
    
    concat_mesh = trimesh.util.concatenate(out)
    concat_mesh_front = trimesh.util.concatenate(front_out)
    
    # concat_mesh.export('bldg_temp.obj')
    # out[0].export('bldg_temp_0.obj')
    # out[1].export('bldg_temp_1.obj')
    
    if bg_output_mesh:
        concat_mesh = trimesh.util.concatenate(concat_mesh, bg_output_mesh)
        concat_mesh_front = trimesh.util.concatenate(concat_mesh_front, bg_output_mesh)
    
    # concat_mesh.export('concat_temp.obj')
    
    concat_mesh_lod1 = trimesh.util.concatenate(bldg_mesh_lod1)
    # concat_mesh_lod1.export('concat_lod1_temp.obj')
    
    if front_mode:
        assert len(bldg_ids) == len(bldg_meshes)

        return concat_mesh, concat_mesh_front, bldg_ids, bldg_meshes, bg_output_mesh

    return concat_mesh
    # concat_mesh.export('test.obj')
    

if __name__ == '__main__':
    # footprint_conditioned_generation()
    vertices, faces, properties = polygon_to_mesh('fs_demo/footprint/group2.geojson')
    params = pd.read_csv('temp_param/params_group2.csv')
    params = params.values.tolist()
    generate(vertices, faces, params=params, seed=1024)
    