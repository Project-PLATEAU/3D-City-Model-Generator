import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
import trimesh
import numpy as np
from models import MeshXL
from dataset import MeshDataset
from models import train_aug
from tqdm import tqdm


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
    


def footprint_conditioned_generation():
    args_dict = {
        "n_discrete_size": 128,
        "llm": 'config/mesh-xl-350m'
    }
    args = argparse.Namespace(**args_dict)
    model = get_model(args)
    model.to("cuda")

    checkpoint = torch.load('ckpt/plateau_lod2_type_mixed.pt', map_location=torch.device("cuda"))
    model.load_state_dict(checkpoint["model"], strict=False)

    dataset_path = './plateau_lod2_type1/plateau_lod2_type1_simple_text.npz'
    dataset = MeshDataset.load(dataset_path)

    prompt_mes = dataset.data[59:60]
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
        # print(data_dict['vertices'][0])
        # aa
        
        data_dict['vertices'], data_dict['faces'], _ = train_aug(data_dict['vertices'].clone(), data_dict['faces'].clone())
        ori = trimesh.Trimesh(vertices=prompt_mes[j]['vertices'].cpu().numpy(), faces=prompt_mes[j]['faces'].cpu().numpy())
        fp = trimesh.Trimesh(vertices=data_dict['vertices'][0].cpu().numpy(), faces=data_dict['faces'][0].cpu().numpy())
        
        ori.vertices = fixed_mesh_scaling(ori.vertices, 5.)
        fp.vertices = fixed_mesh_scaling(fp.vertices, 5.)
        
        # res = trimesh.util.concatenate([ori, fp.apply_translation([20, 0, 0])])
        res = trimesh.util.concatenate([fp])
        # res = None

        # decoder_output = model.generate(num_return_sequences=1, generation_config=dict(do_sample=True, top_k=50, top_p=0.95, ))
        n_samples = 5
        
        for sod in sods:
            texts = [ttext + sod + " design" for ttext in type_texts]
            
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
                    
                    recon.export(f'fs_demo/7/{text[:6]}/{sod}_{i}.obj')
                    
                    res = trimesh.util.concatenate([res, recon.apply_translation([35 * (2 + type_id), 0, 0])])

        out.append(res.apply_translation([0, 0, 35 * j]))

    out = trimesh.util.concatenate(out)
    out.export('plateau_lod2_type_mixed_full/type_mixed_full_test_text_aug.obj')


if __name__ == '__main__':
    footprint_conditioned_generation()