import gradio as gr
import os, time
import json
import numpy as np
import pandas as pd
import random
import argparse
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

import trimesh

from BldgXL.utils.param_gen import random_param
from geoinfo_load import polygon_to_mesh
from generation import generate, get_model


model = None
params = None

def load_model():
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
    
    return model


def reset_all():
    # Return None or default values for all components
    return [
        gr.update(value=None),  # Clear file
        gr.update(value=None), # Clear plot
        gr.update(value=1024), 
        gr.update(value=pd.DataFrame({
            'Bldg_ID': [], 
            'SoD': [], 
            'Height': [], 
            'Roof_Type': []                       
        })),
        gr.update(value=None)
    ]
    

def switch_mode(mode: gr.State):
    new_mode = "wireframe" if mode == "solid" else "solid"
    return gr.Model3D.update(display_mode=new_mode)


def param_load(param_path = ""):
    path = "temp_param/params.csv" if param_path == "" else param_path
    
    param_df = pd.read_csv(path)
    return param_df


def visualize_mesh(file_path):
    """
    Visualize a mesh from a JSON file containing vertices and faces
    """
    global params
    
    if file_path is None:
        return None
    
    try:
        # Load the mesh data
        vertices, faces, properties = polygon_to_mesh(file_path)
        
        vertices = [[[-v[0], v[1], v[2]] for v in mesh] for mesh in vertices]
        
        footprint_mesh = [trimesh.Trimesh(vertices=v, faces=f) for v, f in zip(vertices, faces)]
        footprint_mesh = trimesh.util.concatenate(footprint_mesh)
        
        # vertices = np.array(vertices)
        # faces = np.array(faces)
        
        # Save properties
        params = pd.DataFrame(properties)
        
        # Create 3D figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # print(vertices[:, 0])
        ax.plot_trisurf(footprint_mesh.vertices[:, 0], 
                        footprint_mesh.vertices[:, 1], 
                        triangles=footprint_mesh.faces, 
                        Z=footprint_mesh.vertices[:, 2], 
                        cmap='grey')
        
        ax.view_init(elev=45, azim=-90)
        # plt.margins(x=0.1, y=0.1)
        plt.tight_layout()
        
        # ax.mouse_init()
        
        plt.axis('off')
        
        # Set plot limits and aspects
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_title('3D Mesh Visualization')
        
        # # Auto-scale the axes
        ax.set_xlim3d(footprint_mesh.vertices[:, 0].min() + 1, footprint_mesh.vertices[:, 0].max() - 1)
        ax.set_ylim3d(footprint_mesh.vertices[:, 1].min() + 1, footprint_mesh.vertices[:, 1].max() - 1)
        # ax.set_zlim(-1, 1)  # Adjust Z range as needed
        
        # print(dataframe)
        
        return fig
    
    except Exception as e:
        raise gr.Error(f"Error processing file: {str(e)}")


def mesh_processing(mesh, 
                    height: float = 10.0, 
                    height_assignment: bool = True):
    vertices = mesh.vertices
    faces = mesh.faces
    
    max_y = vertices[:, 1].max()
    min_y = vertices[:, 1].min()
    current_height = max_y - min_y
    
    delta_ratio_height = height / current_height
    
    # print(f'{current_height}, {height}, {delta_height}')
    
    # height assignment
    if height_assignment:
        non_ground_mask = vertices[:, 1] != min_y
        vertices[non_ground_mask, 1] *= delta_ratio_height
    
    # double-sized rendering
    reversed_faces = np.flip(faces, axis=1)
    faces = np.vstack((faces, reversed_faces))
    
    output_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return output_mesh


def pseudo_mesh_generation(file_input,
                           random_seed, 
                           params_df):
    params = params_df.values.tolist()
    
    if len(params) > 1:
        file_name = file_input.split('.')[0].split('/')[-1]
        time.sleep(random.randint(55, 65))
        output_mesh_path = 'fs_demo/group/' + file_name + '.obj'
        
        mesh = trimesh.load(output_mesh_path)
        processed_mesh = mesh_processing(mesh, height_assignment=False)
        
        processed_mesh.export(output_mesh_path)
        
    else:
        param = params[0]
        detail_level = int(param[1])
        height = float(param[2])
        roof_type = int(param[3])
        
        detail_level_dict = {
            1 : "simple", 
            2 : "exact", 
            3 : "detailed", 
            4 : "sophisticated"
        }
        
        filename = os.path.basename(file_input)
        file_id = filename.split('.')[0]
        
        assert isinstance(roof_type, int) and isinstance(detail_level, int)
        
        random_id = random_seed % 5
        
        output_path = f'fs_demo/{file_id}/type {roof_type}/{detail_level_dict[detail_level]}_{random_id}.obj'
        print(output_path)
        
        # height adjustment
        mesh = trimesh.load(output_path)
        processed_mesh = mesh_processing(mesh, height)
        
        output_mesh_path = "Building_Generation_Opening/fs_demo/footprint/temp.obj"
        processed_mesh.export(output_mesh_path)
        
        process_time = 1.5 * detail_level
        time.sleep(process_time)
    
    return output_mesh_path


def mesh_generation(file_input, 
                    random_seed, 
                    params_df):
    global model
    
    p = params_df.values.tolist()
    
    vertices, faces, _ = polygon_to_mesh(file_input)
    
    # generate
    concat_mesh = generate(vertices, faces, p, random_seed, model, 
                           gt_mode=True)
    
    # detail_level_dict = ["simple", "exact", "detailed", "sophisticated"]
    # if len(params) == 1:
    #     file_id = file_input.split('/')[-1].split('.')[0]
    #     roof_type, detail_level, random_id = int(params[0][3]), int(params[0][1]), int(random_seed % 5)
    #     # print(roof_type, detail_level, random_id)
    #     output_mesh = f'fs_demo/{file_id}/type {roof_type}/{detail_level_dict[detail_level - 1]}_{random_id}.obj'
    # else:
    
    output_mesh = "Building_Generation_Opening/fs_demo/footprint/temp.obj"
    
    concat_mesh.export(output_mesh)
    
    return output_mesh


def param_altering(dataframe):
    return dataframe


def param_load(file_input,
               sod_min, 
               sod_max, 
               roof_type):
    with open(file_input) as f:
        file = json.load(f)

    if 'features' in file:
        bldg_num = len(file['features'])
    else:
        bldg_num = 1
        
    params_df = random_param(
        bldg_num=bldg_num, 
        sod_min=sod_min,
        sod_max=sod_max, 
        height_min=0, 
        height_max=50, 
        type_selected=roof_type, 
        output_csv=False
    )
    
    params_df[['Bldg_ID', 'Height']] = params[['id', 'height']]
    
    return params_df
    

def param_generation(file_input, 
                     sod_min, 
                     sod_max, 
                     height_min, 
                     height_max, 
                     roof_type):
    with open(file_input) as f:
        file = json.load(f)

    if 'features' in file:
        bldg_num = len(file['features'])
    else:
        bldg_num = 1
    
    params_df = random_param(
        bldg_num=bldg_num, 
        sod_min=sod_min,
        sod_max=sod_max, 
        height_min=height_min, 
        height_max=height_max, 
        type_selected=roof_type, 
        output_csv=False
    )
    
    return params_df
    
    
print("Model is loading...")
model = load_model()

# Create Gradio interface
with gr.Blocks() as iface:
    display_mode = gr.State('solid')
    
    with gr.Row():
        gr.Markdown("# 三次元都市建物モデル生成")
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="アップロード",
                file_types=[".geojson"], 
                value=None
            )
            examples = gr.Examples(
                examples=[
                    ["Building_Generation_Opening/fs_demo/footprint/1.geojson"], 
                    ["Building_Generation_Opening/fs_demo/footprint/3.geojson"],
                    ["Building_Generation_Opening/fs_demo/footprint/7.geojson"],
                    ["Building_Generation_Opening/fs_demo/footprint/4.geojson"],
                    ["Building_Generation_Opening/fs_demo/footprint/5.geojson"], 
                    ["Building_Generation_Opening/fs_demo/footprint/group1.geojson"], 
                    ["Building_Generation_Opening/fs_demo/footprint/group2.geojson"]
                    # ["fs_demo/footprint/6.geojson", "Detailed_3"]
                ], 
                inputs=file_input, 
                label="サンプルデータ"
            )
            
            plot_output = gr.Plot(
                label="フットプリント", 
                value=None
            )
            
        with gr.Column(scale=1):
            gr.Markdown("### パラメータ")
            with gr.Row():
                random_seed = gr.Slider(minimum=1, maximum=65535, value=1024, 
                                        label="ランダムシード", step=1, interactive=True)
            params_df = gr.DataFrame(
                headers=["Bldg_ID", "SoD", "Height", "Roof_type"], 
                datatype=["str", "number", "number", "number"], 
                col_count=(4, "fixed"), 
                height=250, 
                interactive=True
            )
            
            with gr.Row():
                param_load_button = gr.Button("パラメータ読込")
                param_gen_button = gr.Button("パラメータ乱数生成")
            
            gr.Markdown("")
            
            with gr.Row():
                sod_min = gr.Slider(minimum=1, maximum=4, step=1, value=1, 
                                    label="最小精細度レベル", interactive=True)
                sod_max = gr.Slider(minimum=1, maximum=4, step=1, value=4, 
                                    label="最大精細度レベル", interactive=True)
            
            with gr.Row():
                height_min = gr.Slider(minimum=5.0, maximum=50.0, step=0.1, value=5.0, 
                                    label="高さ最小値", interactive=True)
                height_max = gr.Slider(minimum=5.0, maximum=50.0, step=0.1, value=20.0, 
                                    label="高さ最大値", interactive=True)
                
            with gr.Row():
                roof_type = gr.CheckboxGroup(choices=[1, 2, 3, 4, 5], 
                                             value=[1, 2, 3, 4, 5],
                                             label="屋根種類")
            
                
                # detail_level = gr.Slider(minimum=1, maximum=4, value=1, 
                #                         label="精細度レベル", step=1, interactive=True)
                # height = gr.Slider(minimum=1, maximum=50, value=1, 
                #                 label="高さ", step=0.1, interactive=True)
                # roof_type = gr.Slider(minimum=1, maximum=5, value=1, 
                #                     label="屋根種類", step=1, interactive=True)
            
            
            
        with gr.Column(scale=1):
            vis = gr.Model3D(
                label="生成結果", 
                interactive=False, 
                height=250, 
                clear_color=[0.0, 0.0, 0.0, 0.5], 
                display_mode=display_mode
            )
            reset_button = gr.Button("リセット")
            submit_button = gr.Button("実行", variant="primary")
            
            gr.Markdown("＊実行にエラーが出る場合はリセットをクリック")
            gr.Markdown("")
            
            gr.Markdown("### 様式指定用ランダムシード：")
            gr.Markdown("1 から 65535 まで選択できる")
            
            gr.Markdown("### 精細度レベル：")
            gr.Markdown("最低1、最高4")
            
            gr.Markdown("### 建物高さ：")
            gr.Markdown("最低1、最高50")
            
            gr.Markdown("### 屋根種類：")
            gr.Markdown("1はフラット、2は段差あるフラット、3は複合、")
            gr.Markdown("4は切妻、5は腰折")
            # TODO: type 6
            # with gr.Row():
            #     solid_button = gr.Button("立方モデル")
            #     wireframe_button = gr.Button("線モデル")
    
    file_input.change(
        fn=visualize_mesh,
        inputs=[file_input], 
        outputs=[plot_output]
    )
    
    params_df.change(
        fn=param_altering, 
        inputs=[params_df], 
        outputs=[params_df]
    )
    
    param_load_button.click(
        fn=param_load, 
        inputs=[file_input, sod_min, sod_max, roof_type], 
        outputs=[params_df]
    )
    
    param_gen_button.click(
        fn=param_generation, 
        inputs=[file_input, sod_min, sod_max, height_min, height_max, roof_type], 
        outputs=[params_df]
    )
    
    reset_button.click(
        fn=reset_all, 
        inputs=[], 
        outputs=[file_input, plot_output, random_seed, params_df, vis]
    )
    
    submit_button.click(
        # fn=pseudo_mesh_generation, 
        fn=mesh_generation,
        inputs=[file_input, random_seed, params_df], 
        outputs=[vis]
    )

# iface = gr.Interface(
#     fn=visualize_mesh,
#     inputs=gr.File(label="アップロード", file_types=[".geojson"]),
#     outputs=gr.Plot(label="フットプリント"),
#     title="三次元都市建物モデル生成",
#     description="建物フットプリントの含むGeoJSONをアップロード",
#     examples=[],
#     cache_examples=False, 
#     live=True
# )

if __name__ == "__main__":
    iface.launch()