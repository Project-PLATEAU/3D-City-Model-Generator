import trimesh
import numpy as np

from .mtl import convert_vcolor_to_mtl

def append_mtl(file_name, mtl_name):
    try:
        file_path = f'{file_name}.obj'
        # backup_path = Path(file_path).with_suffix('.obj.backup')
        # shutil.copy2(file_path, backup_path)
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        face_index = -1
        for idx, line in enumerate(lines):
            if line.strip().startswith('f '):
                face_index = idx
                break
            
        if face_index == -1:
            print(f'No {file_name} faces exist. ')
            return
        
        modified_lines = []
        modified_lines.append('mtllib material.mtl')
        modified_lines.extend(lines[:face_index])
        modified_lines.append(f'usemtl {mtl_name}')
        modified_lines.extend(lines[face_index:])
        
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)
    
    except FileNotFoundError:
        print(f"Error: Could not find the file {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("If needed, you can restore from the backup file")
        

def concat_mesh():
    veg_color = [106, 176, 109, 255]
    road_color = [119, 119, 119, 255]
    
    with open('material.mtl', 'a') as f:
        f.write(f'newmtl veg\n')
        f.write(f'Ka {veg_color[0] / 255.0} {veg_color[1] / 255.0} {veg_color[2] / 255.0}\n')
        f.write(f'Kd {veg_color[0] / 255.0} {veg_color[1] / 255.0} {veg_color[2] / 255.0}\n')
        f.write(f'Ks 0.0 0.0 0.0\n')
        f.write(f'Ns 10.0\n')
        f.write(f'd 1.0\n\n')
        
        f.write(f'newmtl road\n')
        f.write(f'Ka {road_color[0] / 255.0} {road_color[1] / 255.0} {road_color[2] / 255.0}\n')
        f.write(f'Kd {road_color[0] / 255.0} {road_color[1] / 255.0} {road_color[2] / 255.0}\n')
        f.write(f'Ks 0.0 0.0 0.0\n')
        f.write(f'Ns 10.0\n')
        f.write(f'd 1.0\n\n')
    
    # concat
    vegs = trimesh.load('veg.obj')
    roads = trimesh.load('road.obj')
    
    with open('buildings.obj', 'r') as f:
        building_lines = f.readlines()
    
    vertex_num = 0
    with open('results.obj', 'w') as file:
        face_index = -1
        for idx, line in enumerate(building_lines):
            if line.strip().startswith('usemtl'):
                face_index = idx
                break
        
        # write building vertices
        file.writelines(building_lines[:face_index])
        vertex_num += len(building_lines[:face_index])

        # write veg vertices
        if vegs.vertices.size != 0:
            if isinstance(vegs, trimesh.Scene):
                vegs = vegs.to_mesh()
            
            vegs.vertices[:, [1, 2]] = vegs.vertices[:, [2, 1]]
            vegs.vertices[:, 2] = -vegs.vertices[:, 2]
            for vertex in vegs.vertices:
                file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
        
        # write road vertices
        if roads.vertices.size != 0:
            if isinstance(roads, trimesh.Scene):
                roads = roads.to_mesh()
            
            roads.vertices[:, [1, 2]] = roads.vertices[:, [2, 1]]
            roads.vertices[:, 2] = -roads.vertices[:, 2]
            for vertex in roads.vertices:
                file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
        
        # write building faces
        file.writelines(building_lines[face_index:])
        
        # write veg faces
        if vegs.vertices.size != 0:
            file.write('usemtl veg\n')
            for face in vegs.faces:
                file.write(f'f {face[0] + vertex_num} {face[1] + vertex_num} {face[2] + vertex_num}\n')
            vertex_num += len(vegs.vertices)
            
        # write road faces
        if roads.vertices.size != 0:
            file.write('usemtl road\n')
            for face in roads.faces:
                file.write(f'f {face[0] + vertex_num} {face[1] + vertex_num} {face[2] + vertex_num}\n')
    
if __name__ == '__main__':
    concat_mesh()