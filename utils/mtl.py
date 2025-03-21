import numpy as np
import trimesh

def convert_vcolor_to_mtl(obj_path, output_path, mtl_path):
    mesh = trimesh.load(obj_path)
    
    vertices = []
    faces = []
    colors = []
    
    with open(obj_path, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = [float(x) for x in parts[1:4]]
                color = [float(x) for x in parts[4:7]]
                vertices.append(vertex)
                colors.append(color)
                
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                face = [int(p.split('/')[0]) for p in parts]
                faces.append(face)
    
    colors = np.array(colors)
    vertices = np.array(vertices)
    
    unique_colors = np.unique(colors.round(8), axis=0)
    
    mtls = {}
    for i, color in enumerate(unique_colors):
        mtl_name = f'mtl_{i}'
        mtls[mtl_name] = {
            'Ka': color, 
            'Kd': color, 
            'Ks': [0, 0, 0], 
            'Ns': 10.0, 
            'd': 1.0
        }
        
    face_mtls = []
    for face in faces:
        face_color = colors[face[0] - 1]  # Use first vertex color
        material_index = np.where((unique_colors == face_color.round(8)).all(axis=1))[0][0]
        face_mtls.append(f'mtl_{material_index}')
            
    with open(mtl_path, 'w') as f:
        for mtl_name, mtl in mtls.items():
            f.write(f'newmtl {mtl_name}\n')
            f.write(f'Ka {mtl["Ka"][0]} {mtl["Ka"][1]} {mtl["Ka"][2]}\n')
            f.write(f'Kd {mtl["Kd"][0]} {mtl["Kd"][1]} {mtl["Kd"][2]}\n')
            f.write(f'Ks {mtl["Ks"][0]} {mtl["Ks"][1]} {mtl["Ks"][2]}\n')
            f.write(f'Ns {mtl["Ns"]}\n')
            f.write(f'd {mtl["d"]}\n\n')
            
    with open(output_path, 'w') as f:
        f.write(f'mtllib {mtl_path.split("/")[-1]}\n')
        
        for v in vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
            
        current_mtl = None
        for face, mtl in zip(faces, face_mtls):
            if mtl != current_mtl:
                f.write(f'usemtl {mtl}\n')
                current_mtl = mtl
            f.write(f'f {" ".join(str(idx) for idx in face)}\n')
            
if __name__ == '__main__':
    convert_vcolor_to_mtl('buildings.obj', 
                          'buildings.obj', 
                          'material.mtl')