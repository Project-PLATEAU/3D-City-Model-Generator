import os, shutil
from pathlib import Path

def flatten_directory(src_dir, tgt_dir):
    src_dir = Path(src_dir)
    tgt_dir = Path(tgt_dir)
    
    tgt_dir.mkdir(parents=True, exist_ok=True)
    
    file_mapping = {}
    
    for root, _, files in os.walk(src_dir):
        root_path = Path(root)
        # print(root_path)
        
        # if root_path == target_dir:
        #     continue
        
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == '.obj':
                print(ext)
                src_file = root_path / filename
                tgt_file = tgt_dir / filename
                
                try:
                    shutil.move(str(src_file), str(tgt_file))
                    file_mapping[str(src_file)] = str(tgt_file)
                    
                except Exception as e:
                    print(f"Failed to move {source_file}: {str(e)}")
            
            
if __name__ == '__main__':
    flatten_directory('/home/sekilab-liao/Documents/gen3d_2_0/mxl/obj_unlabeled', 
                      '/home/sekilab-liao/Documents/gen3d_2_0/mxl/obj_unlabeled_flattened')