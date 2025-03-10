import os
import trimesh.util

from Roof_classification_inference.Building_extraction.Inference import main as extraction
from Roof_classification_inference.Building_extraction.Crop_image import crop_tif_with_geojson
from Roof_classification_inference.Roof_classification.Roof_type_inference import main as classification
from Para_calc.Para_calc import height_calc
from utils.overlaps import remove_overlapping
from utils.mtl import convert_vcolor_to_mtl
from utils.concat import concat_mesh
from bg_extract.bg_inf_onnx import genRoad, genVegetation

from Building_Generation_Opening.bldgGen import building_generation

img_dir = './Roof_classification_inference/Test_image/Kaga_2_1_2.tif'
dst_dir = img_dir.split('/')[-1].split('.')[0]+'.geojson'
route_dir = './Building_Generation_Opening/route_elem/' + img_dir.split('/')[-1].split('.')[0] + '.json'
route_exists = os.path.exists(route_dir)

print("Step 1: Extracting buildings and generating GeoJSON...")
extraction(img_dir)

print("Step 2: Cropping building images based on GeoJSON...")
crop_tif_with_geojson(img_dir)

print("Step 3: Classifying roof types and updating GeoJSON...")
classification(img_dir)

print("Step 4: Calculating the height of the building and updating GeoJSON...")
height_calc(dst_dir)

print("Step 4: Updating GeoJSON by removing overlapped cases...")
remove_overlapping(dst_dir, dst_dir)

print("Building information has been saved in: " + dst_dir)

print("Step 5: Creating mesh for buildings...")
route_path = route_dir if route_exists else ''
building_generation(dst_dir, route_path, route_exists=route_exists)

print("Step 6: Creating mesh for roads...")
gen_road = genRoad(img_path=img_dir)
gen_road.crop_road_lineStr()
mesh_road = gen_road.gen_road_run(road_lod=2, device_lod=2, save_gml=False, gml_root='',
                                  road_width_range=[2, 2], road_sub=0.1,
                                  add_relief=False, srs_epsg='EPSG:30169')
mesh_road = trimesh.util.concatenate(mesh_road)
mesh_road.export('road.obj')

print("Step 6: Creating mesh for vegetation...")
gen_vegetation = genVegetation(img_path=img_dir, high_ratio=0.5)
mesh_veg = gen_vegetation.gen_vege_run(limit_road=None, limit_bdg=None, dense=2000, lod=2,
                                        gml_root='', add_relief=False, gen_relief=None,
                                        srs_epsg='EPSG:6668')
mesh_veg = trimesh.util.concatenate(mesh_veg)
mesh_veg.export('veg.obj')

print("Step 7: Concatenating meshes...")
convert_vcolor_to_mtl('buildings.obj',
                      'buildings.obj',
                      'material.mtl')
concat_mesh()
# mesh_veg_road = trimesh.util.concatenate([mesh_veg, mesh_road])
# mesh_veg_road.export('/fast/zcb/code/cbzhao/bridge2025/veg_road.obj')
