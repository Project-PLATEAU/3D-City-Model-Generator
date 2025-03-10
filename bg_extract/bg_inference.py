import numpy as np
from bg_extract import mmcv_custom
from bg_extract import mmseg_custom
from bg_extract.road_centerline.road_tiff_polygonize import raster_to_vector
from mmseg.apis import inference_segmentor, init_segmentor
from mmcv.runner import load_checkpoint


def mm_road(img, ckpt_path, map_location="cuda:0",
            config_path='bg_extract/ckpt/mm_road/road_config.py'):
    model = init_segmentor(config_path, checkpoint=None, device=map_location)
    load_checkpoint(model, ckpt_path, map_location='cpu')
    result = inference_segmentor(model, img)
    return result[0]

def mm_vegetation(img, ckpt_path, map_location="cuda:0",
            config_path='bg_extract/ckpt/mm_vegetation/vegetation_config.py'):
    model = init_segmentor(config_path, checkpoint=None, device=map_location)
    load_checkpoint(model, ckpt_path, map_location='cpu')
    result = inference_segmentor(model, img)
    result = result[0]
    result[result == 2] = 0
    vec=raster_to_vector(result,'epsg:30169',500)
    res=[np.array(x.exterior.coords) for x in vec.geometry]
    return res
