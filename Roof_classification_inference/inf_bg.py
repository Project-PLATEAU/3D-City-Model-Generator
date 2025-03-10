from osgeo import gdal
import geopandas as gpd
import numpy as np
import cv2
from shapely.geometry import MultiPolygon, Polygon, MultiLineString, LineString, Point, box
from bg_extract.bg_inference import mm_road, mm_vegetation
def read_tif(path):
    dataset = gdal.Open(path)

    cols = dataset.RasterXSize
    rows = (dataset.RasterYSize)
    im_proj = (dataset.GetProjection())
    im_Geotrans = (dataset.GetGeoTransform())
    im_data = dataset.ReadAsArray(0, 0, cols, rows)
    del dataset
    return im_proj, im_Geotrans, im_data

img_path='/fast/zcb/code/cbzhao/bridge2025/Roof_classification_inference/Test_image/Kaga_2_1_2.tif'
im_proj, im_Geotrans, im_data = read_tif(img_path)
img_shape = im_data.shape
if min(img_shape) == img_shape[2]:
    height, width, channel = im_data.shape
    im_data = im_data[:, :, :3]
else:
    channel, height, width = im_data.shape
    im_data = np.transpose(im_data[:3, :, :], (1, 2, 0))

im_data = im_data[:, :, ::-1]

x_min, y_min, resolusion_x, resolusion_y = im_Geotrans[0], im_Geotrans[3], im_Geotrans[1], im_Geotrans[5]
roi_rect = box(x_min, y_min, x_min + width * resolusion_x, y_min + height * resolusion_y)

seg_contours = mm_vegetation(im_data, 'bg_extract/ckpt/mm_vegetation/vegetation.pth')
dst_poly = []
for seg_contour in seg_contours:
    dst_poly.append(Polygon(seg_contour * [resolusion_x, resolusion_y] + [x_min, y_min]))
dst_poly = gpd.array.GeometryArray(np.array(dst_poly))
print(dst_poly)