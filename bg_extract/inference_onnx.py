import onnx
import onnxruntime as ort
import numpy as np
import torch
from osgeo import gdal
from torchvision import transforms
import cv2
from tqdm import tqdm


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

mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]
im_data = cv2.resize(im_data,(512,512))
im_data = (im_data-mean)/std
im_data = np.transpose(im_data, (2, 0, 1))


# im_data = torch.Tensor(im_data.copy()).cuda()

# 加载ONNX模型并检查其有效性
model_path = '/fast/zcb/code/cbzhao/bridge2025/bg_extract/tensorrt_road/end2end.onnx'
# 指定使用 CUDA 作为执行提供者
# providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
# 创建 InferenceSession，并指定执行提供者
session = ort.InferenceSession(model_path)


# 获取模型的输入名称和形状
input_name = session.get_inputs()[0].name

outputs = session.run(None, {input_name: im_data.astype(np.float32)[None,...]})

# 输出结果
print("推理结果:", outputs[0][0,0].shape)
res=outputs[0][0,0].astype(np.int8)
res[res==2]=0
cv2.imwrite('/fast/zcb/code/cbzhao/bridge2025/Roof_classification_inference/bg_extract/deploy/demo2.png',res*255)