import os
import cv2
import glob
import torch
import random
import einops
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from osgeo import gdal
import geopandas as gpd

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

from tiff_to_mesh import xToMesh
from geojson_reader import read_geojson_and_rasterize
import trimesh

from lxml import etree

ckpt_files = ['/fast/zcb/code/ControlNet/lightning_logs/plateau_dataEnhancement_type1/checkpoints/epoch=134-step=20999.ckpt',
              '/fast/zcb/code/ControlNet/lightning_logs/plateau_dataEnhancement_type2/checkpoints/epoch=36-step=28999.ckpt',
              '/fast/zcb/code/ControlNet/lightning_logs/plateau_dataEnhancement_type3/checkpoints/epoch=17-step=25999.ckpt',
              '/fast/zcb/code/ControlNet/lightning_logs/plateau_dataEnhancement_NType5/checkpoints/epoch=95-step=18999.ckpt',
              '/fast/zcb/code/ControlNet/lightning_logs/plateau_dataEnhancement_NType6/checkpoints/epoch=13-step=13999.ckpt']
obj_output_path = '/fast/zcb/data/PLATEAU_obj/obj_geo/obj_geo.obj'
im_proj_epsg = 'epsg:30169'


# 输入图像
def read_tif(path):
    dataset = gdal.Open(path)
    # print(dataset.GetDescription())  # 数据描述

    cols = dataset.RasterXSize  # 图像长度
    rows = (dataset.RasterYSize)  # 图像宽度
    im_proj = (dataset.GetProjection())  # 读取投影
    im_Geotrans = (dataset.GetGeoTransform())  # 读取仿射变换
    im_data = dataset.ReadAsArray(0, 0, cols, rows)  # 转为numpy格式
    # im_data[im_data > 0] = 1 #除0以外都等于1
    del dataset
    return im_proj, im_Geotrans, im_data

def write_img(filename, im_proj, im_geotrans, im_data):
    '''
    :param filename: 存储文件路径
    :param im_proj: 投影系统
    :param im_geotrans: 仿射矩阵
    :param im_data: 栅格矩阵
    :return:
    '''
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset

def get_mask(data):
    _, binary = cv2.threshold(data, 2, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    outline = np.zeros_like(data)
    cv2.drawContours(outline, contours, -1, (255), -1)
    return outline

def cal_normal_vector(im_data):
    height, width = im_data.shape

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    coordinates = np.stack((x, y), axis=-1)

    depth = np.concatenate([coordinates, im_data[..., None]], axis=2)

    depth_dx = np.zeros(depth.shape)
    depth_dx[:, 1:, 2] = depth[:, :-1, 2]
    depth_dx[..., :2] = depth[..., :2]
    depth_dx[..., 1] += 1
    depth_dy = np.zeros(depth.shape)
    depth_dy[1:, :, 2] = depth[:-1, :, 2]
    depth_dy[..., :2] = depth[..., :2]
    depth_dy[..., 0] += 1

    depth_dx = depth_dx - depth
    depth_dy = depth_dy - depth

    nrv = np.cross(depth_dx, depth_dy)
    mode = np.linalg.norm(nrv, axis=2)[..., None]
    nrv = nrv / mode

    return nrv

def det_slope(im_data):
    nrv=cal_normal_vector(im_data)
    # normals_normalized = (nrv + 1) / 2.0

    lower_threshold = 50
    upper_threshold = 150
    edges = cv2.Canny(im_data, lower_threshold, upper_threshold)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    num_labels, labels = cv2.connectedComponents(255 - dilated)
    normalized_labels = (labels * (255.0 / num_labels)).astype(np.uint8)

    cos_theta = nrv[..., 2]  # n[2] 是 n_z
    theta = np.arccos(cos_theta)

    # 将夹角从弧度转为度数，如果需要
    theta_degrees = np.degrees(theta)
    theta_degrees[theta_degrees > 90.] = 180 - theta_degrees[theta_degrees > 90.]

    mask = np.zeros(im_data.shape, dtype=bool)
    for i in range(num_labels):
        tmp = theta_degrees[labels == i]
        if len(tmp[(tmp > 5.) & (tmp < 85.)]) > 10000:
            mask[labels == i] = 1
    return mask

def replace_with_median_at_positions(image, positions, ksize=5):
    """替换指定位置的像素为其邻域的中位数。

    参数:
    - image: 输入图像
    - positions: 要替换的位置列表，格式为 [(y1, x1), (y2, x2), ...]
    - ksize: 中值滤波的核大小，例如 3 表示 3x3 的核
    """
    # 对整个图像进行中值滤波
    median_blurred = cv2.medianBlur(image, ksize)

    # 只在指定的位置替换值
    output = np.copy(image)
    for y, x in positions:
        output[y, x] = median_blurred[y, x]

    return output

def post_processing(im_data):
    image_1D = im_data.reshape((-1, 1))
    image_1D = np.float32(image_1D)
    blurred = cv2.GaussianBlur(im_data, (5, 5), 0)

    # 定义k均值聚类的终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # k表示我们想要的聚类数量或波峰数量
    k = 10
    _, labels, centers = cv2.kmeans(image_1D, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

    # 将中心值转换回uint8
    centers = np.uint8(centers)

    # 根据聚类结果映射回原始图像
    segmented_image = centers[labels.flatten()]

    # var=np.array([np.var(image_1D[labels==i]) for i in range(len(centers))])
    # for i in range(len(var)):
    #     tmp=len(image_1D[labels == i])*var[i]
    #     print(tmp)
    #     if tmp>1e5:
    #         segmented_image[labels==i]=image_1D[labels==i]

    segmented_image = segmented_image.reshape(im_data.shape)

    low_f = cv2.medianBlur(segmented_image, 5)
    high_f = segmented_image - low_f
    mask = np.stack(np.where(high_f > 10), axis=1)

    segmented_image = replace_with_median_at_positions(segmented_image, mask)

    mask_slope = det_slope(im_data)
    segmented_image[mask_slope] = blurred[mask_slope]
    return segmented_image


def keypoint_read(keypoint_path):
    keypoint = gpd.read_file(keypoint_path)
    keypoint_array = np.array(keypoint['geometry'].iloc[0].exterior.coords)

    return keypoint_array


def OBJ_output(obj_file_path, vertices, faces, vertex_num, bldg_lod=1):
    assert bldg_lod in [1, 2]
    with open(obj_file_path, 'a') as obj_file:
        for vertex in vertices:
            formatted_vertex = " ".join(["v"] + [str(i) for i in vertex])
            obj_file.write(formatted_vertex + '\n')

        for face in faces:
            # print(face)
            formatted_face = " ".join(["f"] + [str(i) for i in face]) if bldg_lod == 1 else " ".join(["f"] + [str(i + vertex_num) for i in face])
            obj_file.write(formatted_face + '\n')


def save_citygml(root, file_name):
    tree = etree.ElementTree(root)
    tree.write(file_name, pretty_print=True, xml_declaration=True, encoding='UTF-8')


def arg():
    parser = argparse.ArgumentParser()

    parser.add_argument("cldm_ckpt", help="Input .ckpt file name")
    parser.add_argument("input", help="Input .tiff file name")
    parser.add_argument("--output", help="Output file name")
    parser.add_argument("--cfg", help="Config file name", default='./models/cldm_v21.yaml')

    # 解析命令行参数
    args = parser.parse_args()
    return args


def inference(model, ddim_sampler, im_data, im_Geotrans, keypoint_array, vertex_num, scale=2, seed=1024):
    # im_proj, im_Geotrans, im_data = read_tif(input_tiff)
    mask = get_mask(im_data)
    img = np.stack([mask, mask, mask], axis=2)

    # print(im_proj)
    # img = cv2.imread(src_path[i])
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = HWC3(img)
    H, W, C = img.shape
    num_samples = 1

    control = torch.from_numpy(img).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    # seed = -1
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    save_memory = True
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    prompt = 'digital surface model, depth map, black background, flat peak area'
    a_prompt = ''
    n_prompt = ''
    cond = {"c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
    un_cond = {"c_concat": None, "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    model.control_scales = ([1.] * 13)

    eta = 0.
    ddim_steps = 50
    # scale = random.uniform(1.5, 3.)
    # scale = 2.
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples, shape, cond, verbose=False, eta=eta,
                                                 unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)

    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(
        np.uint8)
    out = cv2.cvtColor(x_samples[0], cv2.COLOR_BGR2RGB)

    mask = img[:, :, 0]
    # out = np.max(out, axis=2)
    out = out[:, :, 0]
    out[mask == 0] = 0

    out = post_processing(out)

    # find contours
    _, binary_out = cv2.threshold(out, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_points = contours[0].reshape(-1, 2)

    contour_points = np.array(contour_points)

    keypoint_array = [
        [int((tuples[0] - im_Geotrans[0]) / im_Geotrans[1]), int((tuples[1] - im_Geotrans[3]) / im_Geotrans[5])]
        for tuples in keypoint_array]
    mesh_converter = xToMesh(out, im_Geotrans)

    mesh_converter.tiffToCloudWithGeoInfo(1)
    mesh_converter.delaunayTriangulation()

    if mesh_converter.meshToArray(contour_points, keypoint_array, True, True) == 0:
        print("Written out vertices and faces to array. ")

    vertices = [mesh_converter.getTriVertices(i) for i in range(mesh_converter.getVertexNum())]
    faces = [mesh_converter.getTriFaces(i) for i in range(mesh_converter.getFaceNum())]

    # OBJ_output(obj_output_path, vertices, faces, vertex_num)

    print(f'Current vertex number: {len(vertices)}, total vertex number: {vertex_num}')

    print("finish one-loop inference.")

    return vertices, faces


def bldg_lod1_gen_realCity(keypoint_array, storey_low, storey_high, vertex_num):
    bldg_vertices, bldg_faces = [], []
    vertex_index = 1
    # print(vertex_num)

    # Roof & bottom
    for r in range(2):
        face = []
        rand_elev = random.randint(storey_low, storey_high)
        for p in keypoint_array[:-1]:
            elev = rand_elev if r == 0 else 0
            bldg_vertices.append([p[0], p[1], elev])

            face.append(vertex_index + vertex_num)
            vertex_index = vertex_index + 1

        bldg_faces.append(face[::-1])

    print(bldg_faces)
    # Facade
    keypoint_num = len(keypoint_array) - 1
    for idx in range(1, keypoint_num + 1):
        idx = idx + vertex_num
        face = [idx, idx + 1, idx + keypoint_num + 1, idx + keypoint_num] \
            if idx < keypoint_num + vertex_num else [idx, vertex_num + 1, vertex_num + keypoint_num + 1, idx + keypoint_num]
        bldg_faces.append(face)

    # OBJ_output(obj_output_path, bldg_vertices, bldg_faces, vertex_num)

    return bldg_vertices, bldg_faces


def bldg_citygml_realCity(vertices, faces, vertex_num=0, lod=2, srs_name="http://www.opengis.net/def/crs/EPSG/0/30169", srsDimension="3"):
    assert lod in [1, 2]

    # print(vertices, faces, len(vertices), len(faces))
    # Header
    nsmap = {
        'core': "http://www.opengis.net/citygml/2.0",
        'bldg': "http://www.opengis.net/citygml/building/2.0",
        'gml': "http://www.opengis.net/gml"
    }
    cityModel = etree.Element("{http://www.opengis.net/citygml/2.0}CityModel", nsmap=nsmap)

    # bounding
    # total_vertices = []
    # for building in np.array(vertices):
    #     total_vertices.extend(building)
    total_vertices = np.vstack(vertices)
    x_max, y_max, z_max = np.max(total_vertices, axis=0)
    x_min, y_min, z_min = np.min(total_vertices, axis=0)

    boundedBy = etree.SubElement(cityModel, "{http://www.opengis.net/gml}boundedBy")
    Envelope = etree.SubElement(boundedBy, "{http://www.opengis.net/gml}Envelope", srsName=srs_name,
                                srsDimension=srsDimension)
    lowerCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}lowerCorner")
    upperCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}upperCorner")
    lowerCorner.text = '{} {} {}'.format(x_min, y_min, z_min)
    upperCorner.text = '{} {} {}'.format(x_max, y_max, z_max)

    # geometry
    if lod == 1:
        for vs, fs in zip(vertices, faces):
            building_member = etree.SubElement(cityModel, "{http://www.opengis.net/citygml/2.0}cityObjectMember")
            building = etree.SubElement(building_member, "{http://www.opengis.net/citygml/building/2.0}Building")

            lod1Solid = etree.SubElement(building, "{http://www.opengis.net/citygml/building/2.0}lod1Solid")
            solid = etree.SubElement(lod1Solid, "{http://www.opengis.net/gml}Solid")
            exterior = etree.SubElement(solid, "{http://www.opengis.net/gml}exterior")
            compositeSurface = etree.SubElement(exterior, "{http://www.opengis.net/gml}CompositeSurface")

            for f in fs:
                surfaceMember = etree.SubElement(compositeSurface, "{http://www.opengis.net/gml}surfaceMember")
                polygon = etree.SubElement(surfaceMember, "{http://www.opengis.net/gml}Polygon")
                exterior = etree.SubElement(polygon, "{http://www.opengis.net/gml}exterior")
                linearRing = etree.SubElement(exterior, "{http://www.opengis.net/gml}LinearRing")
                posList = etree.SubElement(linearRing, "{http://www.opengis.net/gml}posList")

                # print(len(vs), f)
                coords = ' '.join(
                    ['{} {} {}'.format(vs[idx - vertex_num - 1][0], vs[idx - vertex_num - 1][1], vs[idx - vertex_num - 1][2]) for idx in f]
                )
                coords += ' {} {} {}'.format(vs[f[0] - vertex_num - 1][0], vs[f[0] - vertex_num - 1][1], vs[f[0] - vertex_num - 1][2])
                posList.text = coords

            vertex_num = vertex_num + len(vs)
    
    elif lod == 2:
        for vs, fs in zip(vertices, faces):
            vs = np.array(vs)
            z_min, z_max = np.min(vs[:, 2]), np.max(vs[:, 2])
            
            building_member = etree.SubElement(cityModel, "{http://www.opengis.net/citygml/2.0}cityObjectMember")
            building = etree.SubElement(building_member, "{http://www.opengis.net/citygml/building/2.0}Building")
            
            measuredHeight = etree.SubElement(building,
                                                  "{http://www.opengis.net/citygml/building/2.0}measuredHeight")
            measuredHeight.text = str(round(z_max - z_min, 2))

            for f in fs:
                boundedBy = etree.SubElement(building,
                                                 "{http://www.opengis.net/citygml/building/2.0}boundedBy")

                zf = [idx - vertex_num - 1 for idx in f]
                z_face = vs[zf][:, 2]
                if (z_face - z_min < 1.).all():
                    typeSurface = etree.SubElement(boundedBy,
                                                    "{http://www.opengis.net/citygml/building/2.0}GroundSurface")
                elif (z_face - z_min > 1.).all():
                    typeSurface = etree.SubElement(boundedBy,
                                                    "{http://www.opengis.net/citygml/building/2.0}RoofSurface")
                else:
                    typeSurface = etree.SubElement(boundedBy,
                                                    "{http://www.opengis.net/citygml/building/2.0}WallSurface")

                lod2MultiSurface = etree.SubElement(typeSurface,
                                                    "{http://www.opengis.net/citygml/building/2.0}lod2MultiSurface")
                MultiSurface = etree.SubElement(lod2MultiSurface, "{http://www.opengis.net/gml}MultiSurface")
                surfaceMember = etree.SubElement(MultiSurface, "{http://www.opengis.net/gml}surfaceMember")
                polygon = etree.SubElement(surfaceMember, "{http://www.opengis.net/gml}Polygon")
                exterior = etree.SubElement(polygon, "{http://www.opengis.net/gml}exterior")
                linearRing = etree.SubElement(exterior, "{http://www.opengis.net/gml}LinearRing")
                posList = etree.SubElement(linearRing, "{http://www.opengis.net/gml}posList")

                # print(len(vs), f)
                coords = ' '.join(
                    ['{} {} {}'.format(vs[idx - vertex_num - 1][0], vs[idx - vertex_num - 1][1], vs[idx - vertex_num - 1][2]) for idx in f]
                )
                coords += ' {} {} {}'.format(vs[f[0] - vertex_num - 1][0], vs[f[0] - vertex_num - 1][1], vs[f[0] - vertex_num - 1][2])
                posList.text = coords

            vertex_num = vertex_num + len(vs)

    else:
        raise Exception("Building error: Only lod 1 and 2 is implemented. ")

    return cityModel


# def main():
#     # args=arg()
#     # ckpt_file=args.cldm_ckpt
#     # input_tiffs=args.input
#     # output_tiff=args.output
#     # cfg_file=args.cfg


#     ckpt_file = '/fast/zcb/code/ControlNet/lightning_logs/plateau_dataEnhancement_NType6/checkpoints/epoch=13-step=13999.ckpt'
#     input_tiffs = '/fast/zcb/data/pleatau/data/gen_test_data/TIFF/003_92234.tiff'
#     output_tiff = '/fast/zcb/data/pleatau/data/gen_test_data/TIFF_out'
#     cfg_file = '/fast/zcb/code/ControlNet/models/cldm_v21.yaml'
#     geojson_path = '/fast/zcb/data/PLATEAU_obj/gen3d_realCity/gen3d_realCity_testData/test02/footprint/footprint_test_2_selected.geojson'
#     gml_root_path = '/fast/zcb/data/PLATEAU_obj/gml_geo'
#     # obj_output_path = '/fast/zcb/data/PLATEAU_obj/obj_geo/obj_geo.obj'

#     os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#     # apply_uniformer = UniformerDetector()

#     # Create output directory
#     if not os.path.exists(output_tiff):
#         os.mkdir(output_tiff)

#     # Read footprints
#     polygons, names, origin_coords, pixel_sizes, footprint_images = read_geojson_and_rasterize(geojson_path)

#     # Create model
#     model = create_model(cfg_file).cpu()
#     # model.load_state_dict(
#     #     load_state_dict(ckpt_file,
#     #                     location='cuda'))
#     # model = model.cuda()
#     # ddim_sampler = DDIMSampler(model)

#     # Clear OBJ file
#     if os.path.exists(obj_output_path):
#         os.remove(obj_output_path)

#     with open(obj_output_path, 'w'):
#         pass

#     # Main loop
#     bldg_lod = 1

#     index_list = [i for i in range(len(footprint_images))]
#     random.shuffle(index_list)

#     vertex_num = 0
#     ckpt_idx = 0
#     ddim_sampler = ""

#     acc_vertices, acc_faces = [], []
#     if bldg_lod == 2:
#         for i, idx in enumerate(index_list):
#             if not i % 10:
#                 model.load_state_dict(
#                     load_state_dict(ckpt_files[ckpt_idx],
#                                     location='cuda'))
#                 model = model.cuda()
#                 ddim_sampler = DDIMSampler(model)

#                 ckpt_idx += 1
#                 if ckpt_idx >= 5:
#                     ckpt_idx = 0

#             img_Geotrans = np.array(
#                 [origin_coords[idx][0], pixel_sizes[idx][0], 0, origin_coords[idx][1], 0, pixel_sizes[idx][1]],
#                 dtype=np.float32)
#             vertices, faces = inference(model, ddim_sampler, footprint_images[idx], img_Geotrans, polygons[idx][0], vertex_num)

#             vertex_num = vertex_num + len(vertices)
#             acc_vertices.append(vertices)
#             acc_faces.append(faces)
#     elif bldg_lod == 1:
#         for idx, polygon in enumerate(polygons):
#             img_Geotrans = np.array(
#                 [origin_coords[idx][0], pixel_sizes[idx][0], 0, origin_coords[idx][1], 0, pixel_sizes[idx][1]],
#                 dtype=np.float32)
#             vertices, faces = bldg_lod1_gen_realCity(polygon[0], img_Geotrans, vertex_num)

#             vertex_num = vertex_num + len(vertices)
#             # print(vertex_num)
#             acc_vertices.append(vertices)
#             acc_faces.append(faces)
#     else:
#         raise Exception("Not valid LOD. Check again. ")

#     bldg_gml = bldg_citygml_realCity(acc_vertices, acc_faces, bldg_lod)
#     save_citygml(bldg_gml, os.path.join(gml_root_path, f'bldg_test_lod{bldg_lod}.gml'))

#     return 1

# if __name__ == "__main__":
#     main()
