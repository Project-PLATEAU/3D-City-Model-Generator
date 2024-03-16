import numpy as np
import geopandas as gpd
import trimesh
import cv2
import os
import glob
from pyproj import Proj, transform
from shapely.geometry import Polygon
import pandas as pd
from PIL import Image


def write_obj(tmp_texture, tmp_vertices, feat_mesh, out_path):
    obj_v = ['v {} {} {}'.format(x, y, z) for x, y, z in tmp_vertices] + ['v {} {} {}'.format(x, y, z) for x, y, z in
                                                                          feat_mesh.vertices]
    obj_vt = []
    obj_f = []

    feat_faces=feat_mesh.faces+1+len(tmp_vertices)

    for i in range(len(tmp_texture) // 3):
        if tmp_texture[i * 3][1] == -1:
            obj_f.append(
                'f {}/1 {}/1 {}/1'.format(tmp_texture[i * 3][0], tmp_texture[i * 3 + 1][0], tmp_texture[i * 3 + 2][0]))
        else:
            obj_vt.append('vt {} {}'.format(tmp_texture[i * 3][2], tmp_texture[i * 3][3]))
            obj_vt.append('vt {} {}'.format(tmp_texture[i * 3 + 1][2], tmp_texture[i * 3 + 1][3]))
            obj_vt.append('vt {} {}'.format(tmp_texture[i * 3 + 2][2], tmp_texture[i * 3 + 2][3]))
            obj_f.append(
                'f {}/{} {}/{} {}/{}'.format(tmp_texture[i * 3][0], tmp_texture[i * 3][1], tmp_texture[i * 3 + 1][0],
                                             tmp_texture[i * 3 + 1][1], tmp_texture[i * 3 + 2][0],
                                             tmp_texture[i * 3 + 2][1]))
    for i in range(len(feat_faces)):
        obj_f.append('f {} {} {}'.format(feat_faces[i,0],feat_faces[i,1],feat_faces[i,2]))
    with open(out_path, 'w') as f:
        f.write('mtllib material.mtl\nusemtl material_0\n')
        for i in obj_v:
            f.write(i + '\n')
        for i in obj_vt:
            f.write(i + '\n')
        for i in obj_f:
            f.write(i + '\n')


def img_concat(path, concat_path):
    img_path = glob.glob(os.path.join(path, '*.jpg'))
    img_path.sort(reverse=True)

    h_total, w_total = 0, 0
    for i in range(len(img_path)):
        tmp_img = cv2.imread(img_path[i])
        h_total += tmp_img.shape[0]
        w_total = max(tmp_img.shape[1], w_total)
    res = np.zeros((h_total, w_total, 3), dtype=np.uint8)
    h_cnt_tmp = 0
    for i in range(len(img_path)):
        tmp_img = cv2.imread(img_path[i])
        h_tmp, w_tmp, _ = tmp_img.shape
        res[h_cnt_tmp:h_cnt_tmp + h_tmp, :w_tmp] = tmp_img
        h_cnt_tmp += tmp_img.shape[0]
    cv2.imwrite(concat_path, res)
    return h_total, w_total


def create_mtl(path):
    with open(os.path.join(path, 'material.mtl'), 'w') as f:
        f.write(
            'newmtl material_0\nKa 0.40000000 0.40000000 0.40000000\n'
            'Kd 0.40000000 0.40000000 0.40000000\nKs 0.40000000 0.40000000 0.40000000\n'
            'Ns 1.00000000\nmap_Kd texture.jpg\n')


def set_bldg_texture(bldg_mesh_txt, img_folder, height_csv, out_folder):
    create_mtl(out_folder)
    h_total, w_total = img_concat(img_folder, os.path.join(out_folder, 'texture.jpg'))
    src_proj = Proj('epsg:4326')
    dst_proj = Proj('epsg:30169')

    mes = open(bldg_mesh_txt).readlines()
    mes = [i.strip() for i in mes]

    bid = []
    lod0 = []
    lod1_pos = []
    lod1_faces = []
    for i in range(len(mes)):
        if mes[i][0] in '0123456789':
            bid.append(mes[i].split()[-1])
            lod0.append(mes[i + 2].split()[2:])

        if mes[i][:4] == 'lod1':
            tmp_lod1_pos = []
            tmp_lod1_faces = []
            surface_num = int(mes[i].split()[-1])
            for j in range(surface_num):
                tmp_lod1_pos.append([float(x) for x in mes[i + j * 2 + 1].split()[2:]])
                tmp_lod1_faces.append(mes[i + j * 2 + 2].split()[1:])
            lod1_pos.append(tmp_lod1_pos)
            lod1_faces.append(tmp_lod1_faces)

    bldg_height_mes = pd.read_csv(height_csv)
    bldg_height_id = bldg_height_mes['plateau_id'].values
    bldg_height = bldg_height_mes['elevation'].values
    bldg_height_dict = {}
    for i in range(len(bldg_height_id)):
        bldg_height_dict[bldg_height_id[i]] = bldg_height[i]

    bldg_meshes = []
    tmp_num_point = 1
    tmp_vertices, tmp_faces = [], []
    tmp_texture = []
    tmp_num_vt = 0
    h_cnt = 0

    gt_cnt_h, pred_cnt_h = 0., 0.
    acc = 0.
    for i in range(len(lod1_pos)):
        tmp_vertice = []
        img = cv2.imread(os.path.join(img_folder, r'%03d.jpg') % i)
        h, w, _ = img.shape
        for j in range(len(lod1_pos[i])):
            tmp_pos = np.reshape(np.array(lod1_pos[i][j]), (-1, 3))
            tmp_xy = np.array(transform(src_proj, dst_proj, tmp_pos[:, 0], tmp_pos[:, 1]))[::-1].T

            tmp_pos[:, :2] = tmp_xy

            tmp_face_num = int(lod1_faces[i][j][0])
            tmp_face = np.reshape(np.array(lod1_faces[i][j][1:tmp_face_num * 3 + 1], dtype=int),
                                  (-1, 3)) + tmp_num_point
            tmp_num_point += len(tmp_pos)

            tmp_vertice.append(tmp_pos)
            tmp_faces.append(tmp_face)

            for k in range(tmp_face_num):
                if lod1_faces[i][j][tmp_face_num * 3 + 1 + k * 6] == 'None':
                    tmp_texture += [[tmp_face[k, 0], -1, -1, -1], [tmp_face[k, 1], -1, -1, -1],
                                    [tmp_face[k, 2], -1, -1, -1]]
                else:
                    tmp_texture += [
                        [tmp_face[k, 0], tmp_num_vt + 1,
                         max(0, float(lod1_faces[i][j][tmp_face_num * 3 + 1 + k * 6]) / w_total),
                         max(0, (h - float(lod1_faces[i][j][tmp_face_num * 3 + 1 + k * 6 + 1]) + h_cnt) / h_total)],
                        [tmp_face[k, 1], tmp_num_vt + 2,
                         max(0, float(lod1_faces[i][j][tmp_face_num * 3 + 1 + k * 6 + 2]) / w_total),
                         max(0, (h - float(lod1_faces[i][j][tmp_face_num * 3 + 1 + k * 6 + 3]) + h_cnt) / h_total)],
                        [tmp_face[k, 2], tmp_num_vt + 3,
                         max(0, float(lod1_faces[i][j][tmp_face_num * 3 + 1 + k * 6 + 4]) / w_total),
                         max(0, (h - float(lod1_faces[i][j][tmp_face_num * 3 + 1 + k * 6 + 5]) + h_cnt) / h_total)]]
                    tmp_num_vt += 3
        h_cnt += h
        tmp_vertice = np.vstack(tmp_vertice)
        tmp_z = tmp_vertice[:, 2]
        tmp_maxz, tmp_minz = np.max(tmp_z), np.min(tmp_z)
        tmp_bldg_h = tmp_maxz - tmp_minz
        if bid[i] in bldg_height_id:
            tar_h = bldg_height_dict[bid[i]] / 1.5
            tmp_vertice[:, 2][tmp_z > (tmp_minz + 1.)] += (tar_h - tmp_bldg_h)
            gt_cnt_h += tmp_bldg_h
            pred_cnt_h += tar_h
            acc += 1 - abs(tar_h - tmp_bldg_h) / tmp_bldg_h
        tmp_vertice[:, 2] -= tmp_minz
        tmp_vertices.append(tmp_vertice)

    print(pred_cnt_h / gt_cnt_h)
    print('Acc: ', acc / len(bldg_height_dict))
    tmp_vertices = np.vstack(tmp_vertices)
    return tmp_texture, tmp_vertices

# set_bldg_texture(r'C:\zcb\data\plateau\data\svi\texture\route3\bldg_mesh.txt',
#                  r'C:\zcb\data\plateau\data\svi\texture\route3\img',
#                  r'C:\zcb\data\plateau\data\svi\route3_elevation.csv',
#                  r'C:\zcb\data\plateau\data\t\svi\bldg_test')

# h_total, w_total = img_concat(r'C:\zcb\data\plateau\data\svi\texture\route3\img')
# src_proj = Proj('epsg:4326')
# dst_proj = Proj('epsg:30169')
#
# mes = open(r'C:\zcb\data\plateau\data\svi\texture\route3\bldg_mesh.txt').readlines()
# mes = [i.strip() for i in mes]
#
# bid = []
# lod0 = []
# lod1_pos = []
# lod1_faces = []
# for i in range(len(mes)):
#     if mes[i][0] in '0123456789':
#         bid.append(mes[i].split()[-1])
#         lod0.append(mes[i + 2].split()[2:])
#
#     if mes[i][:4] == 'lod1':
#         tmp_lod1_pos = []
#         tmp_lod1_faces = []
#         surface_num = int(mes[i].split()[-1])
#         for j in range(surface_num):
#             tmp_lod1_pos.append([float(x) for x in mes[i + j * 2 + 1].split()[2:]])
#             tmp_lod1_faces.append(mes[i + j * 2 + 2].split()[1:])
#         lod1_pos.append(tmp_lod1_pos)
#         lod1_faces.append(tmp_lod1_faces)
#
# bldg_height_mes = pd.read_csv(r'C:\zcb\data\plateau\data\svi\route3_elevation.csv')
# bldg_height_id = bldg_height_mes['plateau_id'].values
# bldg_height = bldg_height_mes['elevation'].values
# bldg_height_dict = {}
# for i in range(len(bldg_height_id)):
#     bldg_height_dict[bldg_height_id[i]] = bldg_height[i]
#
# bldg_meshes = []
# tmp_num_point = 1
# tmp_vertices, tmp_faces = [], []
# tmp_texture = []
# tmp_num_vt = 0
# h_cnt = 0
#
# gt_cnt_h, pred_cnt_h = 0., 0.
# acc = 0.
# for i in range(len(lod1_pos)):
#     tmp_vertice = []
#     img = cv2.imread(r'C:\zcb\data\plateau\data\svi\texture\route3\img\%03d.jpg' % i)
#     h, w, _ = img.shape
#     for j in range(len(lod1_pos[i])):
#         tmp_pos = np.reshape(np.array(lod1_pos[i][j]), (-1, 3))
#         tmp_xy = np.array(transform(src_proj, dst_proj, tmp_pos[:, 0], tmp_pos[:, 1]))[::-1].T
#
#         tmp_pos[:, :2] = tmp_xy
#
#         tmp_face_num = int(lod1_faces[i][j][0])
#         tmp_face = np.reshape(np.array(lod1_faces[i][j][1:tmp_face_num * 3 + 1], dtype=int), (-1, 3)) + tmp_num_point
#         tmp_num_point += len(tmp_pos)
#
#         tmp_vertice.append(tmp_pos)
#         tmp_faces.append(tmp_face)
#
#         for k in range(tmp_face_num):
#             if lod1_faces[i][j][tmp_face_num * 3 + 1 + k * 6] == 'None':
#                 tmp_texture += [[tmp_face[k, 0], -1, -1, -1], [tmp_face[k, 1], -1, -1, -1],
#                                 [tmp_face[k, 2], -1, -1, -1]]
#             else:
#                 tmp_texture += [
#                     [tmp_face[k, 0], tmp_num_vt + 1,
#                      max(0, float(lod1_faces[i][j][tmp_face_num * 3 + 1 + k * 6]) / w_total),
#                      max(0, (h - float(lod1_faces[i][j][tmp_face_num * 3 + 1 + k * 6 + 1]) + h_cnt) / h_total)],
#                     [tmp_face[k, 1], tmp_num_vt + 2,
#                      max(0, float(lod1_faces[i][j][tmp_face_num * 3 + 1 + k * 6 + 2]) / w_total),
#                      max(0, (h - float(lod1_faces[i][j][tmp_face_num * 3 + 1 + k * 6 + 3]) + h_cnt) / h_total)],
#                     [tmp_face[k, 2], tmp_num_vt + 3,
#                      max(0, float(lod1_faces[i][j][tmp_face_num * 3 + 1 + k * 6 + 4]) / w_total),
#                      max(0, (h - float(lod1_faces[i][j][tmp_face_num * 3 + 1 + k * 6 + 5]) + h_cnt) / h_total)]]
#                 tmp_num_vt += 3
#     h_cnt += h
#     tmp_vertice = np.vstack(tmp_vertice)
#     tmp_z = tmp_vertice[:, 2]
#     tmp_maxz, tmp_minz = np.max(tmp_z), np.min(tmp_z)
#     tmp_bldg_h = tmp_maxz - tmp_minz
#     if bid[i] in bldg_height_id:
#         tar_h = bldg_height_dict[bid[i]] / 1.5
#         tmp_vertice[:, 2][tmp_z > (tmp_minz + 1.)] += (tar_h - tmp_bldg_h)
#         gt_cnt_h += tmp_bldg_h
#         pred_cnt_h += tar_h
#         acc += 1 - abs(tar_h - tmp_bldg_h) / tmp_bldg_h
#     tmp_vertice[:, 2] -= tmp_minz
#     tmp_vertices.append(tmp_vertice)
#
# print(pred_cnt_h / gt_cnt_h)
# print('Acc: ', acc / len(bldg_height_dict))
# tmp_vertices = np.vstack(tmp_vertices)
# write_obj(tmp_texture, tmp_vertices, r'C:\zcb\data\plateau\data\t\svi\res\bldg_test9.obj')

# fp = [Polygon(np.reshape(np.array(i, dtype=float), (-1, 3))[:, :2][:, ::-1]) for i in lod0]
# gpd.GeoDataFrame(data=bid,geometry=fp,crs=4326,columns=['bid']).to_file(r'C:\zcb\data\plateau\data\t\svi\shp\bldg1_lod0.shp')
