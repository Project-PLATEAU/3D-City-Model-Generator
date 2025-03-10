import os
import cv2
import trimesh, pyproj
import math
import numpy as np
import matplotlib.pyplot as plt
from lxml import etree
from earcut.earcut import earcut
from osgeo import gdal
from shapely.geometry import MultiPolygon, Polygon, MultiLineString, LineString, Point, box

from PIL import Image

def check_img_ext(filename):
    _, ext = os.path.splitext(filename)

    ext = ext.lower()

    if ext not in ['.tif', '.tiff']:
        raise ValueError("Input image should be .tif or .tiff")

def read_tif(path):
    dataset = gdal.Open(path)

    cols = dataset.RasterXSize
    rows = (dataset.RasterYSize)
    im_proj = (dataset.GetProjection())
    im_Geotrans = (dataset.GetGeoTransform())
    im_data = dataset.ReadAsArray(0, 0, cols, rows)
    del dataset
    return im_proj, im_Geotrans, im_data


def write_tif(filename, im_geotrans, im_proj, im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset

def vis_polygon(polygon):
    if isinstance(polygon, Polygon):
        x, y = polygon.exterior.xy

        fig, ax = plt.subplots()
        ax.plot(x, y)

        ax.set_title('Shapely Polygon Visualization')
        plt.show()
    elif isinstance(polygon, list):
        fig, ax = plt.subplots()
        for i in polygon:
            x, y = i.exterior.xy
            ax.plot(x, y)

        ax.set_title('Shapely Polygon Visualization')
        plt.show()


def calculate_bearing(polygon):
    # mbr = polygon.minimum_rotated_rectangle

    mbr_coords = list(polygon.exterior.coords)

    edge_lengths = [Point(mbr_coords[i]).distance(Point(mbr_coords[i - 1])) for i in range(len(mbr_coords))]
    max_length_index = edge_lengths.index(max(edge_lengths))

    dx = mbr_coords[max_length_index][0] - mbr_coords[max_length_index - 1][0]
    dy = mbr_coords[max_length_index][1] - mbr_coords[max_length_index - 1][1]

    angle = math.degrees(math.atan2(dy, dx))
    bearing = (450 - angle) % 360

    return bearing


def get_ab(polygon):
    rect = polygon.envelope
    rect = list(rect.exterior.coords)
    edge_lengths = [Point(rect[i]).distance(Point(rect[i - 1])) for i in range(1, len(rect))]

    return [max(edge_lengths), min(edge_lengths)]


def get_polygon(data):
    mesh_point = data.vertices
    min_z = np.min(mesh_point[:, 2])
    plane_origin = [0, 0, min_z + 1]
    plane_normal = [0, 0, 1]
    slice3d = data.section(plane_normal=plane_normal, plane_origin=plane_origin)
    if not slice3d:
        return

    slice2d, affn = slice3d.to_planar()
    slice_poly = slice2d.polygons_full
    if len(slice_poly) == 0:
        return
    slice_poly = np.column_stack((slice_poly[0].exterior.coords.xy))
    homo_array = np.hstack((slice_poly, np.zeros((len(slice_poly), 1)), np.ones((len(slice_poly), 1))))
    homo_3d = homo_array @ (affn.T)

    poly_xy = homo_3d[:, :2] / homo_3d[:, -1, None]

    return Polygon(poly_xy)


def polygon_iou(polygon1, polygon2):
    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area

    iou = intersection / union
    return iou


def polygon_to_mesh(polygon, add_relief=True, gen_relief=None, srs_epsg='EPSG:30169'):
    exterior_coords = np.array(polygon.exterior.coords)
    vertices = np.hstack((exterior_coords, np.zeros((exterior_coords.shape[0], 1))))

    triangles = earcut(exterior_coords.flatten(), dim=2)
    faces = np.reshape(triangles, (-1, 3))
    
    if add_relief:
        source_crs = pyproj.CRS(srs_epsg)
        target_crs = pyproj.CRS('EPSG:6668')
        crs_transformer = pyproj.Transformer.from_crs(source_crs, target_crs)
        
        for vertex in vertices:
            lat, lon = crs_transformer.transform(vertex[1], vertex[0])
            index = [int((lat - gen_relief.dem_geotrans[3]) / gen_relief.dem_geotrans[5]), int((lon - gen_relief.dem_geotrans[0]) / gen_relief.dem_geotrans[1])]
            relief_z = gen_relief.dem_model[index[0], index[1]] + 0.05
            
            vertex[2] += relief_z

    res_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return res_mesh


def polygon_to_mesh_3D(polygon, height=3.):
    exterior_coords = np.array(polygon.exterior.coords)
    vertices_btm = np.hstack((exterior_coords, np.zeros((exterior_coords.shape[0], 1))))

    triangles = earcut(exterior_coords.flatten(), dim=2)
    faces_btm = np.reshape(triangles, (-1, 3))
    faces_btm = np.hstack([faces_btm, faces_btm[:, 0][:, None]])

    l_btm = len(vertices_btm)
    vertices_top = vertices_btm + [0., 0., height]
    faces_top = faces_btm + [l_btm] * 4

    faces_side = []
    for j in range(l_btm - 1):
        faces_side.append([j, j + l_btm, j + l_btm + 1, j + 1])
    faces_side = np.array(faces_side)

    vertices = np.vstack([vertices_btm, vertices_top])
    faces = np.vstack([faces_btm, faces_top, faces_side])

    return vertices, faces


def relief_interpolate(mesh_list, points_relief):
    if (points_relief is None):
        return
    z_points = points_relief[..., 2]
    xy_start_points = points_relief[0, 0, :2]
    xy_dif = points_relief[1, 1, :2] - points_relief[0, 0, :2]
    point_shape = points_relief.shape

    z_points_interpolate = []
    for tmp_mesh in mesh_list:
        tmp_vertices = tmp_mesh.vertices
        tmp_xy_idx = (tmp_vertices[:, :2] - xy_start_points) / xy_dif
        tmp_xy1 = tmp_xy_idx.astype(int)
        tmp_xy2 = tmp_xy_idx.astype(int) + 1
        tmp_xy2[:, 0][tmp_xy2[:, 0] > (point_shape[1] - 1)] = point_shape[1] - 1
        tmp_xy2[:, 1][tmp_xy2[:, 1] > (point_shape[0] - 1)] = point_shape[0] - 1
        tmp_dxy = tmp_xy_idx - tmp_xy1

        V11 = z_points[tmp_xy1[:, 1], tmp_xy1[:, 0]]
        V21 = z_points[tmp_xy1[:, 1], tmp_xy2[:, 0]]
        V12 = z_points[tmp_xy2[:, 1], tmp_xy1[:, 0]]
        V22 = z_points[tmp_xy2[:, 1], tmp_xy2[:, 0]]

        tmp_z_points_interpolate = (V11 * (1 - tmp_dxy[:, 0]) * (1 - tmp_dxy[:, 1]) +
                                    V21 * tmp_dxy[:, 0] * (1 - tmp_dxy[:, 1]) +
                                    V12 * (1 - tmp_dxy[:, 0]) * tmp_dxy[:, 1] +
                                    V22 * tmp_dxy[:, 0] * tmp_dxy[:, 1])
        z_points_interpolate.append(tmp_z_points_interpolate)
    return z_points_interpolate


def flatten_multipolygons(geometry):
    if isinstance(geometry, MultiPolygon):
        return [polygon for polygon in geometry.geoms]
    else:
        return [geometry]

def obj_color(mesh_list, feat_color):
    width, height = 10, 10
    im = Image.new('RGBA', (width, height), feat_color)
    material = trimesh.visual.texture.SimpleMaterial(image=im)
    for x in range(len(mesh_list)):
        uv = np.random.rand(mesh_list[x].vertices.shape[0], 2)
        color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=im, material=material)
        mesh_list[x] = trimesh.Trimesh(vertices=mesh_list[x].vertices, faces=mesh_list[x].faces,
                                       visual=color_visuals, validate=True, process=False)

    return mesh_list

def save_citygml(root, file_name):
    tree = etree.ElementTree(root)
    tree.write(file_name, pretty_print=True, xml_declaration=True, encoding='UTF-8')

def split_multilinestring(geom_list):
    split_list = []
    for geom in geom_list:
        if isinstance(geom, LineString):
            split_list.append(geom)
        elif isinstance(geom, MultiLineString):
            split_list.extend(geom.geoms)
        else:
            raise TypeError(f"Unsupported geometry type: {type(geom)}")
    return split_list