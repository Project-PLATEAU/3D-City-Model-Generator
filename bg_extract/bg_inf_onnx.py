import glob
import cv2
import os, re
import trimesh
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import onnx
import onnxruntime as ort

from tqdm import tqdm
from osgeo import gdal
from shapely import line_merge
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.affinity import rotate, scale, translate

from bg_extract.road_centerline.road_tiff_polygonize import raster_to_vector,road_line_ext
from utils.tools import *


class genRoad:
    def __init__(self,
                 img_path,
                 width=2.,
                 width_sub=0.1,
                 light_ratio=10.,
                 #  tele_ratio=1.,
                 **kwargs):
        self.img_path = img_path
        # self.img_resolution = img_resolution
        self.width = width
        self.width_sub = width_sub
        self.light_ratio = light_ratio
        # self.tele_ratio = tele_ratio

        self.road_limit = None

    def crop_road_lineStr(self):
        im_proj, im_Geotrans, im_data = read_tif(self.img_path)
        img_shape = im_data.shape
        if min(img_shape) == img_shape[2]:
            height, width, channel = im_data.shape
            im_data = im_data[:, :, :3]
        else:
            channel, height, width = im_data.shape
            im_data = np.transpose(im_data[:3, :, :], (1, 2, 0))

        im_data = im_data[:, :, ::-1]
        x_min, y_min, resolusion_x, resolusion_y = im_Geotrans[0], im_Geotrans[3], im_Geotrans[1], im_Geotrans[5]
        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]
        im_data = cv2.resize(im_data, (512, 512))
        im_data = (im_data - mean) / std
        im_data = np.transpose(im_data, (2, 0, 1))

        model_path = './bg_extract/tensorrt_road/end2end.onnx'
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)

        input_name = session.get_inputs()[0].name
        road_masks = session.run(None, {input_name: im_data.astype(np.float32)[None, ...]})
        road_masks = road_masks[0][0,0].astype(np.int8)
        road_masks = cv2.resize(road_masks, (width, height), interpolation=cv2.INTER_NEAREST)

        road_link = road_line_ext(road_masks, [resolusion_x, resolusion_y], x_min, y_min)
        road_link = [line_merge(xx) for xx in road_link]
        road_link_simp = []

        for road_link_ in road_link:
            road_link_simp += [xx.simplify(0.5) for xx in road_link_.geoms] \
                if isinstance(road_link_, MultiLineString) else [road_link_.simplify(0.5)]
        road_link = split_multilinestring(road_link)
        self.roi_road = gpd.array.GeometryArray(np.array(road_link))

        tmp_shp = gpd.GeoDataFrame(geometry=self.roi_road).set_crs(6668)
        tmp_shp.to_crs(epsg=30169, inplace=True)
        self.roi_road = tmp_shp.geometry.simplify(tolerance=0.1, preserve_topology=True)

        return self.roi_road

    def gen_mesh_road(self, shp, buffer, add_relief=True, gen_relief=None, srs_epsg='EPSG:30169'):
        self.mesh_road = []
        self.buffered_line = shp.buffer(buffer)
        self.road_limit = self.buffered_line

        for poly_road in self.buffered_line:
            if isinstance(poly_road, Polygon):
                tmp_mesh = polygon_to_mesh(poly_road, add_relief=add_relief, gen_relief=gen_relief)
                self.mesh_road.append(tmp_mesh)
            elif isinstance(poly_road, MultiPolygon):
                for poly_road_tmp in poly_road.geoms:
                    print(poly_road_tmp)
                    tmp_mesh = polygon_to_mesh(poly_road_tmp, add_relief=add_relief, gen_relief=gen_relief)
                    self.mesh_road.append(tmp_mesh)

    def gen_mesh_road_sub(self, shp, width, width_sub, add_relief=True, gen_relief=None, srs_epsg='EPSG:30169'):
        left_sub, right_sub = [], []
        for tmp_road in shp:
            if isinstance(tmp_road, LineString):
                left_sub.append(
                    tmp_road.parallel_offset(width * (1 + width_sub + 0.5), 'left').buffer(width * width_sub))
                right_sub.append(
                    tmp_road.parallel_offset(width * (1 + width_sub + 0.5), 'right').buffer(width * width_sub))
            elif isinstance(tmp_road, MultiLineString):
                for tmp_road_ in tmp_road.geoms:
                    left_sub.append(
                        tmp_road_.parallel_offset(width * (1 + width_sub + 0.5), 'left').buffer(width * width_sub))
                    right_sub.append(
                        tmp_road_.parallel_offset(width * (1 + width_sub + 0.5), 'right').buffer(
                            width * width_sub))

        self.road_limit = shp.buffer(width * (1 + width_sub * 2. + 0.5))

        for poly_road in left_sub:
            if isinstance(poly_road, Polygon):
                tmp_mesh = polygon_to_mesh(poly_road, add_relief=add_relief, gen_relief=gen_relief)
                self.mesh_road.append(tmp_mesh)
            elif isinstance(poly_road, MultiPolygon):
                for poly_road_tmp in poly_road.geoms:
                    tmp_mesh = polygon_to_mesh(poly_road_tmp, add_relief=add_relief, gen_relief=gen_relief)
                    self.mesh_road.append(tmp_mesh)

        for poly_road in right_sub:
            if isinstance(poly_road, Polygon):
                tmp_mesh = polygon_to_mesh(poly_road, add_relief=add_relief, gen_relief=gen_relief)
                self.mesh_road.append(tmp_mesh)
            elif isinstance(poly_road, MultiPolygon):
                for poly_road_tmp in poly_road.geoms:
                    tmp_mesh = polygon_to_mesh(poly_road_tmp, add_relief=add_relief, gen_relief=gen_relief)
                    self.mesh_road.append(tmp_mesh)

    def generate_poles_along_line(self, line, interval):
        length = line.length
        return [line.interpolate(distance) for distance in range(0, int(length), interval)]

    def gen_device_lod1(self, shp, add_relief=True, gen_relief=None, srs_epsg='EPSG:30169'):
        self.mesh_device = []

        left_sub = []
        for tmp_road in shp:
            if isinstance(tmp_road, LineString):
                left_sub.append(
                    tmp_road.parallel_offset(self.width * (1 + self.width_sub * 2 + 0.5), 'left'))
            elif isinstance(tmp_road, MultiLineString):
                for tmp_road_ in tmp_road.geoms:
                    left_sub.append(
                        tmp_road_.parallel_offset(self.width * (1 + self.width_sub * 2 + 0.5), 'left'))
        self.road_limit = shp.buffer(self.width * (1 + self.width_sub * 2. + 0.5))
        tele_pole_point = []
        for tmp_road in left_sub:
            tele_pole_point += self.generate_poles_along_line(tmp_road, 20)

        source_crs = pyproj.CRS(srs_epsg)
        target_crs = pyproj.CRS('EPSG:6668')
        crs_transformer = pyproj.Transformer.from_crs(source_crs, target_crs)

        for x in range(len(tele_pole_point)):
            half_side = 0.1
            tele_pole_point_xy = [tele_pole_point[x].x, tele_pole_point[x].y]
            tele_pole_square_coords = [
                (tele_pole_point_xy[0] - half_side, tele_pole_point_xy[1] - half_side),
                (tele_pole_point_xy[0] - half_side, tele_pole_point_xy[1] + half_side),
                (tele_pole_point_xy[0] + half_side, tele_pole_point_xy[1] + half_side),
                (tele_pole_point_xy[0] + half_side, tele_pole_point_xy[1] - half_side)
            ]

            tele_pole_square = Polygon(tele_pole_square_coords)
            vertices, faces = polygon_to_mesh_3D(tele_pole_square)
            tmp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            if add_relief:
                device_centroid_xy = np.mean(np.array(vertices)[:, :2], axis=0)
                lat, lon = crs_transformer.transform(device_centroid_xy[1], device_centroid_xy[0])
                index = [int((lat - gen_relief.dem_geotrans[3]) / gen_relief.dem_geotrans[5]),
                         int((lon - gen_relief.dem_geotrans[0]) / gen_relief.dem_geotrans[1])]
                relief_z = gen_relief.dem_model[index[0], index[1]]
                # print(device_centroid_xy, relief_z, lat, lon, index)

            if not tele_pole_point[x].within(self.road_limit).any():
                self.mesh_device.append(tmp_mesh.apply_translation([0, 0, relief_z]))

    def gen_device_lod2(self, shp, pole_ratio=10, add_relief=True, gen_relief=None, srs_epsg='EPSG:30169'):
        self.mesh_device = []
        self.light_ratio = 1. / pole_ratio

        tele_pole_mesh = trimesh.load(os.path.join('./data_src/src_3d/lod3frn/electric_pole',
                                                   'obj_52385618_frn_6697_op_frn_0ece98a1-6070-4315-88d4-3d4546168814__493155_25.obj'))
        tele_pole_mesh = tele_pole_mesh.dump(concatenate=True) if isinstance(tele_pole_mesh,
                                                                             trimesh.Scene) else tele_pole_mesh
        tele_pole_mesh_xy = tele_pole_mesh.centroid[:2]
        tele_pole_mesh_zmin = np.min(tele_pole_mesh.vertices[:, 2])
        tele_pole_mesh_h = np.max(tele_pole_mesh.vertices[:, 2]) - tele_pole_mesh_zmin

        traf_light_mesh = trimesh.load(os.path.join('./data_src/src_3d/lod3frn/traffic_light',
                                                    'obj_52385618_frn_6697_op_frn_25870971-faa3-4677-b281-f192176cfbea__711063_25.obj'))
        traf_light_mesh = traf_light_mesh.dump(concatenate=True) if isinstance(traf_light_mesh,
                                                                               trimesh.Scene) else traf_light_mesh
        traf_light_mesh_xy = traf_light_mesh.centroid[:2]
        traf_light_mesh_zmin = np.min(traf_light_mesh.vertices[:, 2])

        left_sub = []
        for tmp_road in shp:
            if isinstance(tmp_road, LineString):
                left_sub.append(
                    tmp_road.parallel_offset(self.width * (1 + self.width_sub * 2 + 0.5), 'left'))
            elif isinstance(tmp_road, MultiLineString):
                for tmp_road_ in tmp_road.geoms:
                    left_sub.append(
                        tmp_road_.parallel_offset(self.width * (1 + self.width_sub * 2 + 0.5), 'left'))
        self.road_limit = shp.buffer(self.width * (1 + self.width_sub * 2. + 0.5))
        tele_pole_point = []
        for tmp_road in left_sub:
            tele_pole_point += self.generate_poles_along_line(tmp_road, 20)

        source_crs = pyproj.CRS(srs_epsg)
        target_crs = pyproj.CRS('EPSG:6668')
        crs_transformer = pyproj.Transformer.from_crs(source_crs, target_crs)

        res_tele_pole = []
        for x in range(len(tele_pole_point)):
            tmp_tele_pole_mesh = tele_pole_mesh.copy()
            tele_pole_point_xy = [tele_pole_point[x].x, tele_pole_point[x].y]

            if add_relief:
                lat, lon = crs_transformer.transform(tele_pole_point_xy[1], tele_pole_point_xy[0])
                index = [int((lat - gen_relief.dem_geotrans[3]) / gen_relief.dem_geotrans[5]),
                         int((lon - gen_relief.dem_geotrans[0]) / gen_relief.dem_geotrans[1])]
                relief_z = gen_relief.dem_model[index[0], index[1]]
                # print(tele_pole_point_xy, lat, lon, index, relief_z, tele_pole_mesh_zmin)
                # tele_pole_mesh_zmin -= relief_z
            else:
                relief_z = 0.

            trans_tele_mesh = [tele_pole_point_xy[0] - tele_pole_mesh_xy[0],
                               tele_pole_point_xy[1] - tele_pole_mesh_xy[1],
                               relief_z - tele_pole_mesh_zmin]
            if not tele_pole_point[x].within(self.road_limit).any():
                res_tele_pole.append(tmp_tele_pole_mesh.apply_translation(trans_tele_mesh))

        res_traf_light = []
        for x in random.sample(list(range(len(res_tele_pole))),
                               int(len(res_tele_pole) * self.light_ratio)):
            tmp_traf_light_mesh = traf_light_mesh.copy()
            tele_pole_point_xy = res_tele_pole[x].centroid[:2]

            if add_relief:
                lat, lon = crs_transformer.transform(tele_pole_point_xy[1], tele_pole_point_xy[0])
                index = [int((lat - gen_relief.dem_geotrans[3]) / gen_relief.dem_geotrans[5]),
                         int((lon - gen_relief.dem_geotrans[0]) / gen_relief.dem_geotrans[1])]
                relief_z = gen_relief.dem_model[index[0], index[1]]
                # print(tele_pole_point_xy, lat, lon, index, relief_z, traf_light_mesh_zmin)
                # tele_pole_mesh_zmin -= relief_z

            trans_traf_mesh = [tele_pole_point_xy[0] - traf_light_mesh_xy[0],
                               tele_pole_point_xy[1] - traf_light_mesh_xy[1],
                               relief_z - traf_light_mesh_zmin]
            res_traf_light.append(tmp_traf_light_mesh.apply_translation(trans_traf_mesh))

        self.mesh_device += res_tele_pole + res_traf_light

    def create_citygml_road(self, roads, srs_name="http://www.opengis.net/def/crs/EPSG/0/30169",
                            srsDimension="3"):
        nsmap = {
            'core': "http://www.opengis.net/citygml/2.0",
            'tran': "http://www.opengis.net/citygml/transportation/2.0",
            'gml': "http://www.opengis.net/gml"
        }

        cityModel = etree.Element("{http://www.opengis.net/citygml/2.0}CityModel", nsmap=nsmap)

        total_vertices = []
        for road in roads:
            total_vertices.append(road.vertices)
        total_vertices = np.vstack(total_vertices)
        x_max, y_max, z_max = np.max(total_vertices, axis=0)
        x_min, y_min, z_min = np.min(total_vertices, axis=0)
        boundedBy = etree.SubElement(cityModel, "{http://www.opengis.net/gml}boundedBy")
        Envelope = etree.SubElement(boundedBy, "{http://www.opengis.net/gml}Envelope", srsName=srs_name,
                                    srsDimension=srsDimension)
        lowerCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}lowerCorner")
        upperCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}upperCorner")
        lowerCorner.text = '{} {} {}'.format(x_min, y_min, z_min)
        upperCorner.text = '{} {} {}'.format(x_max, y_max, z_max)

        for road_data in roads:
            vertices, faces = road_data.vertices, road_data.faces

            traffic_member = etree.SubElement(cityModel, "{http://www.opengis.net/citygml/2.0}cityObjectMember")
            transportation = etree.SubElement(traffic_member,
                                              "{http://www.opengis.net/citygml/transportation/2.0}Road")

            lod1MultiSurface = etree.SubElement(transportation,
                                                "{http://www.opengis.net/citygml/transportation/2.0}lod2MultiSurface")
            multiSurface = etree.SubElement(lod1MultiSurface, "{http://www.opengis.net/gml}MultiSurface")

            for face in faces:
                surfaceMember = etree.SubElement(multiSurface, "{http://www.opengis.net/gml}surfaceMember")
                polygon = etree.SubElement(surfaceMember, "{http://www.opengis.net/gml}Polygon")
                exterior = etree.SubElement(polygon, "{http://www.opengis.net/gml}exterior")
                linearRing = etree.SubElement(exterior, "{http://www.opengis.net/gml}LinearRing")
                posList = etree.SubElement(linearRing, "{http://www.opengis.net/gml}posList")

                coords = ' '.join(
                    ['{} {} {}'.format(vertices[idx][0], vertices[idx][1], vertices[idx][2]) for idx in face])
                coords += ' {} {} {}'.format(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2])
                posList.text = coords

        return cityModel

    def create_citygml_cityfurniture(self, devices, srs_name="http://www.opengis.net/def/crs/EPSG/0/30169",
                                     srsDimension="3"):
        nsmap = {
            'core': "http://www.opengis.net/citygml/2.0",
            'frn': "http://www.opengis.net/citygml/cityfurniture/2.0",
            'gml': "http://www.opengis.net/gml"
        }

        cityModel = etree.Element("{http://www.opengis.net/citygml/2.0}CityModel", nsmap=nsmap)

        total_vertices = []
        for device in devices:
            total_vertices.append(device.vertices)
        total_vertices = np.vstack(total_vertices)
        x_max, y_max, z_max = np.max(total_vertices, axis=0)
        x_min, y_min, z_min = np.min(total_vertices, axis=0)
        boundedBy = etree.SubElement(cityModel, "{http://www.opengis.net/gml}boundedBy")
        Envelope = etree.SubElement(boundedBy, "{http://www.opengis.net/gml}Envelope", srsName=srs_name,
                                    srsDimension=srsDimension)
        lowerCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}lowerCorner")
        upperCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}upperCorner")
        lowerCorner.text = '{} {} {}'.format(x_min, y_min, z_min)
        upperCorner.text = '{} {} {}'.format(x_max, y_max, z_max)

        for devices_data in devices:
            vertices, faces = devices_data.vertices, devices_data.faces

            furniture_member = etree.SubElement(cityModel, "{http://www.opengis.net/citygml/2.0}cityObjectMember")
            cityFurniture = etree.SubElement(furniture_member,
                                             "{http://www.opengis.net/citygml/cityfurniture/2.0}CityFurniture")

            lod1Geometry = etree.SubElement(cityFurniture,
                                            "{http://www.opengis.net/citygml/cityfurniture/2.0}lod1Geometry")
            multiSurface = etree.SubElement(lod1Geometry, "{http://www.opengis.net/gml}MultiSurface")

            for face in faces:
                surfaceMember = etree.SubElement(multiSurface, "{http://www.opengis.net/gml}surfaceMember")
                polygon = etree.SubElement(surfaceMember, "{http://www.opengis.net/gml}Polygon")
                exterior = etree.SubElement(polygon, "{http://www.opengis.net/gml}exterior")
                linearRing = etree.SubElement(exterior, "{http://www.opengis.net/gml}LinearRing")
                posList = etree.SubElement(linearRing, "{http://www.opengis.net/gml}posList")

                coords = ' '.join(
                    ['{} {} {}'.format(vertices[idx][0], vertices[idx][1], vertices[idx][2]) for idx in face])
                coords += ' {} {} {}'.format(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2])
                posList.text = coords

        return cityModel

    def add_relief(self, points_relief):
        z_points_interpolate = relief_interpolate(self.mesh_road, points_relief)
        if (z_points_interpolate is None):
            return

        for i, tmp_mesh in enumerate(self.mesh_road):
            tmp_vertices = tmp_mesh.vertices
            tmp_vertices[:, 2] += z_points_interpolate[i] + 0.01
            tmp_mesh.vertices = tmp_vertices

    def gen_road_run(self, road_lod=1, device_lod=1, save_gml=True, gml_root='', road_width_range=[1, 10], road_sub=0.1,
                     add_relief=True, gen_relief=None, srs_epsg='EPSG:30169'):
        self.width = random.randint(road_width_range[0], road_width_range[1])
        self.width_sub = road_sub

        if road_lod >= 1:
            self.gen_mesh_road(self.roi_road, self.width, add_relief=add_relief, gen_relief=gen_relief,
                               srs_epsg=srs_epsg)
        if road_lod == 2:
            self.gen_mesh_road_sub(self.roi_road, self.width, self.width_sub, add_relief=add_relief,
                                   gen_relief=gen_relief, srs_epsg=srs_epsg)

        if device_lod == 1:
            self.gen_device_lod1(self.roi_road, add_relief=add_relief, gen_relief=gen_relief, srs_epsg=srs_epsg)
        elif device_lod == 2:
            self.gen_device_lod2(self.roi_road, add_relief=add_relief, gen_relief=gen_relief, srs_epsg=srs_epsg)

        feat_color_road = (255, 253, 230, 255)
        feat_color_device = (240, 128, 128, 255)
        self.mesh_road = obj_color(self.mesh_road, feat_color_road)
        self.mesh_device = obj_color(self.mesh_device, feat_color_device)

        self.mesh_road += self.mesh_device

        # self.add_relief(points_relief)

        road_ori = self.mesh_road.copy()
        if save_gml:
            road_gml = self.create_citygml_road(road_ori)
            save_citygml(road_gml, os.path.join(gml_root, 'road.gml'))
            device_gml = self.create_citygml_cityfurniture(self.mesh_device)
            save_citygml(device_gml, os.path.join(gml_root, 'device.gml'))

        return self.mesh_road


class genVegetation:
    def __init__(self,
                 img_path,
                 vege_root='./data_src/src_3d/lod3veg/SolitaryVegetationObject/',
                 vege_label='./data_src/src_3d/tree_label.csv',
                 #  low_ratio=0.1,
                 high_ratio=10.,
                 **kwargs):
        check_img_ext(img_path)
        self.img_path = img_path
        self.vege_mes = pd.read_csv(vege_label)
        self.vege_id = self.vege_mes['id'].values
        self.vege_type = self.vege_mes['type'].values
        self.vege_root = vege_root
        # self.low_ratio = low_ratio
        self.high_ratio = high_ratio

    def gen_tree_mesh_lod1(self, limit_road, limit_bdg, dense=200, add_relief=True, gen_relief=None,
                           srs_epsg='EPSG:30169'):
        self.mesh_tree = []

        im_proj, im_Geotrans, im_data = read_tif(self.img_path)
        img_shape = im_data.shape
        if min(img_shape) == img_shape[2]:
            height, width, channel = im_data.shape
            im_data = im_data[:, :, :3]
        else:
            channel, height, width = im_data.shape
            im_data = np.transpose(im_data[:3, :, :], (1, 2, 0))

        im_data = im_data[:, :, ::-1]
        x_min, y_min, resolusion_x, resolusion_y = im_Geotrans[0], im_Geotrans[3], im_Geotrans[1], im_Geotrans[5]
        self.roi_rect = box(x_min, y_min, x_min + width * resolusion_x, y_min + height * resolusion_y)

        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]
        im_data = cv2.resize(im_data, (512, 512))
        im_data = (im_data - mean) / std
        im_data = np.transpose(im_data, (2, 0, 1))

        model_path = './bg_extract/tensorrt_veg/end2end.onnx'
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)

        input_name = session.get_inputs()[0].name
        veg_masks = session.run(None, {input_name: im_data.astype(np.float32)[None, ...]})
        veg_masks = veg_masks[0][0,0].astype(np.int8)
        veg_masks = cv2.resize(veg_masks, (width, height), interpolation=cv2.INTER_NEAREST)

        veg_masks[veg_masks == 2] = 0
        vec = raster_to_vector(veg_masks, srs_epsg, 0)
        seg_contours = [np.array(x.exterior.coords) for x in vec.geometry]

        # seg_contours = mm_vegetation(im_data, 'bg_extract/ckpt/mm_vegetation/vegetation.pth')
        dst_poly = []
        for seg_contour in seg_contours:
            dst_poly.append(Polygon(seg_contour * [resolusion_x, resolusion_y] + [x_min, y_min]))
        dst_poly = gpd.array.GeometryArray(np.array(dst_poly))

        tmp_shp = gpd.GeoDataFrame(geometry=dst_poly).set_crs(6668)
        tmp_shp.to_crs(epsg=30169, inplace=True)
        dst_poly = tmp_shp.geometry

        if limit_bdg is not None:
            limit_bdg = limit_bdg.buffer(3.)
        if limit_road is not None:
            limit_road = limit_road.buffer(3.)

        tar_xy = np.array([[random.uniform(x_min, x_min + width * resolusion_x) for _ in range(dense)],
                           [random.uniform(y_min, y_min + height * resolusion_y) for _ in range(dense)]]).T
        source_crs = pyproj.CRS(srs_epsg)
        target_crs = pyproj.CRS('EPSG:30169')
        crs_transformer = pyproj.Transformer.from_crs(source_crs, target_crs)

        for i in range(len(tar_xy)):
            y, x = np.array(crs_transformer.transform(tar_xy[i, 1], tar_xy[i, 0]))
            tar_xy[i] = np.array([x, y])

        tmp_idx = []
        for i in range(len(tar_xy)):
            if limit_road is not None and Point(tar_xy[i]).within(limit_road).any():
                continue
            if limit_bdg is not None and Point(tar_xy[i]).within(limit_bdg).any():
                continue
            if Point(tar_xy[i]).within(dst_poly).any():
                tmp_idx.append(i)
        tar_xy = tar_xy[tmp_idx]

        high_num = int(len(tar_xy) * self.high_ratio)


        for i in range(high_num):
            tree_poly = Point(tar_xy[i]).buffer(random.uniform(1., 3.))
            tree_height = random.uniform(6., 12.)

            if add_relief:
                lat, lon = crs_transformer.transform(tar_xy[i, 1], tar_xy[i, 0])
                index = [int((lat - gen_relief.dem_geotrans[3]) / gen_relief.dem_geotrans[5]),
                         int((lon - gen_relief.dem_geotrans[0]) / gen_relief.dem_geotrans[1])]
                relief_z = gen_relief.dem_model[index[0], index[1]]
            else:
                relief_z = 0.

            vertices, faces = polygon_to_mesh_3D(tree_poly, tree_height)
            tmp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            self.mesh_tree.append(tmp_mesh.apply_translation([0, 0, relief_z]))

        for i in range(high_num, len(tar_xy)):
            tree_poly = Point(tar_xy[i]).buffer(random.uniform(0.5, 2.))
            tree_height = random.uniform(2., 6.)

            if add_relief:
                lat, lon = crs_transformer.transform(tar_xy[i, 1], tar_xy[i, 0])
                index = [int((lat - gen_relief.dem_geotrans[3]) / gen_relief.dem_geotrans[5]),
                         int((lon - gen_relief.dem_geotrans[0]) / gen_relief.dem_geotrans[1])]
                relief_z = gen_relief.dem_model[index[0], index[1]]
            else:
                relief_z = 0.

            vertices, faces = polygon_to_mesh_3D(tree_poly, tree_height)
            tmp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            self.mesh_tree.append(tmp_mesh.apply_translation([0, 0, relief_z]))

    def gen_tree_mesh_lod2(self, limit_bdg, limit_road, dense, add_relief=True, gen_relief=None, srs_epsg='EPSG:30169'):
        self.mesh_tree = []

        im_proj, im_Geotrans, im_data = read_tif(self.img_path)
        img_shape = im_data.shape
        if min(img_shape) == img_shape[2]:
            height, width, channel = im_data.shape
            im_data = im_data[:, :, :3]
        else:
            channel, height, width = im_data.shape
            im_data = np.transpose(im_data[:3, :, :], (1, 2, 0))

        im_data = im_data[:, :, ::-1]
        x_min, y_min, resolusion_x, resolusion_y = im_Geotrans[0], im_Geotrans[3], im_Geotrans[1], im_Geotrans[5]
        self.roi_rect = box(x_min, y_min, x_min + width * resolusion_x, y_min + height * resolusion_y)

        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]
        im_data = cv2.resize(im_data, (512, 512))
        im_data = (im_data - mean) / std
        im_data = np.transpose(im_data, (2, 0, 1))

        model_path = './bg_extract/tensorrt_veg/end2end.onnx'
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)

        input_name = session.get_inputs()[0].name
        veg_masks = session.run(None, {input_name: im_data.astype(np.float32)[None, ...]})
        veg_masks = veg_masks[0][0,0].astype(np.int8)
        veg_masks = cv2.resize(veg_masks, (width, height), interpolation=cv2.INTER_NEAREST)

        veg_masks[veg_masks == 2] = 0
        vec = raster_to_vector(veg_masks, srs_epsg, 0)
        seg_contours = [np.array(x.exterior.coords) for x in vec.geometry]

        # seg_contours = mm_vegetation(im_data, 'bg_extract/ckpt/mm_vegetation/vegetation.pth')
        dst_poly = []
        for seg_contour in seg_contours:
            dst_poly.append(Polygon(seg_contour * [resolusion_x, resolusion_y] + [x_min, y_min]))
        dst_poly = gpd.array.GeometryArray(np.array(dst_poly))

        tmp_shp = gpd.GeoDataFrame(geometry=dst_poly).set_crs(6668)
        tmp_shp.to_crs(epsg=30169, inplace=True)
        dst_poly = tmp_shp.geometry

        if limit_bdg is not None:
            limit_bdg = limit_bdg.buffer(3.)
        if limit_road is not None:
            limit_road = limit_road.buffer(3.)

        tar_xy = np.array([[random.uniform(x_min, x_min + width * resolusion_x) for _ in range(dense)],
                           [random.uniform(y_min, y_min + height * resolusion_y) for _ in range(dense)]]).T
        source_crs = pyproj.CRS(srs_epsg)
        target_crs = pyproj.CRS('EPSG:30169')
        crs_transformer = pyproj.Transformer.from_crs(source_crs, target_crs)

        for i in range(len(tar_xy)):
            y, x = np.array(crs_transformer.transform(tar_xy[i, 1], tar_xy[i, 0]))
            tar_xy[i] = np.array([x, y])

        tmp_idx = []
        for i in range(len(tar_xy)):
            if limit_road is not None and Point(tar_xy[i]).within(limit_road).any():
                continue
            if limit_bdg is not None and Point(tar_xy[i]).within(limit_bdg).any():
                continue
            if Point(tar_xy[i]).within(dst_poly).any():
                tmp_idx.append(i)
        tar_xy = tar_xy[tmp_idx]

        high_num = int(len(tar_xy) * (self.high_ratio / (self.high_ratio + 1)))
        low_num = int(len(tar_xy) * (1 / (self.high_ratio + 1)))
        high_idx = self.vege_id[self.vege_type == 1]
        low_idx = self.vege_id[self.vege_type == 0]
        high_idx_ = random.choices(list(range(len(high_idx))), k=high_num)
        low_idx_ = random.choices(list(range(len(low_idx))), k=low_num)

        for x, i in enumerate(high_idx_):
            tmp_mesh = trimesh.load(os.path.join(self.vege_root, high_idx[i] + '.obj'))
            tmp_mesh = tmp_mesh.dump(concatenate=True) if isinstance(tmp_mesh, trimesh.Scene) else tmp_mesh
            tmp_mesh_xy = tmp_mesh.centroid[:2]

            if add_relief:
                lat, lon = crs_transformer.transform(tar_xy[x, 1], tar_xy[x, 0])
                index = [int((lat - gen_relief.dem_geotrans[3]) / gen_relief.dem_geotrans[5]),
                         int((lon - gen_relief.dem_geotrans[0]) / gen_relief.dem_geotrans[1])]
                relief_z = gen_relief.dem_model[index[0], index[1]]
            else:
                relief_z = 0.

            tmp_mesh_zmin = np.min(tmp_mesh.vertices[:, 2]) - relief_z

            tmp_trans = [tar_xy[x, 0] - tmp_mesh_xy[0], tar_xy[x, 1] - tmp_mesh_xy[1], -tmp_mesh_zmin]
            self.mesh_tree.append(tmp_mesh.apply_translation(tmp_trans))
        for x, i in enumerate(low_idx_):
            tmp_mesh = trimesh.load(os.path.join(self.vege_root, low_idx[i] + '.obj'))
            tmp_mesh = tmp_mesh.dump(concatenate=True) if isinstance(tmp_mesh, trimesh.Scene) else tmp_mesh
            tmp_mesh_xy = tmp_mesh.centroid[:2]

            if add_relief:
                lat, lon = crs_transformer.transform(tar_xy[x + high_num, 1], tar_xy[x + high_num, 0])
                index = [int((lat - gen_relief.dem_geotrans[3]) / gen_relief.dem_geotrans[5]),
                         int((lon - gen_relief.dem_geotrans[0]) / gen_relief.dem_geotrans[1])]
                relief_z = gen_relief.dem_model[index[0], index[1]]
            else:
                relief_z = 0.

            tmp_mesh_zmin = np.min(tmp_mesh.vertices[:, 2]) - relief_z

            tmp_trans = [tar_xy[x + high_num, 0] - tmp_mesh_xy[0], tar_xy[x + high_num, 1] - tmp_mesh_xy[1],
                         -tmp_mesh_zmin]
            self.mesh_tree.append(tmp_mesh.apply_translation(tmp_trans))

    def add_relief(self, points_relief):
        z_points_interpolate = relief_interpolate(self.mesh_tree, points_relief)
        if (z_points_interpolate is None):
            return

        for i, tmp_mesh in enumerate(self.mesh_tree):
            tmp_vertices = tmp_mesh.vertices
            tmp_vertices[:, 2] += z_points_interpolate[i] + 0.01
            tmp_mesh.vertices = tmp_vertices

    def create_citygml_vegetation(self, vegetation, srs_name="http://www.opengis.net/def/crs/EPSG/0/30169",
                                  srsDimension="3"):
        nsmap = {
            'core': "http://www.opengis.net/citygml/2.0",
            'veg': "http://www.opengis.net/citygml/vegetation/2.0",
            'gml': "http://www.opengis.net/gml"
        }

        cityModel = etree.Element("{http://www.opengis.net/citygml/2.0}CityModel", nsmap=nsmap)

        total_vertices = []
        for tree in vegetation:
            total_vertices.append(tree.vertices)
        total_vertices = np.vstack(total_vertices)
        x_max, y_max, z_max = np.max(total_vertices, axis=0)
        x_min, y_min, z_min = np.min(total_vertices, axis=0)
        boundedBy = etree.SubElement(cityModel, "{http://www.opengis.net/gml}boundedBy")
        Envelope = etree.SubElement(boundedBy, "{http://www.opengis.net/gml}Envelope", srsName=srs_name,
                                    srsDimension=srsDimension)
        lowerCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}lowerCorner")
        upperCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}upperCorner")
        lowerCorner.text = '{} {} {}'.format(x_min, y_min, z_min)
        upperCorner.text = '{} {} {}'.format(x_max, y_max, z_max)

        for vegetation_data in vegetation:
            vertices, faces = vegetation_data.vertices, vegetation_data.faces

            vegetation_member = etree.SubElement(cityModel, "{http://www.opengis.net/citygml/2.0}cityObjectMember")
            plantCover = etree.SubElement(vegetation_member,
                                          "{http://www.opengis.net/citygml/vegetation/2.0}SolitaryVegetationObject")

            lod2Geometry = etree.SubElement(plantCover,
                                            "{http://www.opengis.net/citygml/vegetation/2.0}lod2Geometry")
            multiSurface = etree.SubElement(lod2Geometry, "{http://www.opengis.net/gml}MultiSurface")

            for face in faces:
                surfaceMember = etree.SubElement(multiSurface, "{http://www.opengis.net/gml}surfaceMember")
                polygon = etree.SubElement(surfaceMember, "{http://www.opengis.net/gml}Polygon")
                exterior = etree.SubElement(polygon, "{http://www.opengis.net/gml}exterior")
                linearRing = etree.SubElement(exterior, "{http://www.opengis.net/gml}LinearRing")
                posList = etree.SubElement(linearRing, "{http://www.opengis.net/gml}posList")

                coords = ' '.join(
                    ['{} {} {}'.format(vertices[idx][0], vertices[idx][1], vertices[idx][2]) for idx in face])
                coords += ' {} {} {}'.format(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2])
                posList.text = coords

        return cityModel

    def gen_vege_run(self, limit_road, limit_bdg, add_relief=True, gen_relief=None, srs_epsg='EPSG:30169',
                     dense=None,
                     lod=2,
                     save_gml=True, gml_root=''):
        if not dense:
            dense = random.randint(50, 200)
        if lod == 1:
            self.gen_tree_mesh_lod1(limit_road, limit_bdg, dense, add_relief=add_relief, gen_relief=gen_relief,
                                    srs_epsg=srs_epsg)
        elif lod == 2:
            self.gen_tree_mesh_lod2(limit_road, limit_bdg, dense, add_relief=add_relief, gen_relief=gen_relief,
                                    srs_epsg=srs_epsg)
        # self.add_relief(points_relief)
        if save_gml and len(self.mesh_tree):
            vege_gml = self.create_citygml_vegetation(self.mesh_tree)
            save_citygml(vege_gml, os.path.join(gml_root, 'vegetation.gml'))

        feat_color_vege = (137, 179, 95, 255)
        self.mesh_tree = obj_color(self.mesh_tree, feat_color_vege)

        return self.mesh_tree
