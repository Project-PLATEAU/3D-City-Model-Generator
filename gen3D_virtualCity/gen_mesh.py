import glob
import cv2
import os
import trimesh
import math
import random
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from lxml import etree
from earcut.earcut import earcut
from shapely.geometry import MultiPolygon, Polygon, MultiLineString, LineString, Point, box
from shapely.affinity import rotate, scale, translate
from scipy.spatial import Delaunay


class genRelief:
    def __init__(self,
                 reilef_src_root=r'data\src_2d\dem\crop_resize',
                 height_limit=5.,
                 **kwargs):
        self.relief_src_root = reilef_src_root
        self.height_limit = height_limit

        self.relief_src_path = glob.glob(os.path.join(self.relief_src_root, '*.jpg'))
        self.points_relief = None
        self.mesh_relief = []

    def gen_mesh_relief_lod0(self, x_min, y_min, width=200., height=200.):
        l_path = len(self.relief_src_path)
        rand_idx = random.randint(0, l_path - 1)
        rand_dem_path = self.relief_src_path[rand_idx]

        img = cv2.imread(rand_dem_path, 0)

        h_img, w_img = img.shape
        x, y = np.meshgrid(np.arange(w_img), np.arange(h_img))
        coordinates = np.stack((x_min + x * width / (w_img - 1), y_min + y * height / (h_img - 1)), axis=-1)
        points = np.concatenate([coordinates, img[..., None] / 255. * self.height_limit], axis=2)
        points_reshape = np.reshape(points, (-1, 3))

        points_2d = points_reshape[:, :2]
        tri = Delaunay(points_2d)
        faces = tri.simplices

        self.mesh_relief0 = trimesh.Trimesh(vertices=points_reshape, faces=faces)
        self.points_relief = points
        return self.mesh_relief0

    def gen_mesh_relief_lod1(self, x_min, y_min, width=200., height=200.):
        l_path = len(self.relief_src_path)
        rand_idx = random.randint(0, l_path - 1)
        rand_dem_path = self.relief_src_path[rand_idx]

        img = cv2.imread(rand_dem_path, 0)

        h_img, w_img = img.shape
        x, y = np.meshgrid(np.arange(w_img), np.arange(h_img))
        coordinates = np.stack((x_min + x * width / (w_img - 1), y_min + y * height / (h_img - 1)), axis=-1)
        points = np.concatenate([coordinates, img[..., None] / 255. * self.height_limit], axis=2)
        points_reshape = np.reshape(points, (-1, 3))

        points_2d = points_reshape[:, :2]
        tri = Delaunay(points_2d)
        faces = tri.simplices

        self.mesh_relief = trimesh.Trimesh(vertices=points_reshape, faces=faces)
        self.points_relief = points
        return self.mesh_relief

    def create_citygml_relief(self, relief, relief_lod=1, srs_name="http://www.opengis.net/def/crs/EPSG/0/30169",
                              srsDimension="3"):
        nsmap = {
            'core': "http://www.opengis.net/citygml/2.0",
            'dem': "http://www.opengis.net/citygml/relief/2.0",
            'gml': "http://www.opengis.net/gml"
        }

        cityModel = etree.Element("{http://www.opengis.net/citygml/2.0}CityModel", nsmap=nsmap)

        for relief_data in relief:
            vertices, faces = relief_data.vertices, relief_data.faces

            x_max, y_max, z_max = np.max(vertices, axis=0)
            x_min, y_min, z_min = np.min(vertices, axis=0)
            boundedBy = etree.SubElement(cityModel, "{http://www.opengis.net/gml}boundedBy")
            Envelope = etree.SubElement(boundedBy, "{http://www.opengis.net/gml}Envelope", srsName=srs_name,
                                        srsDimension=srsDimension)
            lowerCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}lowerCorner")
            upperCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}upperCorner")
            lowerCorner.text = '{} {} {}'.format(x_min, y_min, z_min)
            upperCorner.text = '{} {} {}'.format(x_max, y_max, z_max)

            relief_member = etree.SubElement(cityModel, "{http://www.opengis.net/citygml/2.0}cityObjectMember")
            reliefFeature = etree.SubElement(relief_member, "{http://www.opengis.net/citygml/relief/2.0}ReliefFeature")

            if relief_lod==0:
                lod1_1 = etree.SubElement(reliefFeature, "{http://www.opengis.net/citygml/relief/2.0}lod")
                lod1_1.text = str(relief_lod)
                reliefComponent = etree.SubElement(reliefFeature,
                                                   "{http://www.opengis.net/citygml/relief/2.0}reliefComponent")

                for vertice in vertices:
                    MassPointRelief = etree.SubElement(reliefComponent, "{http://www.opengis.net/gml}MassPointRelief")
                    posList = etree.SubElement(MassPointRelief, "{http://www.opengis.net/gml}posList")

                    coords = '{} {} {}'.format(vertice[0], vertice[1], vertice[2])
                    posList.text = coords

            elif relief_lod==1:
                lod1_1 = etree.SubElement(reliefFeature, "{http://www.opengis.net/citygml/relief/2.0}lod")
                lod1_1.text = str(relief_lod)
                reliefComponent = etree.SubElement(reliefFeature,
                                                   "{http://www.opengis.net/citygml/relief/2.0}reliefComponent")
                TINRelief = etree.SubElement(reliefComponent, "{http://www.opengis.net/citygml/relief/2.0}TINRelief")
                lod1_2 = etree.SubElement(TINRelief, "{http://www.opengis.net/citygml/relief/2.0}lod")
                lod1_2.text = str(relief_lod)
                tin = etree.SubElement(TINRelief, "{http://www.opengis.net/citygml/relief/2.0}tin")
                triangulatedSurface = etree.SubElement(tin, "{http://www.opengis.net/gml}TriangulatedSurface",
                                                       srsName=srs_name,
                                                       srsDimension=srsDimension)
                trianglePatches = etree.SubElement(triangulatedSurface, "{http://www.opengis.net/gml}trianglePatches")

                for face in faces:
                    polygon = etree.SubElement(trianglePatches, "{http://www.opengis.net/gml}Triangle")
                    exterior = etree.SubElement(polygon, "{http://www.opengis.net/gml}exterior")
                    linearRing = etree.SubElement(exterior, "{http://www.opengis.net/gml}LinearRing")
                    posList = etree.SubElement(linearRing, "{http://www.opengis.net/gml}posList")

                    coords = ' '.join(
                        ['{} {} {}'.format(vertices[idx][0], vertices[idx][1], vertices[idx][2]) for idx in face])
                    coords += ' {} {} {}'.format(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2])
                    posList.text = coords

        return cityModel

    def gen_relief_run(self, x_min, y_min, width=200., height=200., relief_lod=1, save_gml=True, gml_root=''):
        if relief_lod == 0:
            self.gen_mesh_relief_lod0(x_min, y_min, width, height)
        elif relief_lod == 1:
            self.gen_mesh_relief_lod1(x_min, y_min, width, height)
        if save_gml:
            if relief_lod==0:
                relief_gml = self.create_citygml_relief([self.mesh_relief0], relief_lod=relief_lod,
                                                        srs_name="http://www.opengis.net/def/crs/EPSG/0/30169",
                                                        srsDimension="3")
            elif relief_lod==1:
                relief_gml = self.create_citygml_relief([self.mesh_relief], relief_lod=relief_lod,
                                                        srs_name="http://www.opengis.net/def/crs/EPSG/0/30169",
                                                        srsDimension="3")
            save_citygml(relief_gml, os.path.join(gml_root, 'relief.gml'))
            return self.mesh_relief


class genBuilding:
    def __init__(self,
                 bdg_src_path=r'data\src_2d\shp\tatemono_filter1.shp',
                 bdg_obj_label_path=r'data\src_3d\merged_filter1.csv',
                 bdg_obj_root=r'data\src_3d\obj\\',
                 probabilities=[1., 0., 0., 0., 0., 0., 0.],
                 low_storey=2,
                 high_storey=35,
                 **kwargs):

        self.bdg_src_path = bdg_src_path
        self.probabilities = probabilities
        self.bdg_src = gpd.read_file(self.bdg_src_path).geometry.values

        self.obj_mes = pd.read_csv(bdg_obj_label_path)
        self.obj_type = self.obj_mes['type'].values
        # self.obj_type = self.type_map_bdg(self.obj_type)
        self.obj_root = bdg_obj_root

        self.low_storey = low_storey
        self.high_storey = high_storey

    def crop_blg_poly(self, x_min, y_min, width=200., height=200.):
        self.roi_rect = box(x_min, y_min, x_min + width, y_min + height)
        self.roi_building = self.bdg_src[self.bdg_src.within(self.roi_rect)]
        return self.roi_building

    def type_map_bdg(self, data):
        data[(data == 5) | (data == 6) | (data == 7) | (data == 9) | (data == 12) | (data == 13)] = 5
        data[(data == 8) | (data == 11)] = 6
        data[(data == 10)] = 7
        return data

    def get_ab(self, polygon):
        rect = polygon.envelope
        rect = list(rect.exterior.coords)
        edge_lengths = [Point(rect[i]).distance(Point(rect[i - 1])) for i in range(1, len(rect))]

        return [max(edge_lengths), min(edge_lengths)]

    def get_reshape_scale(self, polygon1, polygon2):
        poly1_max, poly1_min = self.get_ab(polygon1)
        poly2_max, poly2_min = self.get_ab(polygon2)

        return min(poly1_max / poly2_max, poly1_min / poly2_min)

    def get_polygon(self, data):
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

    def mesh_poly_iou(self, mesh, rot_shp_poly, center_poly, visualize=False):
        center_poly_xy = [center_poly.x, center_poly.y]
        tmp_mesh_poly = self.get_polygon(mesh)
        if not tmp_mesh_poly:
            return 0, 1
        tmp_mesh_b = calculate_bearing(tmp_mesh_poly)
        rot_mesh_poly = rotate(tmp_mesh_poly, tmp_mesh_b)
        center_mesh = rot_mesh_poly.centroid
        center_mesh_xy = [center_mesh.x, center_mesh.y]

        tmp_mesh_poly = translate(rot_mesh_poly, center_poly_xy[0] - center_mesh_xy[0],
                                  center_poly_xy[1] - center_mesh_xy[1])

        mesh_scale = self.get_reshape_scale(rot_shp_poly, tmp_mesh_poly)
        scaled_mesh_poly = scale(tmp_mesh_poly, xfact=mesh_scale, yfact=mesh_scale, origin=center_poly)

        iou = polygon_iou(rot_shp_poly, scaled_mesh_poly)

        if visualize:
            vis_polygon([rot_shp_poly, scaled_mesh_poly])

        return iou, mesh_scale

    def mesh_poly_transfer(self, mesh, polygon, mp_scale):
        mesh.apply_scale(mp_scale)
        mesh_point = mesh.vertices
        min_z = np.min(mesh_point[:, 2])

        tmp_mesh_poly = get_polygon(mesh)
        if not tmp_mesh_poly:
            return
        cen_tmp_mesh_poly = tmp_mesh_poly.centroid

        tmp_mesh_b = calculate_bearing(tmp_mesh_poly)
        shp_poly_b = calculate_bearing(polygon)
        rotation_matrix = trimesh.transformations.rotation_matrix(
            np.radians(tmp_mesh_b - shp_poly_b), [0, 0, 1], (cen_tmp_mesh_poly.x, cen_tmp_mesh_poly.y, 0))

        mesh.apply_transform(rotation_matrix)
        tmp_mesh_poly = get_polygon(mesh)
        cen_tmp_mesh_poly = tmp_mesh_poly.centroid
        cen_polygon = polygon.centroid
        mesh.apply_translation([cen_polygon.x - cen_tmp_mesh_poly.x, cen_polygon.y - cen_tmp_mesh_poly.y, -min_z])

        return mesh

    def gen_mesh_building_lod0(self, shp_polys, limit=None):
        self.mesh_building = []
        self.poly_building = []

        shp_l = len(shp_polys)
        for i in tqdm(range(shp_l)):
            shp_poly = shp_polys[i]
            if (limit is not None) and limit.intersection(shp_poly).any():
                continue
            self.poly_building.append(shp_poly)

            tmp_mesh = polygon_to_mesh(shp_poly)
            self.mesh_building.append(tmp_mesh)
        self.building_limit = gpd.array.GeometryArray(np.array(self.poly_building))

        return self.mesh_building

    def gen_mesh_building_lod1(self, shp_polys, limit=None):
        self.mesh_building = []
        self.poly_building = []

        shp_l = len(shp_polys)
        for i in tqdm(range(shp_l)):
            shp_poly = shp_polys[i]
            if (limit is not None) and limit.intersection(shp_poly).any():
                continue
            self.poly_building.append(shp_poly)

            vertices, faces = polygon_to_mesh_3D(shp_poly)
            tmp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            self.mesh_building.append(tmp_mesh)
        self.building_limit = gpd.array.GeometryArray(np.array(self.poly_building))

        return self.mesh_building

    def gen_mesh_building_lod2(self, shp_polys, probabilities, limit=None, visualize=False):
        shp_l = len(shp_polys)

        # shp_polys = shp_poly.geometry.values
        self.mesh_building = []
        self.poly_building = []
        for i in tqdm(range(shp_l)):
            shp_poly = shp_polys[i]
            shp_poly_b = calculate_bearing(shp_poly)
            rot_shp_poly = rotate(shp_poly, shp_poly_b)
            center_poly = rot_shp_poly.centroid

            tmp_type_choice = random.choices(range(1, 8), probabilities)
            tmp_type_id = np.where(self.obj_type == tmp_type_choice[0])[0]
            idx_rand = random.sample(list(tmp_type_id), min(100, len(tmp_type_id)))

            mp_iou, mp_scale = 0, 1
            for j in idx_rand:
                tmp_mesh = trimesh.load(os.path.join(self.obj_root, self.obj_mes['id'].values[j] + '.obj'))
                tmp_mesh = tmp_mesh.dump(concatenate=True) if isinstance(tmp_mesh, trimesh.Scene) else tmp_mesh
                tmp_iou, tmp_scale = self.mesh_poly_iou(tmp_mesh, rot_shp_poly, center_poly)
                if tmp_iou > 0.85:
                    mp_iou, mp_scale = tmp_iou, tmp_scale
                    mesh_tar = tmp_mesh
                    break
                if tmp_iou > mp_iou:
                    mp_iou, mp_scale = tmp_iou, tmp_scale
                    mesh_tar = tmp_mesh

            mesh_tar = self.mesh_poly_transfer(mesh_tar, shp_poly, mp_scale)

            if mesh_tar:
                tmp_poly_building = get_polygon(mesh_tar)
                if self.roi_rect.contains(tmp_poly_building) and (not limit.intersection(tmp_poly_building).any()):
                    self.poly_building.append(tmp_poly_building)
                    self.mesh_building.append(mesh_tar)

        self.building_limit = []
        for xx in self.mesh_building:
            self.building_limit.append(get_polygon(xx))
        if visualize:
            vis_polygon(self.building_limit)
        self.building_limit = gpd.array.GeometryArray(np.array(self.building_limit))

        return self.mesh_building

    def set_building_storey(self):
        low, high = self.low_storey * 3, self.high_storey * 3

        for tmp_mesh in self.mesh_building:
            rand_height = random.uniform(low, high)
            mesh_point = tmp_mesh.vertices
            mesh_point_z = mesh_point[:, 2]
            min_z = np.min(mesh_point_z)

            h_trans = rand_height - np.max(mesh_point_z)
            mesh_point[:, 2][mesh_point_z > (min_z + 1.)] += h_trans
            tmp_mesh.vertices = mesh_point

    def add_relief(self, points_relief):
        z_points_interpolate = relief_interpolate(self.mesh_building, points_relief)
        if (z_points_interpolate is None):
            return

        for i, tmp_mesh in enumerate(self.mesh_building):
            tmp_vertices = tmp_mesh.vertices
            tmp_vertices[:, 2] += np.min(z_points_interpolate[i])
            tmp_mesh.vertices = tmp_vertices

    def create_citygml_building(self, buildings, lod=1, srs_name="http://www.opengis.net/def/crs/EPSG/0/30169",
                                srsDimension="3"):
        nsmap = {
            'core': "http://www.opengis.net/citygml/2.0",
            'bldg': "http://www.opengis.net/citygml/building/2.0",
            'gml': "http://www.opengis.net/gml"
        }
        cityModel = etree.Element("{http://www.opengis.net/citygml/2.0}CityModel", nsmap=nsmap)

        total_vertices = []
        for building in buildings:
            total_vertices.append(building.vertices)
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

        if lod == 0:
            for building_data in buildings:
                vertices, faces = building_data.vertices, building_data.faces

                building_member = etree.SubElement(cityModel, "{http://www.opengis.net/citygml/2.0}cityObjectMember")
                building = etree.SubElement(building_member, "{http://www.opengis.net/citygml/building/2.0}Building")

                lod0RoofEdge = etree.SubElement(building, "{http://www.opengis.net/citygml/building/2.0}lod0RoofEdge")
                solid = etree.SubElement(lod0RoofEdge, "{http://www.opengis.net/gml}Solid")
                exterior = etree.SubElement(solid, "{http://www.opengis.net/gml}exterior")
                compositeSurface = etree.SubElement(exterior, "{http://www.opengis.net/gml}CompositeSurface")

                for face in faces:
                    surfaceMember = etree.SubElement(compositeSurface, "{http://www.opengis.net/gml}surfaceMember")
                    polygon = etree.SubElement(surfaceMember, "{http://www.opengis.net/gml}Polygon")
                    exterior = etree.SubElement(polygon, "{http://www.opengis.net/gml}exterior")
                    linearRing = etree.SubElement(exterior, "{http://www.opengis.net/gml}LinearRing")
                    posList = etree.SubElement(linearRing, "{http://www.opengis.net/gml}posList")

                    coords = ' '.join(
                        ['{} {} {}'.format(vertices[idx][0], vertices[idx][1], vertices[idx][2]) for idx in face])
                    coords += ' {} {} {}'.format(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2])
                    posList.text = coords
        if lod == 1:
            for building_data in buildings:
                vertices, faces = building_data.vertices, building_data.faces

                building_member = etree.SubElement(cityModel, "{http://www.opengis.net/citygml/2.0}cityObjectMember")
                building = etree.SubElement(building_member, "{http://www.opengis.net/citygml/building/2.0}Building")

                lod1Solid = etree.SubElement(building, "{http://www.opengis.net/citygml/building/2.0}lod1Solid")
                solid = etree.SubElement(lod1Solid, "{http://www.opengis.net/gml}Solid")
                exterior = etree.SubElement(solid, "{http://www.opengis.net/gml}exterior")
                compositeSurface = etree.SubElement(exterior, "{http://www.opengis.net/gml}CompositeSurface")

                for face in faces:
                    surfaceMember = etree.SubElement(compositeSurface, "{http://www.opengis.net/gml}surfaceMember")
                    polygon = etree.SubElement(surfaceMember, "{http://www.opengis.net/gml}Polygon")
                    exterior = etree.SubElement(polygon, "{http://www.opengis.net/gml}exterior")
                    linearRing = etree.SubElement(exterior, "{http://www.opengis.net/gml}LinearRing")
                    posList = etree.SubElement(linearRing, "{http://www.opengis.net/gml}posList")

                    coords = ' '.join(
                        ['{} {} {}'.format(vertices[idx][0], vertices[idx][1], vertices[idx][2]) for idx in face])
                    coords += ' {} {} {}'.format(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2])
                    posList.text = coords
        elif lod == 2:
            for building_data in buildings:
                vertices, faces = building_data.vertices, building_data.faces
                z_min, z_max = np.min(vertices[:, 2]), np.max(vertices[:, 2])

                building_member = etree.SubElement(cityModel, "{http://www.opengis.net/citygml/2.0}cityObjectMember")
                building = etree.SubElement(building_member, "{http://www.opengis.net/citygml/building/2.0}Building")

                measuredHeight = etree.SubElement(building,
                                                  "{http://www.opengis.net/citygml/building/2.0}measuredHeight")
                measuredHeight.text = str(round(z_max - z_min, 2))

                for face in faces:
                    boundedBy = etree.SubElement(building,
                                                 "{http://www.opengis.net/citygml/building/2.0}boundedBy")
                    z_face = vertices[face][:, 2]
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

                    coords = ' '.join(
                        ['{} {} {}'.format(vertices[idx][0], vertices[idx][1], vertices[idx][2]) for idx in face])
                    coords += ' {} {} {}'.format(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2])
                    posList.text = coords

        return cityModel

    def gen_building_run(self, building_lod=2, limit=None, points_relief=None, visualize=False, save_gml=True,
                         gml_root=''):
        if building_lod == 0:
            self.gen_mesh_building_lod0(self.roi_building, limit)
        elif building_lod == 1:
            self.gen_mesh_building_lod1(self.roi_building, limit)
            self.set_building_storey()
        elif building_lod == 2:
            self.gen_mesh_building_lod2(self.roi_building, self.probabilities, limit, visualize)
            if self.low_storey and self.high_storey:
                self.set_building_storey()
        self.add_relief(points_relief)
        if save_gml:
            building_gml = self.create_citygml_building(self.mesh_building, building_lod)
            save_citygml(building_gml, os.path.join(gml_root, 'building.gml'))

        return self.mesh_building


class genRoad:
    def __init__(self,
                 road_src_path=r'data\src_2d\shp\edges.shp',
                 width=2.,
                 width_sub=0.1,
                 light_ratio=0.1,
                 tele_ratio=1.,
                 **kwargs):
        self.road_src_path = road_src_path
        self.width = width
        self.width_sub = width_sub
        self.light_ratio = light_ratio
        self.tele_ratio = tele_ratio

        self.road_limit = None

    def crop_road_lineStr(self, x_min, y_min, width=200., height=200.):
        self.roi_rect = box(x_min, y_min, x_min + width, y_min + height)
        self.line_shape = gpd.read_file(self.road_src_path)
        self.roi_road = gpd.clip(self.line_shape, self.roi_rect).geometry
        return self.roi_road

    def gen_mesh_road(self, shp, buffer):
        self.mesh_road = []
        self.buffered_line = shp.buffer(buffer)
        self.road_limit = self.buffered_line

        for poly_road in self.buffered_line:
            if isinstance(poly_road, Polygon):
                tmp_mesh = polygon_to_mesh(poly_road)
                self.mesh_road.append(tmp_mesh)
            elif isinstance(poly_road, MultiPolygon):
                for poly_road_tmp in poly_road.geoms:
                    tmp_mesh = polygon_to_mesh(poly_road_tmp)
                    self.mesh_road.append(tmp_mesh)

    def gen_mesh_road_sub(self, shp, width, width_sub):
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
                tmp_mesh = polygon_to_mesh(poly_road)
                self.mesh_road.append(tmp_mesh)
            elif isinstance(poly_road, MultiPolygon):
                for poly_road_tmp in poly_road.geoms:
                    tmp_mesh = polygon_to_mesh(poly_road_tmp)
                    self.mesh_road.append(tmp_mesh)

        for poly_road in right_sub:
            if isinstance(poly_road, Polygon):
                tmp_mesh = polygon_to_mesh(poly_road)
                self.mesh_road.append(tmp_mesh)
            elif isinstance(poly_road, MultiPolygon):
                for poly_road_tmp in poly_road.geoms:
                    tmp_mesh = polygon_to_mesh(poly_road_tmp)
                    self.mesh_road.append(tmp_mesh)

    def generate_poles_along_line(self, line, interval):
        length = line.length
        return [line.interpolate(distance) for distance in range(0, int(length), interval)]

    def gen_device_lod0(self, shp):
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
            tmp_mesh = polygon_to_mesh(tele_pole_square)
            if not tele_pole_point[x].within(self.road_limit).any():
                self.mesh_device.append(tmp_mesh)

    def gen_device_lod1(self, shp):
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
            if not tele_pole_point[x].within(self.road_limit).any():
                self.mesh_device.append(tmp_mesh)

    def gen_device_lod2(self, shp):
        self.mesh_device = []

        tele_pole_mesh = trimesh.load(os.path.join(r'data\src_3d\lod3frn\electric_pole',
                                                   'obj_52385618_frn_6697_op_frn_0ece98a1-6070-4315-88d4-3d4546168814__493155_25.obj'))
        tele_pole_mesh = tele_pole_mesh.dump(concatenate=True) if isinstance(tele_pole_mesh,
                                                                             trimesh.Scene) else tele_pole_mesh
        tele_pole_mesh_xy = tele_pole_mesh.centroid[:2]
        tele_pole_mesh_zmin = np.min(tele_pole_mesh.vertices[:, 2])
        tele_pole_mesh_h = np.max(tele_pole_mesh.vertices[:, 2]) - tele_pole_mesh_zmin

        traf_light_mesh = trimesh.load(os.path.join(r'data\src_3d\lod3frn\traffic_light',
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

        res_tele_pole = []
        for x in range(len(tele_pole_point)):
            tmp_tele_pole_mesh = tele_pole_mesh.copy()
            tele_pole_point_xy = [tele_pole_point[x].x, tele_pole_point[x].y]
            trans_tele_mesh = [tele_pole_point_xy[0] - tele_pole_mesh_xy[0],
                               tele_pole_point_xy[1] - tele_pole_mesh_xy[1],
                               -tele_pole_mesh_zmin]
            if not tele_pole_point[x].within(self.road_limit).any():
                res_tele_pole.append(tmp_tele_pole_mesh.apply_translation(trans_tele_mesh))

        res_traf_light = []
        for x in random.sample(list(range(len(res_tele_pole))),
                               int(len(res_tele_pole) * self.light_ratio / (self.light_ratio + self.tele_ratio))):
            tmp_traf_light_mesh = traf_light_mesh.copy()
            tele_pole_point_xy = res_tele_pole[x].centroid[:2]
            trans_traf_mesh = [tele_pole_point_xy[0] - traf_light_mesh_xy[0],
                               tele_pole_point_xy[1] - traf_light_mesh_xy[1],
                               tele_pole_mesh_h - traf_light_mesh_zmin]
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

    def gen_road_run(self, road_lod=1, device_lod=2, points_relief=None, save_gml=True, gml_root=''):
        if road_lod == 0 or road_lod == 1:
            self.gen_mesh_road(self.roi_road, self.width)
        elif road_lod == 2:
            self.gen_mesh_road(self.roi_road, self.width)
            self.gen_mesh_road_sub(self.roi_road, self.width, self.width_sub)
        if device_lod == 0:
            self.gen_device_lod0(self.roi_road)
        elif device_lod == 1:
            self.gen_device_lod1(self.roi_road)
        elif device_lod == 2:
            self.gen_device_lod2(self.roi_road)
        road_ori = self.mesh_road.copy()

        feat_color_road = (253, 253, 230, 30)
        feat_color_device = (240, 128, 128, 255)
        self.mesh_road = obj_color(self.mesh_road,feat_color_road)
        self.mesh_device = obj_color(self.mesh_device,feat_color_device)

        self.mesh_road += self.mesh_device
        self.add_relief(points_relief)
        if save_gml:
            road_gml = self.create_citygml_road(road_ori)
            save_citygml(road_gml, os.path.join(gml_root, 'road.gml'))
            device_gml = self.create_citygml_cityfurniture(self.mesh_device)
            save_citygml(device_gml, os.path.join(gml_root, 'device.gml'))

        return self.mesh_road


class genVegetation:
    def __init__(self,
                 vege_root=r'data\src_3d\lod3veg\SolitaryVegetationObject\\',
                 vege_label=r'data\src_3d\tree_label.csv',
                 low_ratio=0.1,
                 high_ratio=1.,
                 **kwargs):
        self.vege_mes = pd.read_csv(vege_label)
        self.vege_id = self.vege_mes['id'].values
        self.vege_type = self.vege_mes['type'].values
        self.vege_root = vege_root
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio

    def gen_tree_mesh_lod0(self, limit_road, limit_bdg, x_min, y_min, width=200., height=200., dense=200):
        self.mesh_tree = []
        self.roi_rect = box(x_min, y_min, x_min + width, y_min + height)

        limit_bdg = limit_bdg.buffer(3.)
        limit_road = limit_road.buffer(3.)

        tar_xy = np.array([[random.uniform(x_min, x_min + width) for _ in range(dense)],
                           [random.uniform(y_min, y_min + width) for _ in range(dense)]]).T
        tmp_idx = []
        for i in range(len(tar_xy)):
            if Point(tar_xy[i]).within(limit_road).any() or Point(tar_xy[i]).within(limit_bdg).any():
                continue
            else:
                tmp_idx.append(i)
        tar_xy = tar_xy[tmp_idx]

        for i in range(len(tar_xy)):
            tree_poly = Point(tar_xy[i]).buffer(random.uniform(1., 3.))
            tree_height = random.uniform(6., 12.)

            tmp_mesh = polygon_to_mesh(tree_poly)
            self.mesh_tree.append(tmp_mesh)

    def gen_tree_mesh_lod1(self, limit_road, limit_bdg, x_min, y_min, width=200., height=200., dense=200):
        self.mesh_tree = []
        self.roi_rect = box(x_min, y_min, x_min + width, y_min + height)

        limit_bdg = limit_bdg.buffer(3.)
        limit_road = limit_road.buffer(3.)

        tar_xy = np.array([[random.uniform(x_min, x_min + width) for _ in range(dense)],
                           [random.uniform(y_min, y_min + width) for _ in range(dense)]]).T
        tmp_idx = []
        for i in range(len(tar_xy)):
            if Point(tar_xy[i]).within(limit_road).any() or Point(tar_xy[i]).within(limit_bdg).any():
                continue
            else:
                tmp_idx.append(i)
        tar_xy = tar_xy[tmp_idx]

        high_num = int(len(tar_xy) * self.high_ratio / (self.high_ratio + self.low_ratio))

        for i in range(high_num):
            tree_poly = Point(tar_xy[i]).buffer(random.uniform(1., 3.))
            tree_height = random.uniform(6., 12.)

            vertices, faces = polygon_to_mesh_3D(tree_poly, tree_height)
            tmp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            self.mesh_tree.append(tmp_mesh)

        for i in range(high_num, len(tar_xy)):
            tree_poly = Point(tar_xy[i]).buffer(random.uniform(0.5, 2.))
            tree_height = random.uniform(2., 6.)

            vertices, faces = polygon_to_mesh_3D(tree_poly, tree_height)
            tmp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            self.mesh_tree.append(tmp_mesh)

    def gen_tree_mesh_lod2(self, limit_road, limit_bdg, x_min, y_min, width=200., height=200., dense=200):
        self.mesh_tree = []
        self.roi_rect = box(x_min, y_min, x_min + width, y_min + height)

        limit_bdg = limit_bdg.buffer(3.)
        limit_road = limit_road.buffer(3.)

        tar_xy = np.array([[random.uniform(x_min, x_min + width) for _ in range(dense)],
                           [random.uniform(y_min, y_min + width) for _ in range(dense)]]).T
        tmp_idx = []
        for i in range(len(tar_xy)):
            if Point(tar_xy[i]).within(limit_road).any() or Point(tar_xy[i]).within(limit_bdg).any():
                continue
            else:
                tmp_idx.append(i)
        tar_xy = tar_xy[tmp_idx]

        high_num = int(len(tar_xy) * self.high_ratio / (self.high_ratio + self.low_ratio))
        low_num = int(len(tar_xy) * self.low_ratio / (self.high_ratio + self.low_ratio))
        high_idx = self.vege_id[self.vege_type == 1]
        low_idx = self.vege_id[self.vege_type == 0]
        high_idx_ = random.choices(list(range(len(high_idx))), k=high_num)
        low_idx_ = random.choices(list(range(len(low_idx))), k=low_num)

        for x, i in enumerate(high_idx_):
            tmp_mesh = trimesh.load(os.path.join(self.vege_root, high_idx[i] + '.obj'))
            tmp_mesh = tmp_mesh.dump(concatenate=True) if isinstance(tmp_mesh, trimesh.Scene) else tmp_mesh
            tmp_mesh_xy = tmp_mesh.centroid[:2]
            tmp_mesh_zmin = np.min(tmp_mesh.vertices[:, 2])

            tmp_trans = [tar_xy[x, 0] - tmp_mesh_xy[0], tar_xy[x, 1] - tmp_mesh_xy[1], -tmp_mesh_zmin]
            self.mesh_tree.append(tmp_mesh.apply_translation(tmp_trans))
        for x, i in enumerate(low_idx_):
            tmp_mesh = trimesh.load(os.path.join(self.vege_root, low_idx[i] + '.obj'))
            tmp_mesh = tmp_mesh.dump(concatenate=True) if isinstance(tmp_mesh, trimesh.Scene) else tmp_mesh
            tmp_mesh_xy = tmp_mesh.centroid[:2]
            tmp_mesh_zmin = np.min(tmp_mesh.vertices[:, 2])

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

            lod2Geometry = etree.SubElement(plantCover, "{http://www.opengis.net/citygml/vegetation/2.0}lod2Geometry")
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

    def gen_vege_run(self, limit_road, limit_bdg, x_min, y_min, width=200., height=200., points_relief=None, dense=None,
                     lod_level=2,
                     save_gml=True, gml_root=''):
        if not dense:
            dense = random.randint(50, 200)
        if lod_level == 0:
            self.gen_tree_mesh_lod0(limit_road, limit_bdg, x_min, y_min, width, height, dense)
        elif lod_level == 1:
            self.gen_tree_mesh_lod1(limit_road, limit_bdg, x_min, y_min, width, height, dense)
        elif lod_level == 2:
            self.gen_tree_mesh_lod2(limit_road, limit_bdg, x_min, y_min, width, height, dense)
        self.add_relief(points_relief)
        if save_gml:
            vege_gml = self.create_citygml_vegetation(self.mesh_tree)
            save_citygml(vege_gml, os.path.join(gml_root, 'vegetation.gml'))

        return self.mesh_tree


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


def polygon_to_mesh(polygon):
    exterior_coords = np.array(polygon.exterior.coords)
    vertices = np.hstack((exterior_coords, np.zeros((exterior_coords.shape[0], 1))))

    triangles = earcut(exterior_coords.flatten(), dim=2)
    faces = np.reshape(triangles, (-1, 3))

    res_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return res_mesh


def polygon_to_mesh_3D(polygon, height=3.):
    exterior_coords = np.array(polygon.exterior.coords)
    vertices_btm = np.hstack((exterior_coords, np.zeros((exterior_coords.shape[0], 1))))

    triangles = earcut(exterior_coords.flatten(), dim=2)
    faces_btm = np.reshape(triangles, (-1, 3))
    faces_btm = np.hstack([faces_btm,faces_btm[:,0][:,None]])

    l_btm = len(vertices_btm)
    vertices_top = vertices_btm + [0., 0., height]
    faces_top = faces_btm + [l_btm]*4

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


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("random_seed", help="random seed", type=int, default=1024)

    parser.add_argument("--lod_building", help="building lod", type=int)
    parser.add_argument("--storey_low", help="low limit of building storey", type=int)
    parser.add_argument("--storey_high", help="high limit of building storey", type=int)
    parser.add_argument("--prob_t1", help="probability of type 1, flat", type=float)
    parser.add_argument("--prob_t2", help="probability of type 2, flat with different storey", type=float)
    parser.add_argument("--prob_t3", help="probability of type 3, mixed", type=float)
    parser.add_argument("--prob_t4", help="probability of type 4, abnormal", type=float)
    parser.add_argument("--prob_t5", help="probability of type 5, slope1", type=float)
    parser.add_argument("--prob_t6", help="probability of type 6, slope2", type=float)
    parser.add_argument("--prob_t7", help="probability of type 7, pure flat", type=float)

    parser.add_argument("--lod_road", help="road lod", type=int)
    parser.add_argument("--road_width_low", help="low limit, width of road", type=float)
    parser.add_argument("--road_width_high", help="high limit, width of road", type=float)
    parser.add_argument("--road_width_main", help="width ratio of main road", type=float)
    parser.add_argument("--road_width_sub", help="width ratio of sub road", type=float)

    parser.add_argument("--lod_vegetation", help="vegetation lod", type=int)
    parser.add_argument("--low_tree_ratio", help="low tree ratio", type=float)
    parser.add_argument("--high_tree_ratio", help="high tree ratio", type=float)

    parser.add_argument("--lod_device", help="device lod", type=int)
    parser.add_argument("--telegraph_pole_ratio", help="telegraph pole ratio", type=float)
    parser.add_argument("--traffic_light_ratio", help="traffic light ratio", type=float)

    parser.add_argument("--lod_relief", help="relief lod", type=int)

    parser.add_argument("--output", help="Output file name", type=str)

    args = parser.parse_args()
    return args


def main():
    args = arg()
    random_seed = args.random_seed

    lod_building = args.lod_building
    storey_low = args.storey_low
    storey_high = args.storey_high
    prob_t1 = args.prob_t1
    prob_t2 = args.prob_t2
    prob_t3 = args.prob_t3
    prob_t4 = args.prob_t4
    prob_t5 = args.prob_t5
    prob_t6 = args.prob_t6
    prob_t7 = args.prob_t7

    lod_road = args.lod_road
    road_width_low = args.road_width_low
    road_width_high = args.road_width_high
    road_width_main = args.road_width_main
    road_width_sub = args.road_width_sub

    lod_vegetation = args.lod_vegetation
    low_tree_ratio = args.low_tree_ratio
    high_tree_ratio = args.high_tree_ratio

    lod_device = args.lod_device
    telegraph_pole_ratio = args.telegraph_pole_ratio
    traffic_light_ratio = args.traffic_light_ratio

    lod_relief = args.lod_relief

    output_root = args.output

    random.seed(random_seed)

    probabilities = [prob_t1, prob_t2, prob_t3, prob_t4, prob_t5, prob_t6, prob_t7]
    if road_width_low and road_width_high:
        road_width = random.uniform(road_width_low, road_width_high)
    else:
        road_width = 2.
    width_sub = road_width_sub / road_width_main

    gen_relief = genRelief()
    gen_road = genRoad(width=road_width, width_sub=width_sub, tele_ratio=telegraph_pole_ratio,
                       light_ratio=traffic_light_ratio)
    gen_building = genBuilding(probabilities=probabilities, low_storey=storey_low, high_storey=storey_high)
    gen_vege = genVegetation(low_ratio=low_tree_ratio, high_ratio=high_tree_ratio)

    x_rand = random.uniform(10500., 14250.)
    y_rand = random.uniform(-13000., -20750.)

    gen_relief.gen_relief_run(x_rand, y_rand, relief_lod=lod_relief, gml_root=output_root)
    mesh_relief = gen_relief.mesh_relief
    points_relief = gen_relief.points_relief

    gen_road.crop_road_lineStr(x_rand, y_rand)
    mesh_road = gen_road.gen_road_run(road_lod=lod_road, device_lod=lod_device, points_relief=points_relief,
                                      gml_root=output_root)
    road_limit = gen_road.road_limit

    gen_building.crop_blg_poly(x_rand, y_rand)
    mesh_building = gen_building.gen_building_run(building_lod=lod_building, limit=road_limit,
                                                  points_relief=points_relief, gml_root=output_root)
    bdg_limit = gen_building.building_limit

    mesh_vege = gen_vege.gen_vege_run(road_limit, bdg_limit, x_rand, y_rand, points_relief=points_relief,
                                      lod_level=lod_vegetation, gml_root=output_root)

    feat_color_bldg = (157, 195, 230, 255)
    feat_color_vege = (137, 179, 95, 255)
    feat_color_relief = (53, 53, 53, 255)

    mesh_building = obj_color(mesh_building, feat_color_bldg)
    mesh_vege = obj_color(mesh_vege, feat_color_vege)
    mesh_relief = obj_color([mesh_relief], feat_color_relief)

    for x in range(len(mesh_road)):
        mesh_road[x].vertices[:,2]+=1.

    res = mesh_relief + mesh_building + mesh_road + mesh_vege
    combined_mesh = trimesh.util.concatenate(res)

    if not os.path.exists(output_root):
        os.makedirs(output_root)
    combined_mesh.export(os.path.join(output_root, 'gen_test.obj'))


if __name__ == "__main__":
    import time
    s=time.time()
    main()
    e=time.time()
    print(e-s)
