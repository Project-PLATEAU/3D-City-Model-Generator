import numpy as np
import pandas as pd
import random
import trimesh
import os
import geopandas as gpd
from earcut.earcut import earcut
from lxml import etree
from shapely.geometry import box, Point, Polygon, MultiPolygon, LineString, MultiLineString
from gml_proc import set_bldg_texture,write_obj

class genRoad:
    def __init__(self,
                 road_src_path=r'.\data\src_2d\shp\meguro3.shp',
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

    def crop_road_lineStr(self):
        self.line_shape = gpd.read_file(self.road_src_path)
        self.line_shape.to_crs(30169, inplace=True)
        bounds = self.line_shape.bounds
        self.roi_rect = box(min(bounds.minx), min(bounds.miny), max(bounds.maxx), max(bounds.maxy))
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

        tele_pole_mesh = trimesh.load(os.path.join(r'.\data\src_3d\lod3frn\electric_pole',
                                                   'obj_52385618_frn_6697_op_frn_0ece98a1-6070-4315-88d4-3d4546168814__493155_25.obj'))
        tele_pole_mesh = tele_pole_mesh.dump(concatenate=True) if isinstance(tele_pole_mesh,
                                                                             trimesh.Scene) else tele_pole_mesh
        tele_pole_mesh_xy = tele_pole_mesh.centroid[:2]
        tele_pole_mesh_zmin = np.min(tele_pole_mesh.vertices[:, 2])
        tele_pole_mesh_h = np.max(tele_pole_mesh.vertices[:, 2]) - tele_pole_mesh_zmin

        traf_light_mesh = trimesh.load(os.path.join(r'.\data\src_3d\lod3frn\traffic_light',
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
        # tele_pole_point = []
        # for tmp_road in left_sub:
        #     tele_pole_point += self.generate_poles_along_line(tmp_road, 20)

        tele_pole_point = coords_traf.geometry.values

        res_tele_pole = []
        for x in range(len(tele_pole_point)):
            tmp_tele_pole_mesh = tele_pole_mesh.copy()
            tele_pole_point_xy = [tele_pole_point[x].x, tele_pole_point[x].y]
            trans_tele_mesh = [tele_pole_point_xy[0] - tele_pole_mesh_xy[0],
                               tele_pole_point_xy[1] - tele_pole_mesh_xy[1],
                               -tele_pole_mesh_zmin]
            res_tele_pole.append(tmp_tele_pole_mesh.apply_translation(trans_tele_mesh))
            # if not tele_pole_point[x].within(self.road_limit).any():
            #     res_tele_pole.append(tmp_tele_pole_mesh.apply_translation(trans_tele_mesh))

        res_traf_light = []
        # for x in random.sample(list(range(len(res_tele_pole))),
        #                        int(len(res_tele_pole) * self.light_ratio / (self.light_ratio + self.tele_ratio))):
        for x in range(len(res_tele_pole)):
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
                 vege_root=r'.\data\src_3d\lod3veg\SolitaryVegetationObject\\',
                 vege_label=r'.\data\src_3d\tree_label.csv',
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

    def gen_tree_mesh_lod2(self):
        self.mesh_tree = []

        # self.roi_rect = box(x_min, y_min, x_min + width, y_min + height)
        #
        # limit_bdg = limit_bdg.buffer(3.)
        # limit_road = limit_road.buffer(3.)
        #
        # tar_xy = np.array([[random.uniform(x_min, x_min + width) for _ in range(dense)],
        #                    [random.uniform(y_min, y_min + width) for _ in range(dense)]]).T
        # tmp_idx = []
        # for i in range(len(tar_xy)):
        #     if Point(tar_xy[i]).within(limit_road).any() or Point(tar_xy[i]).within(limit_bdg).any():
        #         continue
        #     else:
        #         tmp_idx.append(i)
        # tar_xy = tar_xy[tmp_idx]
        # print(tar_xy)

        tar_xy = np.array([[p.x, p.y] for p in coords_vege.geometry.values])
        vege_id = self.vege_id
        vege_idx_ = random.choices(list(range(len(vege_id))), k=len(tar_xy))

        for x, i in enumerate(vege_idx_):
            tmp_mesh = trimesh.load(os.path.join(self.vege_root, vege_id[i] + '.obj'))
            tmp_mesh = tmp_mesh.dump(concatenate=True) if isinstance(tmp_mesh, trimesh.Scene) else tmp_mesh

            tmp_x_dif = max(tmp_mesh.vertices[:, 0]) - min(tmp_mesh.vertices[:, 0])
            tmp_y_dif = max(tmp_mesh.vertices[:, 1]) - min(tmp_mesh.vertices[:, 1])
            tmp_r = max([tmp_x_dif,tmp_y_dif])
            tmp_h = max(tmp_mesh.vertices[:, 2]) - min(tmp_mesh.vertices[:, 2])
            tar_r = coords_vege.values[x,4]
            tar_h = coords_vege.values[x,3]

            tmp_h_scale=tar_h/tmp_h
            tmp_mesh.apply_scale(tmp_h_scale)

            # mesh_point = tmp_mesh.vertices
            # mesh_point_z = mesh_point[:, 2]
            # min_z = np.min(mesh_point_z)
            #
            # print(tar_h,np.max(mesh_point_z))
            #
            # h_trans = tar_h - np.max(mesh_point_z)
            # mesh_point[:, 2][mesh_point_z > (min_z + 1.)] += h_trans
            # tmp_mesh.vertices = mesh_point

            tmp_mesh_xy = tmp_mesh.centroid[:2]
            tmp_mesh_zmin = np.min(tmp_mesh.vertices[:, 2])

            tmp_trans = [tar_xy[x, 0] - tmp_mesh_xy[0], tar_xy[x, 1] - tmp_mesh_xy[1], -tmp_mesh_zmin]
            self.mesh_tree.append(tmp_mesh.apply_translation(tmp_trans))

        # high_num = int(len(tar_xy) * self.high_ratio / (self.high_ratio + self.low_ratio))
        # low_num = int(len(tar_xy) * self.low_ratio / (self.high_ratio + self.low_ratio))
        # high_idx = self.vege_id[self.vege_type == 1]
        # low_idx = self.vege_id[self.vege_type == 0]
        # high_idx_ = random.choices(list(range(len(high_idx))), k=high_num)
        # low_idx_ = random.choices(list(range(len(low_idx))), k=low_num)
        #
        #
        # for x, i in enumerate(high_idx_):
        #     tmp_mesh = trimesh.load(os.path.join(self.vege_root, high_idx[i] + '.obj'))
        #     tmp_mesh = tmp_mesh.dump(concatenate=True) if isinstance(tmp_mesh, trimesh.Scene) else tmp_mesh
        #     tmp_mesh_xy = tmp_mesh.centroid[:2]
        #     tmp_mesh_zmin = np.min(tmp_mesh.vertices[:, 2])
        #
        #     tmp_trans = [tar_xy[x, 0] - tmp_mesh_xy[0], tar_xy[x, 1] - tmp_mesh_xy[1], -tmp_mesh_zmin]
        #     self.mesh_tree.append(tmp_mesh.apply_translation(tmp_trans))
        # for x, i in enumerate(low_idx_):
        #     tmp_mesh = trimesh.load(os.path.join(self.vege_root, low_idx[i] + '.obj'))
        #     tmp_mesh = tmp_mesh.dump(concatenate=True) if isinstance(tmp_mesh, trimesh.Scene) else tmp_mesh
        #     tmp_mesh_xy = tmp_mesh.centroid[:2]
        #     tmp_mesh_zmin = np.min(tmp_mesh.vertices[:, 2])
        #
        #     tmp_trans = [tar_xy[x + high_num, 0] - tmp_mesh_xy[0], tar_xy[x + high_num, 1] - tmp_mesh_xy[1],
        #                  -tmp_mesh_zmin]
        #     self.mesh_tree.append(tmp_mesh.apply_translation(tmp_trans))

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
            self.gen_tree_mesh_lod2()
        self.add_relief(points_relief)
        if save_gml:
            vege_gml = self.create_citygml_vegetation(self.mesh_tree)
            save_citygml(vege_gml, os.path.join(gml_root, 'vegetation.gml'))

        return self.mesh_tree


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


def save_citygml(root, file_name):
    tree = etree.ElementTree(root)
    tree.write(file_name, pretty_print=True, xml_declaration=True, encoding='UTF-8')


output_root = r'.\res'
road_src_path = r'.\data\src_2d\shp\meguro1.shp'
vege_pos_path = r'.\data\src_svi\feat\route1\vegetation.txt'
traf_pos_path = r'.\data\src_svi\feat\route1\traffic_light.txt'
mesh_info_path = r'.\data\src_svi\texture\route1\bldg_mesh.txt'
mesh_img_path = r'.\data\src_svi\texture\route1\img'
mesh_height = r'.\data\src_svi\route1_elevation.csv'

data_vege = open(vege_pos_path, 'r').readlines()
data_traf = open(traf_pos_path, 'r').readlines()

col_vege = data_vege[0].strip().split()[1:]
coords_vege = np.array([np.array(i.strip().split()).astype(float) for i in data_vege[1:]])
coords_vege = gpd.GeoDataFrame(coords_vege, geometry=[Point(x) for x in coords_vege[:, :2][:, ::-1]], crs=4326)
coords_vege.to_crs(30169, inplace=True)
col_traf = data_traf[0].strip().split()[1:]
coords_traf = np.array([np.array(i.strip().split()).astype(float) for i in data_traf[1:]])
coords_traf = gpd.GeoDataFrame(coords_traf, geometry=[Point(x) for x in coords_traf[:, :2][:, ::-1]], crs=4326)
coords_traf.to_crs(30169, inplace=True)

gen_road = genRoad(road_src_path=road_src_path, width=2., width_sub=0.1, tele_ratio=1., light_ratio=1.)
gen_road.crop_road_lineStr()
mesh_road = gen_road.gen_road_run(road_lod=2, device_lod=2, points_relief=None,
                                  gml_root=output_root)

gen_vege = genVegetation(low_ratio=1., high_ratio=0.1)
mesh_vege = gen_vege.gen_vege_run(None, None, 0, 0, points_relief=None,
                                  lod_level=2, gml_root=output_root)

res = mesh_road + mesh_vege
combined_mesh = trimesh.util.concatenate(res)

if not os.path.exists(output_root):
    os.makedirs(output_root)
# combined_mesh.export(os.path.join(output_root, 'gen_test.obj'))


tmp_texture, tmp_vertices=set_bldg_texture(mesh_info_path,
                 mesh_img_path,
                 mesh_height,
                 output_root)
write_obj(tmp_texture, tmp_vertices,combined_mesh, os.path.join(output_root, 'bldg_test.obj'))
# pd.DataFrame(coords_vege,columns=col_vege).to_csv(r'C:\zcb\data\plateau\data\t\svi\vegetation3.csv',index=False)
# pd.DataFrame(coords_traf,columns=col_traf).to_csv(r'C:\zcb\data\plateau\data\t\svi\traffic_light3.csv',index=False)
