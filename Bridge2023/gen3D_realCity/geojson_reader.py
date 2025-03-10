import os
import numpy as np
import re

import geojson
from osgeo import ogr, osr, gdal
import cv2

polygon_tag = ['Polygon', 'MultiPolygon']
output_tiff_root = "/Users/konialive/Downloads/Academic_file/master/proj_PLATEAU_bridge/gml_conv/GEO_TIFF"


def read_geojson_and_rasterize(geojson_path):
    try:
        with open(geojson_path, 'r') as geojson_file:
            geojson_data = geojson.load(geojson_file)
            print('file loaded')

            if 'features' not in geojson_data:
                print("Error: No valid features. ")
                return []

            crs_epsg = ''
            if 'crs' in geojson_data:
                crs_properties = geojson_data['crs']['properties']
                if 'name' in crs_properties:
                    crs_name = crs_properties['name']
                    crs_epsg = re.search(r'EPSG::(\d+)', crs_name).group(1)

            polygons = []
            names = []

            print('reading features')
            for feature in geojson_data['features']:
                if 'geometry' in feature and feature['geometry']['type'] in polygon_tag:
                    coordinates = feature['geometry']['coordinates']
                    polygon = [list(point) for point in coordinates[0]]
                    polygons.append(polygon)

                if 'map_id' in feature['properties'] and 'hyosatu_id' in feature['properties']:
                    polygon_name = str(feature['properties']['map_id']) + '_' + feature['properties']['hyosatu_id']
                    names.append(polygon_name)

            assert len(polygons) == len(names)

            shape = [512, 512]
            origin_coords, pixel_sizes, footprint_images = polygon_to_raster(polygons, names, crs_epsg, shape)

            return polygons, names, origin_coords, pixel_sizes, footprint_images

    except FileNotFoundError:
        print(f'Error: Unable to open the GEOJSON file with the path {geojson_path}')

    except Exception as e:
        print(f'Error: Catch exceptions {e}')

    raise Exception('Error occurred. ')


def write_geotiff(image, name, crs_epsg, shape, pixel_size, origin_coords):
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(os.path.join(output_tiff_root, name + '.tiff'), shape[1], shape[0], 1, gdal.GDT_Byte)

    srs = osr.SpatialReference()
    # print(crs_epsg)
    srs.ImportFromEPSG(int(crs_epsg))

    geotransform = (origin_coords[0], pixel_size[0], 0, origin_coords[1], 0, pixel_size[1])
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(srs.ExportToWkt())

    dst_ds.GetRasterBand(1).WriteArray(image)


def polygon_to_raster(polygons, names, crs_epsg, shape):
    '''

    Convert polygons to rasters with geo-infos.

    :param polygons: Shapely polygons.
    :param names: (optional) file names.
    :param crs_epsg: geographic references in EPSG codes.
    :param shape: the returned image shape.
    :return: rasterized footprints, with their names and geo-reference.
    '''

    origin_coord = []
    pixel_size = []
    footprint_images = []

    for polygon, name in zip(polygons, names):
        if len(polygon) == 1:
            polygon = polygon[0]

        polygon_coords = np.array(polygon)

        min_values = np.min(polygon_coords, axis=0)
        max_values = np.max(polygon_coords, axis=0)

        # Compute offset
        span = [max_values[i] - min_values[i] for i in range(len(max_values))]
        max_span_index = span.index(max(span))
        max_span = span[max_span_index]
        pixel_span = [span[i] * shape[max_span_index] / (2 * max_span) for i in range(2)]

        pixel_offset = [((shape[i] - pixel_span[i]) / 2) for i in range(2)]

        # Compute geo-reference
        pixelSize = [(max_values[0] - min_values[0]) / pixel_span[0], (min_values[1] - max_values[1]) / pixel_span[1]]
        originCoords = [min_values[0], max_values[1]]  # [x_min, y_max]

        pixel_polygon = [[int((x - originCoords[0]) / pixelSize[0] + pixel_offset[0]),
                          int((y - originCoords[1]) / pixelSize[1] + pixel_offset[1])] for [x, y] in polygon]
        pixel_coords = np.array(pixel_polygon)
        originCoords = [originCoords[i] - pixel_offset[i] * pixelSize[i] for i in range(2)]

        image = np.zeros((shape[0], shape[1]), dtype=np.uint8)
        cv2.fillPoly(image, [pixel_coords], color=255)

        # draw contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        footprint_image = np.zeros_like(image)
        cv2.drawContours(footprint_image, contours, -1, 255, 1)

        # append geo-info
        origin_coord.append(np.array(originCoords))
        pixel_size.append(np.array(pixelSize))
        footprint_images.append(footprint_image)

    return origin_coord, pixel_size, footprint_images

    # write_geotiff(footprint_image, name, crs_epsg, shape, pixelSize, originCoords)


def main():
    geojson_path = "/Users/konialive/Downloads/Academic_file/master/proj_PLATEAU_bridge/gml_conv/GEO_JSON/test_area_jis.geojson"
    polygons = read_geojson_and_rasterize(geojson_path)

    print(polygons if polygons else "No polygons")

    # if polygons:
    #     print(polygons)
    # else:
    #     print("No polygons")


if __name__ == "__main__":
    main()
