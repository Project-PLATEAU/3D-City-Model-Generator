from osgeo import gdal, ogr
import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint, Polygon, MultiPolygon
from .road_vectorization import construct_centerline



def raster_to_vector(img, crs, thres):
    img[img > 0] = 1
    mem_driver = gdal.GetDriverByName('MEM')
    gdal_dataset = mem_driver.Create('', img.shape[1], img.shape[0], 1, gdal.GDT_Byte)
    gdal_dataset.GetRasterBand(1).WriteArray(img)
    srcBand = gdal_dataset.GetRasterBand(1)

    vector_ds = ogr.GetDriverByName('Memory').CreateDataSource('')
    layer = vector_ds.CreateLayer('vector_layer', geom_type=ogr.wkbPolygon)

    gdal.Polygonize(srcBand, srcBand, layer, -1, [], callback=None)
    print('Successfully polygonized. ')

    geometry_wkt = []
    for feature in layer:
        geometry = feature.GetGeometryRef()
        geometry_wkt.append(geometry.ExportToWkt())

    geometry_wkt = {'Geometry_wkt': geometry_wkt}
    df = pd.DataFrame(geometry_wkt)
    df['geometry'] = df['Geometry_wkt'].apply(wkt.loads)

    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs)

    gdf = gdf[gdf['geometry'].area >= thres]
    return gdf


def flip_linestring_y(line, resolution, x_min, y_min):
    return LineString([(x_min + x * resolution[0], y_min + y * resolution[1]) for x, y in line.coords])


def road_line_ext(img, resolution, x_min, y_min, preset_crs='epsg:30169', preset_thres=1000):
    shapely_gdf = raster_to_vector(img, preset_crs, preset_thres)
    centerlines = []
    for polygon in shapely_gdf['geometry']:
        simplified_polygon = polygon.simplify(5)
        centerline = construct_centerline(simplified_polygon, interpolation_distance=2)

        centerline = MultiLineString(
            [flip_linestring_y(line, resolution, x_min, y_min) for line in centerline.geoms])  # flip over y
        centerlines.append(centerline)

    return centerlines
