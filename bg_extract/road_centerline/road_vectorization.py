import cv2
import numpy as np
from skimage.morphology import skeletonize

import shapely
from shapely.geometry import Polygon, MultiLineString
from centerline.geometry import Centerline

import geopandas as gpd
import matplotlib.pyplot as plt


def construct_centerline(input_geometry, interpolation_distance=0.5):
    borders = input_geometry.segmentize(interpolation_distance)
    voronoied = shapely.voronoi_polygons(borders, only_edges=True)

    # centerlines = gpd.sjoin(gpd.GeoDataFrame(geometry=gpd.GeoSeries(voronoied.geoms)),
    #                         gpd.GeoDataFrame(geometry=gpd.GeoSeries(input_geometry)), op="within")
    centerlines = gpd.sjoin(gpd.GeoDataFrame(geometry=gpd.GeoSeries(voronoied.geoms)), gpd.GeoDataFrame(geometry=gpd.GeoSeries(input_geometry)), predicate="within")

    return centerlines.unary_union


