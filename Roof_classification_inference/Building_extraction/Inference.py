import os
import sys
import glob
import multiprocessing as mp
import cv2
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
import torch
from rasterio.transform import Affine
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from tqdm.auto import tqdm
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config

# Add the Building_extraction directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from maskdino import add_maskdino_config
from Regularization.main_regularization import regularize_geodataframe



# Global variable for configuration file path
CONFIG_PATH = "./Roof_classification_inference/Building_extraction/model/config.yaml"

class Predictor:
    """
    The Predictor class encapsulates the functionality for running predictions 
    on input images using a pre-configured model. It initializes the necessary 
    metadata and a predictor instance for inference.

    Attributes:
        metadata: Metadata of the dataset used for inference.
        cpu_device: Torch device set to CPU for processing.
        predictor: An instance of DefaultPredictor for running inference.
    """
    def __init__(self, cfg):
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.cpu_device = torch.device("cpu")
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Runs inference on a single image and returns the predictions.

        Args:
            image: Input image as a NumPy array.

        Returns:
            Predictions for the input image.
        """
        return self.predictor(image)

def setup_cfg():
    """
    Sets up the configuration for the model based on the global CONFIG_PATH.

    Returns:
        Configured Detectron2 configuration object.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(CONFIG_PATH)
    cfg.MODEL.WEIGHTS = './Roof_classification_inference/Building_extraction/model/model_best.pth'
    cfg.TEST.DETECTIONS_PER_IMAGE = 200
    cfg.freeze()
    return cfg

def inference(img_path):
    """
    Runs inference on a single image using the provided configuration.

    Args:
        img_path: Path to the input image file.

    Returns:
        Predictions for the input image.
    """
    cfg = setup_cfg()
    predictor = Predictor(cfg)
    img = read_image(img_path, format="BGR")
    return predictor.run_on_image(img)

def merge_overlapping_footprints(gdf, iou_threshold=0.2):
    """
    Merge smaller building footprints with larger overlapping footprints when IoU > iou_threshold.

    Args:
        gdf: GeoDataFrame containing building footprints.
        iou_threshold: Intersection over Union (IoU) threshold for merging.

    Returns:
        GeoDataFrame with merged footprints.
    """
    geometries = list(gdf['geometry'])
    merged = []
    visited = set()

    for i, geom1 in enumerate(geometries):
        if i in visited:
            continue

        # 修复无效几何体
        geom1 = make_valid(geom1)
        if geom1.is_empty:
            continue

        union_geom = geom1
        for j, geom2 in enumerate(geometries):
            if i != j and j not in visited:
                geom2 = make_valid(geom2)  # 修复无效几何体
                if geom2.is_empty:
                    continue

                if geom1.intersects(geom2):
                    intersection = geom1.intersection(geom2).area
                    union = geom1.union(geom2).area
                    iou = intersection / union
                    if iou > iou_threshold:
                        union_geom = union_geom.union(geom2)
                        visited.add(j)
        merged.append(union_geom)
        visited.add(i)

    return gpd.GeoDataFrame({'geometry': merged}, crs=gdf.crs)

def save_predictions(predictions, img_path, transform, crs):
    """
    Saves the predictions as a GeoJSON file with geographic coordinates.

    Args:
        predictions: Prediction results from the model.
        img_path: Path to the input image.
        output_dir: Directory where the output GeoJSON file will be saved.
        transform: Affine transformation for pixel-to-geo coordinate conversion.
        crs: Coordinate reference system of the input image.

    Returns:
        GeoDataFrame containing the prediction results.
    """
    # os.makedirs(output_dir, exist_ok=True)
    pred_scores = predictions['instances'].scores
    pred_masks = predictions['instances'].pred_masks
    pred_labels = predictions['instances'].pred_classes

    polygons, scores, classes = [], [], []
    for mask, score, label in zip(pred_masks, pred_scores, pred_labels):
        contours, _ = cv2.findContours(mask.cpu().numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) > 2:
                polygon = Polygon(contour.squeeze())
                polygons.append(polygon)
                scores.append(score.item())
                classes.append(label.item())

    gdf = gpd.GeoDataFrame({'score': scores, 'class': classes, 'geometry': polygons})


    def process_geometry(geom):
        if geom.is_empty or not geom.is_valid:
            return None
        if isinstance(geom, Polygon):
            return Polygon(pixel_to_geo_coords(list(geom.exterior.coords), transform))
        elif isinstance(geom, MultiPolygon):
            return MultiPolygon([Polygon(pixel_to_geo_coords(list(poly.exterior.coords), transform))
                                 for poly in geom.geoms if poly.is_valid])
        else:
            return None

    # Convert pixel coordinates to geographic coordinates
    gdf['geometry'] = gdf['geometry'].apply(process_geometry)
    gdf = gdf.dropna(subset=['geometry'])
    gdf = gdf.set_crs(crs, allow_override=True).to_crs(epsg=3857)
    gdf['area_sqm'] = gdf['geometry'].area
    gdf = gdf[gdf['area_sqm'] >= 7]
    gdf = gdf[gdf['score'] >= 0.25]

    # Merge overlapping footprints
    gdf = merge_overlapping_footprints(gdf)

    # Apply boundary regularization
    gdf = regularize_geodataframe(gdf)

    result_name = os.path.basename(img_path).split('.')[0]
    save_path = f'{result_name}.geojson'
    gdf.to_file(save_path, driver='GeoJSON')

    return gdf

def pixel_to_geo_coords(pixel_coords, transform):
    """
    Converts pixel coordinates to geographic coordinates using an affine transform.

    Args:
        pixel_coords: List of pixel coordinates as (x, y) tuples.
        transform: Affine transformation object from rasterio.

    Returns:
        List of geographic coordinates as (longitude, latitude) tuples.
    """
    return [transform * (x, y) for x, y in pixel_coords]

def convert_to_geocoords(geojson_dir, tif_dir, output_dir):
    """
    Converts building footprint coordinates from pixel space to geographic coordinates.

    Args:
        geojson_dir: Directory containing GeoJSON files with pixel coordinates.
        tif_dir: Directory containing corresponding TIF files.
        output_dir: Directory to save GeoJSON files with geographic coordinates.
    """
    os.makedirs(output_dir, exist_ok=True)
    geojson_files = [f for f in os.listdir(geojson_dir) if f.endswith('.geojson')]

    for geojson_file in tqdm(geojson_files):
        geojson_path = os.path.join(geojson_dir, geojson_file)
        gdf = gpd.read_file(geojson_path)

        tif_path = os.path.join(tif_dir, f"{os.path.splitext(geojson_file)[0]}.tif")
        if not os.path.exists(tif_path):
            print(f"Missing TIF for {geojson_file}")
            continue

        with rasterio.open(tif_path) as src:
            transform = src.transform
            crs = src.crs

        def process_geometry(geom):
            if geom.is_empty or not geom.is_valid:
                geom = make_valid(geom)  # 修复无效几何体
            if geom.is_empty:
                return None
            if isinstance(geom, Polygon):
                return Polygon(pixel_to_geo_coords(list(geom.exterior.coords), transform))
            elif isinstance(geom, MultiPolygon):
                return MultiPolygon([Polygon(pixel_to_geo_coords(list(poly.exterior.coords), transform))
                                    for poly in geom.geoms if poly.is_valid])
            else:
                return None
            
        # Apply processing and drop invalid geometries
        gdf['geometry'] = gdf['geometry'].apply(process_geometry)
        gdf = gdf.dropna(subset=['geometry'])       
        gdf = gdf.set_crs(crs, allow_override=True).to_crs(epsg=3857)
        gdf['area_sqm'] = gdf['geometry'].area
        gdf = gdf[gdf['area_sqm'] >= 7]
        gdf = gdf[gdf['score'] >= 0.25]

        save_path = os.path.join(output_dir, f"{os.path.splitext(geojson_file)[0]}.geojson")
        gdf.to_file(save_path, driver='GeoJSON')

def main(img_dir):
    """
    Main pipeline for extracting building footprints and converting them to geographic coordinates.

    Args:
        img_dir: Directory containing input images.
    """
    predictions = inference(img_dir)

    with rasterio.open(img_dir) as src:
        transform = src.transform
        crs = src.crs

    save_predictions(predictions, img_dir, transform, crs)

    # img_list = glob.glob(os.path.join(img_dir, '*.tif'))
    # for img_path in tqdm(img_list):
    #     predictions = inference(img_path)
    #
    #     with rasterio.open(img_path) as src:
    #         transform = src.transform
    #         crs = src.crs
    #
    #     save_predictions(predictions, img_path, output_dir, transform, crs)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract building footprints and convert to geocoordinates.")
    parser.add_argument("img_dir", type=str, help="Directory containing input images.")

    args = parser.parse_args()
    main(args.img_dir)
