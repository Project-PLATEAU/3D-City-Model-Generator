import os
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import box
from shapely.geometry import shape
from rasterio.features import geometry_mask
import numpy as np
from tqdm.auto import tqdm


def crop_tif_with_geojson(image_dir):
    """
    Crop TIFF images using corresponding GeoJSON files.

    Args:
        image_dir (str): Path to the directory containing TIFF images. GeoJSON files
                         should be in a `prediction_Result` subdirectory.
    """
    # Define paths
    # geojson_dir = os.path.join(image_dir, 'prediction_Result')
    crop_image_dir = 'cropped_image'
    
    # Create output directory
    os.makedirs(crop_image_dir, exist_ok=True)

    # # Get list of GeoJSON files
    # geojson_list = [f for f in os.listdir(geojson_dir) if f.endswith('.geojson')]
    # geojson_list.sort()

    geojson_path = image_dir.split('/')[-1].split('.')[0] + '.geojson'

    geojson_name = image_dir.split('/')[-1].split('.')[0]
    tif_path = image_dir
    save_dir = os.path.join(crop_image_dir, geojson_name)
    os.makedirs(save_dir, exist_ok=True)


    # Read TIFF image
    with rasterio.open(tif_path) as src:
        tif_crs = src.crs

    # Read GeoJSON file
    gdf = gpd.read_file(geojson_path)
    # Remove invalid geometries
    gdf = gdf[gdf['geometry'].notnull() & gdf.is_valid]
    if gdf.crs != tif_crs:
        gdf = gdf.to_crs(tif_crs)

    # Process each building
    with rasterio.open(tif_path) as src:
        for idx, row in tqdm(gdf.iterrows(), total=gdf.shape[0], desc="Processing Buildings", leave=False):
            # Get building geometry with a small buffer
            building_geom = row['geometry'].buffer(0.00001)
            building_id = idx  # Use index as building ID

            # Get bounding box and extend slightly
            minx, miny, maxx, maxy = building_geom.bounds
            bbox = box(minx, miny, maxx, maxy)

            # Crop image
            out_image, out_transform = rasterio.mask.mask(src, [bbox], crop=True, filled=True, nodata=255)
            mask = geometry_mask([building_geom], transform=out_transform, invert=True, out_shape=out_image.shape[1:])

            for i in range(out_image.shape[0]):
                out_image[i, ~mask] = 255  # Set outside mask to white

            # Save cropped image
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            save_path = os.path.join(save_dir, f'{building_id}.tif')
            with rasterio.open(save_path, 'w', **out_meta) as dest:
                dest.write(out_image)

    # for geojson_file in tqdm(geojson_list, desc="Processing GeoJSON Files"):
    #     geojson_path = os.path.join(geojson_dir, geojson_file)
    #     geojson_name = os.path.splitext(geojson_file)[0]
    #     tif_path = os.path.join(image_dir, f'{geojson_name}.tif')
    #     save_dir = os.path.join(crop_image_dir, geojson_name)
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #     if not os.path.exists(tif_path):
    #         print(f"TIFF file not found for {geojson_name}, skipping.")
    #         continue
    #
    #     # Read TIFF image
    #     with rasterio.open(tif_path) as src:
    #         tif_crs = src.crs
    #
    #     # Read GeoJSON file
    #     gdf = gpd.read_file(geojson_path)
    #     # Remove invalid geometries
    #     gdf = gdf[gdf['geometry'].notnull() & gdf.is_valid]
    #     if gdf.crs != tif_crs:
    #         gdf = gdf.to_crs(tif_crs)
    #
    #     # Process each building
    #     with rasterio.open(tif_path) as src:
    #         for idx, row in tqdm(gdf.iterrows(), total=gdf.shape[0], desc="Processing Buildings", leave=False):
    #             # Get building geometry with a small buffer
    #             building_geom = row['geometry'].buffer(0.00001)
    #             building_id = idx  # Use index as building ID
    #
    #             # Get bounding box and extend slightly
    #             minx, miny, maxx, maxy = building_geom.bounds
    #             bbox = box(minx, miny, maxx, maxy)
    #
    #             # Crop image
    #             out_image, out_transform = rasterio.mask.mask(src, [bbox], crop=True, filled=True, nodata=255)
    #             mask = geometry_mask([building_geom], transform=out_transform, invert=True, out_shape=out_image.shape[1:])
    #
    #             for i in range(out_image.shape[0]):
    #                 out_image[i, ~mask] = 255  # Set outside mask to white
    #
    #             # Save cropped image
    #             out_meta = src.meta.copy()
    #             out_meta.update({
    #                 "driver": "GTiff",
    #                 "height": out_image.shape[1],
    #                 "width": out_image.shape[2],
    #                 "transform": out_transform
    #             })
    #             save_path = os.path.join(save_dir, f'{building_id}.tif')
    #             with rasterio.open(save_path, 'w', **out_meta) as dest:
    #                 dest.write(out_image)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop TIFF images using corresponding GeoJSON files.")
    parser.add_argument('image_dir', type=str, help="Path to the directory containing TIFF images.")
    args = parser.parse_args()

    crop_tif_with_geojson(args.image_dir)
