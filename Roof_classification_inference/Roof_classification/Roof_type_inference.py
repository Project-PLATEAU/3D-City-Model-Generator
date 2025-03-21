import os
import json
from argparse import ArgumentParser
from mmengine.fileio import dump
from mmpretrain.apis import ImageClassificationInferencer
import geopandas as gpd
from tqdm.auto import tqdm
import glob

def classify_and_update_geojson(img_dir, model_path, checkpoint_path):
    # Initialize the inferencer
    inferencer = ImageClassificationInferencer(model_path, pretrained=checkpoint_path, device='cuda')

    # Set up paths
    # crop_image_dir = os.path.join(img_dir, 'cropped_image')
    prediction_result_dir = img_dir.split('/')[-1].split('.')[0]

    crop_image_dir = 'cropped_image'

    subfolder_path = crop_image_dir
    geojson_path = f"{prediction_result_dir}.geojson"
    gdf = gpd.read_file(geojson_path)

    # Add a sequential ID column for matching if it doesn't exist
    if 'id' not in gdf.columns:
        gdf['id'] = range(len(gdf))  # Sequential IDs from 0 to n-1

    if 'roof_type' not in gdf.columns:
        gdf['roof_type'] = None

    image_paths = glob.glob('{}/*.tif'.format(os.path.join(subfolder_path, prediction_result_dir)))
    results = inferencer(image_paths)

    for img_path, result in zip(image_paths, results):
        img_id = int(os.path.splitext(os.path.basename(img_path))[0])  # Extract ID from filename
        roof_type = result['pred_label']  # Assuming the model returns a 'pred_label'
        # Update the geojson with the classification result
        gdf.loc[gdf['id'] == img_id, 'roof_type'] = roof_type + 1

    gdf = gdf[(gdf.geometry.area >= 20.) & (gdf.geom_type != 'GeometryCollection')]
    # Save the updated geojson
    gdf.to_file(geojson_path, driver='GeoJSON')

    os.system('rm -rf ' + crop_image_dir)

    return gdf


    # # Iterate through each subfolder in crop_image_dir
    # for subfolder in os.listdir(crop_image_dir):
    #     subfolder_path = os.path.join(crop_image_dir, subfolder)
    #     if not os.path.isdir(subfolder_path):
    #         continue
    #
    #     # Find the corresponding geojson file
    #     geojson_path = os.path.join(prediction_result_dir, f"{subfolder}.geojson")
    #     if not os.path.exists(geojson_path):
    #         # print(f"GeoJSON file not found for {subfolder}, skipping.")
    #         continue
    #
    #     # Load the geojson file
    #     gdf = gpd.read_file(geojson_path)
    #
    #     # Add a sequential ID column for matching if it doesn't exist
    #     if 'id' not in gdf.columns:
    #         gdf['id'] = range(len(gdf))  # Sequential IDs from 0 to n-1
    #
    #     # Initialize the roof_type column if not present
    #     if 'roof_type' not in gdf.columns:
    #         gdf['roof_type'] = None
    #
    #     # Iterate through each cropped image in the subfolder
    #     image_paths = [os.path.join(subfolder_path, img_file) for img_file in os.listdir(subfolder_path) if img_file.endswith(('.jpg', '.png', '.jpeg', 'tif'))]
    #     results = inferencer(image_paths)
    #
    #     # Update the geojson with the classification results
    #     for img_path, result in zip(image_paths, results):
    #         img_id = int(os.path.splitext(os.path.basename(img_path))[0])  # Extract ID from filename
    #         roof_type = result['pred_label']  # Assuming the model returns a 'pred_label'
    #         # Update the geojson with the classification result
    #         gdf.loc[gdf['id'] == img_id, 'roof_type'] = roof_type
    #
    #     # Save the updated geojson
    #     gdf.to_file(geojson_path, driver='GeoJSON')
    #     # print(f"Updated GeoJSON saved: {geojson_path}")

def main(img_dir):

    # Fixed paths for model and checkpoint
    model_path = './Roof_classification_inference/Roof_classification/model/config.py'
    checkpoint_path = './Roof_classification_inference/Roof_classification/model/best_model.pth'

    classify_and_update_geojson(img_dir, model_path, checkpoint_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Path to the image folder containing crop_image and prediction_Result folders.')
    args = parser.parse_args()

    main(args.img_dir)
