import os
import argparse
from Building_extraction.Inference import main as extraction
from Building_extraction.Crop_image import crop_tif_with_geojson
from Roof_classification.Roof_type_inference import main as classification

def process_pipeline(img_dir):
    """
    Complete pipeline for extracting buildings, cropping images, and classifying roof types.

    Args:
        img_dir (str): Path to the input directory containing remote sensing images.

    Outputs:
        Updates the GeoJSON files with building footprints and roof classifications.
    """
    print("Step 1: Extracting buildings and generating GeoJSON...")
    extraction(img_dir)

    print("Step 2: Cropping building images based on GeoJSON...")
    crop_tif_with_geojson(img_dir)

    print("Step 3: Classifying roof types and updating GeoJSON...")
    classification(img_dir)

    print("Pipeline completed successfully. Check the output in the specified directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline for building extraction and roof classification from satellite images.")
    parser.add_argument("img_dir", type=str, help="Path to the input directory containing remote sensing images.")
    args = parser.parse_args()

    process_pipeline(args.img_dir)
