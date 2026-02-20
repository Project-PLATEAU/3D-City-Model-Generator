#!/usr/bin/env python3
"""
End-to-end pipeline script for polygon processing:
1. Fix polygon angles to 90 degrees (polygon_rect_angle.py)
2. Divide polygons into rectangles (polygon_division.py)
3. Merge adjacent rectangles (find_adjacent_rectangles.py)
"""

import argparse
import os
import sys
from pathlib import Path

# Import functions from the three scripts
from .polygon_rect_angle import read_angles
from .polygon_division import find_rectangles
from .find_adjacent_rectangles import (
    load_polygons_from_geojson,
    iterative_merge_by_group
)
import json


def run_pipeline(input_geojson: str,
                 output_dir: str = None,
                 angle_tolerance: float = 3.0,
                 lower_threshold: float = 0.5,
                 merge_tolerance: float = 5e-2,
                 group_by: str = 'original_polygon_id',
                 keep_intermediate: bool = False, 
                 return_json: bool = False):
    """
    Run the complete polygon processing pipeline.

    Args:
        input_geojson: Path to input GeoJSON file
        output_dir: Directory for output files (default: same as input)
        angle_tolerance: Tolerance for angle fixing (default: 3.0)
        lower_threshold: Lower threshold for angle fixing (default: 0.5)
        merge_tolerance: Tolerance for merging rectangles (default: 5e-2)
        group_by: Property to group by when merging (default: 'original_polygon_id')
        keep_intermediate: Keep intermediate files (default: True)

    Returns:
        Path to final merged output file
    """
    # Validate input file exists
    if not os.path.exists(input_geojson):
        raise FileNotFoundError(f"Input file not found: {input_geojson}")

    # Setup output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_geojson) or '.'
    os.makedirs(output_dir, exist_ok=True)

    # Get base name for output files
    base_name = os.path.splitext(os.path.basename(input_geojson))[0]

    # print("=" * 80)
    # print("POLYGON PROCESSING PIPELINE")
    # print("=" * 80)
    # print(f"Input file: {input_geojson}")
    # print(f"Output directory: {output_dir}")
    # print()

    # Step 1: Fix polygon angles
    # print("\n" + "=" * 80)
    # print("STEP 1: Fixing polygon angles to 90 degrees")
    # print("=" * 80)
    fixed_path, angle_results = read_angles(
        input_geojson,
        angle_tolerance=angle_tolerance,
        lower_threshold=lower_threshold
    )
    # print(f"\nFixed polygons saved to: {fixed_path}")

    # Step 2: Divide polygons into rectangles
    # print("\n" + "=" * 80)
    # print("STEP 2: Dividing polygons into rectangles")
    # print("=" * 80)
    divided_path = os.path.join(output_dir, f"{base_name}_divided.geojson")
    find_rectangles(fixed_path, divided_path)
    # print(f"\nDivided polygons saved to: {divided_path}")

    # Step 3: Merge adjacent rectangles
    # print("\n" + "=" * 80)
    # print("STEP 3: Merging adjacent rectangles")
    # print("=" * 80)

    # Load polygons from divided file
    # print(f"Loading polygons from {divided_path}...")
    polygons, properties, crs = load_polygons_from_geojson(divided_path)
    # print(f"Loaded {len(polygons)} polygons")

    # Perform iterative merging
    merged_polygons, merged_properties = iterative_merge_by_group(
        polygons,
        properties,
        tolerance=merge_tolerance,
        group_by_property=group_by
    )

    # Save merged results
    merged_path = os.path.join(output_dir, f"{base_name}_merged.geojson")
    features = []
    for i, poly in enumerate(merged_polygons):
        coords = list(poly.exterior.coords)
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [coords]
            },
            'properties': merged_properties[i]
        }
        features.append(feature)

    geojson_output = {
        'type': 'FeatureCollection',
        'features': features
    }

    if crs:
        geojson_output['crs'] = crs
        
    if return_json:
        return geojson_output

    with open(merged_path, 'w') as f:
        json.dump(geojson_output, f, indent=2)

    # print(f"\nMerged polygons saved to: {merged_path}")

    # Clean up intermediate files if requested
    if not keep_intermediate:
        # print("\nCleaning up intermediate files...")
        if os.path.exists(fixed_path):
            os.remove(fixed_path)
            # print(f"  Removed: {fixed_path}")
        if os.path.exists(divided_path):
            os.remove(divided_path)
            # print(f"  Removed: {divided_path}")

    # Print summary
    # print("\n" + "=" * 80)
    # print("PIPELINE COMPLETE")
    # print("=" * 80)
    # print(f"Input polygons: {len(polygons)}")
    # print(f"Output polygons: {len(merged_polygons)}")
    # print(f"Final output: {merged_path}")
    # print("=" * 80)

    return merged_path


def main():
    """Main function with command-line argument support."""
    parser = argparse.ArgumentParser(
        description='End-to-end pipeline for polygon processing: fix angles, divide, and merge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python polygon_division_merge.py input.geojson

  # Specify output directory
  python polygon_division_merge.py input.geojson -o ./output

  # Custom parameters
  python polygon_division_merge.py input.geojson -a 5.0 -l 1.0 -t 0.1

  # Don't keep intermediate files
  python polygon_division_merge.py input.geojson --no-keep-intermediate
        """
    )

    parser.add_argument(
        'input',
        help='Path to input GeoJSON file'
    )
    parser.add_argument(
        '-o', '--output-dir',
        help='Output directory for processed files (default: same as input file)'
    )
    parser.add_argument(
        '-a', '--angle-tolerance',
        type=float,
        default=3.0,
        help='Angle tolerance for fixing angles (default: 3.0 degrees)'
    )
    parser.add_argument(
        '-l', '--lower-threshold',
        type=float,
        default=0.5,
        help='Lower threshold for angle fixing (default: 0.5 degrees)'
    )
    parser.add_argument(
        '-t', '--merge-tolerance',
        type=float,
        default=5e-2,
        help='Tolerance for merging rectangles (default: 0.05)'
    )
    parser.add_argument(
        '-g', '--group-by',
        default='original_polygon_id',
        help='Property name to group polygons by when merging (default: original_polygon_id)'
    )
    parser.add_argument(
        '--no-keep-intermediate',
        action='store_true',
        help='Remove intermediate files after processing'
    )

    args = parser.parse_args()

    try:
        result_path = run_pipeline(
            input_geojson=args.input,
            output_dir=args.output_dir,
            angle_tolerance=args.angle_tolerance,
            lower_threshold=args.lower_threshold,
            merge_tolerance=args.merge_tolerance,
            group_by=args.group_by,
            keep_intermediate=not args.no_keep_intermediate
        )
        # print(f"\nSuccess! Final output: {result_path}")
        return 0
    except Exception as e:
        # print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
