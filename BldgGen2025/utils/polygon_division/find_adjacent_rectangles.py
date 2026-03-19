"""
Script to find polygon pairs that:
1. Both are rectangles
2. Intersect only with edge attached (no area overlap)
3. Share an identical complete edge
"""

import numpy as np
import json
import os
from shapely.geometry import Polygon, LineString, shape
from typing import List, Tuple, Set, Dict, Any, Optional
from itertools import combinations
import argparse


def is_rectangle(polygon: Polygon, tolerance: float = 1e-6) -> bool:
    """
    Check if a polygon is a rectangle.

    Args:
        polygon: Shapely Polygon object
        tolerance: Numerical tolerance for angle checking

    Returns:
        True if polygon is a rectangle, False otherwise
    """
    coords = list(polygon.exterior.coords[:-1])  # Remove duplicate last point

    # Must have exactly 4 vertices
    if len(coords) != 4:
        return False

    # Check if all interior angles are 90 degrees
    for i in range(4):
        p1 = np.array(coords[i - 1])
        p2 = np.array(coords[i])
        p3 = np.array(coords[(i + 1) % 4])

        # Vectors
        v1 = p1 - p2
        v2 = p3 - p2

        # Dot product should be zero for perpendicular vectors
        dot = np.dot(v1, v2)

        if abs(dot) > tolerance:
            return False

    return True


def get_edges(polygon: Polygon) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Get all edges of a polygon as tuples of coordinate pairs.
    Edges are normalized so that the point with smaller coordinates comes first.

    Args:
        polygon: Shapely Polygon object

    Returns:
        List of edges, each as ((x1, y1), (x2, y2))
    """
    coords = list(polygon.exterior.coords[:-1])
    edges = []

    for i in range(len(coords)):
        p1 = tuple(coords[i])
        p2 = tuple(coords[(i + 1) % len(coords)])

        # Normalize edge direction for comparison
        if p1 < p2:
            edges.append((p1, p2))
        else:
            edges.append((p2, p1))

    return edges


def edges_equal(edge1: Tuple[Tuple[float, float], Tuple[float, float]],
                edge2: Tuple[Tuple[float, float], Tuple[float, float]],
                tolerance: float = 1e-6) -> bool:
    """
    Check if two edges are equal within tolerance.

    Args:
        edge1, edge2: Edges as ((x1, y1), (x2, y2))
        tolerance: Numerical tolerance

    Returns:
        True if edges are equal, False otherwise
    """
    p1_1, p1_2 = edge1
    p2_1, p2_2 = edge2

    return (abs(p1_1[0] - p2_1[0]) < tolerance and
            abs(p1_1[1] - p2_1[1]) < tolerance and
            abs(p1_2[0] - p2_2[0]) < tolerance and
            abs(p1_2[1] - p2_2[1]) < tolerance)


def share_complete_edge(poly1: Polygon, poly2: Polygon, tolerance: float = 1e-6) -> bool:
    """
    Check if two polygons share at least one complete identical edge.

    Args:
        poly1, poly2: Shapely Polygon objects
        tolerance: Numerical tolerance

    Returns:
        True if polygons share a complete edge, False otherwise
    """
    edges1 = get_edges(poly1)
    edges2 = get_edges(poly2)

    for e1 in edges1:
        for e2 in edges2:
            if edges_equal(e1, e2, tolerance):
                return True

    return False


def touch_only_at_edge(poly1: Polygon, poly2: Polygon, tolerance: float = 1e-6) -> bool:
    """
    Check if two polygons touch only at their boundary (no area overlap).

    Args:
        poly1, poly2: Shapely Polygon objects
        tolerance: Numerical tolerance

    Returns:
        True if polygons touch only at edge, False otherwise
    """
    # Check if intersection exists
    if not poly1.intersects(poly2):
        return False

    # Check that intersection has no area (only boundary intersection)
    intersection = poly1.intersection(poly2)

    # Intersection should be a LineString or MultiLineString, not a Polygon
    return intersection.area < tolerance


def load_polygons_from_geojson(filepath: str) -> Tuple[List[Polygon], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load polygons from a GeoJSON file.

    Args:
        filepath: Path to GeoJSON file

    Returns:
        Tuple of (list of Polygon objects, list of feature properties, CRS info)
    """
    with open(filepath, 'r') as f:
        geojson_data = json.load(f)

    polygons = []
    properties = []
    crs = geojson_data.get('crs', None)

    if geojson_data['type'] == 'FeatureCollection':
        for feature in geojson_data['features']:
            geom = shape(feature['geometry'])
            if geom.geom_type == 'Polygon':
                polygons.append(geom)
                properties.append(feature.get('properties', {}))
            elif geom.geom_type == 'MultiPolygon':
                # Handle MultiPolygon by extracting individual polygons
                for poly in geom.geoms:
                    polygons.append(poly)
                    properties.append(feature.get('properties', {}))
    elif geojson_data['type'] == 'Feature':
        geom = shape(geojson_data['geometry'])
        if geom.geom_type == 'Polygon':
            polygons.append(geom)
            properties.append(geojson_data.get('properties', {}))

    return polygons, properties, crs


def find_adjacent_rectangle_pairs(polygons: List[Polygon],
                                  properties: List[Dict[str, Any]] = None,
                                  tolerance: float = 1e-6,
                                  group_by_property: str = None) -> List[Tuple[int, int]]:
    """
    Find all pairs of rectangles that share a complete edge with no area overlap.

    Args:
        polygons: List of Shapely Polygon objects
        properties: List of property dictionaries for each polygon
        tolerance: Numerical tolerance
        group_by_property: Property name to group by (e.g., 'original_polygon_id').
                          If provided, only pairs within the same group are checked.

    Returns:
        List of index pairs (i, j) where i < j
    """
    results = []

    # If grouping is requested, organize indices by group
    if group_by_property and properties:
        groups = {}
        for i, props in enumerate(properties):
            if props and group_by_property in props:
                group_id = props[group_by_property]
                if group_id not in groups:
                    groups[group_id] = []
                groups[group_id].append(i)
            else:
                # If property is missing, add to a special "ungrouped" category
                if None not in groups:
                    groups[None] = []
                groups[None].append(i)

        # print(f"\nGrouping by '{group_by_property}':")
        # for group_id, indices in groups.items():
        #     print(f"  Group '{group_id}': {len(indices)} polygons")

        # Check pairs only within each group
        for group_id, group_indices in groups.items():
            for i, j in combinations(group_indices, 2):
                poly1 = polygons[i]
                poly2 = polygons[j]

                # Check condition 1: both are rectangles
                if not is_rectangle(poly1, tolerance):
                    continue
                if not is_rectangle(poly2, tolerance):
                    continue

                # Check condition 3: share complete edge
                if not share_complete_edge(poly1, poly2, tolerance):
                    continue

                # Check condition 2: touch only at edge (no area overlap)
                if not touch_only_at_edge(poly1, poly2, tolerance):
                    continue

                results.append((i, j))
    else:
        # Check all pairs without grouping
        for i, j in combinations(range(len(polygons)), 2):
            poly1 = polygons[i]
            poly2 = polygons[j]

            # Check condition 1: both are rectangles
            if not is_rectangle(poly1, tolerance):
                continue
            if not is_rectangle(poly2, tolerance):
                continue

            # Check condition 3: share complete edge
            if not share_complete_edge(poly1, poly2, tolerance):
                continue

            # Check condition 2: touch only at edge (no area overlap)
            if not touch_only_at_edge(poly1, poly2, tolerance):
                continue

            results.append((i, j))

    return results


def get_shared_edge_length(poly1: Polygon, poly2: Polygon, tolerance: float = 1e-6) -> float:
    """
    Get the length of the shared edge between two polygons.

    Args:
        poly1, poly2: Shapely Polygon objects
        tolerance: Numerical tolerance

    Returns:
        Length of shared edge, or 0 if no shared edge
    """
    edges1 = get_edges(poly1)
    edges2 = get_edges(poly2)

    for e1 in edges1:
        for e2 in edges2:
            if edges_equal(e1, e2, tolerance):
                # Calculate edge length
                p1, p2 = e1
                length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                return length

    return 0.0


def remove_duplicate_vertices(polygon: Polygon, tolerance: float = 1e-6) -> Polygon:
    """
    Remove duplicate or near-duplicate vertices from a polygon.

    Args:
        polygon: Input polygon
        tolerance: Distance tolerance for considering vertices as duplicates

    Returns:
        Polygon with duplicates removed
    """
    coords = list(polygon.exterior.coords[:-1])  # Remove duplicate last point

    if len(coords) <= 3:
        return polygon

    cleaned_coords = []
    n = len(coords)

    for i in range(n):
        current = np.array(coords[i])
        next_point = np.array(coords[(i + 1) % n])

        # Calculate distance to next point
        dist = np.linalg.norm(next_point - current)

        # Only add if not duplicate of next point
        if dist >= tolerance:
            cleaned_coords.append(coords[i])

    if len(cleaned_coords) < 3:
        # Something went wrong, return original
        return polygon

    return Polygon(cleaned_coords)


def remove_collinear_points(polygon: Polygon, tolerance: float = 1e-6) -> Polygon:
    """
    Remove collinear points from a polygon's exterior.

    Args:
        polygon: Input polygon
        tolerance: Tolerance for collinearity check

    Returns:
        Polygon with collinear points removed
    """
    coords = list(polygon.exterior.coords[:-1])  # Remove duplicate last point

    if len(coords) <= 3:
        return polygon

    # Keep only non-collinear vertices
    cleaned_coords = []
    n = len(coords)

    for i in range(n):
        p1 = np.array(coords[(i - 1) % n])
        p2 = np.array(coords[i])
        p3 = np.array(coords[(i + 1) % n])

        # Vectors from p2 to its neighbors
        v1 = p1 - p2
        v2 = p3 - p2

        # Get vector lengths
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        # Skip degenerate cases
        if len1 < tolerance or len2 < tolerance:
            cleaned_coords.append(coords[i])
            continue

        # Normalize vectors
        v1_norm = v1 / len1
        v2_norm = v2 / len2

        # Check if vectors are pointing in opposite directions (collinear)
        # Dot product close to -1 means opposite directions
        dot = np.dot(v1_norm, v2_norm)

        # Cross product magnitude (normalized by vector lengths)
        # For 2D vectors: cross = x1*y2 - x2*y1
        cross = abs(v1_norm[0] * v2_norm[1] - v1_norm[1] * v2_norm[0])

        # Point is collinear if:
        # - Cross product is near zero (parallel or anti-parallel)
        # - AND dot product is near -1 (opposite directions, meaning p2 is between p1 and p3)
        # Use a more lenient tolerance for collinearity (scaled to the tolerance parameter)
        angle_tolerance = max(tolerance, 0.01)  # At least 0.01 for angle checks
        is_collinear = (cross < angle_tolerance) and (dot < -1 + angle_tolerance)

        if not is_collinear:
            cleaned_coords.append(coords[i])

    if len(cleaned_coords) < 3:
        # Something went wrong, return original
        return polygon

    return Polygon(cleaned_coords)


def merge_rectangles(poly1: Polygon, poly2: Polygon, tolerance: float = 1e-6) -> Tuple[Optional[Polygon], bool]:
    """
    Merge two adjacent rectangles into one larger rectangle.

    Args:
        poly1, poly2: Shapely Polygon objects that share a complete edge
        tolerance: Numerical tolerance

    Returns:
        Tuple of (merged Polygon or None, success boolean)
        If successful, returns (merged_polygon, True)
        If unsuccessful, returns (None, False)
    """
    # Get the union of the two polygons
    union = poly1.union(poly2)

    if union.geom_type != 'Polygon':
        # If union is not a simple polygon, use convex hull as fallback
        union = union.convex_hull

    # First, remove duplicate/near-duplicate vertices that can occur due to floating point precision
    union = remove_duplicate_vertices(union, tolerance)

    # Then remove collinear points from the union
    # When two rectangles share an edge, the union will have collinear points
    # along that edge that should be removed
    union = remove_collinear_points(union, tolerance)

    # Verify that the result is still a rectangle
    if not is_rectangle(union, tolerance):
        # Return None to indicate merge failed
        return None, False

    return union, True


def iterative_merge_by_group(polygons: List[Polygon],
                             properties: List[Dict[str, Any]],
                             tolerance: float = 1e-6,
                             group_by_property: str = 'original_polygon_id') -> Tuple[List[Polygon], List[Dict[str, Any]]]:
    """
    Iteratively merge rectangle pairs within each group, starting with the longest shared edge.

    Args:
        polygons: List of Shapely Polygon objects
        properties: List of property dictionaries
        tolerance: Numerical tolerance
        group_by_property: Property to group by

    Returns:
        Tuple of (merged polygons list, merged properties list)
    """
    # Make copies to avoid modifying originals
    current_polygons = polygons.copy()
    current_properties = properties.copy()

    # Group polygons by property
    groups = {}
    for i, props in enumerate(current_properties):
        if props and group_by_property in props:
            group_id = props[group_by_property]
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(i)
        else:
            if None not in groups:
                groups[None] = []
            groups[None].append(i)

    # print(f"\nStarting iterative merge process...")
    # print(f"Grouping by '{group_by_property}':")
    # for group_id, indices in groups.items():
    #     print(f"  Group '{group_id}': {len(indices)} polygons")

    total_merges = 0
    iteration = 0

    while True:
        iteration += 1
        # print(f"\n--- Iteration {iteration} ---")

        # Find all valid pairs across all groups
        all_pairs_with_info = []

        for group_id, group_indices in groups.items():
            # Update group indices to map to current polygon list
            valid_group_indices = []
            for orig_idx in group_indices:
                # Find current index in current_polygons
                if orig_idx < len(current_polygons) and current_polygons[orig_idx] is not None:
                    valid_group_indices.append(orig_idx)

            # Find pairs within this group
            for i, j in combinations(valid_group_indices, 2):
                if current_polygons[i] is None or current_polygons[j] is None:
                    continue

                poly1 = current_polygons[i]
                poly2 = current_polygons[j]

                # Check all conditions
                if not is_rectangle(poly1, tolerance):
                    continue
                if not is_rectangle(poly2, tolerance):
                    continue
                if not share_complete_edge(poly1, poly2, tolerance):
                    continue
                if not touch_only_at_edge(poly1, poly2, tolerance):
                    continue

                # Get shared edge length
                edge_length = get_shared_edge_length(poly1, poly2, tolerance)
                all_pairs_with_info.append((i, j, edge_length, group_id))

        if not all_pairs_with_info:
            # print("No more valid pairs to merge. Stopping.")
            break

        # Sort by edge length (descending) to get longest edge first
        all_pairs_with_info.sort(key=lambda x: x[2], reverse=True)

        # Process each group, merging the longest edge pair in each
        merged_in_iteration = 0
        processed_groups = set()
        merged_indices = set()

        for i, j, edge_length, group_id in all_pairs_with_info:
            # Skip if this group already processed in this iteration
            if group_id in processed_groups:
                continue

            # Skip if either polygon was already merged
            if i in merged_indices or j in merged_indices:
                continue

            # Skip if polygons were set to None
            if current_polygons[i] is None or current_polygons[j] is None:
                continue

            # Attempt to merge the pair
            # print(f"  Group '{group_id}': Attempting to merge indices {i} and {j} (shared edge length: {edge_length:.4f})")

            merged_poly, success = merge_rectangles(current_polygons[i], current_polygons[j], tolerance)

            if not success:
                # print(f"    Warning: Merge failed - union does not form a rectangle. Skipping this pair.")
                continue

            # Update properties for merged polygon
            merged_props = current_properties[i].copy()
            merged_props['merged_from'] = [i, j]
            merged_props['merge_iteration'] = iteration

            # Replace first polygon with merged, set second to None
            current_polygons[i] = merged_poly
            current_properties[i] = merged_props
            current_polygons[j] = None
            current_properties[j] = None

            # Mark as merged
            merged_indices.add(i)
            merged_indices.add(j)
            processed_groups.add(group_id)
            merged_in_iteration += 1
            total_merges += 1

            # print(f"    Success: Merged into rectangle with {len(list(merged_poly.exterior.coords)) - 1} vertices")

        # print(f"  Merged {merged_in_iteration} pair(s) in this iteration")

        if merged_in_iteration == 0:
            # print("No merges in this iteration. Stopping.")
            break

    # Remove None entries
    final_polygons = []
    final_properties = []
    for i, poly in enumerate(current_polygons):
        if poly is not None:
            final_polygons.append(poly)
            final_properties.append(current_properties[i])

    # print(f"\n=== Merge complete ===")
    # print(f"Total merges: {total_merges}")
    # print(f"Original polygon count: {len(polygons)}")
    # print(f"Final polygon count: {len(final_polygons)}")

    return final_polygons, final_properties


def save_pairs_to_geojson(pairs: List[Tuple[int, int]],
                          polygons: List[Polygon],
                          properties: List[Dict[str, Any]],
                          output_file: str,
                          crs: Dict[str, Any] = None,
                          tolerance: float = 1e-6):
    """
    Save rectangle pairs to a GeoJSON file.

    Args:
        pairs: List of polygon index pairs
        polygons: List of all polygons
        properties: List of properties for each polygon
        output_file: Path to output GeoJSON file
        crs: Coordinate Reference System information
        tolerance: Numerical tolerance for edge comparison
    """
    features = []

    for pair_idx, (i, j) in enumerate(pairs):
        # Find shared edges
        edges_i = get_edges(polygons[i])
        edges_j = get_edges(polygons[j])
        shared_edges = []
        for ei in edges_i:
            for ej in edges_j:
                if edges_equal(ei, ej, tolerance):
                    shared_edges.append(ei)

        # Create feature for each polygon in the pair
        for idx, poly_idx in enumerate([i, j]):
            poly = polygons[poly_idx]
            coords = list(poly.exterior.coords)

            # Create properties with pair information
            feature_props = properties[poly_idx].copy() if properties[poly_idx] else {}
            feature_props.update({
                'pair_id': pair_idx,
                'pair_member': idx,  # 0 or 1
                'original_index': poly_idx,
                'paired_with_index': j if idx == 0 else i,
                'shared_edge': shared_edges[0] if shared_edges else None
            })

            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [coords]
                },
                'properties': feature_props
            }
            features.append(feature)

    # Create GeoJSON FeatureCollection
    geojson_output = {
        'type': 'FeatureCollection',
        'features': features
    }

    # Add CRS if provided
    if crs:
        geojson_output['crs'] = crs

    # Write to file
    with open(output_file, 'w') as f:
        json.dump(geojson_output, f, indent=2)

    # print(f"Saved {len(features)} features ({len(pairs)} pairs) to {output_file}")


def run_example():
    """
    Run example with test rectangles.
    """
    # Create test rectangles
    rectangles = [
        # Two rectangles sharing a vertical edge at x=2
        Polygon([(0, 0), (2, 0), (2, 3), (0, 3)]),  # Left rectangle
        Polygon([(2, 0), (4, 0), (4, 3), (2, 3)]),  # Right rectangle (shares edge)

        # Rectangle above the first one, sharing horizontal edge
        Polygon([(0, 3), (2, 3), (2, 5), (0, 5)]),  # Top rectangle (shares edge with rect 0)

        # Separate rectangle (no shared edge)
        Polygon([(5, 5), (7, 5), (7, 8), (5, 8)]),

        # Overlapping rectangle (has area overlap, not just edge)
        Polygon([(1, 1), (3, 1), (3, 4), (1, 4)]),  # Overlaps with rectangles 0 and 1

        # Not a rectangle (triangle)
        Polygon([(8, 0), (10, 0), (9, 2)]),

        # Rectangle sharing partial edge (not complete)
        Polygon([(2, 1), (4, 1), (4, 2), (2, 2)]),
    ]

    # print("Testing rectangle detection:")
    # for i, rect in enumerate(rectangles):
    #     print(f"  Polygon {i}: is_rectangle = {is_rectangle(rect)}")

    # print("\nFinding adjacent rectangle pairs...")
    pairs = find_adjacent_rectangle_pairs(rectangles)

    # print(f"\nFound {len(pairs)} pairs:")
    # for i, j in pairs:
    #     print(f"  Pair ({i}, {j}):")
    #     print(f"    Rectangle {i}: {list(rectangles[i].exterior.coords[:-1])}")
    #     print(f"    Rectangle {j}: {list(rectangles[j].exterior.coords[:-1])}")

    #     # Find and print the shared edge
    #     edges_i = get_edges(rectangles[i])
    #     edges_j = get_edges(rectangles[j])
    #     for ei in edges_i:
    #         for ej in edges_j:
    #             if edges_equal(ei, ej):
    #                 print(f"    Shared edge: {ei}")
    #     print()


def main():
    """
    Main function with command-line argument support.
    """
    parser = argparse.ArgumentParser(
        description='Find rectangle pairs that share a complete edge with no area overlap'
    )
    parser.add_argument(
        'geojson_file',
        nargs='?',
        help='Path to GeoJSON file containing polygons'
    )
    parser.add_argument(
        '-t', '--tolerance',
        type=float,
        default=5e-2,
        help='Numerical tolerance for geometric comparisons (default: 1e-6)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output GeoJSON file to save rectangle pairs (default: <input>_pairs.geojson)'
    )
    parser.add_argument(
        '--json',
        help='Output JSON file to save results as plain JSON'
    )
    parser.add_argument(
        '-g', '--group-by',
        default='original_polygon_id',
        help='Property name to group polygons by (default: original_polygon_id). Only pairs within the same group will be checked.'
    )
    parser.add_argument(
        '-m', '--merge',
        action='store_true',
        help='Enable iterative merging of rectangle pairs by longest edge within each group'
    )

    args = parser.parse_args()

    if args.geojson_file:
        # Load from GeoJSON file
        # print(f"Loading polygons from {args.geojson_file}...")
        polygons, properties, crs = load_polygons_from_geojson(args.geojson_file)
        # print(f"Loaded {len(polygons)} polygons")
        # if crs:
        #     print(f"CRS: {crs}")

        # Count rectangles
        rectangle_count = sum(1 for p in polygons if is_rectangle(p, args.tolerance))
        # print(f"Found {rectangle_count} rectangles")

        # If merge mode is enabled, do iterative merging
        if args.merge:
            merged_polygons, merged_properties = iterative_merge_by_group(
                polygons, properties, args.tolerance, args.group_by
            )

            # Determine output filename for merged results
            if args.output:
                output_geojson = args.output
            else:
                base = os.path.splitext(args.geojson_file)[0]
                output_geojson = f"{base}_merged.geojson"

            # Save merged results
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

            with open(output_geojson, 'w') as f:
                json.dump(geojson_output, f, indent=2)

            # print(f"\nMerged polygons saved to {output_geojson}")

        else:
            # Original pair-finding mode
            # print("\nFinding adjacent rectangle pairs...")
            pairs = find_adjacent_rectangle_pairs(polygons, properties, args.tolerance, args.group_by)

            # print(f"\nFound {len(pairs)} pairs that meet all conditions:")
            results = []

            for i, j in pairs:
                # print(f"\nPair ({i}, {j}):")

                # Print properties if available
                # if properties[i]:
                #     print(f"  Rectangle {i} properties: {properties[i]}")
                # if properties[j]:
                #     print(f"  Rectangle {j} properties: {properties[j]}")

                # Print coordinates
                coords_i = list(polygons[i].exterior.coords[:-1])
                coords_j = list(polygons[j].exterior.coords[:-1])
                # print(f"  Rectangle {i} coords: {coords_i}")
                # print(f"  Rectangle {j} coords: {coords_j}")

                # Find and print the shared edge
                edges_i = get_edges(polygons[i])
                edges_j = get_edges(polygons[j])
                shared_edges = []
                for ei in edges_i:
                    for ej in edges_j:
                        if edges_equal(ei, ej, args.tolerance):
                            shared_edges.append(ei)
                            # print(f"  Shared edge: {ei}")

                results.append({
                    'indices': [i, j],
                    'properties': [properties[i], properties[j]],
                    'coordinates': [coords_i, coords_j],
                    'shared_edges': [list(e) for e in shared_edges]
                })

            # Determine output filename
            if args.output:
                output_geojson = args.output
            else:
                # Default output name based on input
                base = os.path.splitext(args.geojson_file)[0]
                output_geojson = f"{base}_pairs.geojson"

            # Save to GeoJSON file with CRS
            if pairs:
                save_pairs_to_geojson(pairs, polygons, properties, output_geojson, crs, args.tolerance)
            else:
                pass
                # print(f"\nNo pairs found. Skipping GeoJSON output.")

            # Save to plain JSON file if specified
            if args.json:
                with open(args.json, 'w') as f:
                    json.dump(results, f, indent=2)
                # print(f"Results also saved to {args.json}")

    else:
        # Run example
        # print("No GeoJSON file provided. Running example...")
        # print("Usage: python find_adjacent_rectangles.py <geojson_file> [-t tolerance] [-o output]\n")
        run_example()


if __name__ == "__main__":
    main()
