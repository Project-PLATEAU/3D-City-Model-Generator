from .polygon_division import find_90_degree_vertices
from .polygon_read import PolygonPartitionHandler
import numpy as np
import geojson
from shapely.geometry import Polygon
from shapely.validation import explain_validity


def is_clockwise(polygon):
    """
    Determine if a polygon is ordered clockwise or counterclockwise.
    
    Args:
        polygon: List of (x, y) tuples representing polygon vertices
        
    Returns:
        True if clockwise, False if counterclockwise
    """
    signed_area = 0
    n = len(polygon)
    
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]  # Wrap around to first vertex
        signed_area += (x2 - x1) * (y2 + y1)
    
    return signed_area > 0


def is_polygon_valid(coords):
    """
    Check if a polygon is valid (no self-intersections, simple).

    Args:
        coords: Array of polygon coordinates

    Returns:
        Boolean indicating validity
    """
    try:
        # Remove duplicate closing point if present
        if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
            poly_coords = coords[:-1]
        else:
            poly_coords = coords

        poly = Polygon(poly_coords)
        return poly.is_valid and poly.is_simple
    except:
        return False


def adjust_vertex_to_90_degrees(coords, vertex_idx):
    """
    Adjust a vertex position to make its interior angle exactly 90 degrees.

    Uses the geometric principle that for any point B where angle ABC = 90 degrees,
    B must lie on a circle with AC as diameter (Thales' theorem).

    Tests both possible positions on the circle and chooses the one that:
    1. Creates a valid polygon
    2. Is closest to the original position

    Args:
        coords: Array of polygon coordinates
        vertex_idx: Index of the vertex to adjust

    Returns:
        New position for the vertex, or None if no valid position exists
    """
    n = len(coords)

    # Get the three points: previous, current, next
    prev_idx = (vertex_idx - 1) % n
    next_idx = (vertex_idx + 1) % n

    A = np.array(coords[prev_idx])  # Previous point
    B = np.array(coords[vertex_idx])  # Current point (to be adjusted)
    C = np.array(coords[next_idx])  # Next point

    # By Thales' theorem, all points on a circle with AC as diameter
    # form a 90-degree angle with A and C

    # Calculate the center of the circle (midpoint of AC)
    center = (A + C) / 2

    # Calculate the radius (half the distance AC)
    radius = np.linalg.norm(C - A) / 2

    if radius < 1e-10:
        # A and C are too close, cannot adjust
        return None

    # Find the direction from center to current point B
    direction = B - center
    direction_norm = np.linalg.norm(direction)

    if direction_norm < 1e-10:
        # B is at the center, choose a perpendicular direction
        # Get perpendicular to AC
        AC = C - A
        direction = np.array([-AC[1], AC[0]])
        direction_norm = np.linalg.norm(direction)

    # Normalize the direction
    direction = direction / direction_norm

    # There are two positions on the circle that form 90 degrees
    # One in the current direction, one in the opposite direction
    B_new1 = center + direction * radius
    B_new2 = center - direction * radius

    # Test both positions
    candidates = []

    for B_new in [B_new1, B_new2]:
        # Create test polygon with new position
        test_coords = coords.copy()
        test_coords[vertex_idx] = B_new

        # Check if the polygon is valid
        if is_polygon_valid(test_coords):
            distance = np.linalg.norm(B_new - B)
            candidates.append((B_new, distance))

    # Choose the valid position closest to original
    if len(candidates) == 0:
        # Neither position creates a valid polygon
        return None
    elif len(candidates) == 1:
        return candidates[0][0]
    else:
        # Return the one closest to original position
        return min(candidates, key=lambda x: x[1])[0]


def read_angles(geojson_path, angle_tolerance=3.0, lower_threshold=0.5, max_iterations=10):
    """
    Main function to find and adjust vertices with angles close to 90 degrees.
    Uses a single-pass adjustment with validation to avoid oscillation.

    Args:
        geojson_path: Path to the GeoJSON file
        angle_tolerance: Initial threshold for detecting angles to adjust (default: 3.0 degrees)
        lower_threshold: Stop iterating when all angles are within this threshold (default: 0.5 degrees)
        max_iterations: Maximum number of iterations to prevent infinite loops (default: 10)

    Returns:
        Tuple of (output_path, final_angle_results)
    """
    handler = PolygonPartitionHandler(geojson_path).load()

    iteration = 0
    total_adjustments = 0

    # Track vertices that have been adjusted to avoid oscillation
    adjusted_vertices = set()

    # print(f"Angle tolerance: ±{angle_tolerance}° (adjusting angles between {90-angle_tolerance}° and {90+angle_tolerance}°)")
    # print(f"Lower threshold: ±{lower_threshold}° (stop when all angles within {90-lower_threshold}° to {90+lower_threshold}°)")
    # print("=" * 80)

    while iteration < max_iterations:
        iteration += 1
        adjustments_this_iteration = 0

        # print(f"\n--- Iteration {iteration} ---")

        # Get angle results for current state
        angle_results = handler.get_all_angles()

        # Track feature index for FeatureCollection
        feature_idx = 0

        # Check if any vertices need adjustment
        needs_adjustment = False

        # Process each geometry
        for idx, item in enumerate(angle_results):
            geom_type = item['type']

            # Get the feature from the data
            if isinstance(handler.data, geojson.FeatureCollection):
                feature = handler.data['features'][feature_idx]
                geometry = feature['geometry']
            else:
                # For standalone geometry
                geometry = handler.data

            if geom_type == 'MultiPolygon':
                for poly_idx, polygon in enumerate(item['polygons']):
                    # Get the coordinates for this polygon
                    coords = geometry['coordinates'][poly_idx][0]
                    # if not is_clockwise(coords):
                    #     coords.reverse()

                    # Check if polygon has closing point before converting to numpy
                    has_closing_point = (len(coords) > 1 and coords[0] == coords[-1])

                    coords = np.array(coords)

                    # Track which vertices to adjust in this polygon
                    vertices_to_adjust = []

                    # Identify vertices that need adjustment
                    for angle_info in polygon['exterior_ring_angles']:
                        angle = angle_info['angle_degrees']
                        vertex_idx = angle_info.get('vertex_index', angle_info.get('vertex', None))

                        # Check if angle is within tolerance range but outside lower threshold
                        if abs(angle - 90) <= angle_tolerance and abs(angle - 90) > lower_threshold:
                            vertex_key = (idx, poly_idx, vertex_idx)
                            if vertex_key not in adjusted_vertices:
                                vertices_to_adjust.append((vertex_idx, angle))

                    # Adjust vertices, skipping adjacent ones to avoid conflicts
                    adjusted_in_this_polygon = set()
                    for vertex_idx, angle in vertices_to_adjust:
                        n = len(coords)
                        prev_idx = (vertex_idx - 1) % n
                        next_idx = (vertex_idx + 1) % n

                        # Skip if adjacent vertex was already adjusted in this polygon
                        if prev_idx in adjusted_in_this_polygon or next_idx in adjusted_in_this_polygon:
                            continue

                        # Adjust the vertex position
                        new_position = adjust_vertex_to_90_degrees(coords, vertex_idx)

                        if new_position is not None:
                            # print(f"  MultiPolygon[{idx}][{poly_idx}] vertex {vertex_idx}: {angle:.2f}° -> 90.00°")
                            coords[vertex_idx] = new_position

                            # If we adjusted vertex 0 and there's a closing point, update it
                            if vertex_idx == 0 and has_closing_point:
                                coords[-1] = new_position

                            adjusted_in_this_polygon.add(vertex_idx)
                            adjusted_vertices.add((idx, poly_idx, vertex_idx))
                            adjustments_this_iteration += 1
                            needs_adjustment = True
                        else:
                            pass
                            # print(f"  MultiPolygon[{idx}][{poly_idx}] vertex {vertex_idx}: {angle:.2f}° -> SKIPPED (no valid adjustment)")

                    # Update the geometry with adjusted coordinates
                    geometry['coordinates'][poly_idx][0] = coords.tolist()

            elif geom_type == 'Polygon':
                # Get the coordinates
                coords = geometry['coordinates'][0]
                # if not is_clockwise(coords):
                #     coords.reverse()

                # Check if polygon has closing point before converting to numpy
                has_closing_point = (len(coords) > 1 and coords[0] == coords[-1])

                coords = np.array(coords)

                # Track which vertices to adjust in this polygon
                vertices_to_adjust = []

                # Identify vertices that need adjustment
                for angle_info in item['exterior_ring_angles']:
                    angle = angle_info['angle_degrees']
                    vertex_idx = angle_info.get('vertex_index', angle_info.get('vertex', None))

                    # Check if angle is within tolerance range but outside lower threshold
                    if abs(angle - 90) <= angle_tolerance and abs(angle - 90) > lower_threshold:
                        vertex_key = (idx, -1, vertex_idx)  # -1 for Polygon (not MultiPolygon)
                        if vertex_key not in adjusted_vertices:
                            vertices_to_adjust.append((vertex_idx, angle))

                # Adjust vertices, skipping adjacent ones to avoid conflicts
                adjusted_in_this_polygon = set()
                for vertex_idx, angle in vertices_to_adjust:
                    n = len(coords)
                    prev_idx = (vertex_idx - 1) % n
                    next_idx = (vertex_idx + 1) % n

                    # Skip if adjacent vertex was already adjusted in this polygon
                    if prev_idx in adjusted_in_this_polygon or next_idx in adjusted_in_this_polygon:
                        continue

                    # Adjust the vertex position
                    new_position = adjust_vertex_to_90_degrees(coords, vertex_idx)

                    if new_position is not None:
                        # print(f"  Polygon[{idx}] vertex {vertex_idx}: {angle:.2f}° -> 90.00°")
                        coords[vertex_idx] = new_position

                        # If we adjusted vertex 0 and there's a closing point, update it
                        if vertex_idx == 0 and has_closing_point:
                            coords[-1] = new_position

                        adjusted_in_this_polygon.add(vertex_idx)
                        adjusted_vertices.add((idx, -1, vertex_idx))
                        adjustments_this_iteration += 1
                        needs_adjustment = True
                    else:
                        pass
                        # print(f"  Polygon[{idx}] vertex {vertex_idx}: {angle:.2f}° -> SKIPPED (no valid adjustment)")

                # Update the geometry
                geometry['coordinates'][0] = coords.tolist()

            feature_idx += 1

        total_adjustments += adjustments_this_iteration
        # print(f"Adjustments in iteration {iteration}: {adjustments_this_iteration}")

        # Check if we should stop
        if not needs_adjustment:
            # print(f"\n✓ Converged! All angles within ±{lower_threshold}° of 90°")
            break

        if iteration >= max_iterations:
            # print(f"\n⚠ Reached maximum iterations ({max_iterations})")
            break

    # print("=" * 80)
    # print(f"Total iterations: {iteration}")
    # print(f"Total adjustments: {total_adjustments}")

    # Validate all geometries
    # print("\nValidating geometries...")
    validation_errors = 0
    feature_idx = 0

    for item in angle_results:
        geom_type = item['type']

        if isinstance(handler.data, geojson.FeatureCollection):
            feature = handler.data['features'][feature_idx]
            geometry = feature['geometry']
        else:
            geometry = handler.data

        if geom_type == 'MultiPolygon':
            for poly_idx, coords_list in enumerate(geometry['coordinates']):
                coords = np.array(coords_list[0])
                if not is_polygon_valid(coords):
                    # print(f"  ⚠ Warning: MultiPolygon[{feature_idx}][{poly_idx}] is invalid")
                    validation_errors += 1

        elif geom_type == 'Polygon':
            coords = np.array(geometry['coordinates'][0])
            if not is_polygon_valid(coords):
                # print(f"  ⚠ Warning: Polygon[{feature_idx}] is invalid")
                validation_errors += 1

        feature_idx += 1

    if validation_errors == 0:
        pass
        # print("  ✓ All geometries are valid")
    else:
        pass
        # print(f"  ⚠ Found {validation_errors} invalid geometries")

    # Create output path with _fixed suffix
    import os
    base_name, ext = os.path.splitext(geojson_path)
    output_path = f"{base_name}_fixed{ext}"

    # Save the modified geometries to the new file
    with open(output_path, 'w') as f:
        geojson.dump(handler.data, f, indent=2)

    # print(f"\nGeometry updated and saved to {output_path}")

    # Return the updated angles for verification
    angle_results_updated = handler.get_all_angles()
    return output_path, angle_results_updated


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python polygon_rect_angle.py <geojson_file_path> [angle_tolerance] [lower_threshold]")
        print("  geojson_file_path: Path to the GeoJSON file to process")
        print("  angle_tolerance: Maximum angle deviation to adjust (default: 3.0 degrees)")
        print("  lower_threshold: Stop when all angles within this threshold (default: 0.5 degrees)")
        sys.exit(1)

    geojson_path = sys.argv[1]

    # Optional parameters
    angle_tolerance = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    lower_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    # print(f"Processing {geojson_path}...\n")

    output_path, results = read_angles(geojson_path, angle_tolerance, lower_threshold)

    # print("\nProcessing complete!")
    # print(f"Output saved to: {output_path}")
