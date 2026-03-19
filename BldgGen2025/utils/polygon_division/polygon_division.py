import geojson
from .polygon_read import PolygonPartitionHandler, RemainingPartitionsHandler
from typing import List
import math
import os

import shapely
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon as ShapelyMultiPolygon

def find_90_degree_vertices(polygon_handler: PolygonPartitionHandler, tolerance: float = 0.5) -> List[dict]:
    """Find all vertices with inner angles approximately equal to 90 degrees"""
    angle_results = polygon_handler.get_all_angles()
    # print(f'returned_results: {angle_results}')
    vertices_90 = []

    for result in angle_results:
        if result['type'] == 'Polygon':
            polygon_90_vertices = []
            for angle_data in result['exterior_ring_angles']:
                angle = angle_data['angle_degrees']
                if abs(angle - 90.0) <= tolerance:
                    polygon_90_vertices.append({
                        'vertex': angle_data['vertex'],
                        'angle': angle,
                        'index': angle_data['vertex_index']
                    })

            if polygon_90_vertices:
                vertices_90.append({
                    'type': 'Polygon',
                    'index': result['index'],
                    'properties': result.get('properties', {}),
                    'vertices_90': polygon_90_vertices,
                    'coordinates': polygon_handler.get_polygons()[result['index']]['coordinates']
                })

        elif result['type'] == 'MultiPolygon':
            for poly in result['polygons']:
                polygon_90_vertices = []
                for angle_data in poly['exterior_ring_angles']:
                    angle = angle_data['angle_degrees']
                    if abs(angle - 90.0) <= tolerance:
                        polygon_90_vertices.append({
                            'vertex': angle_data['vertex'],
                            'angle': angle,
                            'index': angle_data['vertex_index']
                        })

                if polygon_90_vertices:
                    vertices_90.append({
                        'type': 'MultiPolygon',
                        'index': result['index'],
                        'properties': result.get('properties', {}),
                        'polygon_index': poly['polygon_index'],
                        'vertices_90': polygon_90_vertices,
                        'coordinates': polygon_handler.get_multipolygons()[result['index']]['coordinates'][poly['polygon_index']]
                    })

    # print(f'90 res: {polygon_90_vertices}')

    return vertices_90

def find_90_degree_vertices_from_coords(coords: List[List[float]], tolerance: float = 1.0) -> List[dict]:
    """Find 90-degree vertices from a single polygon's coordinates"""
    vertices_90 = []
    n = len(coords) - 1  # Exclude closing point

    for i in range(n):
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n

        p0 = coords[prev_idx]
        p1 = coords[i]
        p2 = coords[next_idx]

        # Calculate vectors from p1 to p0 and p1 to p2
        v1 = [p0[0] - p1[0], p0[1] - p1[1]]
        v2 = [p2[0] - p1[0], p2[1] - p1[1]]

        # Calculate the dot product and cross product
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]

        # Calculate angle using atan2 (handles orientation correctly)
        angle_rad = math.atan2(cross_product, dot_product)

        # Convert to degrees and ensure positive
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360

        # Calculate inner angle
        angle_deg = 360.0 - angle_deg

        if abs(angle_deg - 90.0) <= tolerance:
            vertices_90.append({
                'vertex': p1,
                'angle': angle_deg,
                'index': i
            })

    return vertices_90

def find_180_degree_vertices(coords: List[List[float]], tolerance: float = 0.5) -> List[int]:
    """Find vertices with angles nearly equal to 180 degrees (collinear points)"""
    vertices_180_indices = []
    n = len(coords) - 1  # Exclude closing point

    for i in range(n):
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n

        p0 = coords[prev_idx]
        p1 = coords[i]
        p2 = coords[next_idx]

        # Calculate vectors
        v1 = [p0[0] - p1[0], p0[1] - p1[1]]
        v2 = [p2[0] - p1[0], p2[1] - p1[1]]

        # Calculate angle
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if mag1 > 0 and mag2 > 0:
            cos_angle = dot_product / (mag1 * mag2)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle_rad = math.acos(cos_angle)
            angle_deg = math.degrees(angle_rad)

            if abs(angle_deg - 180.0) <= tolerance:
                vertices_180_indices.append(i)

    return vertices_180_indices

def remove_180_degree_vertices(coords: List[List[float]], tolerance: float = 0.5) -> List[List[float]]:
    """Remove vertices with angles nearly equal to 180 degrees from polygon coordinates"""
    vertices_180_indices = find_180_degree_vertices(coords, tolerance)

    if not vertices_180_indices:
        return coords

    # Create new coordinate list without 180-degree vertices
    n = len(coords) - 1  # Exclude closing point
    new_coords = []

    for i in range(n):
        if i not in vertices_180_indices:
            new_coords.append(coords[i])

    # Close the polygon
    if new_coords and new_coords[0] != new_coords[-1]:
        new_coords.append(new_coords[0])

    # print(f"  Removed {len(vertices_180_indices)} vertices with ~180° angles")
    return new_coords

def construct_rectangles_from_vertices(vertices_90_list: List[dict]) -> tuple[List[List[List[float]]], List[List[List[float]]]]:
    """Construct a rectangle for each 90-degree vertex using its two connected edges

    Returns:
        tuple: (all_rectangles, filtered_rectangles_within_bounds)
    """

    all_rectangles = []
    filtered_rectangles = []

    for polygon_data in vertices_90_list:
        vertices_90 = polygon_data['vertices_90']
        coords = polygon_data['coordinates'][0]
        n = len(coords) - 1

        # Create original polygon for boundary check with a small buffer
        original_polygon = ShapelyPolygon(coords[:-1])
        buffered_polygon = original_polygon.buffer(0.1)  # Small buffer for tolerance

        for vertex_data in vertices_90:
            vertex_index = vertex_data['index']

            prev_index = (vertex_index - 1) % n
            next_index = (vertex_index + 1) % n

            p0 = coords[prev_index]
            p1 = coords[vertex_index]
            p2 = coords[next_index]

            v2 = [p2[0] - p1[0], p2[1] - p1[1]]

            p3 = [p0[0] + v2[0], p0[1] + v2[1]]

            rect_coords = [p1, p2, p3, p0, p1]

            # Add to all rectangles
            all_rectangles.append(rect_coords)

            # Check if rectangle is fully within original polygon or completely outside
            rectangle = ShapelyPolygon(rect_coords[:-1])

            if not rectangle.is_valid:
                continue

            # Check if rectangle is fully within buffered bounds
            if rectangle.within(buffered_polygon):
                filtered_rectangles.append(rect_coords)
            # Check if rectangle is completely outside (no intersection with buffered polygon)
            elif not rectangle.intersects(buffered_polygon):
                filtered_rectangles.append(rect_coords)

            # # print("Created. ")

    return all_rectangles, filtered_rectangles

def calculate_polygon_area(coords: List[List[float]]) -> float:
    """Calculate area of a polygon using the shoelace formula"""
    n = len(coords) - 1
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    return abs(area) / 2.0

def calculate_overlap_rate(rect1: List[List[float]], rect2: List[List[float]]) -> float:
    """Calculate the overlap rate between two rectangles"""
    from shapely.geometry import Polygon

    poly1 = Polygon(rect1[:-1])
    poly2 = Polygon(rect2[:-1])

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    intersection = poly1.intersection(poly2)

    if intersection.is_empty:
        return 0.0

    intersection_area = intersection.area
    min_area = min(poly1.area, poly2.area)

    if min_area == 0:
        return 0.0

    return intersection_area / min_area

def remove_high_overlap_rectangles(rectangles: List[List[List[float]]], threshold: float = 0.999, min_area_threshold: float = 0.1):
    """Remove smaller rectangles when overlap rate > threshold, and subtract overlapping areas from larger rectangles"""

    # Calculate areas for all rectangles
    rect_with_areas = [(i, rect, calculate_polygon_area(rect)) for i, rect in enumerate(rectangles)]

    # Sort by area (smallest first, so we process smaller ones first)
    rect_with_areas.sort(key=lambda x: x[2])

    # Convert to Shapely polygons with tracking
    polygons_dict = {}
    for idx, rect, area in rect_with_areas:
        polygons_dict[idx] = ShapelyPolygon(rect[:-1])

    # Track which rectangles to remove completely
    to_remove = set()

    # Process pairs to find overlaps
    for i in range(len(rect_with_areas)):
        idx_i, rect_i, area_i = rect_with_areas[i]

        if idx_i in to_remove:
            continue

        for j in range(i + 1, len(rect_with_areas)):
            idx_j, rect_j, area_j = rect_with_areas[j]

            if idx_j in to_remove:
                continue

            poly_i = polygons_dict[idx_i]
            poly_j = polygons_dict[idx_j]

            if poly_i.intersects(poly_j):
                intersection = poly_i.intersection(poly_j)

                if not intersection.is_empty:
                    intersection_area = intersection.area

                    # Calculate overlap rates
                    overlap_rate_i = intersection_area / poly_i.area if poly_i.area > 0 else 0
                    overlap_rate_j = intersection_area / poly_j.area if poly_j.area > 0 else 0

                    # If smaller rectangle has >threshold overlap, remove it completely
                    if overlap_rate_i > threshold:
                        to_remove.add(idx_i)
                        break

                    # Otherwise, if intersection is significant, subtract from SMALLER rectangle
                    if intersection_area >= min_area_threshold:
                        # Subtract intersection from the smaller polygon (idx_i since sorted by area)
                        new_poly_i = poly_i.difference(poly_j)

                        if not new_poly_i.is_empty and new_poly_i.area >= min_area_threshold:
                            polygons_dict[idx_i] = new_poly_i

    # Build result from remaining polygons
    result = []
    for idx, rect, area in rect_with_areas:
        if idx not in to_remove:
            poly = polygons_dict[idx]
            if not poly.is_empty and poly.area >= min_area_threshold:
                # Convert back to coordinate list
                if isinstance(poly, ShapelyPolygon):
                    coords = list(poly.exterior.coords)
                    result.append(coords)

    return result

def save_rectangles_to_geojson(rectangles: List[List[List[float]]], crs: dict, output_path: str):
    """Save constructed rectangles to a GeoJSON file"""
    features = []

    for i, rect_coords in enumerate(rectangles):
        feature = geojson.Feature(
            geometry=geojson.Polygon([rect_coords]),
            properties={
                'id': str(i),
                'type': 'constructed_rectangle'
            }
        )
        features.append(feature)

    feature_collection = geojson.FeatureCollection(features, crs=crs)

    with open(output_path, 'w') as f:
        geojson.dump(feature_collection, f, indent=2)

def calculate_remaining_polygon(original_coords: List[List[float]], rectangles: List[List[List[float]]]):
    """Calculate the remaining parts of the original polygon after subtracting rectangles"""
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    original_polygon = ShapelyPolygon(original_coords[:-1])

    # Create union of all rectangles
    rect_polygons = [ShapelyPolygon(rect[:-1]) for rect in rectangles]
    if rect_polygons:
        rectangles_union = unary_union(rect_polygons)

        # Subtract rectangles from original polygon
        remaining = original_polygon.difference(rectangles_union)

        return remaining
    else:
        return original_polygon

def save_rectangles_and_remainder_to_geojson(rectangles: List[tuple],
                                              remaining_geometry,
                                              crs: dict,
                                              output_path: str,
                                              min_area_threshold: float = 0.0):
    """Save constructed rectangles and remaining polygon parts to a GeoJSON file

    Args:
        rectangles: List of tuples (rect_coords, original_polygon_id, properties)
        remaining_geometry: List of tuples (geometry, original_polygon_id, properties)
        crs: Coordinate reference system
        output_path: Output file path
        min_area_threshold: Minimum area threshold to filter out negligible slivers
    """
    features = []

    # Add rectangles/polygons (may no longer be perfect rectangles due to overlap subtraction)
    for i, (rect_coords, original_id, original_properties) in enumerate(rectangles):
        # Merge original properties with new properties, preserving height attributes
        properties = original_properties.copy()
        properties.update({
            'id': str(i),
            'type': 'partition_polygon',
            'original_polygon_id': original_id
        })

        feature = geojson.Feature(
            geometry=geojson.Polygon([rect_coords]),
            properties=properties
        )
        features.append(feature)

    # Add remaining polygon(s), filtering out negligible slivers
    if remaining_geometry:
        from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon as ShapelyMultiPolygon

        polygon_id = len(rectangles)

        for geom, original_id, original_properties in remaining_geometry:
            if geom is None or geom.is_empty:
                continue

            if isinstance(geom, ShapelyPolygon):
                if geom.area >= min_area_threshold:
                    coords = [list(geom.exterior.coords)]
                    # Merge original properties with remaining polygon metadata
                    properties = original_properties.copy()
                    properties.update({
                        'id': str(polygon_id),
                        'type': 'remaining_polygon',
                        'original_polygon_id': original_id
                    })
                    feature = geojson.Feature(
                        geometry=geojson.Polygon(coords),
                        properties=properties
                    )
                    features.append(feature)
                    polygon_id += 1
            elif isinstance(geom, ShapelyMultiPolygon):
                for poly in geom.geoms:
                    if poly.area >= min_area_threshold:
                        coords = [list(poly.exterior.coords)]
                        # Merge original properties with remaining polygon metadata
                        properties = original_properties.copy()
                        properties.update({
                            'id': str(polygon_id),
                            'type': 'remaining_polygon',
                            'original_polygon_id': original_id
                        })
                        feature = geojson.Feature(
                            geometry=geojson.Polygon(coords),
                            properties=properties
                        )
                        features.append(feature)
                        polygon_id += 1

    feature_collection = geojson.FeatureCollection(features, crs=crs)

    with open(output_path, 'w') as f:
        geojson.dump(feature_collection, f, indent=2)

def find_rectangles(geojson_path, output_path=None):
    """Main function to find perpendicular vertices and construct rectangles for all geometries"""
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(geojson_path)[0]
        ext = os.path.splitext(geojson_path)[1]
        output_path = f"{base_name}_divided{ext}"

    handler = PolygonPartitionHandler(geojson_path).load()

    crs = handler.data.get('crs', {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}})

    vertices_90_list = find_90_degree_vertices(handler, tolerance=0.5)
    # # print(f'vertices_90_list: {vertices_90_list}')

    # Get ALL polygons from the input, not just those with 90-degree vertices
    all_input_polygons = []
    if isinstance(handler.data, geojson.FeatureCollection):
        for idx, feature in enumerate(handler.data['features']):
            if feature['geometry']['type'] == 'Polygon':
                all_input_polygons.append({
                    'type': 'Polygon',
                    'index': idx,
                    'properties': feature.get('properties', {}),
                    'coordinates': feature['geometry']['coordinates']
                })
            elif feature['geometry']['type'] == 'MultiPolygon':
                # Handle MultiPolygon - add each polygon separately
                for poly_idx, polygon_coords in enumerate(feature['geometry']['coordinates']):
                    all_input_polygons.append({
                        'type': 'MultiPolygon',
                        'index': idx,
                        'polygon_index': poly_idx,
                        'properties': feature.get('properties', {}),
                        'coordinates': [polygon_coords[0]]  # exterior ring
                    })

    # Create a set of identifiers for polygons that have 90-degree vertices
    # Use (index, polygon_index) tuple for MultiPolygons, just index for Polygons
    indices_with_90_vertices = set()
    for p in vertices_90_list:
        if p['type'] == 'MultiPolygon':
            indices_with_90_vertices.add((p['index'], p.get('polygon_index', 0)))
        else:
            indices_with_90_vertices.add(p['index'])

    # Track polygons without 90-degree vertices separately
    polygons_without_90_vertices = []
    for p in all_input_polygons:
        if p['type'] == 'MultiPolygon':
            identifier = (p['index'], p.get('polygon_index', 0))
        else:
            identifier = p['index']

        if identifier not in indices_with_90_vertices:
            polygons_without_90_vertices.append(p)

    # print(f"\nTotal input geometries: {len(all_input_polygons)}")
    # print(f"Geometries with 90° vertices: {len(vertices_90_list)}")
    # print(f"Geometries without 90° vertices: {len(polygons_without_90_vertices)}")
    # print(f"\nProcessing {len(vertices_90_list)} geometries...")

    # Filter out 180-degree vertices from initial polygons
    # print("\n=== Filtering 180-degree vertices from initial polygons ===")
    for polygon_data in vertices_90_list:
        original_coords = polygon_data['coordinates'][0]
        filtered_coords = remove_180_degree_vertices(original_coords, tolerance=1.0)
        polygon_data['coordinates'] = [filtered_coords]

    # # print(f'vertices_90_list_after_180: {vertices_90_list}')

    # Initialize handler for remaining partitions
    remaining_handler = RemainingPartitionsHandler()

    # Process all geometries and collect all partition polygons
    all_partition_polygons = []
    all_initial_rectangles = []
    iteration = 0

    no_recurrent_partition = False

    # Initial processing
    current_polygons = vertices_90_list

    while current_polygons:
        iteration += 1
        if iteration > 5:
            break

        # print(f"\n=== Iteration {iteration} ===")
        # print(f"Processing {len(current_polygons)} geometries...")

        # Clear the remaining handler for this iteration
        remaining_handler.clear()

        for geom_idx, polygon_data in enumerate(current_polygons):
            # print(f"\n--- Geometry {geom_idx} (Iteration {iteration}) ---")

            # Get the original polygon ID from properties
            properties = polygon_data.get('properties', {})
            original_polygon_id = properties.get('id', geom_idx)

            # Find 90-degree vertices for this geometry
            coords = polygon_data['coordinates'][0]
            vertices_90 = find_90_degree_vertices_from_coords(coords, tolerance=0.5)
            # # print(f'recurrent_vertices_90: {vertices_90}')

            if not vertices_90:
                # print(f"  No 90-degree vertices found, adding to final remaining partitions")
                # No 90-degree vertices, add to final remaining partitions
                poly = ShapelyPolygon(coords[:-1])
                remaining_handler.add_partition(poly, original_polygon_id, properties)
                continue

            # Update polygon_data with found vertices
            polygon_data['vertices_90'] = vertices_90

            # Process only this geometry
            all_rects, rects = construct_rectangles_from_vertices([polygon_data])
            if iteration == 1:
                all_initial_rectangles.extend(all_rects)

            filtered_rects = remove_high_overlap_rectangles(rects, threshold=0.95, min_area_threshold=0.1)

            # Debug: Save filtered_rects for target polygon
            # if original_polygon_id == "bldg_b8a2330d-372b-4532-a5df-7f0babd598f6":
            #     debug_features = []
            #     for i, rect_coords in enumerate(filtered_rects):
            #         feature = geojson.Feature(
            #             geometry=geojson.Polygon([rect_coords]),
            #             properties={
            #                 'id': i,
            #                 'type': 'filtered_rect_before_buffering',
            #                 'original_polygon_id': original_polygon_id
            #             }
            #         )
            #         debug_features.append(feature)
            #     debug_fc = geojson.FeatureCollection(debug_features, crs=crs)
            #     with open('debug_filtered_rects.geojson', 'w') as f:
            #         geojson.dump(debug_fc, f, indent=2)
            #     # print(f"  DEBUG: Saved {len(filtered_rects)} filtered_rects to debug_filtered_rects.geojson")

            # Calculate remaining polygon parts for this geometry
            original_coords = polygon_data['coordinates'][0]
            remaining_geom = calculate_remaining_polygon(original_coords, filtered_rects)

            # Store partition polygons with their original polygon ID and properties
            all_partition_polygons.extend([(rect, original_polygon_id, properties) for rect in filtered_rects])

            if remaining_geom is not None and not remaining_geom.is_empty:
                # Debuffer then buffer to remove negligible silver parts
                remaining_geom_debuffered = shapely.buffer(remaining_geom, -0.1, cap_style="flat", join_style="mitre")
                remaining_geom = shapely.buffer(remaining_geom_debuffered, 0.1, cap_style="flat", join_style="mitre")
                remaining_handler.add_partition(remaining_geom, original_polygon_id, properties)

            # Output areas for this geometry
            # print(f"  Partition polygons: {len(filtered_rects)}")
            total_geom_area = sum(calculate_polygon_area(rect) for rect in filtered_rects)
            if remaining_geom is not None and not remaining_geom.is_empty:
                total_geom_area += remaining_geom.area
            # print(f"  Total area: {total_geom_area:.2f} sq units")

        # Save remaining partitions after this iteration
        if remaining_handler.has_partitions():
            # remaining_output_path = f'remaining_partitions_iteration_{iteration}.geojson'
            # remaining_handler.save_to_geojson(remaining_output_path, crs)
            # # print(f"\nSaved remaining partitions to: {remaining_output_path}")

            # Check if there are remaining partitions with 90-degree vertices
            current_polygons = remaining_handler.get_partitions_as_polygon_data()
            # print(f"Remaining partitions for next iteration: {len(current_polygons)}")
        else:
            # print(f"\nNo remaining partitions with 90-degree vertices. Stopping.")
            current_polygons = []
            
        if no_recurrent_partition:
            break

    # Collect all final remaining geometries
    all_remaining_geometries = remaining_handler.partitions

    # Add polygons that never had 90-degree vertices to the remaining geometries
    for polygon_data in polygons_without_90_vertices:
        coords = polygon_data['coordinates'][0]
        poly = ShapelyPolygon(coords[:-1])
        original_id = polygon_data.get('properties', {}).get('id', polygon_data['index'])
        properties = polygon_data.get('properties', {})
        all_remaining_geometries.append((poly, original_id, properties))
        # print(f"Including polygon without 90° vertices: {original_id}")

    # Save ALL initially created rectangles
    # save_rectangles_to_geojson(all_initial_rectangles, crs, 'all_initial_rectangles.geojson')

    # Apply buffering logic to partition polygons to clean up negligible slivers
    buffered_partition_polygons = []
    for partition_coords, original_id, properties in all_partition_polygons:
        partition_poly = ShapelyPolygon(partition_coords[:-1])
        # Debuffer then buffer to remove negligible slivers
        partition_debuffered = shapely.buffer(partition_poly, -0.1, cap_style="flat", join_style="mitre")
        partition_buffered = shapely.buffer(partition_debuffered, 0.1, cap_style="flat", join_style="mitre")

        if not partition_buffered.is_empty and partition_buffered.area >= 0.1:
            # Convert back to coordinate list
            if isinstance(partition_buffered, ShapelyPolygon):
                coords = list(partition_buffered.exterior.coords)
                buffered_partition_polygons.append((coords, original_id, properties))
            elif isinstance(partition_buffered, ShapelyMultiPolygon):
                # Handle MultiPolygon by adding each component polygon separately
                for poly in partition_buffered.geoms:
                    if poly.area >= 0.1:
                        coords = list(poly.exterior.coords)
                        buffered_partition_polygons.append((coords, original_id, properties))

    save_rectangles_and_remainder_to_geojson(buffered_partition_polygons, all_remaining_geometries, crs, output_path, min_area_threshold=0.1)

    # Output summary for all geometries
    # print("\n=== All Polygon Areas (All Geometries) ===")

    total_area = 0.0
    for i, (rect, original_id, properties) in enumerate(buffered_partition_polygons):
        area = calculate_polygon_area(rect)
        total_area += area
        # print(f"Partition polygon {i} (from original {original_id}): {area:.2f} sq units")

    if all_remaining_geometries:
        remaining_idx = 0
        for geom, original_id, properties in all_remaining_geometries:
            if geom is None or geom.is_empty:
                continue
            if isinstance(geom, ShapelyPolygon):
                area = geom.area
                total_area += area
                # print(f"Remaining polygon {remaining_idx} (from original {original_id}): {area:.2f} sq units")
                remaining_idx += 1
            elif isinstance(geom, ShapelyMultiPolygon):
                for poly in geom.geoms:
                    area = poly.area
                    total_area += area
                    # print(f"Remaining polygon {remaining_idx} (from original {original_id}): {area:.2f} sq units")
                    remaining_idx += 1

    # print(f"\nTotal area (all geometries): {total_area:.2f} sq units")

    # Calculate total original area
    total_original_area = 0.0
    for polygon_data in vertices_90_list:
        original_coords = polygon_data['coordinates'][0]
        original_polygon = ShapelyPolygon(original_coords[:-1])
        total_original_area += original_polygon.area

    # Add area of polygons without 90° vertices
    for polygon_data in polygons_without_90_vertices:
        coords = polygon_data['coordinates'][0]
        poly = ShapelyPolygon(coords[:-1])
        total_original_area += poly.area

    # print(f"Original total area: {total_original_area:.2f} sq units")
    # print(f"Difference: {abs(total_original_area - total_area):.6f} sq units")


if __name__ == "__main__":
    find_rectangles("/Users/konialive/Documents/vs_codes/plateauGML/citygml_io_upd/test_geojson/rect_angle_fix/52354600_fixed.geojson")