import geojson
import math
from typing import List, Union, Tuple
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon as ShapelyMultiPolygon

class PolygonPartitionHandler:
    """Reader for GeoJSON files containing Polygons and MultiPolygons"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None
        
    def load(self):
        """Load GeoJSON file"""
        with open(self.filepath, 'r') as f:
            self.data = geojson.load(f)
        return self
    
    def get_polygons(self) -> List[geojson.Polygon]:
        """Extract all Polygon features"""
        polygons = []
        if self.data is None:
            return polygons
            
        if isinstance(self.data, geojson.FeatureCollection):
            for feature in self.data['features']:
                if feature['geometry']['type'] == 'Polygon':
                    polygons.append(feature['geometry'])
        elif isinstance(self.data, geojson.Polygon):
            polygons.append(self.data)
            
        return polygons
    
    def get_multipolygons(self) -> List[geojson.MultiPolygon]:
        """Extract all MultiPolygon features"""
        multipolygons = []
        if self.data is None:
            return multipolygons
            
        if isinstance(self.data, geojson.FeatureCollection):
            for feature in self.data['features']:
                if feature['geometry']['type'] == 'MultiPolygon':
                    multipolygons.append(feature['geometry'])
        elif isinstance(self.data, geojson.MultiPolygon):
            multipolygons.append(self.data)
            
        return multipolygons
    
    def get_all_geometries(self) -> List[Union[geojson.Polygon, geojson.MultiPolygon]]:
        """Extract all Polygon and MultiPolygon geometries"""
        return self.get_polygons() + self.get_multipolygons()
    
    def get_features_with_properties(self):
        """Get all features with their properties"""
        features = []
        if self.data is None:
            return features

        if isinstance(self.data, geojson.FeatureCollection):
            for feature in self.data['features']:
                if feature['geometry']['type'] in ['Polygon', 'MultiPolygon']:
                    features.append({
                        'geometry': feature['geometry'],
                        'properties': feature.get('properties', {})
                    })
        return features

    @staticmethod
    def calculate_inner_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """
        Calculate inner angle at vertex p2 formed by points p1-p2-p3
        Returns angle in degrees
        """
        # Vectors from p2 to p1 and p2 to p3
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # Calculate the dot product and cross product
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cross = v1[0] * v2[1] - v1[1] * v2[0]

        # Calculate angle using atan2
        angle = math.atan2(cross, dot)

        # Convert to degrees and ensure positive
        angle_deg = math.degrees(angle)
        if angle_deg < 0:
            angle_deg += 360

        angle_deg = 360.0 - angle_deg
        # print(angle_deg)
        return angle_deg

    @staticmethod
    def calculate_polygon_angles(coordinates: List[List[float]]) -> List[dict]:
        """
        Calculate inner angles for all vertices of a polygon ring
        coordinates: List of [lon, lat] pairs forming a closed ring
        Returns: List of dicts with vertex coordinates and angles
        """
        angles = []
        n = len(coordinates)

        # Skip if polygon has less than 3 vertices (excluding closing point)
        if n < 4:  # Need at least 3 points + closing point
            return angles

        # Process each vertex (excluding the last duplicate closing point)
        for i in range(n - 1):
            prev_idx = (i - 1) % (n - 1)
            next_idx = (i + 1) % (n - 1)

            p1 = coordinates[prev_idx]
            p2 = coordinates[i]
            p3 = coordinates[next_idx]

            angle = PolygonPartitionHandler.calculate_inner_angle(
                (p1[0], p1[1]),
                (p2[0], p2[1]),
                (p3[0], p3[1])
            )

            angles.append({
                'vertex': p2,
                'angle_degrees': angle,
                'vertex_index': i
            })

        return angles

    def get_all_angles(self) -> List[dict]:
        """
        Calculate inner angles for all polygons and multipolygons
        Returns: List of dicts containing geometry info and angles
        """
        results = []

        # Process features with properties
        if isinstance(self.data, geojson.FeatureCollection):
            polygon_idx = 0
            multipolygon_idx = 0

            for feature in self.data['features']:
                properties = feature.get('properties', {})
                geometry_type = feature['geometry']['type']

                if geometry_type == 'Polygon':
                    coords = feature['geometry']['coordinates']

                    # Process exterior ring
                    exterior_angles = self.calculate_polygon_angles(coords[0])

                    result = {
                        'type': 'Polygon',
                        'index': polygon_idx,
                        'properties': properties,
                        'exterior_ring_angles': exterior_angles,
                        'holes': []
                    }

                    # Process holes (interior rings)
                    for hole_idx, hole in enumerate(coords[1:]):
                        hole_angles = self.calculate_polygon_angles(hole)
                        result['holes'].append({
                            'hole_index': hole_idx,
                            'angles': hole_angles
                        })

                    results.append(result)
                    polygon_idx += 1

                elif geometry_type == 'MultiPolygon':
                    mp_result = {
                        'type': 'MultiPolygon',
                        'index': multipolygon_idx,
                        'properties': properties,
                        'polygons': []
                    }

                    for poly_idx, polygon_coords in enumerate(feature['geometry']['coordinates']):
                        # Process exterior ring
                        exterior_angles = self.calculate_polygon_angles(polygon_coords[0])

                        poly_result = {
                            'polygon_index': poly_idx,
                            'exterior_ring_angles': exterior_angles,
                            'holes': []
                        }

                        # Process holes
                        for hole_idx, hole in enumerate(polygon_coords[1:]):
                            hole_angles = self.calculate_polygon_angles(hole)
                            poly_result['holes'].append({
                                'hole_index': hole_idx,
                                'angles': hole_angles
                            })

                        mp_result['polygons'].append(poly_result)

                    results.append(mp_result)
                    multipolygon_idx += 1
        
        else:
            # Fallback for non-FeatureCollection data
            polygons = self.get_polygons()
            for idx, polygon in enumerate(polygons):
                coords = polygon['coordinates']
                exterior_angles = self.calculate_polygon_angles(coords[0])

                result = {
                    'type': 'Polygon',
                    'index': idx,
                    'properties': {},
                    'exterior_ring_angles': exterior_angles,
                    'holes': []
                }

                for hole_idx, hole in enumerate(coords[1:]):
                    hole_angles = self.calculate_polygon_angles(hole)
                    result['holes'].append({
                        'hole_index': hole_idx,
                        'angles': hole_angles
                    })

                results.append(result)

            multipolygons = self.get_multipolygons()
            for idx, multipolygon in enumerate(multipolygons):
                mp_result = {
                    'type': 'MultiPolygon',
                    'index': idx,
                    'properties': {},
                    'polygons': []
                }

                for poly_idx, polygon_coords in enumerate(multipolygon['coordinates']):
                    exterior_angles = self.calculate_polygon_angles(polygon_coords[0])

                    poly_result = {
                        'polygon_index': poly_idx,
                        'exterior_ring_angles': exterior_angles,
                        'holes': []
                    }

                    for hole_idx, hole in enumerate(polygon_coords[1:]):
                        hole_angles = self.calculate_polygon_angles(hole)
                        poly_result['holes'].append({
                            'hole_index': hole_idx,
                            'angles': hole_angles
                        })

                    mp_result['polygons'].append(poly_result)

                results.append(mp_result)

        return results


class RemainingPartitionsHandler:
    """Handler to manage remaining partitions across recursive iterations"""

    def __init__(self):
        self.partitions = []  # List of (geometry, original_polygon_id, properties)

    def add_partition(self, geometry, original_polygon_id, properties=None):
        """Add a partition to the handler"""
        if geometry is not None and not geometry.is_empty:
            if properties is None:
                properties = {}
            self.partitions.append((geometry, original_polygon_id, properties))

    def get_partitions_as_polygon_data(self) -> List[dict]:
        """Convert remaining partitions to polygon_data format for processing"""
        polygon_data_list = []

        for geom, original_id, properties in self.partitions:
            if isinstance(geom, ShapelyPolygon):
                # Ensure counter-clockwise winding order
                if not geom.exterior.is_ccw:
                    coords_list = list(geom.exterior.coords)
                    coords = [list(reversed(coords_list))]
                else:
                    coords = [list(geom.exterior.coords)]
                polygon_data_list.append({
                    'type': 'Polygon',
                    'index': original_id,
                    'properties': properties,
                    'vertices_90': [],  # Will be populated later
                    'coordinates': coords
                })
            elif isinstance(geom, ShapelyMultiPolygon):
                for poly_idx, poly in enumerate(geom.geoms):
                    # Ensure counter-clockwise winding order
                    if not poly.exterior.is_ccw:
                        coords_list = list(poly.exterior.coords)
                        coords = [list(reversed(coords_list))]
                    else:
                        coords = [list(poly.exterior.coords)]
                    polygon_data_list.append({
                        'type': 'Polygon',
                        'index': original_id,
                        'properties': properties,
                        'vertices_90': [],
                        'coordinates': coords
                    })

        return polygon_data_list

    def has_partitions(self) -> bool:
        """Check if there are any partitions remaining"""
        return len(self.partitions) > 0

    def clear(self):
        """Clear all partitions"""
        self.partitions = []

    def save_to_geojson(self, output_path: str, crs: dict = None):
        """Save remaining partitions to a GeoJSON file"""
        if crs is None:
            crs = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}

        features = []

        for idx, (geom, original_id, properties) in enumerate(self.partitions):
            if geom is None or geom.is_empty:
                continue

            if isinstance(geom, ShapelyPolygon):
                coords = [list(geom.exterior.coords)]
                # Merge original properties with partition metadata
                feature_properties = properties.copy()
                feature_properties.update({
                    'id': idx,
                    'original_polygon_id': original_id,
                    'type': 'remaining_partition'
                })
                feature = geojson.Feature(
                    geometry=geojson.Polygon(coords),
                    properties=feature_properties
                )
                features.append(feature)
            elif isinstance(geom, ShapelyMultiPolygon):
                for poly in geom.geoms:
                    coords = [list(poly.exterior.coords)]
                    # Merge original properties with partition metadata
                    feature_properties = properties.copy()
                    feature_properties.update({
                        'id': idx,
                        'original_polygon_id': original_id,
                        'type': 'remaining_partition'
                    })
                    feature = geojson.Feature(
                        geometry=geojson.Polygon(coords),
                        properties=feature_properties
                    )
                    features.append(feature)

        feature_collection = geojson.FeatureCollection(features, crs=crs)

        with open(output_path, 'w') as f:
            geojson.dump(feature_collection, f, indent=2)

    def load_from_geojson(self, input_path: str):
        """Load remaining partitions from a GeoJSON file"""
        with open(input_path, 'r') as f:
            data = geojson.load(f)

        self.clear()

        if isinstance(data, geojson.FeatureCollection):
            for feature in data['features']:
                if feature['geometry']['type'] == 'Polygon':
                    coords = feature['geometry']['coordinates'][0]
                    geom = ShapelyPolygon(coords[:-1] if coords[0] == coords[-1] else coords)
                    original_id = feature['properties'].get('original_polygon_id', 0)
                    properties = feature.get('properties', {})
                    self.add_partition(geom, original_id, properties)

        return self


# Usage example:
if __name__ == "__main__":
    # Read GeoJSON file
    reader = PolygonPartitionHandler('rectangle_partition_test01.geojson').load()

    # Get all polygons
    polygons = reader.get_polygons()
    # print(f"Found {len(polygons)} polygons")

    # Get all multipolygons
    multipolygons = reader.get_multipolygons()
    # print(f"Found {len(multipolygons)} multipolygons")

    # Get all geometries
    all_geoms = reader.get_all_geometries()
    # print(f"Total geometries: {len(all_geoms)}")

    # Calculate inner angles for all geometries
    # print("\n=== Inner Angles ===")
    angle_results = reader.get_all_angles()

    for result in angle_results:
        if result['type'] == 'Polygon':
            # print(f"\nPolygon {result['index']}:")
            # print(f"  Exterior ring vertices: {len(result['exterior_ring_angles'])}")
            for angle_data in result['exterior_ring_angles']:
                # print(f"    Vertex {angle_data['vertex_index']}: {angle_data['vertex']} -> {angle_data['angle_degrees']:.2f}°")
                pass

            for hole in result['holes']:
                # print(f"  Hole {hole['hole_index']}:")
                for angle_data in hole['angles']:
                    # print(f"    Vertex {angle_data['vertex_index']}: {angle_data['vertex']} -> {angle_data['angle_degrees']:.2f}°")
                    pass

        elif result['type'] == 'MultiPolygon':
            # print(f"\nMultiPolygon {result['index']}:")
            for poly in result['polygons']:
                # print(f"  Polygon {poly['polygon_index']}:")
                # print(f"    Exterior ring vertices: {len(poly['exterior_ring_angles'])}")
                for angle_data in poly['exterior_ring_angles']:
                    # print(f"      Vertex {angle_data['vertex_index']}: {angle_data['vertex']} -> {angle_data['angle_degrees']:.2f}°")
                    pass

                for hole in poly['holes']:
                    # print(f"    Hole {hole['hole_index']}:")
                    for angle_data in hole['angles']:
                        # print(f"      Vertex {angle_data['vertex_index']}: {angle_data['vertex']} -> {angle_data['angle_degrees']:.2f}°")
                        pass