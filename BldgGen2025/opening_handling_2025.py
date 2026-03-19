import json
import numpy as np
from typing import List, Tuple, Dict
import trimesh
from pyproj import Transformer


class LOD1BuildingGenerator:
    """Generate LOD1 buildings with openings (windows, doors) on facades"""

    # Define colors for different opening types (RGB values)
    OPENING_COLORS = {
        'window': [0.055, 0.082, 0.8, 1.0],        # Navy blue
        'entrance': [0.961, 0.808, 0.259, 1.0],    # Yellow
        'door': [0.961, 0.808, 0.259, 1.0],        # Yellow
        'balcony': [0.8, 0.8, 0.8, 1.0],           # Light gray
    }

    WALL_COLOR = [0.506, 0.812, 0.949, 1.0]
    ROOF_COLOR = [0.055, 0.082, 0.8, 1.0]

    def __init__(self):
        self.vertices = []
        self.faces = []
        self.face_colors = []

    @staticmethod
    def is_counter_clockwise(footprint: List[List[float]]) -> bool:
        """
        Determine if a footprint is oriented counter-clockwise.

        Args:
            footprint: List of [x, y] coordinates

        Returns:
            bool: True if counter-clockwise, False if clockwise
        """
        # Remove duplicate last point if present
        if footprint[0] == footprint[-1]:
            footprint = footprint[:-1]

        # Calculate signed area using shoelace formula
        n = len(footprint)
        signed_area = 0
        for i in range(n):
            j = (i + 1) % n
            signed_area += footprint[i][0] * footprint[j][1]
            signed_area -= footprint[j][0] * footprint[i][1]

        # Positive area = counter-clockwise, negative = clockwise
        return signed_area > 0

    @staticmethod
    def normalize_footprint_orientation(footprint: List[List[float]]) -> Tuple[List[List[float]], bool]:
        """
        Ensure footprint is in counter-clockwise order.

        Args:
            footprint: List of [x, y] coordinates

        Returns:
            Tuple of (normalized footprint, was_reversed flag)
        """
        # Remove duplicate last point if present
        has_duplicate = footprint[0] == footprint[-1]
        if has_duplicate:
            footprint = footprint[:-1]

        # Reverse if clockwise
        was_reversed = not LOD1BuildingGenerator.is_counter_clockwise(footprint)
        if was_reversed:
            footprint = footprint[::-1]

        # Add back duplicate if it was there
        if has_duplicate:
            footprint = footprint + [footprint[0]]

        return footprint, was_reversed

    @staticmethod
    def remap_edge_id(old_eid: int, n_vertices: int, was_reversed: bool) -> int:
        """
        Remap edge ID after footprint reversal.

        When a footprint is reversed, edge IDs change because the vertices are in reverse order.
        For a polygon with n vertices, when reversed:
        - old edge i (from vertex i to vertex i+1) becomes new edge (n-2-i)

        Args:
            old_eid: Original edge ID
            n_vertices: Number of vertices in the footprint
            was_reversed: Whether the footprint was reversed

        Returns:
            New edge ID after reversal, or original if not reversed
        """
        if not was_reversed:
            return old_eid

        # Formula: new_eid = (n - 2 - old_eid) % n
        return (n_vertices - 2 - old_eid) % n_vertices

    def create_lod1_building(self, footprint: List[List[float]], height: float) -> trimesh.Trimesh:
        """
        Create a LOD1 building by extruding the footprint to the given height.

        Args:
            footprint: List of [x, y] coordinates defining the building footprint
            height: Building height

        Returns:
            trimesh.Trimesh: The building mesh
        """
        # Remove the last point if it duplicates the first (closed polygon)
        if footprint[0] == footprint[-1]:
            footprint = footprint[:-1]

        n_vertices = len(footprint)
        vertices = []
        faces = []

        # Create bottom vertices (z=0)
        for point in footprint:
            vertices.append([point[0], point[1], 0])

        # Create top vertices (z=height)
        for point in footprint:
            vertices.append([point[0], point[1], height])

        # Create wall faces (triangles only) - OUTWARD NORMALS
        for i in range(n_vertices):
            next_i = (i + 1) % n_vertices
            # Two triangles per wall segment
            faces.append([i, next_i, next_i + n_vertices])
            faces.append([i, next_i + n_vertices, i + n_vertices])

        # Triangulate bottom face (using simple fan triangulation) - OUTWARD (downward)
        for i in range(1, n_vertices - 1):
            faces.append([0, i + 1, i])

        # Triangulate top/roof face - OUTWARD (upward)
        for i in range(1, n_vertices - 1):
            faces.append([n_vertices, n_vertices + i, n_vertices + i + 1])

        vertices = np.array(vertices)
        faces = np.array(faces)

        return trimesh.Trimesh(vertices=vertices, faces=faces)

    def get_wall_coordinates(self, footprint: List[List[float]], height: float,
                            eid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the 3D coordinates of a wall face based on edge id.

        Args:
            footprint: Building footprint
            height: Building height
            eid: Edge ID (0-indexed)

        Returns:
            Tuple of (bottom_left, bottom_right, top_right, top_left) coordinates
        """
        # Remove duplicate last point if present
        if footprint[0] == footprint[-1]:
            footprint = footprint[:-1]

        n_vertices = len(footprint)
        i = eid % n_vertices
        next_i = (i + 1) % n_vertices

        # Wall corners in 3D space
        bottom_left = np.array([footprint[i][0], footprint[i][1], 0])
        bottom_right = np.array([footprint[next_i][0], footprint[next_i][1], 0])
        top_right = np.array([footprint[next_i][0], footprint[next_i][1], height])
        top_left = np.array([footprint[i][0], footprint[i][1], height])

        return bottom_left, bottom_right, top_right, top_left

    def create_opening_on_wall(self, wall_corners: Tuple[np.ndarray, ...],
                              elem: Dict, opening_type: str) -> trimesh.Trimesh:
        """
        Create an opening (window/door) on a wall face.

        Args:
            wall_corners: Tuple of (bottom_left, bottom_right, top_right, top_left)
            elem: Opening element with 'x,y' and 'w,h' in normalized coordinates
            opening_type: Type of opening ('window', 'entrance', etc.)

        Returns:
            trimesh.Trimesh: The opening mesh
        """
        bottom_left, bottom_right, top_right, top_left = wall_corners

        # Get normalized coordinates
        x_norm, y_norm = elem['x,y']
        w_norm, h_norm = elem['w,h']

        # Calculate wall dimensions
        wall_width_vec = bottom_right - bottom_left
        wall_width = np.linalg.norm(wall_width_vec)
        wall_height_vec = top_left - bottom_left
        wall_height = np.linalg.norm(wall_height_vec)

        # Unit vectors along wall
        u_vec = wall_width_vec / wall_width  # horizontal
        v_vec = wall_height_vec / wall_height  # vertical

        # Calculate opening corners in 3D
        x_offset = x_norm * wall_width
        y_offset = y_norm * wall_height
        width = w_norm * wall_width
        height = h_norm * wall_height

        # Opening corners
        opening_bl = bottom_left + u_vec * x_offset + v_vec * y_offset
        opening_br = opening_bl + u_vec * width
        opening_tr = opening_br + v_vec * height
        opening_tl = opening_bl + v_vec * height

        # Create opening mesh (slightly inset to create depth)
        # Reverse cross product order to get inward-pointing normal
        normal = np.cross(v_vec, u_vec)
        normal = normal / np.linalg.norm(normal)
        depth = 0.1  # Opening depth

        # With reversed cross product, normal now points INWARD (into the building)
        # To place openings on exterior:
        # - back face on wall surface (base position)
        # - front face protruding outward (base - normal * depth, since normal points inward)

        # Back face (on the wall surface)
        back_vertices = [opening_bl, opening_br, opening_tr, opening_tl]
        # Front face (protruding outward from wall)
        front_vertices = [v - normal * depth for v in back_vertices]

        all_vertices = front_vertices + back_vertices

        # Create faces for the opening box
        faces = [
            # Front face
            [0, 1, 2], [0, 2, 3],
            # Back face
            [4, 6, 5], [4, 7, 6],
            # Side faces
            [0, 4, 5], [0, 5, 1],  # Bottom
            [1, 5, 6], [1, 6, 2],  # Right
            [2, 6, 7], [2, 7, 3],  # Top
            [3, 7, 4], [3, 4, 0],  # Left
        ]

        vertices = np.array(all_vertices)
        faces = np.array(faces)

        # Get color for this opening type
        color = self.OPENING_COLORS.get(opening_type, [1.0, 0.0, 1.0, 1.0])  # Magenta as default

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.visual.face_colors = color

        return mesh

    def create_wall_with_holes(self, wall_corners: Tuple[np.ndarray, ...],
                               elements: List[Dict]) -> trimesh.Trimesh:
        """
        Create a wall face with holes for openings.

        Args:
            wall_corners: Tuple of (bottom_left, bottom_right, top_right, top_left)
            elements: List of opening elements

        Returns:
            trimesh.Trimesh: Wall mesh with holes
        """
        bottom_left, bottom_right, top_right, top_left = wall_corners

        # For simplicity, we'll create the wall as a simple quad
        # A more sophisticated approach would use polygon boolean operations
        vertices = np.array([bottom_left, bottom_right, top_right, top_left])
        faces = np.array([[0, 1, 2], [0, 2, 3]])

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.visual.face_colors = self.WALL_COLOR

        return mesh

    def generate_building_with_openings(self, building_data: Dict) -> trimesh.Trimesh:
        """
        Generate a complete building with openings from JSON data.

        Args:
            building_data: Dictionary containing building data

        Returns:
            trimesh.Trimesh: Complete building mesh with openings
        """
        footprint = building_data['footprint']

        # Convert footprint from EPSG:6677 to EPSG:30169
        transformer = Transformer.from_crs("EPSG:6677", "EPSG:30169", always_xy=True)
        footprint = [list(transformer.transform(x, y)) for x, y in footprint]

        height = building_data['height']
        faces_data = building_data.get('faces', [])

        # Normalize footprint to counter-clockwise orientation
        footprint, was_reversed = self.normalize_footprint_orientation(footprint)

        # Get number of vertices for edge ID remapping
        temp_footprint = footprint[:-1] if footprint[0] == footprint[-1] else footprint
        n_vertices = len(temp_footprint)

        print(f"  Footprint orientation: {'counter-clockwise' if self.is_counter_clockwise(footprint) else 'clockwise'}")
        if was_reversed:
            print(f"  Footprint was reversed (was clockwise, now counter-clockwise)")

        # Create base LOD1 building
        building_mesh = self.create_lod1_building(footprint, height)
        building_mesh.visual.face_colors = self.WALL_COLOR

        # Set roof color (top face is typically the second face)
        if len(building_mesh.faces) > 1:
            face_colors = np.array([self.WALL_COLOR] * len(building_mesh.faces))
            face_colors[1] = self.ROOF_COLOR  # Top face
            building_mesh.visual.face_colors = face_colors

        meshes = [building_mesh]

        # Create openings for each face
        for face_data in faces_data:
            # Skip if no 'elems' key exists
            if 'elems' not in face_data:
                continue

            old_eid = face_data['eid']
            elements = face_data['elems']

            # Skip if no elements on this face
            if not elements:
                continue

            # Remap edge ID if footprint was reversed
            eid = self.remap_edge_id(old_eid, n_vertices, was_reversed)
            if was_reversed:
                print(f"  Remapped edge {old_eid} -> {eid}")

            # Get wall coordinates
            wall_corners = self.get_wall_coordinates(footprint, height, eid)

            # Create openings
            for elem in elements:
                opening_type = elem.get('type', 'window')
                opening_mesh = self.create_opening_on_wall(wall_corners, elem, opening_type)
                meshes.append(opening_mesh)

        # Combine all meshes
        combined_mesh = trimesh.util.concatenate(meshes)
        # combined_mesh.vertices[:, [1, 2]] = combined_mesh.vertices[:, [2, 1]]
        
        return combined_mesh


def main():
    """Main function to demonstrate the building generation"""

    # Parse JSON
    with open('updated_bldg_data1.json', 'r') as json_data:  
      buildings = json.load(json_data)

    # Generate building with openings
    generator = LOD1BuildingGenerator()

    all_meshes = []
    for building_data in buildings['bldgs']:
        print(f"Generating building: {building_data['id']}")
        mesh = generator.generate_building_with_openings(building_data)
        all_meshes.append(mesh)

        # Export to OBJ file with materials
        building_id = building_data['id'].split('_')[-1]  # Get short ID
        output_file = f"building_{building_id}_with_openings.obj"

        # print(f"Exporting to OBJ file: {output_file}")
        # mesh.export(output_file)
        # print(f"✓ Successfully exported to: {output_file}")

        # Print statistics
        print(f"\nBuilding Statistics:")
        print(f"  Building ID: {building_data['id']}")
        print(f"  Height: {building_data['height']} meters")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        print(f"  Bounding Box: {mesh.bounds}")

        # Count openings (safely handle missing 'elems' keys)
        total_openings = sum(len(face.get('elems', [])) for face in building_data.get('faces', []) if 'elems' in face)
        print(f"  Total Openings: {total_openings}")
        print()

    print(f"All buildings exported successfully!")

    combined_all_meshes = trimesh.util.concatenate(all_meshes)
    combined_all_meshes.export('akabane_lod3_test.obj')

    # Visualize (optional, requires display)
    try:
        mesh.show()
    except:
        print("Note: 3D visualization not available in headless environment")


if __name__ == "__main__":
    main()
