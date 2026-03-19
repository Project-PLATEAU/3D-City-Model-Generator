import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon
from shapely import validation

def edges_to_polygons(edges, vertices, min_vertices=3, fallback=True, logging=True):
    """
    Convert edge list to enclosed polygons using a cleaner cycle detection approach.
    If no cycles found, connects boundary vertices directly.
    
    Args:
        edges: numpy array of edges [[v1, v2], ...]
        vertices: numpy array of vertex coordinates [[x, y, z], ...]
        min_vertices: minimum number of vertices for a valid polygon
    
    Returns:
        list of shapely Polygon objects
    """
    if len(edges) == 0:
        return []
    
    # Build adjacency graph
    graph = defaultdict(list)
    vertex_set = set()
    
    for edge in edges:
        v1, v2 = edge
        graph[v1].append(v2)
        graph[v2].append(v1)
        vertex_set.add(v1)
        vertex_set.add(v2)
    
    # Find cycles using a simple traversal approach
    visited_edges = set()
    cycles = []
    
    def find_simple_cycles():
        """Find cycles by following connected paths"""
        for start_vertex in vertex_set:
            if len(graph[start_vertex]) < 2:
                continue  # Skip vertices with degree < 2
                
            # Try to find cycles starting from this vertex
            for first_neighbor in graph[start_vertex]:
                if (start_vertex, first_neighbor) in visited_edges or (first_neighbor, start_vertex) in visited_edges:
                    continue
                    
                path = [start_vertex, first_neighbor]
                current = first_neighbor
                used_edges = {(start_vertex, first_neighbor), (first_neighbor, start_vertex)}
                
                while True:
                    # Find next unvisited neighbor
                    next_options = []
                    for neighbor in graph[current]:
                        edge_key = tuple(sorted([current, neighbor]))
                        if edge_key not in used_edges:
                            next_options.append(neighbor)
                    
                    # Check if we can close the cycle
                    if start_vertex in graph[current] and len(path) >= min_vertices:
                        edge_key = tuple(sorted([current, start_vertex]))
                        if edge_key not in used_edges:
                            # Found a valid cycle
                            cycle = path[:]
                            if is_valid_cycle(cycle):
                                cycles.append(cycle)
                            break
                    
                    # Continue path if possible
                    if len(next_options) == 0:
                        break  # Dead end
                    elif len(next_options) == 1:
                        next_vertex = next_options[0]
                        if next_vertex in path:  # Would create a loop
                            break
                        path.append(next_vertex)
                        used_edges.add(tuple(sorted([current, next_vertex])))
                        current = next_vertex
                    else:
                        # Multiple options - take the one that goes "most clockwise" or just the first
                        next_vertex = next_options[0]
                        if next_vertex in path:
                            break
                        path.append(next_vertex)
                        used_edges.add(tuple(sorted([current, next_vertex])))
                        current = next_vertex
                    
                    if len(path) > len(vertex_set):  # Prevent infinite loops
                        break
    
    def is_valid_cycle(cycle):
        """Check if cycle is valid (no repeated vertices except start/end)"""
        return len(cycle) >= min_vertices and len(set(cycle)) == len(cycle)
    
    find_simple_cycles()
    
    # Remove duplicate cycles using a better method
    unique_cycles = []
    seen_cycle_sets = set()
    
    for cycle in cycles:
        # Convert to set for comparison (ignoring order and direction)
        cycle_set = frozenset(cycle)
        if cycle_set not in seen_cycle_sets and len(cycle_set) >= min_vertices:
            seen_cycle_sets.add(cycle_set)
            unique_cycles.append(cycle)
    
    if logging:
        print(f"Debug: Found {len(cycles)} total cycles, {len(unique_cycles)} unique cycles")
    
    polygons = []
    
    # Convert cycles to polygons
    for cycle in unique_cycles:
        try:
            # Get coordinates for the cycle
            coords = []
            for vertex_idx in cycle:
                vertex_coord = vertices[vertex_idx]
                coords.append((vertex_coord[0], vertex_coord[2]))  # Use x, y coordinates
            
            # Close the polygon
            if len(coords) >= 3:
                coords.append(coords[0])
                poly = Polygon(coords)
                
                if poly.is_valid and poly.area > 1e-10:  # Valid and non-degenerate
                    polygons.append(poly)
                    
        except Exception as e:
            print(f"Warning: Could not create polygon from cycle {cycle}: {e}")
            continue
    
    if logging:
        print(f"Debug: Created {len(polygons)} valid polygons from cycles")
    
    # If no polygons found, create fallback
    if not polygons and fallback:
        polygons = create_fallback_polygon(edges, vertices)
        if logging:
            print(f"Debug: Created {len(polygons)} fallback polygons")
    
    return polygons


def create_fallback_polygon(edges, vertices):
    """Create a simple polygon when no cycles are detected."""
    if len(edges) == 0:
        return []
    
    # Get all vertices involved
    all_vertices = set()
    for edge in edges:
        all_vertices.add(edge[0])
        all_vertices.add(edge[1])
    
    if len(all_vertices) < 3:
        return []
    
    # Create convex hull of all vertices
    try:
        from shapely.geometry import MultiPoint
        
        coords = [(vertices[v][0], vertices[v][1]) for v in all_vertices]
        points = MultiPoint(coords)
        hull = points.convex_hull
        
        if isinstance(hull, Polygon) and hull.is_valid and hull.area > 1e-10:
            return [hull]
            
    except Exception as e:
        print(f"Warning: Could not create fallback polygon: {e}")
    
    return []


# Test with simple example
if __name__ == "__main__":
    # Simple test case: a square and triangle
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Square (should create 1 polygon)
        [4, 5], [5, 6], [6, 4],          # Triangle (should create 1 polygon)
    ])
    
    vertices = np.array([
        [0, 0, 0],    # 0
        [1, 0, 0],    # 1  
        [1, 1, 0],    # 2
        [0, 1, 0],    # 3
        [2, 0, 0],    # 4
        [3, 0, 0],    # 5
        [2.5, 1, 0],  # 6
    ])
    
    polygons = edges_to_polygons(edges, vertices)
    
    print(f"\nFinal result: {len(polygons)} polygon(s)")
    for i, poly in enumerate(polygons):
        print(f"Polygon {i+1}: {len(poly.exterior.coords)-1} vertices, area = {poly.area:.3f}")