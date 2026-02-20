# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import trimesh

from trimesh.visual.texture import TextureVisuals, SimpleMaterial

import geopandas as gpd
import shapely
from shapely.geometry import (
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon,
)
from shapely.ops import polygonize

from PIL import Image, ImageDraw


# ===================== Global parameters =====================

# Footprint extraction parameters
DEFAULT_Z_OFFSETS = (0.05, 0.10, 0.20, 0.30, 0.50)
DEFAULT_MIN_AREA = 0.2
DEFAULT_SIMPLIFY_TOL = 0.02

# Window parameters (meters)
WINDOW_W = 1.0
WINDOW_H = 1.2
H_GAP = 1.8              # horizontal gap between windows (center to center)
V_GAP = 1.8              # vertical gap between window rows (center to center)
EDGE_GAP = 0.5           # horizontal margin from wall corners for windows
SILL_Z = 3.9             # bottom height of the first window row
HEIGHT_RATIO_MAX = 0.65  # max ratio of building height used for windows
MIN_VERTICAL_CLEAR = 0.30  # minimal vertical clearance above doors

# Door parameters (meters)
DOOR_W = 1.0
DOOR_H = 2.1
DOOR_GAP = 100.0         # horizontal spacing between doors on the same wall edge
DOOR_EDGE_GAP = 0.8      # margin from wall corners for doors
DOOR_MIN_EDGE_LENGTH = DOOR_W + 2 * DOOR_EDGE_GAP

# Other parameters
EPSILON_OUT = 0.1  # small outward offset to reduce z-fighting
HEIGHT_KEYS = ["height", "Height", "H", "h", "building:height", "bldg_height", "levels"]
MIN_EDGE_LENGTH_FOR_WINDOWS = WINDOW_W + 2 * EDGE_GAP

# Colors (RGBA 0-255) for the palette texture
COLOR_BUILDING = (210, 210, 210, 255)  # light gray
COLOR_WINDOWS = (60, 120, 220, 255)    # blue
COLOR_DOORS = (220, 80, 60, 255)       # red


# ============================================================
# 0. Palette material helpers (MTL + PNG)
# ============================================================

def _build_palette_material(
    keys,
    color_map,
    default_rgba=(253, 253, 230, 255),
    tex_h=8,
):
    """
    Build a 1-row palette image (N blocks) + a single SimpleMaterial.
    Returns:
        im         : PIL.Image (RGBA)
        material   : SimpleMaterial with embedded image
        uv_lookup  : dict {key -> (u, v)}
    """
    keys = list(keys)
    N = max(1, len(keys))
    W, H = N * tex_h, tex_h

    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    for i, k in enumerate(keys):
        rgba = color_map.get(k, default_rgba)
        draw.rectangle(
            [i * tex_h, 0, (i + 1) * tex_h - 1, H - 1],
            fill=tuple(rgba),
        )

    mat = SimpleMaterial(image=im)
    try:
        mat.name = "palette_mat"
    except Exception:
        pass

    uv_lookup = {}
    for i, k in enumerate(keys):
        u = (i + 0.5) / N
        v = 0.5
        uv_lookup[k] = (u, v)

    return im, mat, uv_lookup


def _apply_single_material(
    mesh: trimesh.Trimesh,
    im,
    material,
    uv_const=(0.5, 0.5),
):
    """
    Assign one shared material + constant UV to a mesh.
    This allows multiple meshes to share a single PNG + MTL,
    while using different (u, v) for different categories.
    """
    uv = np.tile(np.asarray(uv_const, dtype=float), (len(mesh.vertices), 1))
    vis = TextureVisuals(uv=uv, image=im, material=material)
    mesh.visual = vis
    return mesh


# ============================================================
# 1. Footprint extraction from LOD2 OBJ
# ============================================================

def _section_to_polys_world_xy(sec, min_area: float = 0.05):
    """
    Convert a mesh section to XY polygons in world coordinates (no planar transform).
    """
    if sec is None:
        return []
    lines = []
    for seg3d in sec.discrete:  # list of (Ni, 3)
        if len(seg3d) >= 2:
            xy = seg3d[:, :2]
            lines.append(LineString(xy))
    if not lines:
        return []
    mls = MultiLineString(lines)
    polys = [p.buffer(0) for p in polygonize(mls)]
    polys = [p for p in polys if (not p.is_empty) and p.area >= min_area]
    return polys


def _choose_outermost(polygons):
    """
    Choose the outermost (largest) polygon by area (and then by perimeter).
    """
    if not polygons:
        return None
    return sorted(polygons, key=lambda p: (p.area, p.length), reverse=True)[0]


def extract_footprint_from_mesh(
    mesh: trimesh.Trimesh,
    z_offsets=DEFAULT_Z_OFFSETS,
    min_area: float = DEFAULT_MIN_AREA,
    simplify_tol: float = DEFAULT_SIMPLIFY_TOL,
):
    """
    Extract a single footprint (shapely.Polygon in world XY coordinates) from one LOD2 building mesh.
    """
    if mesh.is_empty:
        return None

    zmin = mesh.bounds[0, 2]
    candidates = []

    # Try horizontal sections slightly above z_min
    for dz in z_offsets:
        z = zmin + dz
        sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        polys = _section_to_polys_world_xy(sec, min_area=min_area)
        if polys:
            candidates.append(_choose_outermost(polys))

    # Fallback: project near-ground triangles onto XY
    if not candidates:
        faces = mesh.faces
        verts = mesh.vertices
        z = verts[:, 2]
        near = (z >= zmin - 1e-6) & (z <= zmin + 0.5)
        tris = verts[faces]
        tri_mask = near[faces].any(axis=1)
        tris_xy = tris[tri_mask][:, :, :2]
        lines = []
        for tri in tris_xy:
            lines += [
                LineString([tri[0], tri[1]]),
                LineString([tri[1], tri[2]]),
                LineString([tri[2], tri[0]]),
            ]
        mls = shapely.ops.unary_union(MultiLineString(lines))
        polys = [p.buffer(0) for p in polygonize(mls)]
        polys = [p for p in polys if (not p.is_empty) and p.area >= min_area]
        if not polys:
            return None
        candidates = polys

    footprint = _choose_outermost(candidates).buffer(0)
    if simplify_tol and simplify_tol > 0:
        footprint = footprint.simplify(simplify_tol, preserve_topology=True).buffer(0)

    return footprint


def parse_multi_building_obj(obj_path: str):
    """
    Parse an OBJ file where buildings are stored as blocks:
      - Each block is: a sequence of 'v ...' lines followed by a sequence of 'f ...' lines.
      - The next 'v' starts a new building block.
    Faces use global indices (1-based in OBJ); we convert to local 0-based indices per building.

    Returns:
        List of (vertices_np, faces_np) for each building.
    """
    buildings: List[Tuple[np.ndarray, np.ndarray]] = []
    cur_vs: List[List[float]] = []
    cur_fs_global: List[List[int]] = []
    seen_face = False
    global_v_count = 0
    block_start_idx: Optional[int] = None  # 1-based index

    def finalize_current():
        nonlocal cur_vs, cur_fs_global, block_start_idx, seen_face, buildings
        if not cur_vs or not cur_fs_global:
            cur_vs, cur_fs_global, block_start_idx, seen_face = [], [], None, False
            return
        verts = np.array(cur_vs, dtype=float)
        offset = block_start_idx  # 1-based
        faces_local = []
        for face in cur_fs_global:
            faces_local.append([gi - offset for gi in face])
        faces = np.array(faces_local, dtype=int)
        buildings.append((verts, faces))
        cur_vs, cur_fs_global, block_start_idx, seen_face = [], [], None, False

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                # if we already started reading faces and see a new vertex, the previous building ends
                if seen_face:
                    finalize_current()
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                if block_start_idx is None:
                    block_start_idx = global_v_count + 1  # 1-based start index
                cur_vs.append([x, y, z])
                global_v_count += 1

            elif line.startswith("f "):
                seen_face = True
                parts = line.split()[1:]
                idxs = []
                for tok in parts:
                    # handle "i" or "i/j/k" and take the first part as index
                    gi = int(tok.split("/")[0])
                    idxs.append(gi)
                # Triangulate if necessary
                if len(idxs) == 3:
                    cur_fs_global.append(idxs)
                else:
                    for k in range(1, len(idxs) - 1):
                        cur_fs_global.append([idxs[0], idxs[k], idxs[k + 1]])

            else:
                # ignore vt, vn, usemtl, etc.
                continue

    finalize_current()
    return buildings


def extract_footprints_gdf_from_lod2(
    obj_path: str,
    src_epsg: str = "EPSG:30169",
    save_shapefile: Optional[str] = None,
    z_offsets=DEFAULT_Z_OFFSETS,
    min_area: float = DEFAULT_MIN_AREA,
    simplify_tol: float = DEFAULT_SIMPLIFY_TOL,
) -> gpd.GeoDataFrame:
    """
    High-level function:
    - Parse the LOD2 OBJ into multiple building meshes (v/f blocks).
    - Extract one footprint per building.
    - Compute zmin, zmax, and height from vertices.
    - Return everything as a GeoDataFrame (and optionally save to a Shapefile).

    Columns:
        id, area, zmin, zmax, height, geometry
    """
    print("[Step 1] Parsing LOD2 OBJ into building blocks for footprints...")
    buildings = parse_multi_building_obj(obj_path)
    print(f"  Found {len(buildings)} building blocks in OBJ (for footprints).")

    records: List[Dict[str, Any]] = []
    geoms: List[Polygon] = []

    bid = 0
    for verts, faces in buildings:
        if len(verts) == 0 or len(faces) == 0:
            continue

        zmin = float(np.min(verts[:, 2]))
        zmax = float(np.max(verts[:, 2]))
        height = float(zmax - zmin)

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        poly = extract_footprint_from_mesh(
            mesh,
            z_offsets=z_offsets,
            min_area=min_area,
            simplify_tol=simplify_tol,
        )

        if poly is None or poly.is_empty:
            continue

        bid += 1
        geoms.append(poly)
        records.append(
            {
                "id": bid,
                "area": float(poly.area),
                "zmin": zmin,
                "zmax": zmax,
                "height": height,
            }
        )

    if not geoms:
        raise ValueError("No valid footprints were extracted from the OBJ.")

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=src_epsg)
    print(f"  Extracted {len(gdf)} valid footprints.")

    if save_shapefile is not None:
        out_path = Path(save_shapefile)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out_path, driver="ESRI Shapefile", encoding="utf-8")
        print(f"  Footprints saved to Shapefile: {out_path}")

    return gdf


# ============================================================
# 2. Generate doors & windows meshes from footprints GDF
# ============================================================

def polygon_orientation(ring_coords) -> float:
    """
    Signed area: >0 means CCW.
    """
    a = 0.0
    pts = list(ring_coords)
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a


def edge_unit_vector(p0, p1):
    ux = p1[0] - p0[0]
    uy = p1[1] - p0[1]
    L = float(np.hypot(ux, uy))
    if L == 0.0:
        return (0.0, 0.0), 0.0
    return (ux / L, uy / L), L


def outward_normal(u, is_ccw: bool):
    """
    For a CCW polygon, outward normal is (y, -x); for CW, (-y, x).
    """
    ux, uy = u
    return (uy, -ux) if is_ccw else (-uy, ux)


def detect_height(props: Dict[str, Any]) -> float:
    """
    Try to detect building height from attribute dict using common keys.
    - "levels" is interpreted as number of floors * 3.0m.
    """
    for k in HEIGHT_KEYS:
        if k in props and props[k] is not None:
            v = props[k]
            # numeric value
            if isinstance(v, (int, float)):
                return float(v) * 3.0 if k == "levels" else float(v)
            # string, possibly with "m"
            if isinstance(v, str):
                s = v.strip().lower().replace("m", "")
                try:
                    val = float(s)
                    return val * 3.0 if k == "levels" else val
                except Exception:
                    continue
    raise ValueError("No valid height field found in attributes.")


def iter_exterior_edges(poly: Polygon):
    coords = list(poly.exterior.coords)
    for i in range(len(coords) - 1):
        yield (coords[i][0:2], coords[i + 1][0:2])


def place_doors_on_edge(p0, p1, is_ccw: bool) -> List[List[Tuple[float, float, float]]]:
    """
    Place doors on a single exterior wall edge.
    Door Z range is [0, DOOR_H].
    Returns a list of quads, each quad is 4 vertices (x, y, z).
    """
    u, L = edge_unit_vector(p0, p1)
    if L < DOOR_MIN_EDGE_LENGTH:
        return []

    ux, uy = u
    nx, ny = outward_normal(u, is_ccw)

    pitch = DOOR_W + DOOR_GAP
    s0 = DOOR_EDGE_GAP + DOOR_W / 2.0
    centers_s = []
    s = s0
    while s <= L - (DOOR_EDGE_GAP + DOOR_W / 2.0) + 1e-8:
        centers_s.append(s)
        s += pitch
    if not centers_s:
        return []

    wx = DOOR_W / 2.0
    wz = DOOR_H / 2.0
    zc = wz  # center of the door in [0, DOOR_H]

    doors = []
    for s in centers_s:
        cx = p0[0] + ux * s + EPSILON_OUT * nx
        cy = p0[1] + uy * s + EPSILON_OUT * ny
        v0 = (cx - ux * wx, cy - uy * wx, zc - wz)  # bottom-left
        v1 = (cx + ux * wx, cy + uy * wx, zc - wz)  # bottom-right
        v2 = (cx + ux * wx, cy + uy * wx, zc + wz)  # top-right
        v3 = (cx - ux * wx, cy - uy * wx, zc + wz)  # top-left
        doors.append([v0, v1, v2, v3])
    return doors


def place_windows_on_edge(
    p0,
    p1,
    H: float,
    is_ccw: bool,
) -> List[List[Tuple[float, float, float]]]:
    """
    Place windows on a single exterior wall edge:
    - only up to HEIGHT_RATIO_MAX * H
    - keep vertical clearance from doors.
    """
    u, L = edge_unit_vector(p0, p1)
    if L < max(MIN_EDGE_LENGTH_FOR_WINDOWS, WINDOW_W + 2 * EDGE_GAP):
        return []

    ux, uy = u
    nx, ny = outward_normal(u, is_ccw)

    # horizontal placement
    pitch_h = WINDOW_W + H_GAP
    s0 = EDGE_GAP + WINDOW_W / 2.0
    centers_s = []
    s = s0
    while s <= L - (EDGE_GAP + WINDOW_W / 2.0) + 1e-8:
        centers_s.append(s)
        s += pitch_h
    if not centers_s:
        return []

    # vertical range and clearance: window bottom >= max(SILL_Z, DOOR_H + MIN_VERTICAL_CLEAR)
    z_bottom_min = max(SILL_Z, DOOR_H + MIN_VERTICAL_CLEAR)
    z_top_limit = HEIGHT_RATIO_MAX * H
    pitch_v = WINDOW_H + V_GAP

    centers_z = []
    zc = z_bottom_min + WINDOW_H / 2.0
    while zc + WINDOW_H / 2.0 <= z_top_limit + 1e-8:
        centers_z.append(zc)
        zc += pitch_v
    if not centers_z:
        return []

    wx = WINDOW_W / 2.0
    wz = WINDOW_H / 2.0

    windows = []
    for s in centers_s:
        cx = p0[0] + ux * s + EPSILON_OUT * nx
        cy = p0[1] + uy * s + EPSILON_OUT * ny
        for zc in centers_z:
            v0 = (cx - ux * wx, cy - uy * wx, zc - wz)
            v1 = (cx + ux * wx, cy + uy * wx, zc - wz)
            v2 = (cx + ux * wx, cy + uy * wx, zc + wz)
            v3 = (cx - ux * wx, cy - uy * wx, zc + wz)
            windows.append([v0, v1, v2, v3])
    return windows


def generate_door_window_meshes_from_footprints(
    gdf: gpd.GeoDataFrame,
) -> Tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh]]:
    """
    Generate door and window meshes directly from a building footprint GeoDataFrame.

    Input:
        gdf: must contain 'geometry' (Polygon/MultiPolygon) and at least one height-like attribute.
             The earlier footprint extraction function already creates 'height' column.

    Returns:
        door_mesh:   Trimesh or None
        window_mesh: Trimesh or None
    """
    print("[Step 2] Generating doors and windows from footprints...")

    door_vertices: List[Tuple[float, float, float]] = []
    door_faces: List[Tuple[int, int, int]] = []

    window_vertices: List[Tuple[float, float, float]] = []
    window_faces: List[Tuple[int, int, int]] = []

    n_bldg = 0
    n_door = 0
    n_window = 0

    geom_col = gdf.geometry.name  # usually "geometry"

    for _, row in gdf.iterrows():
        geom = row[geom_col]
        if geom is None or geom.is_empty:
            continue

        props = row.drop(labels=[geom_col]).to_dict()

        # Ensure 'height' exists for compatibility with detect_height
        if "height" not in props and "Height" not in props:
            if "height" in row:
                props["height"] = row["height"]

        try:
            H = detect_height(props)
        except Exception:
            # skip buildings without height information
            continue

        polys: List[Polygon] = []
        if isinstance(geom, Polygon):
            polys.append(geom)
        elif isinstance(geom, MultiPolygon):
            polys.extend(
                [
                    g
                    for g in geom.geoms
                    if isinstance(g, Polygon) and not g.is_empty
                ]
            )
        else:
            continue

        if not polys:
            continue

        n_bldg += 1

        for poly in polys:
            is_ccw = polygon_orientation(poly.exterior.coords) > 0.0

            for p0, p1 in iter_exterior_edges(poly):
                # Place doors along the edge
                door_quads = place_doors_on_edge(p0, p1, is_ccw)
                for quad in door_quads:
                    base_idx = len(door_vertices)
                    door_vertices.extend(quad)
                    door_faces.append((base_idx + 0, base_idx + 1, base_idx + 2))
                    door_faces.append((base_idx + 0, base_idx + 2, base_idx + 3))
                    n_door += 1

                # Place windows along the edge
                window_quads = place_windows_on_edge(p0, p1, H, is_ccw)
                for quad in window_quads:
                    base_idx = len(window_vertices)
                    window_vertices.extend(quad)
                    window_faces.append((base_idx + 0, base_idx + 1, base_idx + 2))
                    window_faces.append((base_idx + 0, base_idx + 2, base_idx + 3))
                    n_window += 1

    print(
        f"  Buildings used for openings: {n_bldg}, "
        f"doors: {n_door}, windows: {n_window}"
    )

    door_mesh = None
    window_mesh = None

    if door_vertices and door_faces:
        door_mesh = trimesh.Trimesh(
            vertices=np.array(door_vertices, dtype=float),
            faces=np.array(door_faces, dtype=int),
            process=False,
        )
        door_mesh.metadata = {"name": "doors"}

    if window_vertices and window_faces:
        window_mesh = trimesh.Trimesh(
            vertices=np.array(window_vertices, dtype=float),
            faces=np.array(window_faces, dtype=int),
            process=False,
        )
        window_mesh.metadata = {"name": "windows"}

    return door_mesh, window_mesh


# ============================================================
# 3. Rebuild building mesh per block, snap each building to ground
# ============================================================

def build_building_mesh_from_blocks(obj_path: str) -> trimesh.Trimesh:
    """
    Build one combined building mesh from v/f blocks.
    For each building block:
        - Shift its vertices so that its OWN z_min becomes 0.
    This ensures each building sits on the ground independently.

    Returns:
        Trimesh with all buildings, z_min = 0 per building.
    """
    print("[Step 3.1] Parsing LOD2 OBJ into building blocks for geometry...")
    buildings = parse_multi_building_obj(obj_path)
    print(f"  Found {len(buildings)} building blocks in OBJ (for geometry).")

    all_verts = []
    all_faces = []
    offset = 0
    n_valid = 0

    for verts, faces in buildings:
        if len(verts) == 0 or len(faces) == 0:
            continue
        verts_shifted = verts.copy()
        zmin = float(np.min(verts_shifted[:, 2]))
        verts_shifted[:, 2] -= zmin  # snap this building to ground (z_min = 0)
        all_verts.append(verts_shifted)
        all_faces.append(faces + offset)
        offset += verts_shifted.shape[0]
        n_valid += 1

    if not all_verts:
        raise RuntimeError("No valid building geometry blocks found in OBJ.")

    verts_combined = np.vstack(all_verts)
    faces_combined = np.vstack(all_faces)

    mesh = trimesh.Trimesh(
        vertices=verts_combined,
        faces=faces_combined,
        process=False,
    )
    mesh.metadata = {"name": "buildings"}

    zmin_global = float(mesh.bounds[0, 2])
    zmax_global = float(mesh.bounds[1, 2])
    print(
        f"  Combined building mesh built: "
        f"{n_valid} buildings, global z_min={zmin_global:.3f}, z_max={zmax_global:.3f}"
    )

    return mesh


# ============================================================
# 4. Apply shared palette material to building/doors/windows
# ============================================================

def apply_palette_to_meshes(
    building_mesh: trimesh.Trimesh,
    door_mesh: Optional[trimesh.Trimesh],
    window_mesh: Optional[trimesh.Trimesh],
):
    """
    Build a single palette PNG + MTL and assign different UVs for
    building, doors and windows.
    """
    print("[Step 4] Building palette texture and assigning materials...")

    keys = []
    if building_mesh is not None:
        keys.append("building")
    if door_mesh is not None:
        keys.append("door")
    if window_mesh is not None:
        keys.append("window")

    color_map = {
        "building": COLOR_BUILDING,
        "door": COLOR_DOORS,
        "window": COLOR_WINDOWS,
    }

    im, mat, uv_lookup = _build_palette_material(
        keys=keys,
        color_map=color_map,
        default_rgba=(253, 253, 230, 255),
        tex_h=8,
    )

    if building_mesh is not None:
        _apply_single_material(
            building_mesh,
            im,
            mat,
            uv_const=uv_lookup["building"],
        )

    if door_mesh is not None:
        _apply_single_material(
            door_mesh,
            im,
            mat,
            uv_const=uv_lookup["door"],
        )

    if window_mesh is not None:
        _apply_single_material(
            window_mesh,
            im,
            mat,
            uv_const=uv_lookup["window"],
        )

    print("  Palette material assigned to building, doors and windows.")


# ============================================================
# 5. Merge and export
# ============================================================

def merge_and_export(
    building_mesh: trimesh.Trimesh,
    door_mesh: Optional[trimesh.Trimesh],
    window_mesh: Optional[trimesh.Trimesh],
    out_obj_path: str,
):
    """
    Merge building mesh with door and window meshes and export as OBJ (+MTL+PNG).
    """
    print("[Step 5] Merging meshes and exporting OBJ + MTL + PNG...")

    parts = []
    if building_mesh is not None:
        parts.append(building_mesh)
    if door_mesh is not None:
        parts.append(door_mesh)
    if window_mesh is not None:
        parts.append(window_mesh)

    if not parts:
        raise RuntimeError("No geometry to export (all meshes are None).")

    scene = trimesh.Scene()
    for m in parts:
        name = (m.metadata or {}).get("name", "mesh")
        scene.add_geometry(m, node_name=name, geom_name=name)

    out_path = Path(out_obj_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # This will write:
    #   - out_obj_path (OBJ)
    #   - out_obj_path with .mtl extension (MTL)
    #   - one PNG file for the palette material (typically palette_mat.png)
    scene.export(str(out_path))
    print(f"  Exported merged OBJ (+MTL+PNG): {out_path}")


# ============================================================
# 6. High-level pipeline function
# ============================================================

def generate_lod2_with_doors_and_windows(
    lod2_obj_path: str,
    out_obj_path: str,
    *,
    src_epsg: str = "EPSG:30169",
    footprint_shp_path: Optional[str] = None,
):
    """
    High-level convenience function:

    Input:
        lod2_obj_path: path to input LOD2 OBJ (multi-building; each block is v... then f...).
        out_obj_path:  path to output colored OBJ with doors and windows.
        src_epsg:      CRS of OBJ coordinates for footprint GeoDataFrame.
        footprint_shp_path: optional; if given, footprints will be saved as a Shapefile.

    Steps:
        1) Extract footprints (GeoDataFrame) from LOD2 OBJ.
        2) Generate door/window meshes from footprints.
        3) Build LOD2 mesh from v/f blocks, shifting EACH building so that its own z_min = 0.
        4) Apply palette texture to building + doors + windows.
        5) Merge and export final OBJ (+MTL+PNG).
    """
    print("========== LOD2 → LOD2+doors+windows pipeline started ==========")

    # Step 1: footprints
    gdf = extract_footprints_gdf_from_lod2(
        lod2_obj_path,
        src_epsg=src_epsg,
        save_shapefile=footprint_shp_path,
    )

    # Step 2: doors & windows
    door_mesh, window_mesh = generate_door_window_meshes_from_footprints(gdf)

    # Step 3: build building mesh from v/f blocks with per-building z_min=0
    building_mesh = build_building_mesh_from_blocks(lod2_obj_path)

    # Step 4: apply shared palette material
    apply_palette_to_meshes(building_mesh, door_mesh, window_mesh)

    # Step 5: merge and export
    merge_and_export(building_mesh, door_mesh, window_mesh, out_obj_path)

    print("========== Pipeline finished successfully ==========")


# ============================================================
# Example usage (modify paths as needed)
# ============================================================

# if __name__ == "__main__":
#     # Example paths (replace with your actual data paths)
#     LOD2_OBJ_PATH = r"C:\zcb\data\project\plateau\draft\yokohama_obj_20250806\LoD2\merged_model_53391530_lod2.obj"
#     OUTPUT_OBJ_PATH = r"C:\zcb\code\python\gen_bg\draft\building\merged_lod2_doors_windows.obj"
#     # FOOTPRINT_SHP_PATH = r"C:\zcb\code\python\gen_bg\draft\building\footprints_30169.shp"
#
#     generate_lod2_with_doors_and_windows(
#         lod2_obj_path=LOD2_OBJ_PATH,
#         out_obj_path=OUTPUT_OBJ_PATH,
#         src_epsg="EPSG:30169",
#         footprint_shp_path=None,  # or None if you do not want shp
#     )
