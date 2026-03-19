# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, glob, uuid, math, random, warnings, shutil
from typing import Iterable, Union, Literal, Optional

import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import box, LineString, MultiLineString, Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from shapely.errors import TopologicalError
from pyproj import Transformer
from lxml import etree
from PIL import Image, ImageDraw

import osmnx as ox
import trimesh
import rasterio
from rasterio.warp import transform_bounds

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Module directory for relative paths
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_MODULE_DIR, "data")

GeoLike = Union[LineString, MultiLineString, Polygon, MultiPolygon]
LODType = Literal[1, 2, 3]


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _sanitize_str(x, fallback="unknown"):
    """Return a filesystem/ID safe string; map NaN/None to fallback."""
    if x is None:
        return fallback
    try:
        if isinstance(x, float) and x != x:  # NaN
            return fallback
    except Exception:
        pass
    s = str(x) or fallback
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", s) or fallback


def _fmt_poslist(coords, precision=3):
    """(N,3) → 'x y z x y z ...' with given precision."""
    a = np.asarray(coords, dtype=float).reshape(-1)
    fmt = f"%.{precision}f"
    return " ".join(fmt % v for v in a)


# -----------------------------------------------------------------------------
# OSM roads fetch (bbox)
# -----------------------------------------------------------------------------
def crop_osm_roads_online(
        x_min, y_min, x_max, y_max,
        *,
        input_crs="EPSG:4326",
        highway_tags=None,
) -> gpd.GeoDataFrame:
    """
    Fetch OSM/Overpass features within bbox as linework and clip to bbox.
    Result is projected to EPSG:30169 (meters) for later processing.
    """
    # 1) Transform bbox to WGS84
    if input_crs and input_crs != "EPSG:4326":
        transformer = Transformer.from_crs(input_crs, "EPSG:4326", always_xy=True)
        xs = [x_min, x_min, x_max, x_max]
        ys = [y_min, y_max, y_min, y_max]
        lon, lat = transformer.transform(xs, ys)
        west, east = float(min(lon)), float(max(lon))
        south, north = float(min(lat)), float(max(lat))
    else:
        west, south, east, north = map(float, (x_min, y_min, x_max, y_max))
        if east < west:
            west, east = east, west
        if north < south:
            south, north = north, south

    # 2) OSMnx config
    ox.settings.use_cache = False
    ox.settings.log_console = False
    if highway_tags is None:
        highway_tags = {"highway": True}

    # 3) Query by polygon (more robust than bbox call variants)
    bbox_poly = box(west, south, east, north)
    if bbox_poly.is_empty or not bbox_poly.is_valid or bbox_poly.area == 0.0:
        raise ValueError("Invalid bbox polygon: check coordinates and CRS.")

    gdf = ox.features_from_polygon(bbox_poly, tags=highway_tags)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    # 4) Keep only line features, non-empty
    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_crs("EPSG:30169")

    # 5) Clip and project to EPSG:30169
    try:
        gdf = gpd.clip(gdf, bbox_poly)
    except TopologicalError:
        gdf = gdf.assign(geometry=gdf.buffer(0.0))
        gdf = gpd.clip(gdf, bbox_poly)
    return gdf.to_crs("EPSG:30169")


# -----------------------------------------------------------------------------
# Road → Scene (one palette material for all classes)
# -----------------------------------------------------------------------------
def _extrude_polygon_robust(poly: Polygon, h: float) -> Optional[trimesh.Trimesh]:
    """Extrude a polygon; try buffer(0) on failure."""
    if poly.is_empty:
        return None
    try:
        return trimesh.creation.extrude_polygon(poly, height=h)
    except Exception:
        try:
            return trimesh.creation.extrude_polygon(poly.buffer(0.0), height=h)
        except Exception:
            return None


def _build_palette_material(keys, color_map, default_rgba=(253, 253, 230, 255), tex_h=8):
    """
    Build a 1-row palette image (N blocks) + single SimpleMaterial.
    Returns (image, material, uv_lookup: {key -> (u,v)}).
    """
    keys = list(keys)
    N = max(1, len(keys))
    W, H = N * tex_h, tex_h

    im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    for i, k in enumerate(keys):
        rgba = color_map.get(k, default_rgba)
        draw.rectangle([i * tex_h, 0, (i + 1) * tex_h - 1, H - 1], fill=tuple(rgba))

    mat = trimesh.visual.texture.SimpleMaterial(image=im)
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


def _apply_single_material(mesh: trimesh.Trimesh, im, material, uv_const=(0.5, 0.5)):
    """Assign one material + constant UV to a mesh."""
    uv = np.tile(np.asarray(uv_const, dtype=float), (len(mesh.vertices), 1))
    vis = trimesh.visual.texture.TextureVisuals(uv=uv, image=im, material=material)
    mesh.visual = vis
    return mesh


def roads_to_scene_with_attrs(
        roads_gdf: gpd.GeoDataFrame,
        *,
        carriage_half_width: float = 3.5,
        thickness: float = 0.06,
        color_map: dict | None = None,
        default_rgba=(253, 253, 230, 255),
        tex_h: int = 8,
        z_offset_map: dict[str, float] | None = None,  # e.g. {'motorway':0.015, 'trunk':0.012, ...} in meters
        z_offset_default: float = 0.0,
        width_map: dict[str, float] | None = None,     # NEW: per-class half-width (meters)
):
    """
    Convert road lines to a Scene with one palette material and per-road attributes.
    Returns (scene, node_attr_map).
    """
    gdf = roads_gdf
    # gather highway classes
    hw_set = set()
    for _, row in gdf.iterrows():
        hw_raw = row.get("highway")
        if isinstance(hw_raw, (list, tuple, set)):
            hw_raw = next(iter(hw_raw), None)
        hw_set.add(_sanitize_str(hw_raw, "unknown"))

    im_palette, single_mat, uv_lookup = _build_palette_material(
        hw_set, color_map or {}, default_rgba=default_rgba, tex_h=tex_h
    )

    scene = trimesh.Scene()
    node_attr_map = {}

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # ---- per-class half width ----
        hw_raw = row.get("highway")
        if isinstance(hw_raw, (list, tuple, set)):
            hw_raw = next(iter(hw_raw), None)
        hw = _sanitize_str(hw_raw, "unknown")
        this_half_width = float(width_map.get(hw, carriage_half_width)) if width_map else float(carriage_half_width)
        # ------------------------------

        polys = []
        if isinstance(geom, LineString):
            polys.append(geom.buffer(this_half_width, join_style=2))
        elif isinstance(geom, MultiLineString):
            for seg in geom.geoms:
                polys.append(seg.buffer(this_half_width, join_style=2))
        elif isinstance(geom, (Polygon, MultiPolygon)):
            polys.extend(list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom])
        else:
            continue

        merged = unary_union([p for p in polys if p and not p.is_empty])
        if merged.is_empty:
            continue

        meshes = []
        if isinstance(merged, Polygon):
            merged = MultiPolygon([merged])

        for poly in merged.geoms:
            poly = shapely.make_valid(poly)
            if poly.is_empty:
                continue
            m = _extrude_polygon_robust(poly, thickness)
            if m is not None:
                meshes.append(m)
        if not meshes:
            continue

        feat_mesh = trimesh.util.concatenate(meshes)

        uv_const = uv_lookup.get(hw, (0.5, 0.5))
        feat_mesh = _apply_single_material(feat_mesh, im_palette, single_mat, uv_const=uv_const)

        z_off = (z_offset_map or {}).get(hw, z_offset_default)
        if abs(z_off) > 0:
            T = np.eye(4)
            T[2, 3] = float(z_off)
            feat_mesh.apply_transform(T)

        osmid_raw = row.get("osmid") or row.get("id") or row.get("@id") or uuid.uuid4().hex[:8]
        osmid = _sanitize_str(osmid_raw, "noid")
        node_name = f"road_{hw}_{osmid}"

        scene.add_geometry({node_name: feat_mesh})

        node_attr_map[node_name] = {
            "highway": hw,
            "name": row.get("name"),
            "osmid": osmid if osmid != "noid" else None,
            "length_m": float(row.get("length", 0)) if "length" in gdf.columns else None,
            "thickness_m": float(thickness),
            "carriage_half_width_m": float(this_half_width),
        }

    return scene, node_attr_map



# -----------------------------------------------------------------------------
# City furniture sampling and placement
# -----------------------------------------------------------------------------
def _iter_lines(geoms) -> tuple[list[LineString], Optional[str]]:
    """Normalize to LineString list."""
    if hasattr(geoms, "geometry"):  # GeoDataFrame
        iterable = geoms.geometry
        crs = geoms.crs
    elif isinstance(geoms, gpd.GeoSeries):
        iterable = geoms
        crs = geoms.crs
    else:
        iterable = geoms
        crs = None

    lines = []
    for g in iterable:
        if g is None or g.is_empty:
            continue
        if isinstance(g, LineString):
            lines.append(g)
        elif isinstance(g, MultiLineString):
            lines.extend(list(g.geoms))
    return lines, crs


def _heading_from_line(line: LineString, dist: float, eps: float = 0.5) -> float:
    """Tangent heading (deg) at arc-length dist, fallback to backward diff."""
    L = line.length
    d2 = min(dist + eps, L)
    p1 = np.asarray(line.interpolate(dist).coords[0])
    p2 = np.asarray(line.interpolate(d2).coords[0])
    v = p2 - p1
    if np.allclose(v[:2], 0.0):
        d0 = max(dist - eps, 0.0)
        p0 = np.asarray(line.interpolate(d0).coords[0])
        v = p1 - p0
    return float(math.degrees(math.atan2(v[1], v[0])))


def _sample_along_offset(line: LineString, offset: float, spacing: float, side: str, start_shift: float):
    """Sample points on a parallel offset line with equal spacing."""
    try:
        off = line.parallel_offset(offset, side=side, join_style=2, resolution=16)
    except Exception:
        return []
    segs = [off] if isinstance(off, LineString) else (list(off.geoms) if isinstance(off, MultiLineString) else [])
    pts = []
    for seg in segs:
        L = seg.length
        if L <= 0:
            continue
        dists = np.arange(start_shift, L, spacing)
        pts.extend([seg.interpolate(float(d)) for d in dists])
    return pts


def sample_city_furniture_points(
        geoms,
        *,
        carriage_half_width: float = 3.5,
        spacing_map: dict | None = None,
        side_map: dict | None = None,
        jitter_seed: int | None = 42,
        output_shp: str | None = None,
        width_map: dict[str, float] | None = None,   # NEW: per-highway half-width
        highway_field: str = "highway",              # NEW: highway column name
) -> gpd.GeoDataFrame:
    """
    Sample 3 types of city furniture along left/right offsets of road centerlines.
    Returns GeoDataFrame with columns: geometry, kind, side, heading, road_id.

    If width_map is provided and geoms is a GeoDataFrame with `highway_field`,
    each road uses its own half-width; otherwise falls back to `carriage_half_width`.
    """
    # default spacing & side config
    if spacing_map is None:
        spacing_map = {
            "electric_pole": 250.0,
            "street_light": 350.0,
            "traffic_light": 1100.0,
        }
    if side_map is None:
        side_map = {
            "electric_pole": "both",
            "street_light": "both",
            "traffic_light": "right",
        }

    # RNG for jitter
    rng = np.random.default_rng(jitter_seed) if jitter_seed is not None else None

    records = []

    # ------------------------------------------------------------------
    # Case 1: use per-road widths (GeoDataFrame with highway + width_map)
    # ------------------------------------------------------------------
    use_per_road_width = (
        isinstance(geoms, gpd.GeoDataFrame)
        and width_map is not None
        and highway_field in geoms.columns
    )

    if use_per_road_width:
        crs = geoms.crs
        for road_idx, row in geoms.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # determine half-width for this road
            hw_raw = row.get(highway_field)
            if isinstance(hw_raw, (list, tuple, set)):
                hw_raw = next(iter(hw_raw), None)
            hw_key = _sanitize_str(hw_raw, "unknown")
            this_half_width = float(width_map.get(hw_key, carriage_half_width))

            # normalize geometry to list of LineString
            if isinstance(geom, LineString):
                lines = [geom]
            elif isinstance(geom, MultiLineString):
                lines = list(geom.geoms)
            else:
                # ignore polygons or other geometries here
                continue

            # per-road/per-kind sampling
            for ln in lines:
                L = ln.length
                if L <= 0:
                    continue

                # pre-sample points along the centerline for nearest-point lookup
                nseg = max(50, int(L / 2.0))
                ds = np.linspace(0, L, nseg)
                coords = np.asarray([ln.interpolate(float(di)).coords[0] for di in ds])

                # per-kind starting jitter to avoid exact overlaps
                jitter = {
                    k: (float(rng.uniform(0.25 * sp, 0.75 * sp)) if rng is not None else sp / 2.0)
                    for k, sp in spacing_map.items()
                }

                for kind, spacing in spacing_map.items():
                    side_pref = side_map.get(kind, "both")
                    sides = ["left", "right"] if side_pref == "both" else [side_pref]

                    for side in sides:
                        pts = _sample_along_offset(
                            ln,
                            offset=this_half_width,             # << use per-road width
                            spacing=float(spacing),
                            side=side,
                            start_shift=jitter[kind],
                        )
                        for p in pts:
                            x, y = p.x, p.y
                            # find nearest centerline sample for heading
                            dists = np.hypot(coords[:, 0] - x, coords[:, 1] - y)
                            kmin = int(np.argmin(dists))
                            d_est = float(ds[kmin])
                            heading_deg = _heading_from_line(ln, d_est, eps=0.5)

                            records.append({
                                "geometry": Point(x, y),
                                "kind": kind,
                                "side": side,
                                "heading": heading_deg,
                                "road_id": road_idx,
                            })

    # ------------------------------------------------------------------
    # Case 2: fallback — original behavior using a single carriage_half_width
    # ------------------------------------------------------------------
    else:
        lines, crs = _iter_lines(geoms)
        if not lines:
            raise ValueError("No LineString/MultiLineString features found.")

        for idx, ln in enumerate(lines):
            L = ln.length
            if L <= 0:
                continue

            # per-kind starting jitter to avoid exact overlaps
            jitter = {
                k: (float(rng.uniform(0.25 * sp, 0.75 * sp)) if rng is not None else sp / 2.0)
                for k, sp in spacing_map.items()
            }

            nseg = max(50, int(L / 2.0))
            ds = np.linspace(0, L, nseg)
            coords = np.asarray([ln.interpolate(float(di)).coords[0] for di in ds])

            for kind, spacing in spacing_map.items():
                side_pref = side_map.get(kind, "both")
                sides = ["left", "right"] if side_pref == "both" else [side_pref]
                for side in sides:
                    pts = _sample_along_offset(
                        ln,
                        offset=carriage_half_width,           # << global width as before
                        spacing=float(spacing),
                        side=side,
                        start_shift=jitter[kind],
                    )
                    for p in pts:
                        x, y = p.x, p.y
                        dists = np.hypot(coords[:, 0] - x, coords[:, 1] - y)
                        kmin = int(np.argmin(dists))
                        d_est = float(ds[kmin])
                        heading_deg = _heading_from_line(ln, d_est, eps=0.5)

                        records.append({
                            "geometry": Point(x, y),
                            "kind": kind,
                            "side": side,
                            "heading": heading_deg,
                            "road_id": idx,
                        })

    # build GeoDataFrame result
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs if crs else "EPSG:30169")
    if output_shp:
        gdf.to_file(output_shp, driver="ESRI Shapefile", encoding="utf-8")
    return gdf


# -----------------------------------------------------------------------------
# Furniture placement from points
# -----------------------------------------------------------------------------
def _load_mesh_bank(base_dir: str) -> list[trimesh.Trimesh]:
    paths = []
    for ext in ("*.obj", "*.OBJ"):
        paths.extend(glob.glob(os.path.join(base_dir, "**", ext), recursive=True))
    bank = []
    for p in paths:
        try:
            m = trimesh.load(p, force='mesh')
            if isinstance(m, trimesh.Scene):
                m = trimesh.util.concatenate([g for g in m.geometry.values()])
            if isinstance(m, trimesh.Trimesh) and m.vertices.size and m.faces.size:
                m.remove_unreferenced_vertices()
                bank.append(m)
        except Exception:
            continue
    return bank


def _normalize_to_origin(mesh: trimesh.Trimesh, anchor_xy_mode: str = "base_center") -> trimesh.Trimesh:
    """
    Normalize a mesh so that zmin→0, and chosen XY anchor → (0,0).
    anchor_xy_mode: 'base_center' | 'centroid_xy' | 'bounds_min_xy' | 'origin'
    """
    m = mesh.copy()

    # ground to Z=0
    zmin = m.bounds[0][2]
    if abs(zmin) > 1e-12:
        Tz = np.eye(4);
        Tz[2, 3] = -zmin
        m.apply_transform(Tz)

    # XY anchor
    if anchor_xy_mode == "origin":
        ax, ay = 0.0, 0.0
    elif anchor_xy_mode == "centroid_xy":
        c = m.center_mass;
        ax, ay = float(c[0]), float(c[1])
    elif anchor_xy_mode == "bounds_min_xy":
        ax, ay = float(m.bounds[0][0]), float(m.bounds[0][1])
    else:
        V = m.vertices
        eps = max(1e-6, 1e-4 * (m.bounds[1][2] + 1.0))
        base_mask = np.abs(V[:, 2]) <= eps
        if base_mask.any():
            ax = float(V[base_mask, 0].mean())
            ay = float(V[base_mask, 1].mean())
        else:
            c = m.center_mass;
            ax, ay = float(c[0]), float(c[1])

    if abs(ax) > 1e-12 or abs(ay) > 1e-12:
        Txy0 = np.eye(4);
        Txy0[0, 3] = -ax;
        Txy0[1, 3] = -ay
        m.apply_transform(Txy0)
    return m


def _make_solid_material(rgba=(240, 128, 128, 255), tex_size=8):
    """Create a single-color SimpleMaterial with a tiny RGBA texture."""
    im = Image.new('RGBA', (tex_size, tex_size), rgba)
    mat = trimesh.visual.texture.SimpleMaterial(image=im)
    return im, mat


def _colorize_mesh(mesh: trimesh.Trimesh, rgba=(240, 128, 128, 255)):
    """Apply a solid color material to a mesh."""
    im, mat = _make_solid_material(rgba)
    uv = np.zeros((len(mesh.vertices), 2), dtype=float)
    vis = trimesh.visual.texture.TextureVisuals(uv=uv, image=im, material=mat)
    mesh.visual = vis
    return mesh


def export_furniture_from_points(
        points_gdf: gpd.GeoDataFrame,
        *,
        base_dir_map: dict | None = None,
        kind_field: str = "kind",
        heading_field: str = "heading",
        use_heading: bool = True,
        random_seed: int | None = 42,
        heading_offset_deg: dict | None = None,
        anchor_xy_mode: str = "base_center",
        color_rgba: tuple[int, int, int, int] | None = (240, 128, 128, 255),
) -> list[trimesh.Trimesh]:
    """
    Place furniture meshes on given points.
    Returns a list of placed Trimesh instances (not merged).
    """
    if base_dir_map is None:
        base_dir_map = {
            "electric_pole": os.path.join(_DATA_DIR, "city_furniture", "electric_pole"),
            "street_light": os.path.join(_DATA_DIR, "city_furniture", "street_light"),
            "traffic_light": os.path.join(_DATA_DIR, "city_furniture", "traffic_light")
        }
    if heading_offset_deg is None:
        heading_offset_deg = {}

    if not {"geometry", kind_field}.issubset(points_gdf.columns):
        raise ValueError(f"points_gdf must contain geometry and {kind_field}")
    if not all(points_gdf.geometry.geom_type == "Point"):
        raise ValueError("points_gdf.geometry must be all Point")

    # load & pre-normalize banks
    banks = {}
    for cls, d in base_dir_map.items():
        bank = _load_mesh_bank(d)
        if not bank:
            raise FileNotFoundError(f"No OBJ found in: {d}")
        banks[cls] = [_normalize_to_origin(m, anchor_xy_mode=anchor_xy_mode) for m in bank]

    rng = random.Random(random_seed)
    instances = []
    for _, row in points_gdf.iterrows():
        kind = row[kind_field]
        if kind not in banks:
            continue
        m = rng.choice(banks[kind]).copy()

        if use_heading and (heading_field in points_gdf.columns):
            hd = float(row[heading_field]) + float(heading_offset_deg.get(kind, 0.0))
            th = math.radians(hd)
            cz, sz = math.cos(th), math.sin(th)
            Rz = np.array([[cz, -sz, 0, 0],
                           [sz, cz, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
            m.apply_transform(Rz)

        x, y = row.geometry.x, row.geometry.y
        Txy = np.eye(4);
        Txy[0, 3] = x;
        Txy[1, 3] = y
        m.apply_transform(Txy)

        if color_rgba is not None:
            _colorize_mesh(m, color_rgba)

        instances.append(m)

    if not instances:
        raise RuntimeError("No instances placed. Check kinds/points.")
    return instances


def instances_cylinders_from_points(
        points_gdf: gpd.GeoDataFrame,
        *,
        diameter_m: float = 0.30,
        height_m: float = 3.0,
        segments: int = 24,
        color_rgba: tuple[int, int, int, int] | None = (240, 128, 128, 255),
) -> list[trimesh.Trimesh]:
    """
    Create one vertical cylinder per point.
    - Cylinder is centered at Z=0 by trimesh; we shift it up by +height/2 so base lies on Z=0.
    - Heading is ignored (cylindrical symmetry).
    """
    if points_gdf is None or len(points_gdf) == 0:
        return []

    radius = float(diameter_m) / 2.0
    base = trimesh.creation.cylinder(radius=radius, height=float(height_m), sections=int(segments))
    # lift so base sits on Z=0
    Tz = np.eye(4);
    Tz[2, 3] = height_m / 2.0
    base = base.copy()
    base.apply_transform(Tz)

    instances = []
    for _, row in points_gdf.iterrows():
        x, y = row.geometry.x, row.geometry.y
        m = base.copy()
        Txy = np.eye(4);
        Txy[0, 3] = x;
        Txy[1, 3] = y
        m.apply_transform(Txy)
        if color_rgba is not None:
            # solid color
            im = Image.new('RGBA', (8, 8), color_rgba)
            mat = trimesh.visual.texture.SimpleMaterial(image=im)
            uv = np.zeros((len(m.vertices), 2), dtype=float)
            m.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=im, material=mat)
        instances.append(m)
    return instances


# -----------------------------------------------------------------------------
# Unified CityGML (3.0-style) writer for roads & furniture
# -----------------------------------------------------------------------------
def _iter_meshes(geometry):
    """Yield (name, mesh) from Scene | dict[name->mesh] | list[mesh]."""
    if isinstance(geometry, trimesh.Scene):
        for name, mesh in geometry.geometry.items():
            yield name, mesh
    elif isinstance(geometry, dict):
        for name, mesh in geometry.items():
            yield str(name), mesh
    elif isinstance(geometry, (list, tuple)):
        for i, mesh in enumerate(geometry):
            yield f"mesh_{i}", mesh
    else:
        raise TypeError(f"Unsupported geometry type: {type(geometry)}")


def export_citygml(
        geometry,
        *,
        attr_map: dict | None = None,
        lod: LODType = 1,
        feature: Literal["road", "cityfurniture"] = "road",
        srs_name: str = "urn:ogc:def:crs:EPSG::30169",
        srs_dimension: str = "3",
        precision: int = 3,
) -> etree._Element:
    """
    Unified CityGML 3.0-style writer.
    - feature='road'  → tran:Road + tran:lod{n}MultiSurface
    - feature='cityfurniture' → frn:CityFurniture + frn:lod{n}Geometry/ MultiSurface
    Geometry is written as gml:MultiSurface of triangle polygons.
    """
    NS_CORE = "http://www.opengis.net/citygml/2.0"
    NS_TRAN = "http://www.opengis.net/citygml/transportation/2.0"
    NS_FRN = "http://www.opengis.net/citygml/cityfurniture/2.0"
    NS_GML = "http://www.opengis.net/gml"

    nsmap = {None: NS_CORE, 'tran': NS_TRAN, 'frn': NS_FRN, 'gml': NS_GML}
    cityModel = etree.Element(f"{{{NS_CORE}}}CityModel", nsmap=nsmap)

    # Compute bounds
    all_xyz = []
    items = list(_iter_meshes(geometry))
    for _, mesh in items:
        if isinstance(mesh, trimesh.Trimesh) and mesh.vertices.size:
            all_xyz.append(mesh.vertices)
    if all_xyz:
        V = np.vstack(all_xyz)
        (x_min, y_min, z_min) = np.min(V, axis=0)
        (x_max, y_max, z_max) = np.max(V, axis=0)
    else:
        x_min = y_min = z_min = 0.0
        x_max = y_max = z_max = 0.0

    boundedBy = etree.SubElement(cityModel, f"{{{NS_GML}}}boundedBy")
    envelope = etree.SubElement(
        boundedBy, f"{{{NS_GML}}}Envelope",
        srsName=srs_name, srsDimension=srs_dimension
    )
    lowerCorner = etree.SubElement(envelope, f"{{{NS_GML}}}lowerCorner")
    upperCorner = etree.SubElement(envelope, f"{{{NS_GML}}}upperCorner")
    lowerCorner.text = f"{x_min:.{precision}f} {y_min:.{precision}f} {z_min:.{precision}f}"
    upperCorner.text = f"{x_max:.{precision}f} {y_max:.{precision}f} {z_max:.{precision}f}"

    if lod not in (1, 2, 3):
        raise ValueError("lod must be 1, 2, or 3.")
    lod_tag = f"lod{lod}MultiSurface"

    for name, mesh in items:
        if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0 or mesh.faces.size == 0:
            continue

        member = etree.SubElement(cityModel, f"{{{NS_CORE}}}cityObjectMember")

        if feature == "road":
            elem = etree.SubElement(member, f"{{{NS_TRAN}}}Road")
            elem.set(f"{{{NS_GML}}}id", f"rid_{uuid.uuid4().hex[:12]}")
            gml_name = etree.SubElement(elem, f"{{{NS_GML}}}name")
            gml_name.text = str((attr_map or {}).get(name, {}).get("name", name))
            # function from highway class if available
            tran_function = etree.SubElement(elem, f"{{{NS_TRAN}}}function")
            tran_function.text = str((attr_map or {}).get(name, {}).get("highway", "road"))
            lod_container = etree.SubElement(elem, f"{{{NS_TRAN}}}{lod_tag}")
        else:
            elem = etree.SubElement(member, f"{{{NS_FRN}}}CityFurniture")
            elem.set(f"{{{NS_GML}}}id", f"cf_{uuid.uuid4().hex[:12]}")
            gml_name = etree.SubElement(elem, f"{{{NS_GML}}}name")
            gml_name.text = str(name)
            # CityFurniture in C3 often uses frn:lodXGeometry → gml:MultiSurface
            lod_geom = etree.SubElement(elem, f"{{{NS_FRN}}}lod{lod}Geometry")
            lod_container = lod_geom  # will contain gml:MultiSurface

        multiSurface = etree.SubElement(
            lod_container, f"{{{NS_GML}}}MultiSurface",
            srsName=srs_name, srsDimension=srs_dimension
        )
        V = mesh.vertices
        F = mesh.faces

        for face in F:
            idxs = np.asarray(face, dtype=int).ravel()
            ring = V[idxs]
            ring_closed = np.vstack([ring, ring[0:1, :]])

            surfaceMember = etree.SubElement(multiSurface, f"{{{NS_GML}}}surfaceMember")
            polygon = etree.SubElement(
                surfaceMember, f"{{{NS_GML}}}Polygon",
                srsName=srs_name, srsDimension=srs_dimension
            )
            exterior = etree.SubElement(polygon, f"{{{NS_GML}}}exterior")
            linearRing = etree.SubElement(exterior, f"{{{NS_GML}}}LinearRing")
            posList = etree.SubElement(linearRing, f"{{{NS_GML}}}posList")
            posList.text = _fmt_poslist(ring_closed, precision=precision)

    return cityModel


import shutil


def align_road_obj_mtl(
        obj_path: str,
        *,
        desired_mtl_filename: str = "road_palette.mtl",
        desired_material_name: str = "palette_mat",
        desired_texture_png: str = "palette_mat.png",
        verbose: bool = True,
):
    """
    After exporting an OBJ via trimesh, rewrite the MTL and OBJ so that:
    - OBJ: 'mtllib road_palette.mtl' and all 'usemtl palette_mat'
    - MTL: 'newmtl palette_mat' and 'map_Kd palette_mat.png'
    Also ensures the texture file exists (rename/copy if needed).
    """

    def _log(s):
        if verbose:
            print(s)

    obj_dir = os.path.dirname(obj_path)
    obj_base = os.path.basename(obj_path)

    # Read OBJ
    with open(obj_path, "r", encoding="utf-8") as f:
        obj_txt = f.read()

    # Find original mtllib
    m = re.search(r'^\s*mtllib\s+([^\r\n]+)', obj_txt, flags=re.IGNORECASE | re.MULTILINE)
    orig_mtl_filename = m.group(1).strip() if m else None
    orig_mtl_path = os.path.join(obj_dir, orig_mtl_filename) if orig_mtl_filename else None
    _log(f"[MTL] Original mtllib: {orig_mtl_filename}")

    # Read original MTL (if any)
    orig_material_name = None
    orig_texture_name = None
    mtl_txt = ""
    if orig_mtl_path and os.path.exists(orig_mtl_path):
        with open(orig_mtl_path, "r", encoding="utf-8") as f:
            mtl_txt = f.read()
        m2 = re.search(r'^\s*newmtl\s+([^\r\n]+)', mtl_txt, flags=re.IGNORECASE | re.MULTILINE)
        if m2:
            orig_material_name = m2.group(1).strip()
        m3 = re.search(r'^\s*map_Kd\s+([^\r\n]+)', mtl_txt, flags=re.IGNORECASE | re.MULTILINE)
        if m3:
            orig_texture_name = m3.group(1).strip()

    # If no MTL, create a minimal one
    if not mtl_txt:
        _log("[MTL] No original MTL found. Creating a new one.")
        mtl_txt = ""

    # Ensure texture file exists: if exporter wrote a different name, reuse it
    candidates = [orig_texture_name, desired_texture_png, "material_0.png", "material_1.png"]
    candidates = [c for c in candidates if c]
    texture_src = None
    for cand in candidates:
        p = os.path.join(obj_dir, cand)
        if os.path.exists(p):
            texture_src = p
            break

    # If palette_mat.png doesn't exist but another texture exists, copy/rename to desired
    desired_tex_path = os.path.join(obj_dir, desired_texture_png)
    if texture_src and os.path.abspath(texture_src) != os.path.abspath(desired_tex_path):
        shutil.copyfile(texture_src, desired_tex_path)
        _log(f"[MTL] Copied texture → {desired_texture_png}")
    elif not os.path.exists(desired_tex_path):
        _log("[WARN] Desired texture not found and no source to copy from. "
             "Exporter may not have written texture. Check export settings.")

    # Build NEW MTL text (single material)
    new_mtl_txt = [
        "# rewritten by align_road_obj_mtl",
        f"newmtl {desired_material_name}",
        "Ka 0.40000000 0.40000000 0.40000000",
        "Kd 0.40000000 0.40000000 0.40000000",
        "Ks 0.40000000 0.40000000 0.40000000",
        "Ns 1.00000000",
        f"map_Kd {desired_texture_png}",
        ""
    ]
    new_mtl_txt = "\n".join(new_mtl_txt)

    # Write NEW MTL file
    new_mtl_path = os.path.join(obj_dir, desired_mtl_filename)
    with open(new_mtl_path, "w", encoding="utf-8") as f:
        f.write(new_mtl_txt)
    _log(f"[MTL] Wrote new MTL → {desired_mtl_filename}")

    # Rewrite OBJ:
    # 1) mtllib line → desired_mtl_filename
    if orig_mtl_filename:
        obj_txt = re.sub(r'^\s*mtllib\s+[^\r\n]+',
                         f"mtllib {desired_mtl_filename}",
                         obj_txt, flags=re.IGNORECASE | re.MULTILINE)
    else:
        # insert at the top if missing
        obj_txt = f"mtllib {desired_mtl_filename}\n{obj_txt}"

    # 2) replace all 'usemtl <old>' to 'usemtl palette_mat' (or force-insert if none)
    if orig_material_name:
        obj_txt = re.sub(rf'^\s*usemtl\s+{re.escape(orig_material_name)}\s*$',
                         f"usemtl {desired_material_name}",
                         obj_txt, flags=re.IGNORECASE | re.MULTILINE)
    # If there was no usemtl or names vary, normalize all usemtl lines:
    obj_txt = re.sub(r'^\s*usemtl\s+[^\r\n]+',
                     f"usemtl {desired_material_name}",
                     obj_txt, flags=re.IGNORECASE | re.MULTILINE)

    # Persist updated OBJ
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(obj_txt)
    _log(f"[OBJ] Updated OBJ to use '{desired_material_name}' and mtllib '{desired_mtl_filename}'.")


def save_citygml(root: etree._Element, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tree = etree.ElementTree(root)
    tree.write(file_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def generate_osm_roads_and_furniture(
        tif_path: str,
        *,
        input_crs: str = "EPSG:4326",
        lod: LODType = 1,
        out_dir: str = "./draft",
        # outputs (defaults match your request)
        export_shp: bool = False,
        export_obj_roads: bool = True,
        color_roads: bool = True,  # palette coloring on roads
        export_obj_furniture: bool = True,
        color_furniture: bool = True,
        furniture_rgba=(240, 128, 128, 255),
        export_gml_roads: bool = True,
        export_gml_furniture: bool = True,
        # road appearance
        carriage_half_width: float = 3.5,
        thickness: float = 0.06,
        road_color_map: dict | None = None,  # optional per-highway palette
        # furniture sampling
        furniture_spacing_map: dict | None = None,
        furniture_side_map: dict | None = None,
        furniture_bank_dirs: dict | None = None,
        cyl_diameter_m: float = 0.30,
        cyl_height_m: float = 5.0,
        cyl_segments: int = 24,
        # logging
        verbose: bool = True,
):
    """
    One-call pipeline:
    - Fetch OSM roads in bbox
    - Sample city furniture points
    - Build meshes
    - Export requested outputs under out_dir:
        road.shp (optional, default False)
        gen_road.obj (default True)
        gen_city_furniture.obj (default True) with color (240,128,128,255) by default
        gen_road.gml (default True)
        gen_city_furniture.gml (default True)
    """

    def _log(msg: str):
        if verbose:
            print(msg)

    with rasterio.open(tif_path) as src:
        # original bbox in native CRS
        bounds = src.bounds
        src_crs = src.crs
        if src_crs is None:
            raise ValueError("Input TIF has no CRS information.")
        # transform to EPSG:4326
        x_min, y_min, x_max, y_max = transform_bounds(src_crs, "EPSG:4326",
                                                      bounds.left, bounds.bottom,
                                                      bounds.right, bounds.top)

    if lod == 1:
        road_color_map = None
    else:
        road_color_map = {
            "motorway": (220, 80, 80, 255),
            "trunk": (230, 130, 70, 255),
            "primary": (240, 170, 70, 255),
            "secondary": (250, 210, 80, 255),
            "tertiary": (250, 240, 140, 255),
            "residential": (200, 200, 200, 255),
            "footway": (120, 200, 120, 255),
            "cycleway": (120, 160, 220, 255),
        }

    os.makedirs(out_dir, exist_ok=True)
    _log(f"[Step 0] Prepare output directory: {os.path.abspath(out_dir)}")

    # 1) fetch roads
    _log(f"[Step 1] Fetching OSM roads in bbox "
         f"(x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}, crs={input_crs}) ...")
    roads_gdf = crop_osm_roads_online(
        x_min, y_min, x_max, y_max, input_crs=input_crs
    )
    _log(f"        Roads fetched: {len(roads_gdf)} features, CRS={roads_gdf.crs}")

    # 2) optional shapefile of road lines
    if export_shp:
        shp_path = os.path.join(out_dir, "road.shp")
        _log(f"[Step 2] Writing road shapefile → {shp_path}")
        roads_gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
        _log("        Shapefile written.")

    # Default per-highway half widths in Japan (meters, *half*-width)
    road_width_map = {
        "motorway":    4.5,   # ~9 m full width (2–3 lanes each direction often split; this is 片側)
        "trunk":       3.5,   # ~7 m (2 lanes)
        "primary":     3.0,   # ~6 m
        "secondary":   2.7,   # ~5.4 m
        "tertiary":    2.4,   # ~4.8 m
        "residential": 2.0,   # ~4 m
        "service":     1.8,   # ~3.6 m
        "unclassified":1.8,   # ~3.6 m
        "footway":     0.9,   # ~1.8 m 歩道
        "cycleway":    0.9,   # ~1.8 m 自転車道
    }

    # 3) build road scene (+ attributes)
    _log(f"[Step 3] Building road meshes (carriage_half_width={carriage_half_width} m, thickness={thickness} m)...")
    if color_roads:
        _log("        Using palette coloring for roads.")
        scene, attr_map = roads_to_scene_with_attrs(
            roads_gdf,
            carriage_half_width=carriage_half_width,
            thickness=thickness,
            color_map=road_color_map or {},
            z_offset_map={
                "motorway": 0.015,
                "trunk": 0.012,
                "primary": 0.009,
                "secondary": 0.006,
                "tertiary": 0.004,
                "residential": 0.002,
                "footway": -0.002,
            },
            z_offset_default=0.0,
            width_map=road_width_map,   # NEW
        )
    else:
        _log("        Building road geometry without custom colors (default material).")
        scene, attr_map = roads_to_scene_with_attrs(
            roads_gdf,
            carriage_half_width=carriage_half_width,
            thickness=thickness,
            color_map={},
            width_map=road_width_map,   # NEW
        )
    _log(f"        Road meshes built: {len(scene.geometry)} mesh objects.")

    # 4) furniture points
    _log("[Step 4] Sampling city furniture points along road offsets...")
    pts_gdf = sample_city_furniture_points(
        roads_gdf,
        carriage_half_width=carriage_half_width,
        spacing_map=furniture_spacing_map,
        side_map=furniture_side_map,
        output_shp=None,
        width_map=road_width_map,      # NEW: use per-highway half-width
        highway_field="highway",       # NEW: which column stores road class
    )
    _log(
        f"        Furniture points sampled: {len(pts_gdf)} points, kinds={sorted(pts_gdf['kind'].unique()) if len(pts_gdf) > 0 else []}")

    # 5) furniture meshes
    if lod == 1:
        _log(f"[Step 5] LOD=1 → use cylinders (Ø={cyl_diameter_m} m, H={cyl_height_m} m, seg={cyl_segments})")
        furniture_meshes = instances_cylinders_from_points(
            pts_gdf,
            diameter_m=cyl_diameter_m,
            height_m=cyl_height_m,
            segments=cyl_segments,
            color_rgba=(furniture_rgba if color_furniture else None),
        )
    else:
        _log("[Step 5] LOD>1 → place meshes from banks")
        furniture_meshes = export_furniture_from_points(
            pts_gdf,
            base_dir_map=(furniture_bank_dirs or {
                "electric_pole": os.path.join(_DATA_DIR, "city_furniture", "electric_pole"),
                "street_light": os.path.join(_DATA_DIR, "city_furniture", "street_light"),
                "traffic_light": os.path.join(_DATA_DIR, "city_furniture", "traffic_light"),
            }),
            use_heading=True,
            color_rgba=(furniture_rgba if color_furniture else None),
        )
    _log(f"        Furniture instances: {len(furniture_meshes)}")

    # 6) export OBJ
    if export_obj_roads:
        road_obj = os.path.join(out_dir, "gen_road.obj")
        _log(f"[Step 6] Exporting road OBJ → {road_obj}")
        scene.export(road_obj)
        # Retarget MTL + PNG names specifically for roads
        align_road_obj_mtl(
            road_obj,
            desired_mtl_filename="road_palette.mtl",  # <- you can choose another name
            desired_material_name="palette_mat",  # <- the material name you prefer
            desired_texture_png="palette_mat.png",  # <- ensure this matches the actual PNG file
            verbose=True
        )
        _log("        Road OBJ+MTL retargeted.")
    if export_obj_furniture:
        frn_obj = os.path.join(out_dir, "gen_city_furniture.obj")
        _log(f"[Step 6] Exporting furniture OBJ → {frn_obj}")
        merged_frn = trimesh.util.concatenate([m for m in furniture_meshes])
        merged_frn.export(frn_obj)
        _log("        Furniture OBJ exported.")

    # 7) export CityGML
    if export_gml_roads:
        gml_path = os.path.join(out_dir, "gen_road.gml")
        _log(f"[Step 7] Writing CityGML for roads (lod={lod}) → {gml_path}")
        roads_gml = export_citygml(scene, attr_map=attr_map, lod=lod, feature="road")
        save_citygml(roads_gml, gml_path)
        _log("        Road GML written.")

    if export_gml_furniture:
        gml_path = os.path.join(out_dir, "gen_city_furniture.gml")
        _log(f"[Step 7] Writing CityGML for furniture (lod={lod}) → {gml_path}")
        frn_dict = {f"frn_{i:05d}": m for i, m in enumerate(furniture_meshes)}
        frn_gml = export_citygml(frn_dict, lod=lod, feature="cityfurniture")
        save_citygml(frn_gml, gml_path)
        _log("        Furniture GML written.")

    _log("[Done] All tasks completed.")
    return {
        "roads_gdf": roads_gdf,
        "points_gdf": pts_gdf,
        "scene": scene,
        "attr_map": attr_map,
        "furniture_meshes": furniture_meshes
    }


# -----------------------------------------------------------------------------
# Example (disabled)
# -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     generate_osm_roads_and_furniture(
#         139.625000, 35.441667, 139.637500, 35.450000,
#         input_crs="EPSG:4326",
#         lod=1,
#         out_dir=r"C:\zcb\code\python\gen_road\draft",
#         export_shp=False,
#         export_obj_roads=True,
#         color_roads=True,
#         export_obj_furniture=True,
#         color_furniture=True,
#         furniture_rgba=(240,128,128,255),
#         export_gml_roads=True,
#         export_gml_furniture=True,
#         carriage_half_width=3.5,
#         thickness=0.06,
#     )
