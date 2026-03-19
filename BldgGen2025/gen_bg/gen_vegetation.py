import os
import math
import random
import warnings

import cv2
import geopandas as gpd
import numpy as np
import onnxruntime as ort
import rasterio
import trimesh
from PIL import Image
from lxml import etree
from pyproj import CRS
from rasterio.features import shapes
from shapely import ops as shapely_ops
from shapely.geometry import shape, Point, Polygon, MultiPolygon

# ----------------- Optional tqdm (progress bar) -----------------
try:
    from tqdm import tqdm
except ImportError:  # fallback if tqdm is not installed
    def tqdm(iterable=None, **kwargs):
        return iterable

# ----------------- Global settings -----------------
warnings.filterwarnings("ignore", category=DeprecationWarning)
ort.set_default_logger_severity(4)


# ================================================================
# 1. Segmentation: image -> vegetation polygons (GeoDataFrame / Shapefile)
# ================================================================
def predict_vegetation_to_shp(
    img_path: str,
    out_shp_path: str = None,
    onnx_path: str = "./data/model/vegetation_seg.onnx",
    *,
    tile_size: int = 512,
    stride: int = 512,  # can be < tile_size for overlapping smoothing; tile_size for faster non-overlap
    mean=(123.675, 116.28, 103.53),
    std=(58.395, 57.12, 57.375),
    vegetation_class_ids=(1,),  # class IDs considered as vegetation
    drop_class_ids=(2,),        # class IDs to be forced to background (0)
    min_area_m2: float = 1.0,   # minimum area threshold in target CRS (m^2)
    target_epsg: int = 30169    # target CRS for output polygons
) -> gpd.GeoDataFrame:
    """
    Run semantic segmentation on a georeferenced raster using a 512x512 ONNX model,
    extract vegetation regions as polygons, reproject to target_epsg, and optionally
    save as a Shapefile.

    Returns
    -------
    gdf : GeoDataFrame
        Vegetation polygons in target_epsg coordinate system.
    """

    print("[Segmentation] Loading ONNX model...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    def _preprocess_rgb_to_bgr_chw(img_rgb):
        # img_rgb: HWC, uint8, RGB
        img_bgr = img_rgb[:, :, ::-1].astype(np.float32)  # RGB -> BGR
        img_bgr = (img_bgr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
        chw = np.transpose(img_bgr, (2, 0, 1))  # CHW
        return chw

    def _infer_512(img_rgb_512):
        """
        Input 512x512 RGB uint8, output 512x512 class map (int).
        Handles two common ONNX output formats:
          (1) [1,1,H,W] class map
          (2) [1,C,H,W] logits
        """
        x = _preprocess_rgb_to_bgr_chw(img_rgb_512)
        y = session.run(None, {input_name: x[None, ...]})[0]
        pred = y

        if pred.ndim == 4:
            if pred.shape[1] == 1:
                mask = pred[0, 0].astype(np.int32)
            else:
                mask = np.argmax(pred[0], axis=0).astype(np.int32)
        elif pred.ndim == 3:
            mask = pred[0].astype(np.int32)
        else:
            raise ValueError(f"Unexpected ONNX output shape: {pred.shape}")

        # Force specific classes to background (0)
        if drop_class_ids:
            for cid in drop_class_ids:
                mask[mask == cid] = 0
        return mask

    # ---------- 1) Read input image ----------
    print("[Segmentation] Reading input raster...")
    with rasterio.open(img_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        H, W = src.height, src.width

        bands = min(3, src.count)
        img = src.read(list(range(1, bands + 1)))  # (C,H,W)
        if bands < 3:
            pad = np.zeros((3 - bands, H, W), dtype=img.dtype)
            img = np.concatenate([img, pad], axis=0)
        img = np.transpose(img, (1, 2, 0))  # (H,W,C), assume band order is RGB

    # ---------- 2) Run segmentation (small image vs tiled) ----------
    print("[Segmentation] Running vegetation segmentation...")
    if H <= tile_size and W <= tile_size:
        # Directly resize to tile_size, run model, then resize back
        resized = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        mask512 = _infer_512(resized)
        full_mask = cv2.resize(mask512.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        # Tiled inference
        if stride <= 0 or stride > tile_size:
            stride = tile_size

        grid_h = math.ceil((H - tile_size) / stride) + 1
        grid_w = math.ceil((W - tile_size) / stride) + 1

        pad_h = max(0, tile_size + (grid_h - 1) * stride - H)
        pad_w = max(0, tile_size + (grid_w - 1) * stride - W)
        img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        prob_accum = np.zeros((img_pad.shape[0], img_pad.shape[1]), dtype=np.float32)
        count_accum = np.zeros_like(prob_accum, dtype=np.float32)

        total_tiles = grid_h * grid_w
        pbar = tqdm(total=total_tiles, desc="[Segmentation] Tiled inference")

        for gy in range(grid_h):
            for gx in range(grid_w):
                y0 = gy * stride
                x0 = gx * stride
                tile = img_pad[y0:y0 + tile_size, x0:x0 + tile_size, :]
                pred = _infer_512(tile)

                veg_mask = np.zeros_like(pred, dtype=np.uint8)
                for cid in vegetation_class_ids:
                    veg_mask[pred == cid] = 1

                prob_accum[y0:y0 + tile_size, x0:x0 + tile_size] += veg_mask.astype(np.float32)
                count_accum[y0:y0 + tile_size, x0:x0 + tile_size] += 1.0
                pbar.update(1)
        pbar.close()

        avg_prob = prob_accum / np.clip(count_accum, 1e-6, None)
        bin_pad = (avg_prob >= 0.5).astype(np.uint8)
        full_mask = bin_pad[:H, :W].astype(np.uint8)

    if full_mask.dtype != np.uint8:
        full_mask = full_mask.astype(np.uint8)

    # ---------- 3) Raster mask -> polygons ----------
    print("[Segmentation] Vectorizing mask to polygons...")
    results = []
    with rasterio.open(img_path) as src:
        transform = src.transform
        for geom, val in shapes(full_mask, mask=(full_mask == 1), transform=transform):
            if int(val) != 1:
                continue
            results.append(shape(geom))

    if not results:
        print("[Segmentation] No vegetation detected; returning empty GeoDataFrame.")
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs=src_crs)
        empty_gdf = empty_gdf.to_crs(epsg=target_epsg)
        if out_shp_path:
            out_dir = os.path.dirname(out_shp_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            empty_gdf.to_file(out_shp_path, driver="ESRI Shapefile")
        return empty_gdf

    gdf = gpd.GeoDataFrame(geometry=results, crs=src_crs)
    gdf = gdf.to_crs(epsg=target_epsg)

    if min_area_m2 is not None and min_area_m2 > 0:
        print(f"[Segmentation] Filtering polygons with area < {min_area_m2} m²...")
        gdf["area"] = gdf.geometry.area
        gdf = gdf[gdf["area"] >= float(min_area_m2)].drop(columns=["area"])

    # ---------- 4) Write Shapefile if requested ----------
    if out_shp_path:
        print(f"[Segmentation] Saving Shapefile to: {out_shp_path}")
        out_dir = os.path.dirname(out_shp_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        gdf.to_file(out_shp_path, driver="ESRI Shapefile")

    return gdf


# ================================================================
# 2. Sampling trees and exporting OBJ
# ================================================================
def _uniform_points_in_polygon(poly, n, rng):
    minx, miny, maxx, maxy = poly.bounds
    pts, attempts, limit = [], 0, n * 50 + 1000
    while len(pts) < n and attempts < limit:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if poly.contains(Point(x, y)):
            pts.append((x, y))
        attempts += 1
    return pts


def _sample_points_from_gdf_by_total_count(gdf, total_points, seed=42):
    rng = random.Random(seed)
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros((0, 2), dtype=float)

    polys, areas = [], []
    for g in geoms:
        if isinstance(g, Polygon):
            polys.append(g)
            areas.append(g.area)
        elif isinstance(g, MultiPolygon):
            for p in g.geoms:
                polys.append(p)
                areas.append(p.area)

    areas = np.array(areas, dtype=float)
    if areas.sum() <= 0:
        return np.zeros((0, 2), dtype=float)

    weights = areas / areas.sum()
    counts = (weights * total_points).astype(int)
    deficit = total_points - counts.sum()
    if deficit > 0:
        idx = np.argsort(-weights)[:deficit]
        counts[idx] += 1

    all_pts = []
    for p, k in zip(polys, counts):
        if k > 0:
            all_pts.extend(_uniform_points_in_polygon(p, k, rng))

    if len(all_pts) < total_points:
        union_poly = shapely_ops.unary_union(polys)
        rest = total_points - len(all_pts)
        all_pts.extend(_uniform_points_in_polygon(union_poly, rest, rng))

    return np.array(all_pts[:total_points], dtype=float)


def _make_cylinder(height, radius, sections=24):
    return trimesh.creation.cylinder(radius=radius, height=height, sections=sections)


def _solid_color_png(path_png, rgb=(137, 179, 95), size=4):
    Image.new("RGB", (size, size), rgb).save(path_png)


def _write_mtl(path_mtl, material_name, rgb=(137, 179, 95), texture_png=None):
    r, g, b = [c / 255.0 for c in rgb]
    lines = [
        f"newmtl {material_name}",
        f"Kd {r:.6f} {g:.6f} {b:.6f}",
        "Ka 0.000000 0.000000 0.000000",
        "Ks 0.000000 0.000000 0.000000",
        "d 1.0",
    ]
    if texture_png:
        lines.append(f"map_Kd {os.path.basename(texture_png)}")
    with open(path_mtl, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _export_obj_single_material(out_obj_path, mesh: trimesh.Trimesh, mtl_path, material_name):
    """
    Export OBJ with a single material and external MTL file.
    """
    with open(out_obj_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {os.path.basename(mtl_path)}\n")
        f.write(f"usemtl {material_name}\n")
        V, F = mesh.vertices, mesh.faces
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in F:
            a, b, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
            f.write(f"f {a} {b} {c}\n")


def _export_obj_geometry_only(out_obj_path, mesh: trimesh.Trimesh):
    """
    Export OBJ without any material information (geometry only).
    """
    with open(out_obj_path, "w", encoding="utf-8") as f:
        V, F = mesh.vertices, mesh.faces
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in F:
            a, b, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
            f.write(f"f {a} {b} {c}\n")


def sample_trees_and_export_obj(
    gdf: gpd.GeoDataFrame,
    *,
    # density (choose one)
    density_per_ha: float = None,
    density_per_m2: float = None,
    # LOD and distribution
    lod: int = 2,
    high_ratio: float = 10.0,
    high_height_range=(6.0, 12.0),
    low_height_range=(2.0, 6.0),
    high_radius_range=(1.0, 3.0),
    low_radius_range=(0.5, 2.0),
    vege_root: str = "./data/vegetation/SolitaryVegetationObject/",
    vege_label_csv: str = "./data/vegetation/tree_label.csv",
    seed: int = 1234,
    # terrain (optional): if z_func is None, trees are placed on z=0
    z_func=None,
    # OBJ export options
    export_obj_path: str = None,
    export_mtl_filename: str = "vegetation_materials.mtl",
    export_texture_png: str = "vegetation_diffuse.png",
    material_name: str = "veg_green_mat",
    material_rgb=(137, 179, 95),
    colorize_obj: bool = True,
):
    """
    Sample tree positions inside polygons and generate 3D meshes.

    Parameters
    ----------
    gdf : GeoDataFrame
        Vegetation polygons in a metric CRS (expected EPSG:30169).
    density_per_ha : float, optional
        Trees per hectare. Choose either density_per_ha or density_per_m2.
    density_per_m2 : float, optional
        Trees per square meter. Choose either density_per_ha or density_per_m2.
    lod : int
        1: use cylinders, 2/3: use pre-modeled tree meshes from vege_root.
    z_func : callable, optional
        Function z = f(x,y) to place trees on terrain; if None, z = 0.

    Returns
    -------
    meshes_list : list[trimesh.Trimesh]
        One mesh per tree.
    merged_mesh : trimesh.Trimesh
        Concatenated mesh of all trees.
    """
    assert isinstance(gdf, gpd.GeoDataFrame), "gdf must be a GeoDataFrame"
    if gdf.crs is None or CRS.from_user_input(gdf.crs).to_epsg() != 30169:
        raise ValueError("gdf must be in EPSG:30169 (metric)")

    if (density_per_ha is None) == (density_per_m2 is None):
        raise ValueError("Exactly one of density_per_ha or density_per_m2 must be provided.")

    total_area_m2 = float(gdf.geometry.area.sum())
    if density_per_ha is not None:
        total_points = int(round(total_area_m2 * (density_per_ha / 10000.0)))
    else:
        total_points = int(round(total_area_m2 * float(density_per_m2)))

    if total_points <= 0 and total_area_m2 > 0:
        total_points = 1

    print(f"[Trees] Total vegetation area: {total_area_m2:.2f} m²")
    print(f"[Trees] Target number of trees: {total_points}")

    print("[Trees] Sampling tree positions...")
    pts_xy = _sample_points_from_gdf_by_total_count(gdf, total_points, seed=seed)
    if len(pts_xy) == 0:
        print("[Trees] No sample points generated; returning empty meshes.")
        merged_mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))
        return [], merged_mesh

    frac_high = float(high_ratio) / (float(high_ratio) + 1.0)
    high_num = int(round(len(pts_xy) * frac_high))
    low_num = len(pts_xy) - high_num

    print(f"[Trees] High trees: {high_num}, Low trees: {low_num}")

    rng = random.Random(seed)
    meshes_list: list[trimesh.Trimesh] = []

    # ---------- LOD1: cylinders ----------
    if lod == 1:
        print("[Trees] Generating LOD1 cylinder trees (high)...")
        for i in tqdm(range(high_num), desc="[Trees] High trees"):
            x, y = pts_xy[i]
            h = rng.uniform(*high_height_range)
            r = rng.uniform(*high_radius_range)
            cyl = _make_cylinder(height=h, radius=r, sections=24)
            base_z = float(z_func(x, y)) if z_func else 0.0
            cyl.apply_translation([x, y, base_z + h / 2.0])
            meshes_list.append(cyl)

        print("[Trees] Generating LOD1 cylinder trees (low)...")
        for i in tqdm(range(high_num, high_num + low_num), desc="[Trees] Low trees"):
            x, y = pts_xy[i]
            h = rng.uniform(*low_height_range)
            r = rng.uniform(*low_radius_range)
            cyl = _make_cylinder(height=h, radius=r, sections=20)
            base_z = float(z_func(x, y)) if z_func else 0.0
            cyl.apply_translation([x, y, base_z + h / 2.0])
            meshes_list.append(cyl)

    # ---------- LOD2 / LOD3: pre-modeled meshes ----------
    elif lod in (2, 3):
        import pandas as pd

        print("[Trees] Loading pre-modeled tree library...")
        meta = pd.read_csv(vege_label_csv)
        ids, types = meta["id"].values, meta["type"].values
        high_ids = [ids[i] for i in range(len(ids)) if int(types[i]) == 1]
        low_ids = [ids[i] for i in range(len(ids)) if int(types[i]) == 0]

        high_pick = [rng.choice(high_ids) for _ in range(high_num)] if high_ids else []
        low_pick = [rng.choice(low_ids) for _ in range(low_num)] if low_ids else []

        print("[Trees] Generating high trees from library...")
        for i, tid in tqdm(list(enumerate(high_pick)), desc="[Trees] High trees"):
            mesh = trimesh.load(os.path.join(vege_root, f"{tid}.obj"), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            x, y = pts_xy[i]
            base_z = float(z_func(x, y)) if z_func else 0.0
            vmin_z = float(mesh.vertices[:, 2].min())
            mesh.apply_translation([x - mesh.centroid[0], y - mesh.centroid[1], (base_z - vmin_z)])

            if (mesh.faces.ndim != 2) or (mesh.faces.shape[1] != 3):
                mesh = mesh.triangulate()
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            meshes_list.append(mesh)

        print("[Trees] Generating low trees from library...")
        for j, tid in tqdm(list(enumerate(low_pick)), desc="[Trees] Low trees"):
            mesh = trimesh.load(os.path.join(vege_root, f"{tid}.obj"), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            x, y = pts_xy[high_num + j]
            base_z = float(z_func(x, y)) if z_func else 0.0
            vmin_z = float(mesh.vertices[:, 2].min())
            mesh.apply_translation([x - mesh.centroid[0], y - mesh.centroid[1], (base_z - vmin_z)])

            if (mesh.faces.ndim != 2) or (mesh.faces.shape[1] != 3):
                mesh = mesh.triangulate()
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            meshes_list.append(mesh)
    else:
        raise ValueError("lod must be 1, 2, or 3")

    # ---------- Merge meshes ----------
    print("[Trees] Merging all tree meshes into a single mesh...")
    if len(meshes_list) == 0:
        merged_mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))
    else:
        merged_mesh = trimesh.util.concatenate(meshes_list)

    # ---------- Export OBJ if requested ----------
    if export_obj_path:
        out_dir = os.path.dirname(export_obj_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        if colorize_obj:
            print(f"[Trees] Exporting colored OBJ to: {export_obj_path}")
            mtl_path = os.path.join(out_dir, export_mtl_filename)
            tex_path = os.path.join(out_dir, export_texture_png)
            _solid_color_png(tex_path, rgb=material_rgb, size=4)
            _write_mtl(mtl_path, material_name, rgb=material_rgb, texture_png=os.path.basename(tex_path))
            _export_obj_single_material(export_obj_path, merged_mesh, mtl_path, material_name)
        else:
            print(f"[Trees] Exporting geometry-only OBJ to: {export_obj_path}")
            _export_obj_geometry_only(export_obj_path, merged_mesh)

    return meshes_list, merged_mesh


# ================================================================
# 3. CityGML generation from meshes
# ================================================================
def create_citygml_vegetation_from_meshes(
    meshes,
    *,
    lod: int = 2,
    srs_epsg: int = 30169,
    srsDimension: str = "3",
    gml_id_prefix: str = "veg_",
):
    """
    Convert a list of vegetation meshes into a CityGML 2.0 CityModel
    with SolitaryVegetationObject elements.

    Parameters
    ----------
    meshes : Trimesh or Scene or list
        Single Trimesh/Scene or a list/tuple of them.
    lod : int
        Level of detail for CityGML geometry tag (1/2/3).
    """
    if isinstance(meshes, (trimesh.Trimesh, trimesh.Scene)):
        meshes = [meshes]
    elif isinstance(meshes, (list, tuple)):
        pass
    else:
        raise TypeError(
            f"'meshes' must be Trimesh/Scene or a list/tuple of them, got {type(meshes)}"
        )

    nsmap = {
        "core": "http://www.opengis.net/citygml/2.0",
        "veg": "http://www.opengis.net/citygml/vegetation/2.0",
        "gml": "http://www.opengis.net/gml",
    }
    srs_name = f"http://www.opengis.net/def/crs/EPSG/0/{int(srs_epsg)}"

    cityModel = etree.Element("{http://www.opengis.net/citygml/2.0}CityModel", nsmap=nsmap)

    tri_meshes = []
    all_vertices = []

    for i, m in enumerate(meshes):
        if isinstance(m, trimesh.Scene):
            m = m.dump(concatenate=True)
        if not isinstance(m, trimesh.Trimesh):
            raise TypeError(f"meshes[{i}] is not Trimesh or Scene, got {type(m)}")

        if (m.faces.ndim != 2) or (m.faces.shape[1] != 3):
            m = m.triangulate()

        tri_meshes.append(m)
        all_vertices.append(m.vertices)

    if not tri_meshes:
        return cityModel

    total_vertices = np.vstack(all_vertices)
    x_min, y_min, z_min = np.min(total_vertices, axis=0)
    x_max, y_max, z_max = np.max(total_vertices, axis=0)

    # boundedBy
    boundedBy = etree.SubElement(cityModel, "{http://www.opengis.net/gml}boundedBy")
    Envelope = etree.SubElement(
        boundedBy,
        "{http://www.opengis.net/gml}Envelope",
        srsName=srs_name,
        srsDimension=srsDimension,
    )
    lowerCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}lowerCorner")
    upperCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}upperCorner")
    lowerCorner.text = f"{x_min} {y_min} {z_min}"
    upperCorner.text = f"{x_max} {y_max} {z_max}"

    # one SolitaryVegetationObject per mesh
    lod_tag = f"{{http://www.opengis.net/citygml/vegetation/2.0}}lod{lod}Geometry"

    print("[CityGML] Building CityModel from meshes...")
    for idx, m in enumerate(tqdm(tri_meshes, desc="[CityGML] Meshes")):
        V = m.vertices
        F = m.faces

        member = etree.SubElement(
            cityModel,
            "{http://www.opengis.net/citygml/2.0}cityObjectMember",
        )

        veg_obj = etree.SubElement(
            member,
            "{http://www.opengis.net/citygml/vegetation/2.0}SolitaryVegetationObject",
        )
        veg_obj.set("{http://www.opengis.net/gml}id", f"{gml_id_prefix}{idx:05d}")

        lodGeom = etree.SubElement(veg_obj, lod_tag)
        multiSurface = etree.SubElement(
            lodGeom,
            "{http://www.opengis.net/gml}MultiSurface",
        )

        for f in F:
            i0, i1, i2 = int(f[0]), int(f[1]), int(f[2])
            p0, p1, p2 = V[i0], V[i1], V[i2]

            surfaceMember = etree.SubElement(
                multiSurface,
                "{http://www.opengis.net/gml}surfaceMember",
            )
            polygon = etree.SubElement(
                surfaceMember,
                "{http://www.opengis.net/gml}Polygon",
                srsName=srs_name,
                srsDimension=srsDimension,
            )

            exterior = etree.SubElement(
                polygon,
                "{http://www.opengis.net/gml}exterior",
            )
            linearRing = etree.SubElement(
                exterior,
                "{http://www.opengis.net/gml}LinearRing",
            )
            posList = etree.SubElement(
                linearRing,
                "{http://www.opengis.net/gml}posList",
            )

            coords = np.vstack([p0, p1, p2, p0]).astype(float)
            posList.text = " ".join(f"{c:.6f}" for xyz in coords for c in xyz)

    return cityModel


def save_citygml(
    root_element,
    out_path: str,
    *,
    pretty_print: bool = True,
    xml_declaration: bool = True,
    encoding: str = "UTF-8",
):
    tree = etree.ElementTree(root_element)
    tree.write(
        out_path,
        pretty_print=pretty_print,
        xml_declaration=xml_declaration,
        encoding=encoding,
    )


# ================================================================
# 4. Main pipeline function
# ================================================================
def generate_vegetation_assets(
    img_path: str,
    lod: int,
    out_dir: str,
    *,
    # segmentation / shapefile options
    onnx_path: str = "./data/model/vegetation_seg.onnx",
    tile_size: int = 512,
    stride: int = 512,
    vegetation_class_ids=(1,),
    drop_class_ids=(2,),
    min_area_m2: float = 1.0,
    target_epsg: int = 30169,
    save_shp: bool = False,
    # tree sampling / OBJ options
    density_per_ha: float = 150.0,
    density_per_m2: float = None,
    vege_root: str = "./data/vegetation/SolitaryVegetationObject/",
    vege_label_csv: str = "./data/vegetation/tree_label.csv",
    high_ratio: float = 10.0,
    high_height_range=(6.0, 12.0),
    low_height_range=(2.0, 6.0),
    high_radius_range=(1.0, 3.0),
    low_radius_range=(0.5, 2.0),
    z_func=None,
    save_obj: bool = True,
    save_gml: bool = True,
    colorize_obj: bool = True,
):
    """
    Full pipeline:
      1. Segment vegetation from ortho image.
      2. Sample trees and generate OBJ.
      3. Export CityGML.

    Parameters
    ----------
    img_path : str
        Input ortho image path (georeferenced).
    lod : int
        1/2/3, controlling tree model type and CityGML LOD tag.
    out_dir : str
        Output folder. Inside it:
          - vegetation.shp (optional)
          - gen_vegetation.obj (default)
          - gen_vegetation.gml (default)
    """
    os.makedirs(out_dir, exist_ok=True)
    shp_path = os.path.join(out_dir, "vegetation.shp") if save_shp else None
    obj_path = os.path.join(out_dir, "gen_vegetation.obj") if save_obj else None
    gml_path = os.path.join(out_dir, "gen_vegetation.gml") if save_gml else None

    print("===================================================")
    print("[Pipeline] Vegetation generation pipeline started")
    print("===================================================")

    # ---------- Step 1: segmentation ----------
    print("[Pipeline] Step 1/3: Vegetation segmentation...")
    ext = os.path.splitext(img_path)[1].lower()

    if ext in [".geojson", ".json"]:
        print("[Pipeline] Input is GeoJSON; skip vegetation segmentation.")
        vegetation_gdf = gpd.GeoDataFrame(
            geometry=[],
            crs=f"EPSG:{target_epsg}"
        )
    else:
        vegetation_gdf = predict_vegetation_to_shp(
            img_path,
            out_shp_path=shp_path,
            onnx_path=onnx_path,
            tile_size=tile_size,
            stride=stride,
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            vegetation_class_ids=vegetation_class_ids,
            drop_class_ids=drop_class_ids,
            min_area_m2=min_area_m2,
            target_epsg=target_epsg,
        )

    if vegetation_gdf.empty:
        print("[Pipeline] No vegetation polygons; skipping tree generation and GML export.")
        return {
            "shapefile_path": shp_path,
            "obj_path": None,
            "gml_path": None,
            "vegetation_gdf": vegetation_gdf,
            "meshes_list": [],
            "merged_mesh": None,
        }

    # ---------- Step 2: sample trees + OBJ ----------
    print("[Pipeline] Step 2/3: Sampling trees and exporting OBJ...")
    if save_obj or save_gml:
        meshes_list, merged_mesh = sample_trees_and_export_obj(
            vegetation_gdf,
            density_per_ha=density_per_ha if density_per_m2 is None else None,
            density_per_m2=density_per_m2,
            lod=lod,
            high_ratio=high_ratio,
            high_height_range=high_height_range,
            low_height_range=low_height_range,
            high_radius_range=high_radius_range,
            low_radius_range=low_radius_range,
            vege_root=vege_root,
            vege_label_csv=vege_label_csv,
            seed=1234,
            z_func=z_func,
            export_obj_path=obj_path,
            export_mtl_filename="vegetation_materials.mtl",
            export_texture_png="vegetation_diffuse.png",
            material_name="veg_green_mat",
            material_rgb=(137, 179, 95),
            colorize_obj=colorize_obj,
        )
    else:
        meshes_list, merged_mesh = [], None

    # ---------- Step 3: CityGML ----------
    print("[Pipeline] Step 3/3: Exporting CityGML...")
    if save_gml and meshes_list:
        gml_root = create_citygml_vegetation_from_meshes(
            meshes_list,
            lod=lod,
            srs_epsg=target_epsg,
            srsDimension="3",
            gml_id_prefix="veg_",
        )
        print(f"[Pipeline] Saving CityGML to: {gml_path}")
        save_citygml(gml_root, gml_path)
    else:
        gml_path = None
        print("[Pipeline] CityGML export skipped (no meshes or save_gml=False).")

    print("===================================================")
    print("[Pipeline] Vegetation generation pipeline finished")
    print("===================================================")

    return {
        "shapefile_path": shp_path,
        "obj_path": obj_path,
        "gml_path": gml_path,
        "vegetation_gdf": vegetation_gdf,
        "meshes_list": meshes_list,
        "merged_mesh": merged_mesh,
    }
