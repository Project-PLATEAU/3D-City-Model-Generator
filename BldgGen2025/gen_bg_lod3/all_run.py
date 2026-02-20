import os
import time
import trimesh

import numpy as np

from .gen_road import generate_osm_roads_and_furniture
from .gen_vegetation import generate_vegetation_assets
from .gen_building import generate_lod2_with_doors_and_windows


def generate_city_assets(
        tif_path: str,
        lod2_obj_path: str = None,
        directly_concat: bool = False,
        output_dir: str = "./output/separate"
):
    """
    One-click pipeline to generate:
        - OSM-based roads and street furniture
        - Vegetation assets
        - (Optional) LOD2 buildings with doors & windows
        - Final merged OBJ model

    Inputs:
        tif_path       : Path to orthoimage (GeoTIFF)
        lod2_obj_path  : Path to LOD2 OBJ, or None to skip building generation
        output_dir     : Output folder for intermediate files

    Outputs:
        merged OBJ at ./output/gen_mesh.obj
    """

    os.makedirs(output_dir, exist_ok=True)
    time_start = time.time()

    # ------------------------------------------------------------------
    print("\n=== Step 1: Generate OSM Roads & Street Furniture ===")
    road_data = generate_osm_roads_and_furniture(
        tif_path,
        # carriage_half_width=2.5,
        export_shp=True,
        lod=3,
        out_dir=output_dir,
    )

    # ------------------------------------------------------------------
    print("\n=== Step 2: Generate Vegetation Assets ===")
    vegetation_data = generate_vegetation_assets(
        img_path=tif_path,
        lod=3,
        out_dir=output_dir,
        save_shp=True,
        save_obj=True,
        save_gml=True,
        colorize_obj=True,
    )

    # ------------------------------------------------------------------
    # Building pipeline (optional)
    building_mesh = None
    out_lod2_with_door_window = os.path.join(output_dir, "merged_lod2_doors_windows.obj")

    if lod2_obj_path is not None:
        if directly_concat:
            building_mesh = trimesh.load(lod2_obj_path)
            print("\n=== Step 3: Directly Concatenate LOD2 building ===")
        else:
            print("\n=== Step 3: Generate LOD2 Buildings (with Doors & Windows) ===")
            generate_lod2_with_doors_and_windows(
                lod2_obj_path=lod2_obj_path,
                out_obj_path=out_lod2_with_door_window,
                src_epsg="EPSG:30169",
                footprint_shp_path=None,
            )
            building_mesh = trimesh.load(out_lod2_with_door_window)
    else:
        print("\n=== Step 3: No LOD2 Path Provided → Skipping Building Generation ===")

    # ------------------------------------------------------------------
    print("\n=== Step 4: Merge OBJ Models ===")

    # ---- Roads & Street Furniture ----
    roads_gdf = road_data.get("roads_gdf", None)
    if roads_gdf is not None and not roads_gdf.empty:
        road_mesh = trimesh.load(os.path.join(output_dir, "gen_road.obj"))
        city_furniture_mesh = trimesh.load(os.path.join(output_dir, "gen_city_furniture.obj"))
    else:
        road_mesh, city_furniture_mesh = None, None

    # ---- Vegetation ----
    veg_gdf = vegetation_data.get("vegetation_gdf", None)
    if veg_gdf is not None and not veg_gdf.empty:
        vegetation_mesh = trimesh.load(os.path.join(output_dir, "gen_vegetation.obj"))
    else:
        vegetation_mesh = None

    # ---- Merge list (skip None) ----
    mesh_list = [
        m for m in [
            road_mesh,
            vegetation_mesh,
            city_furniture_mesh,
            building_mesh
        ] if m is not None
    ]

    # Merge them
    output_mesh = trimesh.util.concatenate(mesh_list)
    
    output_mesh_vertices = output_mesh.vertices.copy()
    output_mesh.vertices = np.column_stack([
        output_mesh_vertices[:, 0],
        output_mesh_vertices[:, 2],
        -output_mesh_vertices[:, 1]
    ])
    
    if isinstance(output_mesh.visual, trimesh.visual.TextureVisuals):
        material = output_mesh.visual.material
        if hasattr(material, 'image') and material.image is not None:
            texture = np.array(material.image)
            
            # Get UV coordinates for each face
            uv = output_mesh.visual.uv
            face_uvs = uv[output_mesh.faces]  # Shape: (n_faces, 3, 2)
            
            # Sample texture at face centers (average UV of 3 vertices)
            face_uv_centers = face_uvs.mean(axis=1)  # Shape: (n_faces, 2)
            
            # Convert UV (0-1) to pixel coordinates
            h, w = texture.shape[:2]
            pixel_coords = face_uv_centers * [w-1, h-1]
            pixel_coords = pixel_coords.astype(int)
            
            # Clamp coordinates
            pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, w-1)
            pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, h-1)
            
            # Sample colors (note: UV v-coordinate is flipped)
            face_colors = texture[h-1-pixel_coords[:, 1], pixel_coords[:, 0]]
            
            # Convert to ColorVisuals
            output_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=output_mesh,
                face_colors=face_colors
            )
        else:
            # No texture, use material diffuse color
            Kd = material.diffuse[:3] if hasattr(material, 'diffuse') else [0.4, 0.4, 0.4]
            color = (np.array(Kd) * 255).astype(np.uint8)
            output_mesh.visual.face_colors = np.tile(
                np.append(color, 255), 
                (len(output_mesh.faces), 1)
            )

    merged_output_path = "./output/gen_mesh.obj"
    os.makedirs("./output", exist_ok=True)
    output_mesh.export(merged_output_path)

    time_end = time.time()
    print(f"\n=== Pipeline Completed! Total Time: {time_end - time_start:.2f} sec ===")
    print(f"Merged city model saved at: {merged_output_path}")

    return output_mesh

# ===========================
# Example usage
# ===========================
if __name__ == "__main__":
    tif_path = r"./data/gsi_data/route1.tif"

    # Case 1: Use LOD2
    # lod2_path = r"./data/test_data/obj/akabane_lod2_nocolor.obj"
    # generate_city_assets(tif_path, lod2_path,directly_concat=True)

    # Case 2: No LOD2 → skip building
    generate_city_assets(tif_path, lod2_obj_path=None)
