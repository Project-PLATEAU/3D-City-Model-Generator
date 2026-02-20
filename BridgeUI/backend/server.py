#!/usr/bin/env python3
"""
BridgeUI Backend Server
Simulates file processing for the BridgeUI frontend application
"""

import os, random
import sys
import json
import time
import uuid
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add BldgGen2024 to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "BldgGen2025"))

import torch
import numpy as np
import trimesh
from BldgXL_3x.models import MeshXL
from geoinfo_load import polygon_to_mesh
from generation import generate
from utils.tiff_crop.tiff_crop_new import extract_raster_by_polygons
from gen_bg.all_run import generate_city_assets

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global model variable
model = None

def load_model(distributed=True):
    """
    Load the BldgXL PyTorch model for 3D building generation
    """
    print("🔄 Loading BldgXL model...")

    # Model configuration - use path relative to script location
    llm_config_path = Path(__file__).parent.parent.parent / "BldgGen2025" / "BldgXL_3x" / "config" / "mesh-xl-125m"
    args_dict = {
        "n_discrete_size": 128,
        "llm": str(llm_config_path)
    }
    args = argparse.Namespace(**args_dict)

    # Initialize model
    model = MeshXL(args)

    # Determine device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   ├─ Using device: {device}")

    model.to(device)
    model.eval()

    # Load checkpoint
    checkpoint_path = Path(__file__).parent.parent.parent / "BldgGen2025" / "BldgXL_3x" / "checkpoints" / 'mesh-transformer.ckpt.epoch_490_avg_loss_0.159.pt'
    print(f"   ├─ Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, strict=True)

    print("   └─ ✅ Model loaded successfully!")

    return model

def find_geojson_job_by_layer(layer_name: str) -> Optional[Dict[str, Any]]:
    """
    Find a GeoJSON job by layer name

    Args:
        layer_name: The layer name to search for

    Returns:
        Job dict if found, None otherwise
    """
    for job_id, job in active_jobs.items():
        if job.get('file_type') == 'geojson' and job.get('layer_name') == layer_name:
            return job
    return None

def run_model_inference(geojson_job_id: str, layer_name: str, params: list = None, orthophoto_path: str = None, job_id: str = None):
    """
    Run BldgXL model inference on GeoJSON file to generate LoD2 model

    Args:
        geojson_job_id: Job ID of the GeoJSON to use cached data
        layer_name: Name of the layer
        params: Optional list of [building_id, sod, height, roof_type]
                If None, uses default parameters
        orthophoto_path: Optional path to the uploaded satellite image (orthophoto)
        job_id: Optional job ID for progress tracking

    Returns:
        tuple: (output_mesh, output_mesh_bmqi, bldg_ids, bldg_meshes, properties, bg_output_mesh)
    """
    global model

    print(f"🤖 Running BldgXL model inference for layer: {layer_name}")
    print(f"   ├─ GeoJSON job ID: {geojson_job_id}")
    if orthophoto_path:
        print(f"   ├─ Orthophoto path: {orthophoto_path}")

    # Retrieve cached processed GeoJSON data (much faster than reprocessing!)
    if geojson_job_id not in geojson_cache:
        print(f"   ⚠️  No cached data found, processing GeoJSON...")
        geojson_path = active_jobs[geojson_job_id]['file_path']
        vertices, faces, properties, polygons = polygon_to_mesh(str(geojson_path), partition=True)
    else:
        print(f"   ✅ Using cached GeoJSON data (fast path)")
        cached_data = geojson_cache[geojson_job_id]
        vertices = cached_data['vertices']
        faces = cached_data['faces']
        properties = cached_data['properties']
        polygons = cached_data['polygons']
        geojson_path = cached_data['file_path']

    # Extract geojson filename without extension for image path construction
    # Use original filename (not the saved filename with job_id prefix)
    original_filename = active_jobs[geojson_job_id]['filename']
    geojson_name = Path(original_filename).stem
    print(f"   ├─ GeoJSON name (for image path): {geojson_name}")

    # Use default params if not provided
    if params is None:
        # Default: [building_id, height]
        # sod: 1=simple, 2=exact, 3=detailed, 4=sophisticated
        # roof_type: 1=flat, 2=stepped, 3=hybrid, 4=hipped, 5=gable
        params = []
        for i, prop in enumerate(properties):
            # Extract height from properties if available, otherwise use default
            bldg_id = prop.get('id', '')
            
            height = prop.get('height', 10.0)
            if isinstance(height, str) and height is not None:
                try:
                    height = float(height)
                except:
                    height = 10.0
            
            if isinstance(height, str) and height == '':
                height = 10.0

            # Default to detailed design (sod=3) and flat roof (roof_type=1)
            # roof_type_index = random.randint(0, 2)
            # possible_roof_type = [1, 4, 5]
            # selected_roof_type = possible_roof_type[roof_type_index]
            
            params.append([bldg_id, height])

    print(f"   ├─ Number of buildings: {len(vertices)}")
    # print(f"   ├─ Parameters: {params}")

    # Check model device
    device = next(model.parameters()).device
    print(f"   ├─ Model device: {device}")
    print(f"   ├─ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ├─ CUDA device: {torch.cuda.get_device_name(0)}")

    # Define progress callback for granular progress updates
    def progress_callback(current: int, total: int):
        """Update job progress based on building generation progress"""
        if job_id and job_id in active_jobs:
            # Map building progress to 40-90% range (model inference phase)
            base_progress = 40
            progress_range = 50  # 90 - 40 = 50
            building_progress = (current / total) * progress_range if total > 0 else 0
            active_jobs[job_id]['progress'] = int(base_progress + building_progress)
            print(f"   📊 Progress: {current}/{total} buildings ({active_jobs[job_id]['progress']}%)")

    # Run generation with timing
    import time
    start_time = time.time()

    # Note: color_mode=True enables quality loop
    output_mesh_bmqi, output_mesh, bldg_ids, bldg_meshes, bg_output_mesh = generate(
        vertices=vertices,
        faces=faces,
        params=params,
        footprints=polygons,
        model=model,
        gt_mode=False,
        color_mode=True,
        front_mode=True,
        orthophoto_path=orthophoto_path,
        geojson_name=geojson_name,
        progress_callback=progress_callback if job_id else None
    )

    elapsed = time.time() - start_time
    print(f"   └─ ✅ Model inference completed in {elapsed:.2f}s")

    return output_mesh, output_mesh_bmqi, bldg_ids, bldg_meshes, properties, bg_output_mesh

# Configuration
# Use parent directory (BridgeUI) for uploads and outputs
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
ALLOWED_EXTENSIONS = {
    'geojson': {'.geojson', '.json'},
    'orthophoto': {'.tif', '.tiff', '.jpg', '.jpeg', '.png'},
    'pointcloud': {'.ply', '.las', '.laz', '.pcd', '.xyz'},
    'streetview': {'.jpg', '.jpeg', '.png'},
    'lod3-data': {'.ply', '.las', '.laz', '.pcd', '.xyz', '.jpg', '.jpeg', '.png'}  # Combined LoD3 data
}

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job tracking
active_jobs: Dict[str, Dict[str, Any]] = {}

# Cache for processed GeoJSON data to avoid reprocessing
geojson_cache: Dict[str, Dict[str, Any]] = {}  # job_id -> {vertices, faces, properties, polygons}

def allowed_file(filename: str, file_type: str) -> bool:
    """Check if file extension is allowed for the given type"""
    if file_type not in ALLOWED_EXTENSIONS:
        return False
    
    suffix = Path(filename).suffix.lower()
    return suffix in ALLOWED_EXTENSIONS[file_type]

def process_uploaded_file(job_id: str, file_path: Path, file_type: str, layer_name: str):
    """Actually process the uploaded file and extract real information"""
    
    # Update job status
    active_jobs[job_id]['status'] = 'processing'
    active_jobs[job_id]['progress'] = 0
    
    print(f"\n🔍 Job {job_id}: Starting real file processing...")
    print(f"📁 File: {file_path.name}")
    print(f"📊 Type: {file_type}")
    print(f"🏷️  Layer: {layer_name}")
    print(f"💾 Size: {file_path.stat().st_size / 1024:.2f} KB")
    
    try:
        if file_type == 'geojson':
            print("GeoJSON received. ")
            process_geojson_file(job_id, file_path)
        elif file_type == 'orthophoto':
            process_orthophoto_file(job_id, file_path)
        elif file_type == 'pointcloud':
            process_pointcloud_file(job_id, file_path)
        elif file_type == 'streetview':
            process_streetview_file(job_id, file_path)
        elif file_type == 'lod3-data':
            # LoD3 requires both pointcloud and streetview - should not reach here
            # Use /upload-lod3 endpoint instead
            print("⚠️  Warning: lod3-data type should use /upload-lod3 endpoint with both files")
            raise ValueError("LoD3 generation requires both pointcloud and streetview files. Use /upload-lod3 endpoint.")
        else:
            process_generic_file(job_id, file_path)
            
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        active_jobs[job_id]['status'] = 'failed'
        active_jobs[job_id]['error'] = str(e)
        return
    
    # Generate output file
    print(f"🔨 Generating output files...")
    active_jobs[job_id]['progress'] = 90
    output_file = generate_output_file(job_id, file_type, layer_name)
    
    # Mark as completed
    active_jobs[job_id]['status'] = 'completed'
    active_jobs[job_id]['progress'] = 100
    active_jobs[job_id]['output_file'] = str(output_file)
    active_jobs[job_id]['completed_at'] = datetime.now().isoformat()
    
    print(f"✅ Job {job_id}: Processing completed - Output: {output_file}")

def generate_lod1_model(job_id: str, vertices: list, faces: list, properties: list, polygons: list, layer_name: str) -> Path:
    """
    Generate LoD1 model by extruding building footprints to their heights

    Args:
        job_id: The job ID for output directory
        vertices: List of vertex arrays for each building footprint (triangulated)
        faces: List of face arrays for each building footprint (triangulated)
        properties: List of property dictionaries with height information
        polygons: List of Shapely Polygon objects for proper extrusion
        layer_name: Name for the output file

    Returns:
        Path to the generated LoD1 OBJ file
    """
    print(f"\n🏗️  Generating LoD1 model...")
    print(f"   ├─ Number of buildings: {len(polygons)}")

    all_meshes = []

    for i, (verts, face_indices, props, polygon) in enumerate(zip(vertices, faces, properties, polygons)):
        # Get height from properties, default to 10.0 if not available
        height = props.get('height', 10.0)
        if isinstance(height, str):
            try:
                height = float(height) if height else 10.0
            except:
                height = 10.0

        # Get perimeter vertices from polygon exterior
        exterior_coords = np.array(polygon.exterior.coords[:-1])  # Exclude duplicate last point
        n_perimeter = len(exterior_coords)

        # Create base vertices (z=0) and top vertices (z=height)
        # Note: vertices from polygon_to_mesh have y negated, so we match that
        base_perimeter = np.array([[coord[0], -coord[1], 0.0] for coord in exterior_coords])
        top_perimeter = np.array([[coord[0], -coord[1], height] for coord in exterior_coords])

        # Use triangulated vertices for top and bottom faces
        base_triangulated = np.array(verts)
        top_triangulated = base_triangulated.copy()
        top_triangulated[:, 2] = height

        # Combine all vertices: triangulated base, triangulated top, perimeter base, perimeter top
        n_triangulated = len(base_triangulated)
        offset_top_tri = n_triangulated
        offset_base_peri = 2 * n_triangulated
        offset_top_peri = offset_base_peri + n_perimeter

        all_verts = np.vstack([base_triangulated, top_triangulated, base_perimeter, top_perimeter])

        # Bottom faces (use original triangulation)
        bottom_faces = face_indices

        # Top faces (offset by n_triangulated)
        top_faces = face_indices + offset_top_tri

        # Side faces (connect perimeter base to perimeter top)
        side_faces = []
        for j in range(n_perimeter):
            next_j = (j + 1) % n_perimeter
            base_j = offset_base_peri + j
            base_next = offset_base_peri + next_j
            top_j = offset_top_peri + j
            top_next = offset_top_peri + next_j

            # Two triangles for each side quad
            side_faces.append([base_j, base_next, top_j])
            side_faces.append([base_next, top_next, top_j])

        side_faces = np.array(side_faces)

        # Combine all faces
        all_faces = np.vstack([bottom_faces, top_faces, side_faces])

        # Create mesh for this building
        try:
            building_mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
            all_meshes.append(building_mesh)
        except Exception as e:
            print(f"   ⚠️  Failed to create mesh for building {i}: {e}")
            import traceback
            traceback.print_exc()

    # Combine all building meshes
    if not all_meshes:
        raise Exception("No valid building meshes created")

    combined_mesh = trimesh.util.concatenate(all_meshes)
    print(f"   ├─ Combined mesh: {len(combined_mesh.vertices)} vertices, {len(combined_mesh.faces)} faces")

    # Save to output directory
    job_output_dir = OUTPUT_DIR / job_id
    job_output_dir.mkdir(exist_ok=True)
    output_file = job_output_dir / f"{layer_name}_lod1.obj"

    combined_mesh.export(output_file)
    print(f"   └─ ✅ LoD1 model saved: {output_file}")

    return output_file


def process_geojson_file(job_id: str, file_path: Path):
    """Process GeoJSON file and extract real data"""
    print(f"🗺️  Processing GeoJSON file...")
    active_jobs[job_id]['progress'] = 10

    # Load and cache the processed GeoJSON data for later use
    print(f"   ├─ Loading and processing GeoJSON for caching...")
    vertices, faces, properties, polygons = polygon_to_mesh(str(file_path), partition=True)

    # Store in cache for fast retrieval during LoD2 generation
    geojson_cache[job_id] = {
        'vertices': vertices,
        'faces': faces,
        'properties': properties,
        'polygons': polygons,
        'file_path': str(file_path)
    }
    print(f"   ├─ ✅ Cached processed GeoJSON data for {len(vertices)} buildings")

    # Also analyze the raw GeoJSON for display purposes
    with open(file_path, 'r', encoding='utf-8') as f:
        import json
        geojson_data = json.load(f)

    active_jobs[job_id]['progress'] = 30
    print(f"   ├─ GeoJSON type: {geojson_data.get('type', 'Unknown')}")

    if 'features' in geojson_data:
        features = geojson_data['features']
        print(f"   ├─ Number of features: {len(features)}")

        # Analyze feature types
        geometry_types = {}
        for feature in features:
            geom_type = feature.get('geometry', {}).get('type', 'Unknown')
            geometry_types[geom_type] = geometry_types.get(geom_type, 0) + 1

        print(f"   ├─ Geometry types:")
        for geom_type, count in geometry_types.items():
            print(f"   │  └─ {geom_type}: {count}")

        # Analyze properties
        if features and 'properties' in features[0]:
            properties_keys = features[0]['properties'].keys()
            print(f"   ├─ Properties: {list(properties_keys)}")

        active_jobs[job_id]['progress'] = 50

        # Check for height information
        has_height = any('height' in str(feature.get('properties', {})).lower()
                        for feature in features[:5])  # Check first 5 features
        print(f"   ├─ Has height data: {has_height}")

    active_jobs[job_id]['progress'] = 60

    # Generate city assets (roads, vegetation, street furniture) with LoD1
    try:
        print(f"\n🌆 Generating city assets (roads, vegetation, furniture) with LoD1...")
        layer_name = active_jobs[job_id].get('layer_name', 'untitled')

        # Create output directory for city assets in job output folder
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(exist_ok=True)
        city_assets_output_dir = job_output_dir / "city_assets"

        # Call generate_city_assets
        print(f"   ├─ Input GeoJSON: {file_path}")
        print(f"   ├─ Output directory: {city_assets_output_dir}")
        print(f"   └─ LoD level: 1")

        active_jobs[job_id]['progress'] = 65

        # Generate city assets
        generate_city_assets(
            input_path=str(file_path),
            lod=1,
            lod2_obj_path=None,  # Don't include buildings in this layer
            output_dir=str(city_assets_output_dir)
        )

        active_jobs[job_id]['progress'] = 75

        # The output will be in ./output/gen_mesh.obj (as per all_run.py line 110)
        city_assets_mesh_path = Path("./output/gen_mesh.obj").resolve()
        output_dir_path = Path("./output").resolve()

        # Move the generated mesh to the job output directory
        final_city_assets_path = job_output_dir / f"{layer_name}_city_assets.obj"

        if city_assets_mesh_path.exists():
            import shutil

            # Fix coordinate system and export with vertex colors
            print(f"   ├─ Fixing coordinate system (swapping y and z)...")
            try:
                # Load the mesh
                city_mesh = trimesh.load(str(city_assets_mesh_path))

                # Swap y and z columns in vertices (x, y, z) -> (x, z, y)
                vertices = city_mesh.vertices.copy()
                city_mesh.vertices[:, 1] = vertices[:, 2]  # y = old z
                city_mesh.vertices[:, 2] = -vertices[:, 1]  # z = old y

                print(f"   ├─ Coordinate system fixed")
                print(f"   │  └─ Swapped y and z axes")

                # Export with original materials and textures preserved
                print(f"   ├─ Exporting mesh (preserving materials and textures)...")
                city_mesh.export(str(city_assets_mesh_path))
                print(f"   ├─ ✅ Corrected mesh saved")
            except Exception as e:
                print(f"   ├─ ⚠️  Coordinate fix failed: {e}")
                import traceback
                traceback.print_exc()
                print(f"   └─ Continuing with original mesh...")

            # Move the main OBJ file
            shutil.move(str(city_assets_mesh_path), str(final_city_assets_path))
            print(f"   ✅ City assets mesh saved: {final_city_assets_path}")

            # Also move and fix the material file and textures if they exist
            material_mtl_path = output_dir_path / "material.mtl"
            if material_mtl_path.exists():
                # Fix material colors to be white (1.0) instead of grey (0.4)
                # This prevents darkening of texture colors
                print(f"   ├─ Fixing material colors to preserve texture brightness...")
                try:
                    with open(material_mtl_path, 'r') as f:
                        mtl_content = f.read()

                    # Replace grey (0.4) with white (1.0) for Ka, Kd, Ks
                    mtl_content = mtl_content.replace(
                        'Ka 0.40000000 0.40000000 0.40000000',
                        'Ka 1.00000000 1.00000000 1.00000000'
                    )
                    mtl_content = mtl_content.replace(
                        'Kd 0.40000000 0.40000000 0.40000000',
                        'Kd 1.00000000 1.00000000 1.00000000'
                    )
                    mtl_content = mtl_content.replace(
                        'Ks 0.40000000 0.40000000 0.40000000',
                        'Ks 0.20000000 0.20000000 0.20000000'  # Keep specular lower to avoid over-brightness
                    )

                    with open(material_mtl_path, 'w') as f:
                        f.write(mtl_content)

                    print(f"   │  └─ Material colors fixed (Ka, Kd = white, Ks = 0.2)")
                except Exception as e:
                    print(f"   │  └─ ⚠️ Failed to fix material colors: {e}")

                final_mtl_path = job_output_dir / "material.mtl"
                shutil.move(str(material_mtl_path), str(final_mtl_path))
                print(f"   ├─ Material file moved: {final_mtl_path}")

            # Move texture files (material_0.png, material_1.png, etc.)
            for texture_file in output_dir_path.glob("material_*.png"):
                final_texture_path = job_output_dir / texture_file.name
                shutil.move(str(texture_file), str(final_texture_path))
                print(f"   ├─ Texture file moved: {final_texture_path}")

            # Store the path in job metadata
            active_jobs[job_id]['city_assets_output'] = str(final_city_assets_path)
            print(f"   ✅ City assets path stored in job metadata")
        else:
            print(f"   ⚠️  Expected output file not found: {city_assets_mesh_path}")

    except Exception as e:
        print(f"   ⚠️  City assets generation failed: {e}")
        import traceback
        traceback.print_exc()

    active_jobs[job_id]['progress'] = 80
    print(f"   └─ GeoJSON analysis complete")

def process_orthophoto_file(job_id: str, file_path: Path):
    """Process orthophoto/image file and generate real LoD2 model"""
    print(f"\n{'='*60}")
    print(f"📸 PROCESSING ORTHOPHOTO FILE")
    print(f"{'='*60}")
    active_jobs[job_id]['progress'] = 10

    layer_name = active_jobs[job_id].get('layer_name', 'untitled')

    file_ext = file_path.suffix.lower()
    print(f"   ├─ Job ID: {job_id}")
    print(f"   ├─ File format: {file_ext}")
    print(f"   ├─ Layer name: '{layer_name}'")
    print(f"   └─ File path: {file_path}")

    # Find linked GeoJSON job
    print(f"\n🔍 SEARCHING FOR LINKED GEOJSON JOB")
    print(f"   ├─ Looking for layer: '{layer_name}'")
    print(f"   ├─ Active jobs: {len(active_jobs)}")

    # Debug: List all jobs and their layer names
    geojson_jobs = [(jid, job.get('layer_name'), job.get('file_type'))
                    for jid, job in active_jobs.items()
                    if job.get('file_type') == 'geojson']
    print(f"   ├─ GeoJSON jobs found: {len(geojson_jobs)}")
    for jid, lname, ftype in geojson_jobs:
        print(f"   │  └─ Job {jid[:8]}...: layer='{lname}', type={ftype}")

    geojson_job = find_geojson_job_by_layer(layer_name)

    if geojson_job is None:
        print(f"\n   ⚠️  NO GEOJSON JOB FOUND for layer '{layer_name}'")
        print(f"   ├─ Possible reasons:")
        print(f"   │  1. GeoJSON not uploaded yet")
        print(f"   │  2. Layer name mismatch")
        print(f"   │  3. GeoJSON job was deleted")
        print(f"   └─ Will generate pseudo LoD2 model instead")
        active_jobs[job_id]['progress'] = 50
        time.sleep(1)
        active_jobs[job_id]['progress'] = 80
        print(f"\n{'='*60}")
        return

    print(f"   ✅ FOUND LINKED GEOJSON JOB: {geojson_job['job_id']}")
    geojson_job_id = geojson_job['job_id']
    geojson_path = Path(geojson_job['file_path'])
    print(f"   └─ GeoJSON path: {geojson_path}")

    if not geojson_path.exists():
        print(f"\n   ❌ GEOJSON FILE NOT FOUND at: {geojson_path}")
        print(f"   └─ Will generate pseudo LoD2 model instead")
        active_jobs[job_id]['progress'] = 80
        print(f"\n{'='*60}")
        return

    # Extract bounding box from orthophoto
    bbox = None
    try:
        print(f"\n📊 Extracting bounding box from orthophoto...")
        import rasterio
        from rasterio.warp import transform_bounds

        with rasterio.open(file_path) as src:
            # Get bounds in the source CRS
            bounds = src.bounds  # (left, bottom, right, top)
            src_crs = src.crs

            # Transform bounds to WGS84 (EPSG:4326) for web map compatibility
            if src_crs:
                bbox_wgs84 = transform_bounds(src_crs, 'EPSG:4326', *bounds)
                bbox = {
                    'west': bbox_wgs84[0],
                    'south': bbox_wgs84[1],
                    'east': bbox_wgs84[2],
                    'north': bbox_wgs84[3],
                    'crs': 'EPSG:4326'
                }
                print(f"   ├─ Source CRS: {src_crs}")
                print(f"   ├─ Original bounds: {bounds}")
                print(f"   ├─ WGS84 bbox: [{bbox['west']:.6f}, {bbox['south']:.6f}, {bbox['east']:.6f}, {bbox['north']:.6f}]")

                # Store bbox in job metadata
                active_jobs[job_id]['bbox'] = bbox
                print(f"   └─ ✅ Bounding box extracted and stored")
            else:
                print(f"   └─ ⚠️  No CRS information in image")
    except Exception as e:
        print(f"   └─ ⚠️  Could not extract bounding box: {e}")

    active_jobs[job_id]['progress'] = 20

    # Extract geojson filename (without extension) for output directory naming
    original_geojson_filename = geojson_job['filename']
    geojson_name = Path(original_geojson_filename).stem
    print(f"\n📐 GeoJSON name for image cropping: '{geojson_name}'")

    # Load partitioned GeoJSON data from cache first
    print(f"\n📦 Loading partitioned GeoJSON data from cache...")
    if geojson_job_id not in geojson_cache:
        print(f"   ⚠️  No cached data found, processing GeoJSON with partition...")
        vertices, faces, properties, polygons = polygon_to_mesh(str(geojson_path), partition=True)
        # Store in cache
        geojson_cache[geojson_job_id] = {
            'vertices': vertices,
            'faces': faces,
            'properties': properties,
            'polygons': polygons,
            'file_path': str(geojson_path)
        }
    else:
        print(f"   ✅ Using cached partitioned data")

    cached_data = geojson_cache[geojson_job_id]
    partitioned_polygons = cached_data['polygons']
    partitioned_properties = cached_data['properties']
    print(f"   └─ Loaded {len(partitioned_polygons)} partitioned polygons")

    # Check if partitioned file already exists from run_pipeline
    geojson_dir = geojson_path.parent
    geojson_basename = geojson_path.stem
    partitioned_file_path = geojson_dir / f"{geojson_basename}_merged.geojson"

    print(f"\n📄 Checking for partitioned GeoJSON file...")
    print(f"   ├─ Expected path: {partitioned_file_path}")

    if partitioned_file_path.exists():
        print(f"   └─ ✅ Found existing partitioned file")
    else:
        print(f"   ├─ ⚠️  Partitioned file not found, creating it...")
        # Save partitioned data to file for TIFF cropping
        try:
            from shapely.geometry import mapping
            from shapely.ops import transform as shapely_transform
            from pyproj import Transformer
            import json as json_module

            # Determine source CRS from original GeoJSON
            source_crs = "EPSG:6677"  # Default for Japanese data
            try:
                with open(geojson_path, 'r') as f:
                    original_data = json_module.load(f)
                    if 'crs' in original_data:
                        crs_info = original_data['crs']
                        if 'properties' in crs_info and 'name' in crs_info['properties']:
                            crs_name = crs_info['properties']['name']
                            # Extract EPSG code from URN format
                            if 'EPSG' in crs_name:
                                epsg_code = crs_name.split('EPSG::')[-1].split(':')[-1]
                                source_crs = f"EPSG:{epsg_code}"
            except Exception as e:
                print(f"   ├─ Could not read CRS from original: {e}")

            print(f"   ├─ Transforming from {source_crs} to EPSG:4326...")

            # Create transformer for coordinate reprojection
            transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)

            # Transform polygons and create features
            features = []
            for polygon, prop in zip(partitioned_polygons, partitioned_properties):
                # Transform polygon coordinates to EPSG:4326
                transformed_polygon = shapely_transform(transformer.transform, polygon)

                feature = {
                    'type': 'Feature',
                    'geometry': mapping(transformed_polygon),
                    'properties': prop
                }
                features.append(feature)

            geojson_data = {
                'type': 'FeatureCollection',
                'features': features,
                'crs': {
                    'type': 'name',
                    'properties': {
                        'name': 'urn:ogc:def:crs:EPSG::4326'
                    }
                }
            }

            with open(partitioned_file_path, 'w') as f:
                json_module.dump(geojson_data, f, indent=2)

            print(f"   ├─ Transformed {len(features)} polygons to EPSG:4326")
            print(f"   └─ ✅ Saved partitioned GeoJSON: {partitioned_file_path}")
        except Exception as e:
            print(f"   └─ ⚠️  Failed to save partitioned file: {e}")
            import traceback
            traceback.print_exc()

    # Define output directory for cropped images in BldgGen2025
    bldggen_dir = Path(__file__).parent.parent.parent / "BldgGen2025"
    output_crop_dir = bldggen_dir / f"tiff_{geojson_name}_buf1"
    print(f"\n📁 Output directory: {output_crop_dir}")

    # Crop satellite image using partitioned GeoJSON file
    try:
        print(f"\n✂️  CROPPING SATELLITE IMAGE BY PARTITIONED BUILDING POLYGONS")
        print(f"   ├─ Vector (Partitioned GeoJSON): {partitioned_file_path}")
        print(f"   ├─ File exists: {partitioned_file_path.exists()}")
        print(f"   ├─ File size: {partitioned_file_path.stat().st_size if partitioned_file_path.exists() else 0} bytes")
        print(f"   ├─ Raster (Orthophoto): {file_path}")
        print(f"   ├─ Raster exists: {file_path.exists()}")
        print(f"   └─ Output directory: {output_crop_dir}")

        # Verify partitioned file has valid content
        if partitioned_file_path.exists():
            with open(partitioned_file_path, 'r') as f:
                import json as json_module
                partitioned_data = json_module.load(f)
                num_features = len(partitioned_data.get('features', []))
                print(f"   ├─ Number of features in partitioned file: {num_features}")
                if num_features > 0:
                    first_feature = partitioned_data['features'][0]
                    print(f"   ├─ First feature ID: {first_feature.get('properties', {}).get('id', 'NO ID')}")
                    print(f"   ├─ First feature geometry type: {first_feature.get('geometry', {}).get('type', 'UNKNOWN')}")
        else:
            print(f"   ⚠️  CRITICAL: Partitioned file does not exist!")
            raise FileNotFoundError(f"Partitioned file not found: {partitioned_file_path}")

        # Buffer distance in degrees (EPSG:4326)
        # 1e-5 degrees ≈ 1.1 meters at equator
        buffer_distance = 1e-5

        print(f"\n   🔄 Starting TIFF extraction...")
        extract_raster_by_polygons(
            vector_path=str(partitioned_file_path),
            raster_path=str(file_path),
            output_dir=str(output_crop_dir),
            buffer_distance=buffer_distance
        )

        # Check if any files were created
        if output_crop_dir.exists():
            output_files = list(output_crop_dir.glob("*.tif"))
            print(f"\n   ✅ IMAGE CROPPING COMPLETED")
            print(f"   ├─ Output directory: {output_crop_dir}")
            print(f"   ├─ Number of cropped images: {len(output_files)}")
            if output_files:
                print(f"   └─ Sample file: {output_files[0].name}")
            else:
                print(f"   └─ ⚠️  No .tif files were created!")
        else:
            print(f"   ⚠️  Output directory was not created!")

    except Exception as e:
        print(f"\n   ⚠️  IMAGE CROPPING FAILED: {e}")
        print(f"   ├─ Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print(f"   └─ Will continue without cropped images")

    # Process orthophoto metadata if needed
    if file_ext in ['.tif', '.tiff']:
        print(f"\n📊 Reading TIFF metadata...")
        time.sleep(0.5)

    # active_jobs[job_id]['progress'] = 30

    # Run BldgXL model inference to generate real LoD2
    try:
        print(f"\n🚀 STARTING REAL LOD2 MODEL GENERATION")
        print(f"   └─ Using BldgXL PyTorch model")
        active_jobs[job_id]['progress'] = 40

        # Use geojson_job_id instead of path to leverage caching
        # Pass the orthophoto file path (from the current job) to the model
        # Pass job_id for granular progress tracking during model inference
        output_mesh, output_mesh_bmqi, bldg_ids, bldg_meshes, properties, bg_output_mesh = run_model_inference(
            geojson_job_id,
            layer_name,
            orthophoto_path=str(file_path),
            job_id=job_id
        )

        active_jobs[job_id]['progress'] = 90

        # Create job output directory
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(exist_ok=True)

        # Save background mesh (roads, vegetation) separately for LOD3 use
        if bg_output_mesh is not None:
            bg_mesh_path = job_output_dir / f"{layer_name}_background.obj"
            print(f"\n   💾 Saving background mesh (roads + vegetation)...")
            try:
                bg_output_mesh.export(bg_mesh_path)
                print(f"   └─ ✅ Background mesh saved: {bg_mesh_path}")
            except Exception as e:
                print(f"   └─ ⚠️  Failed to save background mesh: {e}")

        # First create MTL file for materials
        # mtl_file = job_output_dir / "material.mtl"
        # create_mock_mtl_file(mtl_file, "LoD2")
        # print(f"\n💾 Preparing to save generated mesh...")
        # print(f"   ├─ MTL file created: {mtl_file}")

        # Save the first mesh (regular output)
        output_file = job_output_dir / f"{layer_name}_lod2.obj"
        print(f"   ├─ Output path (regular): {output_file}")
        print(f"   ├─ Writing OBJ file manually (demo-compatible format)...")

        # Extract vertices and faces from trimesh
        vertices = output_mesh.vertices
        faces = output_mesh.faces

        print(f"   ├─ Vertices: {len(vertices)}, Faces: {len(faces)}")
        output_mesh.export(output_file)

        # Save the second mesh (BMQI output)
        output_file_bmqi = job_output_dir / f"{layer_name}_lod2_bmqi.obj"
        print(f"   ├─ Output path (BMQI): {output_file_bmqi}")

        # Extract vertices and faces from bmqi mesh
        vertices_bmqi = output_mesh_bmqi.vertices
        faces_bmqi = output_mesh_bmqi.faces

        print(f"   ├─ BMQI Vertices: {len(vertices_bmqi)}, Faces: {len(faces_bmqi)}")
        output_mesh_bmqi.export(output_file_bmqi)

        # Save individual building meshes grouped by original building ID
        if bldg_meshes and properties:
            individual_meshes_dir = job_output_dir / "individual_buildings"
            individual_meshes_dir.mkdir(exist_ok=True)
            print(f"\n   📦 Saving {len(bldg_meshes)} individual building meshes...")
            print(f"   ├─ Output directory: {individual_meshes_dir}")

            # Group meshes by original_polygon_id
            from collections import defaultdict
            building_groups = defaultdict(list)

            for idx, (bldg_id, mesh, prop) in enumerate(zip(bldg_ids, bldg_meshes, properties)):
                # Get original building ID from properties
                original_id = prop.get('original_polygon_id', bldg_id)
                if not original_id:
                    original_id = f"unknown_{idx:04d}"

                building_groups[str(original_id)].append({
                    'partition_id': bldg_id,
                    'mesh': mesh,
                    'index': idx
                })

            print(f"   ├─ Found {len(building_groups)} original buildings with partitions")

            # Save each building's partitions in its own subfolder
            for original_id, partitions in building_groups.items():
                building_dir = individual_meshes_dir / str(original_id)
                building_dir.mkdir(exist_ok=True)

                print(f"   ├─ Building '{original_id}': {len(partitions)} partition(s)")

                for part_idx, part_data in enumerate(partitions):
                    partition_id = part_data['partition_id']
                    mesh = part_data['mesh']

                    # Use partition ID if available, otherwise use part index
                    if partition_id:
                        filename = f"{partition_id}.obj"
                    else:
                        filename = f"part_{part_idx:04d}.obj"

                    mesh_output_path = building_dir / filename

                    try:
                        mesh.export(mesh_output_path)
                        print(f"   │  └─ Saved: {original_id}/{filename} ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)")
                    except Exception as e:
                        print(f"   │  └─ ⚠️  Failed to save {original_id}/{filename}: {e}")

            print(f"   └─ ✅ Individual building meshes saved to: {individual_meshes_dir}")

        # Write OBJ file in simple format matching demo
        # with open(output_file, 'w') as f:
        #     # Write mtllib directive
        #     f.write('mtllib material.mtl\n')

        #     # Write vertices (only x, y, z - no colors)
        #     for v in vertices:
        #         f.write(f'v {v[0]} {v[1]} {v[2]}\n')

        #     # Write usemtl before faces
        #     f.write('usemtl mtl_0\n')

        #     # Write faces (1-indexed in OBJ format)
        #     for face in faces:
        #         f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')

        print(f"   ├─ File size: {output_file.stat().st_size / 1024:.2f} KB")
        print(f"   ├─ ✅ OBJ file written in demo-compatible format")

        # Store both output paths in job metadata
        active_jobs[job_id]['model_output'] = str(output_file)
        active_jobs[job_id]['model_output_bmqi'] = str(output_file_bmqi)
        print(f"   ├─ Stored in job metadata: 'model_output'")
        print(f"   └─ Stored in job metadata: 'model_output_bmqi'")

        print(f"\n   ✅ REAL LOD2 MODELS SAVED SUCCESSFULLY (Regular + BMQI)")

    except Exception as e:
        print(f"\n   ❌ MODEL INFERENCE FAILED!")
        print(f"   ├─ Error: {e}")
        print(f"   ├─ Error type: {type(e).__name__}")
        print(f"   └─ Full traceback:")
        import traceback
        traceback.print_exc()
        print(f"\n   ⚠️  Will fall back to pseudo LoD2 model")

    active_jobs[job_id]['progress'] = 95
    print(f"\n   └─ Orthophoto processing complete")
    print(f"{'='*60}\n")

def process_pointcloud_file(job_id: str, file_path: Path):
    """Process point cloud file and extract real data"""
    print(f"☁️  Processing point cloud file...")
    active_jobs[job_id]['progress'] = 10
    
    file_ext = file_path.suffix.lower()
    print(f"   ├─ Point cloud format: {file_ext}")
    
    # Simulate reading file header/metadata
    print(f"   ├─ Reading file header...")
    active_jobs[job_id]['progress'] = 20
    time.sleep(0.5)
    
    if file_ext == '.ply':
        print(f"   ├─ PLY format detected")
        print(f"   ├─ Parsing PLY header...")
        active_jobs[job_id]['progress'] = 40
        time.sleep(1)
        
        # Could actually parse PLY header here
        print(f"   ├─ Estimated points: ~{file_path.stat().st_size // 50}")
        
    elif file_ext in ['.las', '.laz']:
        print(f"   ├─ LAS/LAZ format detected")
        print(f"   ├─ Reading LAS header...")
        active_jobs[job_id]['progress'] = 40
        time.sleep(1)
        
    else:
        print(f"   ├─ Generic point cloud format")
        active_jobs[job_id]['progress'] = 40
        time.sleep(0.5)
    
    print(f"   ├─ Analyzing point density...")
    active_jobs[job_id]['progress'] = 60
    time.sleep(1.5)
    
    print(f"   ├─ Detecting building structures...")
    active_jobs[job_id]['progress'] = 80
    time.sleep(1)
    
    print(f"   └─ Point cloud analysis complete")

def process_streetview_file(job_id: str, file_path: Path):
    """Process streetview image file and extract real data"""
    print(f"🏙️  Processing streetview image file...")
    active_jobs[job_id]['progress'] = 10
    
    file_ext = file_path.suffix.lower()
    print(f"   ├─ Image format: {file_ext}")
    
    # Read basic image information
    print(f"   ├─ Reading image metadata...")
    active_jobs[job_id]['progress'] = 20
    time.sleep(0.5)
    
    try:
        # Try to read image with PIL if available
        # from PIL import Image, ExifTags
        # img = Image.open(file_path)
        # print(f"   ├─ Image dimensions: {img.size}")
        
        # For now, simulate image processing
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"   ├─ File size: {file_size_mb:.2f} MB")
        
        active_jobs[job_id]['progress'] = 40
        time.sleep(1)
        
        print(f"   ├─ Analyzing street-level features...")
        active_jobs[job_id]['progress'] = 60
        time.sleep(1.5)
        
        print(f"   ├─ Detecting building facades...")
        active_jobs[job_id]['progress'] = 75
        time.sleep(1)
        
        print(f"   ├─ Extracting architectural details...")
        active_jobs[job_id]['progress'] = 80
        time.sleep(0.5)
        
    except Exception as e:
        print(f"   ├─ Streetview processing error: {e}")
        active_jobs[job_id]['progress'] = 60
    
    print(f"   └─ Streetview image analysis complete")

def process_lod3_combined(job_id: str, pointcloud_dir: Path, streetview_dir: Path, layer_name: str):
    """Process combined LoD3 data (pointcloud folder + streetview folder) into single model"""
    try:
        print(f"🏗️  Processing combined LoD3 data for job {job_id}...")
        active_jobs[job_id]['status'] = 'processing'
        active_jobs[job_id]['progress'] = 10

        # List files in directories
        pc_files = list(pointcloud_dir.glob('*')) if pointcloud_dir.exists() else []
        pc_files = [f for f in pc_files if f.suffix.lower() in ['.ply', '.las']]
        sv_files = list(streetview_dir.glob('*')) if streetview_dir.exists() else []
        sv_files = [f for f in sv_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]

        print(f"   ├─ Point cloud folder: {pointcloud_dir.name} ({len(pc_files)} PLY/LAS files)")
        print(f"   ├─ Streetview folder: {streetview_dir.name} ({len(sv_files)} images)")

        # Find the LOD2 job for this layer to get individual_buildings folder
        print(f"\n🔍 Looking for LOD2 job with layer: '{layer_name}'")
        lod2_job = None
        lod2_job_id = None

        # First try exact layer name match
        for jid, job in active_jobs.items():
            if job.get('file_type') == 'orthophoto' and job.get('layer_name') == layer_name:
                lod2_job = job
                lod2_job_id = jid
                break

        # If no exact match, find the most recent LOD2 job with individual_buildings folder
        if not lod2_job:
            print(f"   ⚠️  No exact layer name match")
            print(f"   ├─ Searching for most recent LOD2 job with individual_buildings...")

            lod2_jobs = [(jid, job) for jid, job in active_jobs.items()
                         if job.get('file_type') == 'orthophoto' and job.get('status') == 'completed']

            # Sort by creation time (most recent first)
            lod2_jobs.sort(key=lambda x: x[1].get('created_at', ''), reverse=True)

            # Find the first one with individual_buildings folder
            for jid, job in lod2_jobs:
                potential_dir = OUTPUT_DIR / jid / "individual_buildings"
                if potential_dir.exists():
                    lod2_job = job
                    lod2_job_id = jid
                    print(f"   ├─ Found LOD2 job: {jid}")
                    print(f"   └─ Layer: '{job.get('layer_name')}'")
                    break

        if not lod2_job:
            print(f"   ⚠️  No LOD2 job found with individual_buildings folder")
            print(f"   └─ Falling back to demo file")
            output_file = generate_output_file(job_id, 'lod3-data', layer_name)
            active_jobs[job_id]['status'] = 'completed'
            active_jobs[job_id]['progress'] = 100
            active_jobs[job_id]['output_file'] = str(output_file)
            active_jobs[job_id]['completed_at'] = datetime.now().isoformat()
            return

        print(f"   ✅ Found LOD2 job: {lod2_job_id}")
        print(f"   ├─ Layer name: '{lod2_job.get('layer_name')}'")

        # Get individual_buildings folder
        individual_buildings_dir = OUTPUT_DIR / lod2_job_id / "individual_buildings"
        print(f"   └─ Individual buildings folder: {individual_buildings_dir}")
        active_jobs[job_id]['progress'] = 20

        # Load building data JSON
        bldg_data_path = Path(__file__).parent.parent.parent / "BldgGen2025" / "updated_bldg_data.json"
        print(f"\n📄 Loading building data from: {bldg_data_path}")

        building_data_dict = {}
        if bldg_data_path.exists():
            with open(bldg_data_path, 'r') as f:
                bldg_data = json.load(f)
                # Create lookup dictionary by building ID
                for bldg in bldg_data.get('bldgs', []):
                    building_data_dict[bldg['id']] = bldg
            print(f"   ✅ Loaded {len(building_data_dict)} buildings from JSON")
        else:
            print(f"   ⚠️  Building data JSON not found")

        active_jobs[job_id]['progress'] = 30

        # Import LOD1 generator
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "BldgGen2025"))
        from opening_handling_2025 import LOD1BuildingGenerator
        generator = LOD1BuildingGenerator()

        # Collect all meshes
        all_meshes = []

        # Iterate through individual_buildings folder
        print(f"\n🏢 Processing individual buildings...")
        building_folders = [f for f in individual_buildings_dir.iterdir() if f.is_dir()]
        total_buildings = len(building_folders)
        print(f"   ├─ Found {total_buildings} building folders")

        for idx, building_folder in enumerate(building_folders):
            building_id = building_folder.name
            print(f"\n   [{idx+1}/{total_buildings}] Processing building: {building_id}")

            # Check if this building has data in JSON
            if building_id in building_data_dict:
                print(f"      ├─ Found in JSON - creating LOD1 with openings")
                try:
                    mesh = generator.generate_building_with_openings(building_data_dict[building_id])
                    mesh.vertices = mesh.vertices[:, [0, 2, 1]] * [1, 1, -1]
                    all_meshes.append(mesh)
                    print(f"      └─ ✅ LOD1 with openings created ({len(mesh.vertices)} vertices)")
                except Exception as e:
                    print(f"      └─ ⚠️  Failed to create LOD1: {e}")
                    # Fall back to reading partitions
                    print(f"      └─ Falling back to reading partition files")
                    partition_meshes = load_partition_meshes(building_folder)
                    all_meshes.extend(partition_meshes)
            else:
                print(f"      ├─ Not in JSON - using partition files")
                partition_meshes = load_partition_meshes(building_folder)
                all_meshes.extend(partition_meshes)
                print(f"      └─ ✅ Loaded {len(partition_meshes)} partition(s)")

            # Update progress
            progress = 30 + int((idx + 1) / total_buildings * 50)
            active_jobs[job_id]['progress'] = progress
            
            time.sleep(1.0)

        active_jobs[job_id]['progress'] = 85

        # Load background mesh (roads, vegetation) from LOD2 job if available
        bg_mesh_path = OUTPUT_DIR / lod2_job_id / f"{lod2_job.get('layer_name')}_background.obj"
        print(f"\n🌳 Loading background mesh (roads + vegetation)...")
        print(f"   ├─ Expected path: {bg_mesh_path}")

        if bg_mesh_path.exists():
            try:
                bg_mesh = trimesh.load(bg_mesh_path)
                all_meshes.append(bg_mesh)
                print(f"   └─ ✅ Background mesh loaded ({len(bg_mesh.vertices)} vertices)")
            except Exception as e:
                print(f"   └─ ⚠️  Failed to load background mesh: {e}")
        else:
            print(f"   └─ ⚠️  Background mesh not found")

        # Combine all meshes
        print(f"\n🔨 Combining {len(all_meshes)} meshes...")
        if not all_meshes:
            raise Exception("No meshes to combine")

        combined_mesh = trimesh.util.concatenate(all_meshes)
        print(f"   ✅ Combined mesh created ({len(combined_mesh.vertices)} vertices, {len(combined_mesh.faces)} faces)")

        # Export to output file
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(exist_ok=True)

        output_file = job_output_dir / f"{layer_name}_lod3.obj"
        combined_mesh.export(output_file)
        print(f"   ✅ Exported to: {output_file}")

        active_jobs[job_id]['status'] = 'completed'
        active_jobs[job_id]['progress'] = 100
        active_jobs[job_id]['output_file'] = str(output_file)
        active_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        print(f"   └─ ✅ Combined LoD3 model generated: {output_file}")

    except Exception as e:
        print(f"   └─ ❌ Error processing combined LoD3: {str(e)}")
        import traceback
        traceback.print_exc()
        active_jobs[job_id]['status'] = 'failed'
        active_jobs[job_id]['error'] = str(e)


def load_partition_meshes(building_folder: Path) -> list:
    """Load all partition mesh files from a building folder"""
    meshes = []
    obj_files = list(building_folder.glob("*.obj"))

    for obj_file in obj_files:
        try:
            mesh = trimesh.load(obj_file)
            meshes.append(mesh)
        except Exception as e:
            print(f"         ⚠️  Failed to load {obj_file.name}: {e}")

    return meshes



def process_generic_file(job_id: str, file_path: Path):
    """Process any other file type"""
    print(f"📄 Processing generic file...")
    active_jobs[job_id]['progress'] = 20
    time.sleep(0.5)
    
    # Read first few lines/bytes to analyze content
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_lines = [f.readline().strip() for _ in range(5)]
            print(f"   ├─ First few lines:")
            for i, line in enumerate(first_lines):
                if line:
                    preview = line[:60] + "..." if len(line) > 60 else line
                    print(f"   │  {i+1}: {preview}")
        
        active_jobs[job_id]['progress'] = 60
        time.sleep(1)
        
    except Exception as e:
        print(f"   ├─ Could not read as text: {e}")
        print(f"   ├─ Processing as binary file...")
        active_jobs[job_id]['progress'] = 60
        time.sleep(0.5)
    
    print(f"   └─ Generic file analysis complete")

def generate_output_file(job_id: str, file_type: str, layer_name: str) -> Path:
    """Generate a mock output file based on input type"""

    print(f"\n🔨 GENERATE_OUTPUT_FILE called")
    print(f"   ├─ Job ID: {job_id}")
    print(f"   ├─ File type: {file_type}")
    print(f"   └─ Layer name: '{layer_name}'")

    # Check if model has already generated output
    if job_id in active_jobs and 'model_output' in active_jobs[job_id]:
        model_output_path = Path(active_jobs[job_id]['model_output'])
        print(f"\n   ✅ MODEL OUTPUT FOUND IN JOB METADATA")
        print(f"   ├─ Path: {model_output_path}")
        print(f"   └─ Exists: {model_output_path.exists()}")

        if model_output_path.exists():
            print(f"   ✅ USING MODEL-GENERATED OUTPUT")
            return model_output_path
        else:
            print(f"   ⚠️  Model output path exists in metadata but file not found!")

    else:
        print(f"\n   ℹ️  No 'model_output' in job metadata")
        if job_id in active_jobs:
            print(f"   ├─ Job metadata keys: {list(active_jobs[job_id].keys())}")

    # Create job-specific output directory
    job_output_dir = OUTPUT_DIR / job_id
    job_output_dir.mkdir(exist_ok=True)

    if file_type == 'geojson':
        # GeoJSON processing complete - no backend LoD1 file generated
        # LoD1 buildings are generated in frontend from GeoJSON data
        print(f"\n   ℹ️  GeoJSON processing complete - LoD1 buildings handled by frontend")

        # Check if city assets were generated
        if 'city_assets_output' in active_jobs[job_id]:
            city_assets_path = Path(active_jobs[job_id]['city_assets_output'])
            print(f"   ├─ City assets available: {city_assets_path}")
            # Return city assets as the primary output for GeoJSON jobs
            output_file = city_assets_path
        else:
            # No output file for pure GeoJSON processing
            print(f"   └─ No backend-generated output file")
            # Create a simple JSON status file to satisfy the output requirement
            output_file = job_output_dir / f"{layer_name}_status.json"
            with open(output_file, 'w') as f:
                import json
                json.dump({
                    'type': 'geojson_processing_complete',
                    'layer_name': layer_name,
                    'job_id': job_id,
                    'message': 'GeoJSON processed successfully. LoD1 buildings generated in frontend.',
                    'city_assets_generated': 'city_assets_output' in active_jobs[job_id]
                }, f, indent=2)

    elif file_type == 'orthophoto':
        # Generate LoD2 model (OBJ + MTL files)
        # Note: Real LoD2 should already be generated by model in process_orthophoto_file
        output_file = job_output_dir / f"{layer_name}_lod2.obj"

        print(f"\n   🔍 Checking if real LoD2 model exists at: {output_file}")
        # Check if real model output exists
        if not output_file.exists():
            print(f"   ⚠️  REAL LOD2 MODEL NOT FOUND")
            print(f"   └─ Creating MOCK/DEMO file as fallback")
            create_mock_obj_file(output_file, "LoD2")
            mtl_file = job_output_dir / "material.mtl"
            create_mock_mtl_file(mtl_file, "LoD2")
        else:
            print(f"   ✅ REAL LOD2 MODEL ALREADY EXISTS")
            print(f"   └─ File size: {output_file.stat().st_size / 1024:.2f} KB")
        
    elif file_type == 'pointcloud':
        # Generate LoD3 model (OBJ + MTL files)
        output_file = job_output_dir / f"{layer_name}_lod3.obj"
        mtl_file = job_output_dir / "material.mtl"
        create_mock_obj_file(output_file, "LoD3")
        create_mock_mtl_file(mtl_file, "LoD3")
        
    elif file_type == 'streetview':
        # Generate LoD3 model from streetview (OBJ + MTL files)
        output_file = job_output_dir / f"{layer_name}_lod3.obj"
        mtl_file = job_output_dir / "material.mtl"
        create_mock_obj_file(output_file, "LoD3")
        create_mock_mtl_file(mtl_file, "LoD3")
        
    elif file_type == 'lod3-data':
        # Load demo LoD3 model from demo-lod3-new directory
        print(f"\n📦 Loading demo LoD3 model from demo-lod3-new...")
        demo_lod3_path = Path(__file__).parent.parent / "demo-lod3-new" / "akabane_lod3_route1.obj"

        # Use "Combined_LoD3" as default layer name if not provided
        final_layer_name = layer_name if layer_name else "Combined_LoD3"
        output_file = job_output_dir / f"{final_layer_name}_lod3.obj"
        mtl_file = job_output_dir / "material.mtl"

        try:
            # Copy the demo file to the output directory
            shutil.copy2(demo_lod3_path, output_file)
            print(f"   ├─ Demo file: {demo_lod3_path}")
            print(f"   ├─ Copied to: {output_file}")
            print(f"   ├─ File size: {output_file.stat().st_size / 1024:.2f} KB")

            # Create a material.mtl file for LOD3 (even though OBJ has vertex colors)
            # This ensures consistent behavior with LOD2 models
            create_mock_mtl_file(mtl_file, "LoD3")
            print(f"   ├─ Created material file: {mtl_file}")
            print(f"   └─ ✅ Demo LoD3 model loaded successfully!")
        except FileNotFoundError:
            print(f"   ⚠️  Demo LoD3 file not found at: {demo_lod3_path}")
            print(f"   └─ Creating mock LoD3 file as fallback...")
            # Fallback to mock file if demo not found
            create_mock_obj_file(output_file, "LoD3")
            create_mock_mtl_file(mtl_file, "LoD3")
        except Exception as e:
            print(f"   ❌ Error copying demo LoD3 file: {e}")
            print(f"   └─ Creating mock LoD3 file as fallback...")
            # Fallback to mock file on any error
            create_mock_obj_file(output_file, "LoD3")
            create_mock_mtl_file(mtl_file, "LoD3")
        
    else:
        # Generic processed file
        output_file = job_output_dir / f"{layer_name}_processed.txt"
        with open(output_file, 'w') as f:
            f.write(f"Processed file: {layer_name}\nType: {file_type}\nJob ID: {job_id}\n")
    
    return output_file

def create_mock_obj_file(file_path: Path, lod_level: str):
    """Create a mock OBJ file using real route1 building data"""

    # Select the appropriate source file based on LoD level
    if lod_level == "LoD3":
        real_obj_path = Path(__file__).parent.parent / "demo-lod3" / "results_route1_lod3.obj"
    else:
        # LoD1 and LoD2 use the LoD2 demo file
        real_obj_path = Path(__file__).parent.parent / "demo-lod2" / "results_route1_lod2.obj"
    
    print(f"🔧 DEBUG: Creating mock OBJ file for {lod_level} at {file_path}")
    
    try:
        # Read the real OBJ file content
        with open(real_obj_path, 'r') as f:
            real_content = f.read()
        
        # Add a header with the LoD level info
        header = f"""# {lod_level} Model - Route1 Building
# Generated by BridgeUI Backend (Real Data Mock)
# Source: {real_obj_path.name}

"""
        
        final_content = header + real_content
        
        # Count vertices (fix the backslash issue)
        vertex_count = len([l for l in real_content.split('\n') if l.strip().startswith('v ')])
        
        print(f"   ├─ Using real OBJ data from: {real_obj_path.name}")
        print(f"   ├─ Real OBJ file size: {len(real_content)} characters")
        print(f"   ├─ Vertices in source: {vertex_count}")
        
    except FileNotFoundError:
        print(f"   ⚠️  Real OBJ file not found at: {real_obj_path}")
        print(f"   ├─ Falling back to simple cube geometry")
        
        # Fallback to original simple cube if real file not found
        final_content = f"""# Mock {lod_level} Model
# Generated by BridgeUI Backend
# Vertices (simple cube)
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0
v 1.0 0.0 1.0
v 1.0 1.0 1.0
v 0.0 1.0 1.0

# Faces
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
"""
    
    except Exception as e:
        print(f"   ❌ Error reading real OBJ file: {e}")
        print(f"   ├─ Falling back to simple cube geometry")
        
        # Fallback to original simple cube if any error occurs
        final_content = f"""# Mock {lod_level} Model (Fallback)
# Generated by BridgeUI Backend
# Vertices (simple cube)
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0
v 1.0 0.0 1.0
v 1.0 1.0 1.0
v 0.0 1.0 1.0

# Faces
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
"""
    
    # Write the final content
    with open(file_path, 'w') as f:
        f.write(final_content)
    
    # Debug: Verify what was written
    print(f"   ├─ Output file written: {file_path}")
    print(f"   ├─ Output file size: {file_path.stat().st_size} bytes")
    
    # Count vertices in the written file
    with open(file_path, 'r') as f:
        written_content = f.read()
        vertex_lines = [line for line in written_content.split('\n') if line.strip().startswith('v ')]
        print(f"   ├─ Vertices written: {len(vertex_lines)}")
        if vertex_lines:
            print(f"   ├─ First vertex: {vertex_lines[0]}")
            print(f"   └─ Last vertex: {vertex_lines[-1] if len(vertex_lines) > 1 else 'Same as first'}")
        else:
            print(f"   └─ ❌ NO VERTICES FOUND IN OUTPUT FILE!")

def create_mock_mtl_file(file_path: Path, lod_level: str = "LoD2"):
    """Create a material file by copying the real material file from demo-lod2 or demo-lod3"""

    # Select the appropriate source MTL file based on LoD level
    if lod_level == "LoD3":
        real_mtl_path = Path(__file__).parent.parent / "demo-lod3" / "material.mtl"
    else:
        # LoD1 and LoD2 use the LoD2 material file
        real_mtl_path = Path(__file__).parent.parent / "demo-lod2" / "material.mtl"

    print(f"🎨 Creating MTL file for {lod_level} at {file_path}")

    try:
        # Copy the real MTL file content
        with open(real_mtl_path, 'r') as f:
            mtl_content = f.read()

        # Write to output location
        with open(file_path, 'w') as f:
            f.write(mtl_content)

        print(f"   ├─ Using real MTL data from: {real_mtl_path.name}")
        print(f"   └─ MTL file size: {file_path.stat().st_size} bytes")

    except FileNotFoundError:
        print(f"   ⚠️  Real MTL file not found at: {real_mtl_path}")
        print(f"   ├─ Creating basic MTL file with default materials")

        # Fallback to basic material if real file not found
        basic_mtl = """# Basic Material File
newmtl default
Ka 0.8 0.8 0.8
Kd 0.8 0.8 0.8
Ks 0.0 0.0 0.0
Ns 10.0
d 1.0
"""
        with open(file_path, 'w') as f:
            f.write(basic_mtl)

        print(f"   └─ Basic MTL file created")

    except Exception as e:
        print(f"   ❌ Error creating MTL file: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get additional parameters
    file_type = request.form.get('type', 'unknown')
    layer_name = request.form.get('layerName', 'untitled')
    
    # Validate file type
    if not allowed_file(file.filename, file_type):
        return jsonify({'error': f'File type not allowed for {file_type}'}), 400
    
    # Generate job ID and secure filename
    job_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{job_id}_{filename}"
    file.save(str(file_path))
    
    # Create job record
    active_jobs[job_id] = {
        'job_id': job_id,
        'filename': filename,
        'file_type': file_type,
        'layer_name': layer_name,
        'status': 'uploaded',
        'progress': 0,
        'created_at': datetime.now().isoformat(),
        'file_path': str(file_path)
    }
    
    # Start real file processing in background
    import threading
    processing_thread = threading.Thread(
        target=process_uploaded_file, 
        args=(job_id, file_path, file_type, layer_name)
    )
    processing_thread.daemon = True
    processing_thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'uploaded',
        'message': f'File {filename} uploaded successfully and processing started'
    })

@app.route('/upload-lod3', methods=['POST'])
def upload_lod3_files():
    """Handle combined LoD3 folder upload (pointcloud folder + streetview folder)"""

    # Get file counts from form data
    pointcloud_count = int(request.form.get('pointcloud_count', 0))
    streetview_count = int(request.form.get('streetview_count', 0))

    if pointcloud_count == 0 or streetview_count == 0:
        return jsonify({'error': 'Both pointcloud and streetview folders are required'}), 400

    # Get folder names
    pointcloud_folder = request.form.get('pointcloud_folder', 'pointcloud')
    streetview_folder = request.form.get('streetview_folder', 'streetview')
    layer_name = request.form.get('layerName', 'Combined_LoD3')

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Create directories for uploaded files
    pc_dir = UPLOAD_DIR / job_id / "pointcloud"
    sv_dir = UPLOAD_DIR / job_id / "streetview"
    pc_dir.mkdir(parents=True, exist_ok=True)
    sv_dir.mkdir(parents=True, exist_ok=True)

    # Save all pointcloud files
    pc_files = []
    for i in range(pointcloud_count):
        file_key = f'pointcloud_{i}'
        if file_key in request.files:
            file = request.files[file_key]
            filename = secure_filename(file.filename)
            file_path = pc_dir / filename
            file.save(str(file_path))
            pc_files.append(str(file_path))

    # Save all streetview files
    sv_files = []
    for i in range(streetview_count):
        file_key = f'streetview_{i}'
        if file_key in request.files:
            file = request.files[file_key]
            filename = secure_filename(file.filename)
            file_path = sv_dir / filename
            file.save(str(file_path))
            sv_files.append(str(file_path))

    print(f"📁 Received LoD3 folders:")
    print(f"   ├─ Pointcloud: {pointcloud_folder} ({len(pc_files)} files)")
    print(f"   └─ Streetview: {streetview_folder} ({len(sv_files)} files)")

    # Create job record
    active_jobs[job_id] = {
        'job_id': job_id,
        'pointcloud_folder': pointcloud_folder,
        'streetview_folder': streetview_folder,
        'pointcloud_files': pc_files,
        'streetview_files': sv_files,
        'pointcloud_count': len(pc_files),
        'streetview_count': len(sv_files),
        'file_type': 'lod3-data',
        'layer_name': layer_name,
        'status': 'uploaded',
        'progress': 0,
        'created_at': datetime.now().isoformat(),
        'pointcloud_dir': str(pc_dir),
        'streetview_dir': str(sv_dir)
    }

    # Start combined LoD3 processing in background
    import threading
    processing_thread = threading.Thread(
        target=process_lod3_combined,
        args=(job_id, pc_dir, sv_dir, layer_name)
    )
    processing_thread.daemon = True
    processing_thread.start()

    return jsonify({
        'job_id': job_id,
        'status': 'uploaded',
        'message': f'LoD3 folders uploaded successfully ({len(pc_files)} point cloud files, {len(sv_files)} images)'
    })

@app.route('/jobs/<job_id>/status', methods=['GET'])
def get_job_status(job_id: str):
    """Get the status of a processing job"""
    
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = active_jobs[job_id]
    return jsonify(job)

@app.route('/jobs/<job_id>/download', methods=['GET'])
def download_result(job_id: str):
    """Download the processed result file"""
    
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = active_jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({'error': 'Job not completed yet'}), 400
    
    if 'output_file' not in job:
        return jsonify({'error': 'No output file available'}), 500
    
    output_file = Path(job['output_file'])
    
    if not output_file.exists():
        return jsonify({'error': 'Output file not found'}), 500
    
    return send_file(str(output_file), as_attachment=True)

@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all jobs and their status"""
    
    return jsonify({
        'jobs': list(active_jobs.values()),
        'total': len(active_jobs)
    })

@app.route('/jobs/<job_id>', methods=['DELETE'])
def delete_job(job_id: str):
    """Delete a job and its associated files"""
    
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = active_jobs[job_id]
    
    # Clean up files
    try:
        # Remove uploaded file
        if 'file_path' in job:
            file_path = Path(job['file_path'])
            if file_path.exists():
                file_path.unlink()
        
        # Remove output files
        job_output_dir = OUTPUT_DIR / job_id
        if job_output_dir.exists():
            shutil.rmtree(job_output_dir)
        
        # Remove job record
        del active_jobs[job_id]
        
        return jsonify({'message': 'Job deleted successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Failed to delete job: {str(e)}'}), 500

@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    """Serve files from the outputs directory"""
    try:
        # Use absolute path to outputs directory (parent of backend dir)
        outputs_dir = Path(__file__).parent.parent / 'outputs'
        return send_from_directory(str(outputs_dir), filename)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/cleanup', methods=['POST'])
def cleanup_directories():
    """Clear uploads and outputs directories and reset active jobs"""
    global active_jobs

    try:
        # Clear uploads directory
        if UPLOAD_DIR.exists():
            for item in UPLOAD_DIR.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

        # Clear outputs directory
        if OUTPUT_DIR.exists():
            for item in OUTPUT_DIR.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

        # Reset active jobs
        active_jobs = {}

        print("🧹 Cleaned up uploads and outputs directories")
        return jsonify({'message': 'Cleanup completed successfully'})

    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

# Load model at startup (only in the reloader subprocess, not the parent watcher)
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    print("=" * 60)
    print("🤖 Initializing BldgXL Model...")
    print("=" * 60)
    model = load_model()
    print("=" * 60)

if __name__ == '__main__':
    # Only print startup messages in the main process
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        print("🚀 Starting BridgeUI Backend Server...")
        print(f"📁 Upload directory: {UPLOAD_DIR.absolute()}")
        print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
        print("🌐 Server will be available at: http://localhost:5001")

    # Bind to localhost only - backend not accessible from network
    # Frontend will proxy requests to backend on the same machine
    app.run(debug=True, host='127.0.0.1', port=5001)