import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.io import MemoryFile
from rasterio.crs import CRS
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box
import numpy as np

def extract_raster_by_polygons(vector_path, raster_path, output_dir, buffer_distance=0):
    """
    Extract raster areas covered by polygons and save as separate files.
    Vector and raster are reprojected to EPSG:4326 for clipping.
    Each clip is then reprojected to EPSG:6668 and output as 3-band RGB.

    Args:
        vector_path: Path to vector file
        raster_path: Path to raster file
        output_dir: Output directory for cropped rasters
        buffer_distance: Buffer distance in degrees (applied in EPSG:4326). Default is 0.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    target_crs = "EPSG:4326"
    
    # Read vector layer and reproject to EPSG:4326
    gdf = gpd.read_file(vector_path)
    print(f"Original Vector CRS: {gdf.crs}")
    gdf_4326 = gdf.to_crs(target_crs)
    print(f"Vector bounds in EPSG:4326: {gdf_4326.total_bounds}")
    print(f"Number of polygons: {len(gdf_4326)}")
    print(f"Buffer distance: {buffer_distance} degrees")
    
    # Open raster file
    with rasterio.open(raster_path) as src:
        print(f"\nOriginal Raster CRS: {src.crs}")
        print(f"Raster size: {src.width} x {src.height}")
        print(f"Raster bounds: {src.bounds}")
        
        # Since raster is already in geographic coordinates (degrees),
        # we just need to ensure proper CRS metadata
        # The raster appears to already be in a geographic CRS (JGD2011)
        
        # Get raster bounds in EPSG:4326
        raster_bbox = box(*src.bounds)
        raster_gdf = gpd.GeoDataFrame({'geometry': [raster_bbox]}, crs=src.crs)
        raster_bbox_4326 = raster_gdf.to_crs(target_crs)
        bounds_4326 = raster_bbox_4326.total_bounds
        
        print(f"Raster bounds in EPSG:4326: {bounds_4326}")
        
        # Use calculate_default_transform with pyproj workaround
        from rasterio.crs import CRS as RioCRS
        from affine import Affine
        
        # Manual calculation of transform preserving original resolution
        src_width = src.width
        src_height = src.height
        
        # Calculate pixel size in degrees
        pixel_width = (bounds_4326[2] - bounds_4326[0]) / src_width
        pixel_height = (bounds_4326[3] - bounds_4326[1]) / src_height
        
        print(f"\nPixel size: {pixel_width} x {pixel_height} degrees")
        print(f"Output dimensions: {src_width} x {src_height}")
        
        # Create transform for EPSG:4326
        transform_4326 = Affine(
            pixel_width, 0.0, bounds_4326[0],
            0.0, -pixel_height, bounds_4326[3]
        )
        
        # Prepare output metadata
        out_meta = src.meta.copy()
        out_meta.update({
            'crs': target_crs,
            'transform': transform_4326,
            'width': src_width,
            'height': src_height
        })
        
        print(f"\nReprojecting raster to EPSG:4326...")
        
        # Create in-memory reprojected raster
        with MemoryFile() as memfile:
            with memfile.open(**out_meta) as dst:
                # Reproject each band
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform_4326,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear
                    )
                
                print("Reprojection complete!")
                
                # Check for overlap
                raster_bbox_4326_geom = box(*dst.bounds)
                overlapping = gdf_4326[gdf_4326.intersects(raster_bbox_4326_geom)]
                print(f"\nNumber of overlapping polygons: {len(overlapping)}")
                
                if len(overlapping) == 0:
                    print("ERROR: No polygons overlap with the raster!")
                    return
                
                # Process overlapping polygons
                success_count = 0
                for idx, row in overlapping.iterrows():
                    polygon_id = row.get('id', row.get('ID', row.get('fid', idx)))

                    # Apply buffer to the geometry if buffer_distance > 0
                    if buffer_distance > 0:
                        buffered_geom = row.geometry.buffer(buffer_distance)
                        geom = [buffered_geom.__geo_interface__]
                    else:
                        geom = [row.geometry.__geo_interface__]

                    try:
                        out_image, out_transform = mask(dst, geom, crop=True,
                                                       all_touched=True, nodata=src.nodata)

                        # Replace nodata pixels with 0
                        if src.nodata is not None:
                            out_image[out_image == src.nodata] = 0

                        # Ensure 3 bands (RGB)
                        if out_image.shape[0] == 1:
                            # Grayscale - duplicate to 3 bands
                            out_image = np.repeat(out_image, 3, axis=0)
                        elif out_image.shape[0] == 4:
                            # RGBA - take only RGB
                            out_image = out_image[:3, :, :]
                        elif out_image.shape[0] != 3:
                            # Other band counts - take first 3 or duplicate
                            if out_image.shape[0] < 3:
                                out_image = np.repeat(out_image[:1, :, :], 3, axis=0)
                            else:
                                out_image = out_image[:3, :, :]

                        # Calculate transform for EPSG:6668
                        dst_transform, dst_width, dst_height = calculate_default_transform(
                            target_crs,
                            "EPSG:6668",
                            out_image.shape[2],
                            out_image.shape[1],
                            *rasterio.transform.array_bounds(out_image.shape[1], out_image.shape[2], out_transform)
                        )

                        # Create reprojected output
                        reprojected = np.zeros((3, dst_height, dst_width), dtype=out_image.dtype)

                        # Reproject each band to EPSG:6668
                        for i in range(3):
                            reproject(
                                source=out_image[i],
                                destination=reprojected[i],
                                src_transform=out_transform,
                                src_crs=target_crs,
                                dst_transform=dst_transform,
                                dst_crs="EPSG:6668",
                                resampling=Resampling.bilinear
                            )

                        # Prepare output metadata
                        out_meta_clip = {
                            "driver": "GTiff",
                            "height": dst_height,
                            "width": dst_width,
                            "count": 3,
                            "dtype": reprojected.dtype,
                            "crs": "EPSG:6668",
                            "transform": dst_transform,
                            "nodata": None
                        }

                        output_path = Path(output_dir) / f"{polygon_id}.tif"
                        with rasterio.open(output_path, "w", **out_meta_clip) as dest:
                            dest.write(reprojected)

                        success_count += 1
                        if success_count % 100 == 0:
                            print(f"Processed {success_count} polygons...")

                    except ValueError as e:
                        print(f"Error with polygon {polygon_id}: {e}")
                
                print(f"\nCompleted! Successfully saved {success_count} rasters in EPSG:6668 (3-band RGB).")


if __name__ == "__main__":
    # Usage
    vector_file = "/Users/konialive/Downloads/Academic_file/doctoral/proj_PLATEAU_bridge/FY2025/202511_12/確認会１/data/確認会向け/kyoto/route1.geojson"
    raster_file = "/Users/konialive/Downloads/Academic_file/doctoral/proj_PLATEAU_bridge/FY2025/202511_12/確認会１/data/確認会向け/kyoto/route1.tif"
    output_directory = "tiff_kyoto_route1_buf1"

    # Buffer distance in degrees (EPSG:4326)
    # For example: 0.0001 degrees ≈ 11 meters at equator
    # Adjust this value based on your needs
    buffer_distance = 1e-5

    extract_raster_by_polygons(vector_file, raster_file, output_directory, buffer_distance)