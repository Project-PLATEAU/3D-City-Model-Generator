
import os
import glob
import csv
import pandas as pd
import numpy as np

def type_map_bdg(data):
    data[(data == 5) | (data == 6) | (data == 7) | (data == 9) | (data == 12) | (data == 13)] = 4
    data[(data == 8) | (data == 11)] = 5
    data[(data == 10)] = 6
    return data

def get_building_type_name(type_id):
    """
    Map building type ID to descriptive name.

    Args:
        type_id (int): Building type ID

    Returns:
        str: Human-readable building type name
    """
    type_mapping = {
        1: "type 1",
        2: "type 2",
        3: "type 3",
        4: "type 4",  # mapped from 5,6,7,9,12,13
        5: "type 5",  # mapped from 8,11
        6: "type 6"  # mapped from 10
    }
    return type_mapping.get(type_id, "unknown building type")

def load_building_labels(csv_path):
    """
    Load building labels from CSV file and apply type mapping.

    Args:
        csv_path (str): Path to the CSV file containing building labels

    Returns:
        dict: Dictionary mapping building IDs to mapped type IDs
    """
    labels = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                building_id = row['id']
                original_type = int(row['type'])

                # Apply type mapping using existing function
                mapped_type = type_map_bdg(np.array([original_type]))[0]
                labels[building_id] = mapped_type

    except Exception as e:
        print(f"Error loading labels from {csv_path}: {e}")

    return labels

def read_obj_file(filepath):
    """
    Read an OBJ file and extract basic information about vertices, faces, and materials.

    Args:
        filepath (str): Path to the OBJ file

    Returns:
        dict: Dictionary containing OBJ file information
    """
    obj_info = {
        'vertices': [],
        'faces': [],
        'materials': [],
        'comments': [],
        'vertex_count': 0,
        'face_count': 0,
        'bounding_box': {'min': [float('inf')] * 3, 'max': [float('-inf')] * 3}
    }

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if not parts:
                    continue

                # Parse vertices
                if parts[0] == 'v':
                    if len(parts) >= 4:
                        vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                        obj_info['vertices'].append(vertex)
                        obj_info['vertex_count'] += 1

                        # Update bounding box
                        for i in range(3):
                            obj_info['bounding_box']['min'][i] = min(obj_info['bounding_box']['min'][i], vertex[i])
                            obj_info['bounding_box']['max'][i] = max(obj_info['bounding_box']['max'][i], vertex[i])

                # Parse faces
                elif parts[0] == 'f':
                    obj_info['faces'].append(parts[1:])
                    obj_info['face_count'] += 1

                # Parse material library
                elif parts[0] == 'mtllib':
                    if len(parts) > 1:
                        obj_info['materials'].append(parts[1])

                # Parse comments
                elif parts[0] == '#':
                    obj_info['comments'].append(' '.join(parts[1:]))

        # Calculate dimensions
        if obj_info['vertex_count'] > 0:
            obj_info['dimensions'] = [
                obj_info['bounding_box']['max'][i] - obj_info['bounding_box']['min'][i]
                for i in range(3)
            ]
            obj_info['volume_estimate'] = obj_info['dimensions'][0] * obj_info['dimensions'][1] * obj_info['dimensions'][2]
        else:
            obj_info['dimensions'] = [0, 0, 0]
            obj_info['volume_estimate'] = 0

    except Exception as e:
        obj_info['error'] = str(e)

    return obj_info

def generate_text_description(obj_info, filename, building_type=None):
    """
    Generate a text description for an OBJ file based on its properties.

    Args:
        obj_info (dict): Dictionary containing OBJ file information
        filename (str): Name of the OBJ file
        building_type (str, optional): Building type description

    Returns:
        str: Generated text description
    """
    if 'error' in obj_info:
        return f"Error processing {filename}: {obj_info['error']}"

    # Extract building ID from filename
    building_id = filename.replace('.obj', '')
    # .replace('bldg_', '').replace('_lod2', '')

    # Generate description based on file properties
    description_parts = []

    # Basic building description with type
    # if building_type:
    #     description_parts.append(f"Building {building_id} is a {building_type} represented as a 3D architectural model")
    # else:
    #     description_parts.append(f"Building {building_id} is a 3D architectural model")

    # Geometric complexity
    # if obj_info['vertex_count'] > 0:
    #     if obj_info['vertex_count'] < 30:
    #         complexity = "simple"
    #     elif obj_info['vertex_count'] < 60:
    #         complexity = "moderate"
    #     elif obj_info['vertex_count'] < 100:
    #         complexity = "detailed"
    #     else:
    #         complexity = "highly detailed"

    #     description_parts.append(f"with {complexity} geometry containing {obj_info['vertex_count']} vertices and {obj_info['face_count']} faces")

    # Dimensions description
    if obj_info['dimensions'][0] > 0:
        dims = obj_info['dimensions']
        width, depth, height = dims[0], dims[1], dims[2]

        # Convert to more readable units (assuming meters)
        # description_parts.append(f"measuring approximately {width:.1f}m × {depth:.1f}m × {height:.1f}m")

        # Building type inference based on dimensions
        if height > width and height > depth:
            dimensional_type = "tall building"
        elif width > height * 2 or depth > height * 2:
            dimensional_type = "low-profile building"
        else:
            dimensional_type = "mid-rise building"

        description_parts.append(f"{dimensional_type}")

    # LOD information
    # if 'lod2' in filename.lower():
    #     description_parts.append("created at Level of Detail 2 (LOD2) with textured surfaces and basic roof structures")

    # # Coordinate system information
    # coordinate_info = ""
    # for comment in obj_info.get('comments', []):
    #     if 'COORDINATE_SYSTEM' in comment and 'Japan' in comment:
    #         coordinate_info = "positioned using Japanese Geodetic Datum 2011 coordinate system"
    #         break

    # if coordinate_info:
    #     description_parts.append(coordinate_info)

    # # Material information
    # if obj_info.get('materials'):
    #     description_parts.append(f"with material definitions from {', '.join(obj_info['materials'])}")

    return ". ".join(description_parts) + "."

def process_obj_directory(directory_path, labels_csv_path=None, output_csv=None):
    """
    Process all OBJ files in a directory and generate text descriptions for them.

    Args:
        directory_path (str): Path to directory containing OBJ files
        labels_csv_path (str, optional): Path to CSV file containing building labels
        output_csv (str, optional): Path to output CSV file. If None, prints to console.

    Returns:
        list: List of dictionaries containing results
    """
    obj_files = glob.glob(os.path.join(directory_path, "*.obj"))
    results = []

    # Load building labels if provided
    building_labels = {}
    if labels_csv_path and os.path.exists(labels_csv_path):
        building_labels = load_building_labels(labels_csv_path)
        print(f"Loaded {len(building_labels)} building labels from {labels_csv_path}")

    print(f"Found {len(obj_files)} OBJ files in {directory_path}")

    for i, obj_file in enumerate(obj_files):
        filename = os.path.basename(obj_file)
        building_id = filename.replace('.obj', '')

        # Skip files not in the labels CSV if provided
        if labels_csv_path and building_id not in building_labels:
            print(f"Skipping {i+1}/{len(obj_files)}: {filename} (not found in labels)")
            continue

        print(f"Processing {i+1}/{len(obj_files)}: {filename}")

        # Read OBJ file
        obj_info = read_obj_file(obj_file)

        # Get building type if available
        building_type = None
        type_id = None
        if building_id in building_labels:
            type_id = building_labels[building_id]
            building_type = get_building_type_name(type_id)

        # Generate text description
        description = generate_text_description(obj_info, filename, building_type)

        # Create enhanced description that includes building type
        enhanced_description = description
        # if building_type and building_type not in description:
        enhanced_description = f"{building_type}, {description.rstrip('.')}."

        result = {
            # 'filename': filename,
            'building_id': building_id,
            # 'type_id': type_id,
            # 'building_type': building_type,
            'description': enhanced_description
            # 'vertex_count': obj_info.get('vertex_count', 0),
            # 'face_count': obj_info.get('face_count', 0),
            # 'dimensions_x': obj_info.get('dimensions', [0, 0, 0])[0],
            # 'dimensions_y': obj_info.get('dimensions', [0, 0, 0])[1],
            # 'dimensions_z': obj_info.get('dimensions', [0, 0, 0])[2]
        }

        results.append(result)

        if not output_csv:
            print(f"{filename}: {description}\n")

    # Save to CSV file if specified
    if output_csv and results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"Results saved to {output_csv}")
    elif output_csv:
        # Create empty CSV with headers
        pd.DataFrame(columns=['filename', 'building_id', 'type_id', 'building_type', 'description',
                             'vertex_count', 'face_count', 'dimensions_x', 'dimensions_y', 'dimensions_z']).to_csv(output_csv, index=False)
        print(f"No matching files found. Empty CSV created at {output_csv}")

    return results

def main():
    """
    Main function to process OBJ files in the current directory.
    """
    # current_dir = os.getcwd()
    # print(f"Processing OBJ files in: {current_dir}")

    # Find labels CSV file
    labels_csv_path = "/home/sekilab-liao/Documents/gen3d_2_0/BldgGen2024/BldgXL/obj_label.csv"
    current_dir = "/home/sekilab-liao/Documents/gen3d_2_0/BldgGen2024/BldgXL/obj_unlabeled_flattened"

    # Process all OBJ files and save to CSV file
    results = process_obj_directory(current_dir, labels_csv_path, "obj_descriptions.csv")

    print(f"Generated descriptions for {len(results)} OBJ files")
    return results

if __name__ == "__main__":
    main()