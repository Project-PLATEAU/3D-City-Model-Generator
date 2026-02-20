import pandas as pd
import shutil
import os
from pathlib import Path

def copy_labeled_obj_files(csv_path, source_folder, destination_folder, target_label=1):
    """
    Copy OBJ files from source to destination folder based on labels in CSV.
    
    Args:
        csv_path (str): Path to CSV file with 'ID' and 'label' columns
        source_folder (str): Path to folder containing OBJ files
        destination_folder (str): Path to destination folder
        target_label (int): Label value to filter for (default: 1)
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Filter rows with target label
    target_files = df[df['label'] == target_label]['id'].tolist()
    
    # Create destination folder if it doesn't exist
    Path(destination_folder).mkdir(parents=True, exist_ok=True)
    
    # Convert paths to Path objects for easier handling
    source_path = Path(source_folder)
    dest_path = Path(destination_folder)
    
    copied_count = 0
    missing_files = []
    
    print(f"Found {len(target_files)} files with label {target_label}")
    
    for file_id in target_files:
        # Try different possible filename formats
        possible_names = [
            f"{file_id}.obj",
            f"{file_id}.OBJ",
            file_id if file_id.endswith(('.obj', '.OBJ')) else None
        ]
        
        file_found = False
        for filename in possible_names:
            if filename is None:
                continue
                
            source_file = source_path / filename
            if source_file.exists():
                destination_file = dest_path / filename
                shutil.copy2(source_file, destination_file)
                print(f"Copied: {filename}")
                copied_count += 1
                file_found = True
                break
        
        if not file_found:
            missing_files.append(file_id)
    
    # Summary
    print(f"\nSummary:")
    print(f"Successfully copied: {copied_count} files")
    if missing_files:
        print(f"Missing files: {len(missing_files)}")
        print("Missing file IDs:", missing_files[:10])  # Show first 10
        if len(missing_files) > 10:
            print(f"... and {len(missing_files) - 10} more")

# Example usage
if __name__ == "__main__":
    # Update these paths according to your setup
    csv_file = "obj_clean_labeled/combined_data.csv"  # Your CSV file with ID and label columns
    source_dir = "obj_unlabeled_flattened"  # Folder containing OBJ files
    dest_dir = "obj_unlabeled_cleaned"  # Destination folder
    
    copy_labeled_obj_files(csv_file, source_dir, dest_dir, target_label=1)