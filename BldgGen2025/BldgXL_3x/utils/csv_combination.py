import pandas as pd
import glob
import os
from pathlib import Path

def combine_csv_files(folder_path, output_file, pattern="*.csv"):
    """
    Combine multiple CSV files with the same columns into one file.
    
    Args:
        folder_path (str): Path to folder containing CSV files
        output_file (str): Path for the combined output CSV file
        pattern (str): File pattern to match (default: "*.csv")
    
    Returns:
        pd.DataFrame: Combined dataframe
    """
    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, pattern))
    
    if not csv_files:
        print(f"No CSV files found in {folder_path} with pattern {pattern}")
        return None
    
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # Read and combine all CSV files
    dataframes = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Add source file column (optional)
            df['source_file'] = os.path.basename(file)
            dataframes.append(df)
            print(f"Loaded {file}: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not dataframes:
        print("No valid CSV files could be loaded")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Remove source_file column if you don't want it
    # combined_df = combined_df.drop('source_file', axis=1)
    
    # Save combined dataframe
    combined_df.to_csv(output_file, index=False)
    
    print(f"\nCombined CSV saved to: {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Columns: {list(combined_df.columns)}")
    
    return combined_df


# Example usage
if __name__ == "__main__":
    # Basic combination
    folder = "obj_clean_labeled"  # Update with your folder path
    output = "obj_clean_labeled/combined_data.csv"
    
    combined_data = combine_csv_files(folder, output)
    
    # Advanced combination with options
    # combined_data = combine_csv_files_advanced(
    #     folder_path="csv_folder",
    #     output_file="combined_advanced.csv",
    #     remove_duplicates=True,
    #     sort_by="ID",  # or ["ID", "label"] for multiple columns
    #     pattern="*.csv"
    # )