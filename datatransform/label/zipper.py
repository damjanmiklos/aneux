import os
import pandas as pd
import zipfile

# --- CONFIGURATION ---
CSV_PATH = r"C:\Users\miklo\OneDrive\UQ\aneux\datatransform\label\hascap.csv"
SOURCE_DIR = r"C:\Users\miklo\OneDrive\UQ\aneux\rawdata\models-v1.0\models\vessels\original"

# The output zip file will be created in your label folder
OUTPUT_ZIP_PATH = r"C:\Users\miklo\OneDrive\UQ\aneux\datatransform\label\kept_vessels.zip"

SUPPORTED_EXTENSIONS = ['.vtp', '.stl']
# ---------------------

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    print(f"Reading labels from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # Ensure the Label column is treated as a string and strip any spaces
    df['Label'] = df['Label'].astype(str).str.strip()
    
    # Filter for files marked with '1'
    kept_files_df = df[df['Label'] == '1']
    kept_filenames = kept_files_df['Filename'].tolist()
    
    if not kept_filenames:
        print("No files with label '1' found in the CSV. Nothing to zip.")
        return
        
    print(f"Found {len(kept_filenames)} files marked as 'Keep' (1). Creating zip archive...")
    
    added_count = 0
    missing_count = 0
    
    # Open a new ZIP file with standard ZIP_DEFLATED compression
    with zipfile.ZipFile(OUTPUT_ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for base_name in kept_filenames:
            file_found = False
            
            # Since the CSV only contains the name without the extension, 
            # we check for both .vtp and .stl
            for ext in SUPPORTED_EXTENSIONS:
                target_file_path = os.path.join(SOURCE_DIR, base_name + ext)
                
                if os.path.exists(target_file_path):
                    # arcname defines the name of the file inside the zip archive.
                    # This ensures we don't zip your entire C:\Users\miklo\... folder path.
                    zipf.write(target_file_path, arcname=base_name + ext)
                    added_count += 1
                    file_found = True
                    break  # Stop checking extensions once we find the file
            
            if not file_found:
                print(f"  -> Warning: Could not find physical file for '{base_name}' in {SOURCE_DIR}")
                missing_count += 1

    print("\n" + "="*40)
    print("FINISHED")
    print("="*40)
    print(f"Successfully copied and zipped {added_count} files.")
    if missing_count > 0:
        print(f"Missing files: {missing_count} (Check warnings above)")
    print(f"Archive saved to: {OUTPUT_ZIP_PATH}")

if __name__ == "__main__":
    main()