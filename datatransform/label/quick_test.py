import os
import pandas as pd

# Paths (update the CSV path if needed)
FOLDER_PATH = r"C:\Users\miklo\OneDrive\UQ\aneux\rawdata\models-v1.0\models\vessels\original"
METADATA_CSV_PATH = r"C:\Users\miklo\OneDrive\UQ\aneux\rawdata\data-v1.0\data\clinical.csv" 
FILTER_LOCATIONS = ['ICA pcom', 'ICA oph', 'ICA cav', 'ICA bif']

# 1. Get the list of expected datasets from the CSV
df = pd.read_csv(METADATA_CSV_PATH)
df['location'] = df['location'].astype(str).str.strip()
df_filtered = df[df['location'].isin(FILTER_LOCATIONS)]

# Create a set of lowercase, stripped dataset IDs
expected_datasets = set(df_filtered['dataset'].astype(str).str.strip().str.lower())

# 2. Get the list of actual 3D files in the folder
actual_files = []
if os.path.exists(FOLDER_PATH):
    for f in os.listdir(FOLDER_PATH):
        if f.lower().endswith(('.vtp', '.stl')):
            # Strip extension and whitespace, convert to lowercase
            name_without_ext = os.path.splitext(f)[0].strip().lower()
            actual_files.append(name_without_ext)
actual_datasets = set(actual_files)

# 3. Calculate the difference
missing_from_folder = expected_datasets - actual_datasets
found_in_folder = expected_datasets.intersection(actual_datasets)

print(f"Total datasets expected (CSV): {len(expected_datasets)}")
print(f"Total valid 3D files found matching CSV: {len(found_in_folder)}")
print(f"Missing files: {len(missing_from_folder)}")

if missing_from_folder:
    print("\nHere are a few of the missing IDs you should look for in your folder:")
    for missing in list(missing_from_folder)[:10]:
        print(f" - Expected file: {missing}.vtp")