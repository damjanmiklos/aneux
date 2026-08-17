import os
import csv
import pandas as pd
from paraview.simple import *

# --- CONFIGURATION ---
# Use raw strings (r"...") to safely handle Windows backslashes
FOLDER_PATH = r"C:\Users\miklo\OneDrive\UQ\aneux\rawdata\models-v1.0\models\vessels\original"
OUTPUT_CSV_PATH = r"C:\Users\miklo\OneDrive\UQ\aneux\datatransform\label\hascap.csv"

# Add the path to the CSV containing the 'dataset' and 'location' columns
METADATA_CSV_PATH = r"C:\Users\miklo\OneDrive\UQ\aneux\rawdata\data-v1.0\data\clinical.csv" 

SUPPORTED_EXTENSIONS = ('.vtp', '.stl')
FILTER_LOCATIONS = ['ICA pcom', 'ICA oph', 'ICA cav', 'ICA bif']
# ---------------------

state = {
    'files': [],
    'index': 0,
    'results': [],
    'current_reader': None,
    'text_source': None
}

def load_current_file(view):
    """Loads the next 3D file and updates the on-screen text."""
    # Clean up the previous model
    if state['current_reader']:
        Hide(state['current_reader'], view)
        Delete(state['current_reader'])
        state['current_reader'] = None

    # Check if we are done
    if state['index'] >= len(state['files']):
        save_results()
        state['text_source'].Text = "Finished! All files labeled.\nResults saved to CSV.\n(Press Left Arrow if you need to go back, or close this window)"
        Render(view)
        return

    # Get the current file info
    file_path = state['files'][state['index']]
    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]

    # Update the text on the screen
    progress = f"[{state['index'] + 1}/{len(state['files'])}]"
    
    # --- UPDATED: Added category '2' to the instructions ---
    instructions = "Press '1' (Keep), '0' (Discard), or '2' (Category 3) | Press 'Left Arrow' to go back."
    state['text_source'].Text = f"{progress} {filename}\n{instructions}"

    # Load and display the new file
    state['current_reader'] = OpenDataFile(file_path, registrationName=name_without_ext)
    Show(state['current_reader'], view)
    ResetCamera(view)
    Render(view)

def save_results():
    """Writes the accumulated labels to the CSV."""
    with open(OUTPUT_CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Label'])
        writer.writerows(state['results'])
    print(f"\nResults successfully saved to: {OUTPUT_CSV_PATH}")

def on_key_press(caller, event):
    """Callback function triggered every time a key is pressed in the 3D window."""
    key = caller.GetKeySym()
    view = GetActiveView()

    # Go Back Logic
    if key == 'Left':
        if state['index'] > 0:
            state['index'] -= 1        
            if state['results']:
                state['results'].pop() 
            load_current_file(view)    
        return 

    if state['index'] >= len(state['files']):
        return 

    # --- UPDATED: Accept 0, 1, and 2 (standard and numpad) ---
    if key in ['0', '1', '2', 'KP_0', 'KP_1', 'KP_2']:
        if key in ['1', 'KP_1']:
            label = '1'
        elif key in ['2', 'KP_2']:
            label = '2'
        else:
            label = '0'
        
        # Record the result
        file_path = state['files'][state['index']]
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]
        state['results'].append((name_without_ext, label))
        
        # Advance to the next file
        state['index'] += 1
        
        # Update the view
        load_current_file(view)

def main():
    # --- Load and filter metadata ---
    print("Loading metadata and filtering datasets...")
    df = pd.read_csv(METADATA_CSV_PATH)
    df['location'] = df['location'].astype(str).str.strip()
    df_filtered = df[df['location'].isin(FILTER_LOCATIONS)]
    
    # Extract expected IDs and convert to lowercase for safe matching
    expected_ids = df_filtered['dataset'].astype(str).str.strip().tolist()
    expected_ids_lower = [eid.lower() for eid in expected_ids]
    
    # Gather the files
    all_files = os.listdir(FOLDER_PATH)
    for f in all_files:
        if f.lower().endswith(SUPPORTED_EXTENSIONS):
            name_without_ext = os.path.splitext(f)[0]
            name_lower = name_without_ext.lower()
            
            # Flexible matching: Check if the CSV ID starts with the file's base name
            # e.g., 'p163_abc123_1'.startswith('p163_abc123') -> True
            is_match = any(eid.startswith(name_lower) or name_lower.startswith(eid) for eid in expected_ids_lower)
            
            if is_match:
                state['files'].append(os.path.join(FOLDER_PATH, f))
            
    if not state['files']:
        print(f"No .vtp or .stl files found in {FOLDER_PATH} that match the target locations.")
        return

    print(f"Found {len(state['files'])} valid files. Opening interactive window...")

    # Set up the render window
    view = GetActiveViewOrCreate('RenderView')
    view.Background = [0.2, 0.2, 0.2]
    view.ViewSize = [1600, 1200]

    # Set up the on-screen text overlay
    state['text_source'] = Text()
    text_display = Show(state['text_source'], view)
    text_display.FontSize = 20
    text_display.WindowLocation = 'Upper Left Corner' 

    # Load the very first file
    load_current_file(view)

    # Attach the keyboard listener to the window
    interactor = view.GetRenderWindow().GetInteractor()
    interactor.AddObserver("KeyPressEvent", on_key_press)

    # Start the ParaView interaction loop
    Interact()

if __name__ == "__main__":
    main()