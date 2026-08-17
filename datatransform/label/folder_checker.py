import os
from paraview.simple import *

# --- CONFIGURATION ---
# Folder 1: Your cleaned/filtered data
FOLDER_1 = r"C:\Users\miklo\OneDrive\UQ\aneux\datatransform\cleaned_data\vessels_cleaned_and_decapped"

# Folder 2: Your original/unfiltered data
FOLDER_2 = r"C:\Users\miklo\OneDrive\UQ\aneux\rawdata\models-v1.0\models\vessels\remeshed\area-001"


SUPPORTED_EXTENSIONS = ('.vtp', '.stl')
# ---------------------

state = {
    'files': [],
    'index': 0,
    'reader1': None,
    'reader2': None,
    'text1': None,
    'text2': None,
    'view1': None,
    'view2': None,
    'is_busy': False  # NEW: Safety lock to prevent pipeline crashes
}

def load_current_file():
    """Loads the linked files into their respective views."""
    view1 = state['view1']
    view2 = state['view2']

    # Clean up the previous models
    if state['reader1']:
        Hide(state['reader1'], view1)
        Delete(state['reader1'])
        state['reader1'] = None
    if state['reader2']:
        Hide(state['reader2'], view2)
        Delete(state['reader2'])
        state['reader2'] = None

    # Get the current filename
    filename = state['files'][state['index']]
    
    # Construct exact paths for both folders
    path1 = os.path.join(FOLDER_1, filename)
    path2 = os.path.join(FOLDER_2, filename)
    name_without_ext = os.path.splitext(filename)[0]

    # Update the text overlays
    progress = f"[{state['index'] + 1}/{len(state['files'])}]"
    state['text1'].Text = f"FOLDER 1 (Cleaned)\n{progress} {filename}\n< Left Arrow (Back) | Right Arrow (Next) >"
    state['text2'].Text = f"FOLDER 2 (Original)\n{progress} {filename}"

    # Load and display in View 1 (Left)
    state['reader1'] = OpenDataFile(path1, registrationName=f"{name_without_ext}_F1")
    Show(state['reader1'], view1)

    # Load and display in View 2 (Right)
    state['reader2'] = OpenDataFile(path2, registrationName=f"{name_without_ext}_F2")
    Show(state['reader2'], view2)

    # Reset cameras to center the models
    ResetCamera(view1)
    ResetCamera(view2)
    RenderAllViews()

def on_key_press(caller, event):
    """Handles keyboard navigation safely."""
    # NEW: If the script is already loading a file, ignore new key presses.
    # This prevents crashes if you hold down the arrow key.
    if state['is_busy']:
        return
        
    state['is_busy'] = True 
    
    try:
        key = caller.GetKeySym()
        
        if key == 'Right':
            if state['index'] < len(state['files']) - 1:
                state['index'] += 1
                load_current_file()
                
        elif key == 'Left':
            if state['index'] > 0:
                state['index'] -= 1
                load_current_file()
    finally:
        # Unlock the keyboard only after the entire rendering process finishes
        state['is_busy'] = False

def main():
    # 1. Gather and intersect files
    files1 = {f for f in os.listdir(FOLDER_1) if f.lower().endswith(SUPPORTED_EXTENSIONS)}
    files2 = {f for f in os.listdir(FOLDER_2) if f.lower().endswith(SUPPORTED_EXTENSIONS)}
    
    state['files'] = sorted(list(files1.intersection(files2)))
    
    if not state['files']:
        print(f"No common .vtp or .stl files found between the two folders.")
        return

    print(f"Found {len(state['files'])} matching files in both folders. Opening side-by-side viewer...")

    # 2. Set up the Layout and Views
    state['view1'] = GetActiveViewOrCreate('RenderView')
    state['view1'].Background = [0.2, 0.2, 0.2]
    
    layout = GetLayout()
    # --- FIXED: Set the size of the whole layout, not just view1 ---
    layout.SetSize(1800, 1000) 
    layout.SplitHorizontal(0, 0.5)
    
    state['view2'] = CreateView('RenderView')
    state['view2'].Background = [0.25, 0.25, 0.25]
    layout.AssignView(2, state['view2'])

    # 3. Synchronize the cameras
    AddCameraLink(state['view1'], state['view2'], 'CameraSync')

    # 4. Set up on-screen text
    state['text1'] = Text()
    text_display1 = Show(state['text1'], state['view1'])
    text_display1.FontSize = 16
    text_display1.WindowLocation = 'Upper Left Corner'

    state['text2'] = Text()
    text_display2 = Show(state['text2'], state['view2'])
    text_display2.FontSize = 16
    text_display2.WindowLocation = 'Upper Left Corner'

    # Load the very first file
    load_current_file()

    # 5. Attach the keyboard listener
    interactor = state['view1'].GetRenderWindow().GetInteractor()
    interactor.AddObserver("KeyPressEvent", on_key_press)

    # Start the ParaView interaction loop
    Interact()

if __name__ == "__main__":
    main()