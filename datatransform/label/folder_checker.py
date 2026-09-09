import os
import sys

import pyvista as pv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from aneux_paths import CLEANED_VESSELS, VESSELS_AREA001

# --- CONFIGURATION ---
FOLDER_1 = CLEANED_VESSELS
FOLDER_2 = VESSELS_AREA001

SUPPORTED_EXTENSIONS = ('.vtp', '.stl')
# ---------------------

state = {
    'files': [],
    'index': 0,
    'plotter': None,
    'is_busy': False,
}


def _overlay_text(index, filename):
    progress = f"[{index + 1}/{len(state['files'])}]"
    left = (
        f"FOLDER 1 (Cleaned)\n{progress} {filename}\n"
        "< Left Arrow (Back) | Right Arrow (Next) >"
    )
    right = f"FOLDER 2 (Original)\n{progress} {filename}"
    return left, right


def load_current_file():
    """Loads the linked files into their respective views."""
    plotter = state['plotter']
    filename = state['files'][state['index']]
    path1 = os.path.join(FOLDER_1, filename)
    path2 = os.path.join(FOLDER_2, filename)
    left_text, right_text = _overlay_text(state['index'], filename)

    plotter.subplot(0, 0)
    plotter.add_mesh(pv.read(path1), name="mesh1", show_scalar_bar=False)
    plotter.add_text(left_text, name="text1", font_size=10, position="upper_left")
    plotter.reset_camera()

    plotter.subplot(0, 1)
    plotter.add_mesh(pv.read(path2), name="mesh2", show_scalar_bar=False)
    plotter.add_text(right_text, name="text2", font_size=10, position="upper_left")
    plotter.reset_camera()

    plotter.link_views()
    plotter.render()


def _step(delta):
    if state['is_busy']:
        return
    new_index = state['index'] + delta
    if not (0 <= new_index < len(state['files'])):
        return
    state['is_busy'] = True
    try:
        state['index'] = new_index
        load_current_file()
    finally:
        state['is_busy'] = False


def main():
    files1 = {f for f in os.listdir(FOLDER_1) if f.lower().endswith(SUPPORTED_EXTENSIONS)}
    files2 = {f for f in os.listdir(FOLDER_2) if f.lower().endswith(SUPPORTED_EXTENSIONS)}

    state['files'] = sorted(files1.intersection(files2))

    if not state['files']:
        print("No common .vtp or .stl files found between the two folders.")
        return

    print(f"Found {len(state['files'])} matching files in both folders. Opening side-by-side viewer...")

    plotter = pv.Plotter(shape=(1, 2), window_size=(1800, 1000))
    plotter.set_background([0.2, 0.2, 0.2], top=[0.25, 0.25, 0.25])
    state['plotter'] = plotter

    load_current_file()

    plotter.add_key_event("Right", lambda: _step(1))
    plotter.add_key_event("Left", lambda: _step(-1))
    plotter.show()


if __name__ == "__main__":
    main()
