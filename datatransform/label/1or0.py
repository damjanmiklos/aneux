import csv
import importlib.util
import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor

# VTK 9.2 in vmtk_env has no vtkRenderingMatplotlib; PyVista 0.48 imports it
# only for optional LaTeX math text.
if importlib.util.find_spec("vtkmodules.vtkRenderingMatplotlib") is None:
    sys.modules["vtkmodules.vtkRenderingMatplotlib"] = types.ModuleType(
        "vtkmodules.vtkRenderingMatplotlib"
    )

import pyvista as pv
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from aneux_paths import VESSELS_ORIGINAL, HASCAPOREXTENSION_CSV

# --- CONFIGURATION ---
FOLDER_PATH = VESSELS_ORIGINAL
OUTPUT_CSV_PATH = HASCAPOREXTENSION_CSV

SUPPORTED_EXTENSIONS = ('.vtp', '.stl')
LOAD_WORKERS = min(8, os.cpu_count() or 4)
# ---------------------

state = {
    'files': [],
    'meshes': [],
    'bounds': [],
    'index': 0,
    'results': [],
    'plotter': None,
    'actor': None,
    'is_busy': False,
}


def _read_mesh(path):
    mesh = pv.read(path)
    return mesh, tuple(mesh.bounds)


def preload_meshes(paths):
    """Load every mesh and cache bounds before the viewer opens."""
    n = len(paths)
    print(f"Preloading {n} meshes ({LOAD_WORKERS} workers)...")
    with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as pool:
        loaded = list(tqdm(pool.map(_read_mesh, paths), total=n, desc="meshes"))
    state['meshes'] = [mesh for mesh, _ in loaded]
    state['bounds'] = [bounds for _, bounds in loaded]


def _set_mesh(mesh, bounds):
    plotter = state['plotter']
    actor = state['actor']
    if actor is None:
        state['actor'] = plotter.add_mesh(
            mesh,
            name="mesh",
            show_scalar_bar=False,
            reset_camera=False,
            render=False,
        )
        plotter.view_isometric(render=False, bounds=bounds)
        return
    actor.mapper.dataset = mesh
    actor.visibility = True
    plotter.reset_camera(bounds=bounds, render=False)


def load_current_file():
    """Show the current preloaded mesh (or the finished overlay)."""
    plotter = state['plotter']

    if state['index'] >= len(state['files']):
        save_results()
        if state['actor'] is not None:
            state['actor'].visibility = False
        plotter.add_text(
            "Finished! All files labeled.\nResults saved to CSV.\n"
            "(Press Left Arrow if you need to go back, or close this window)",
            name="overlay",
            font_size=12,
            position="upper_left",
            render=False,
        )
        plotter.render()
        return

    file_path = state['files'][state['index']]
    filename = os.path.basename(file_path)
    progress = f"[{state['index'] + 1}/{len(state['files'])}]"
    instructions = "Press '1' (Keep), '0' (Discard), or '2' (Category 3) | Press 'Left Arrow' to go back."

    _set_mesh(state['meshes'][state['index']], state['bounds'][state['index']])
    plotter.add_text(
        f"{progress} {filename}\n{instructions}",
        name="overlay",
        font_size=12,
        position="upper_left",
        render=False,
    )
    plotter.render()


def save_results():
    """Writes the accumulated labels to the CSV."""
    with open(OUTPUT_CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Label'])
        writer.writerows(state['results'])
    print(f"\nResults successfully saved to: {OUTPUT_CSV_PATH}")


def _go_back():
    if state['is_busy'] or state['index'] <= 0:
        return
    state['is_busy'] = True
    try:
        state['index'] -= 1
        if state['results']:
            state['results'].pop()
        load_current_file()
    finally:
        state['is_busy'] = False


def _record(label):
    if state['is_busy'] or state['index'] >= len(state['files']):
        return
    state['is_busy'] = True
    try:
        file_path = state['files'][state['index']]
        name_without_ext = os.path.splitext(os.path.basename(file_path))[0]
        state['results'].append((name_without_ext, label))
        state['index'] += 1
        load_current_file()
    finally:
        state['is_busy'] = False


def main():
    print(f"Loading all meshes from {FOLDER_PATH}...")
    if not os.path.isdir(FOLDER_PATH):
        print(f"Folder not found: {FOLDER_PATH}")
        return

    names = [
        f for f in os.listdir(FOLDER_PATH)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    names.sort(key=str.casefold)
    state['files'] = [os.path.join(FOLDER_PATH, f) for f in names]

    if not state['files']:
        print(f"No .vtp or .stl files found in {FOLDER_PATH}.")
        return

    print(f"Found {len(state['files'])} files.")
    preload_meshes(state['files'])
    print("Opening interactive window...")

    plotter = pv.Plotter(window_size=(1600, 1200))
    plotter.set_background([0.2, 0.2, 0.2])
    state['plotter'] = plotter

    load_current_file()

    plotter.add_key_event("Left", _go_back)
    for key in ("0", "KP_0"):
        plotter.add_key_event(key, lambda: _record("0"))
    for key in ("1", "KP_1"):
        plotter.add_key_event(key, lambda: _record("1"))
    for key in ("2", "KP_2"):
        plotter.add_key_event(key, lambda: _record("2"))
    plotter.show()


if __name__ == "__main__":
    main()
