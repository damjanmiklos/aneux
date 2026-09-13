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
from aneux_paths import UNCAPPED_VESSELS, VESSELS_ORIGINAL

# --- CONFIGURATION ---
FOLDER_1 = UNCAPPED_VESSELS
FOLDER_2 = VESSELS_ORIGINAL

SUPPORTED_EXTENSIONS = ('.vtp', '.stl', '.vtk')
LOAD_WORKERS = min(8, os.cpu_count() or 4)
# ---------------------

state = {
    'files': [],
    'paths1': {},
    'paths2': {},
    'meshes1': [],
    'meshes2': [],
    'bounds1': [],
    'bounds2': [],
    'index': 0,
    'plotter': None,
    'actor1': None,
    'actor2': None,
    'is_busy': False,
}


def _mesh_stems(folder):
    stems = {}
    if not os.path.isdir(folder):
        return stems
    for name in os.listdir(folder):
        stem, ext = os.path.splitext(name)
        if ext.lower() in SUPPORTED_EXTENSIONS:
            stems[stem] = os.path.join(folder, name)
    return stems


def _read_mesh(path):
    mesh = pv.read(path)
    return mesh, tuple(mesh.bounds)


def preload_meshes(paths):
    """Load every mesh and cache bounds before the viewer opens."""
    n = len(paths)
    print(f"Preloading {n} meshes ({LOAD_WORKERS} workers)...")
    with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as pool:
        loaded = list(tqdm(pool.map(_read_mesh, paths), total=n, desc="meshes"))
    return [mesh for mesh, _ in loaded], [bounds for _, bounds in loaded]


def _set_mesh(col, actor_key, mesh, bounds):
    plotter = state['plotter']
    plotter.subplot(0, col)
    actor = state[actor_key]
    if actor is None:
        state[actor_key] = plotter.add_mesh(
            mesh,
            name=actor_key,
            show_scalar_bar=False,
            reset_camera=False,
            render=False,
        )
        plotter.view_isometric(render=False, bounds=bounds)
        return
    actor.mapper.dataset = mesh
    actor.visibility = True
    plotter.reset_camera(bounds=bounds, render=False)


def _overlay_text(index, name1, name2):
    progress = f"[{index + 1}/{len(state['files'])}]"
    left = (
        f"FOLDER 1 (Uncapped)\n{progress} {name1}\n"
        "< Left Arrow (Back) | Right Arrow (Next) >"
    )
    right = f"FOLDER 2 (Original)\n{progress} {name2}"
    return left, right


def load_current_file():
    """Show the current preloaded pair."""
    plotter = state['plotter']
    idx = state['index']
    stem = state['files'][idx]
    left_text, right_text = _overlay_text(
        idx,
        os.path.basename(state['paths1'][stem]),
        os.path.basename(state['paths2'][stem]),
    )

    _set_mesh(0, 'actor1', state['meshes1'][idx], state['bounds1'][idx])
    plotter.add_text(left_text, name="text1", font_size=10, position="upper_left", render=False)

    _set_mesh(1, 'actor2', state['meshes2'][idx], state['bounds2'][idx])
    plotter.add_text(right_text, name="text2", font_size=10, position="upper_left", render=False)

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
    paths1 = _mesh_stems(FOLDER_1)
    paths2 = _mesh_stems(FOLDER_2)
    state['paths1'] = paths1
    state['paths2'] = paths2
    state['files'] = sorted(set(paths1) & set(paths2), key=str.casefold)

    if not state['files']:
        print("No common mesh names found between the two folders (.vtk/.vtp/.stl).")
        return

    print(f"Found {len(state['files'])} matching files in both folders.")
    left_paths = [paths1[stem] for stem in state['files']]
    right_paths = [paths2[stem] for stem in state['files']]
    all_meshes, all_bounds = preload_meshes(left_paths + right_paths)
    n = len(state['files'])
    state['meshes1'] = all_meshes[:n]
    state['bounds1'] = all_bounds[:n]
    state['meshes2'] = all_meshes[n:]
    state['bounds2'] = all_bounds[n:]
    print("Opening interactive window...")

    plotter = pv.Plotter(shape=(1, 2), window_size=(1800, 1000))
    plotter.set_background([0.2, 0.2, 0.2], top=[0.25, 0.25, 0.25])
    state['plotter'] = plotter

    load_current_file()

    plotter.add_key_event("Right", lambda: _step(1))
    plotter.add_key_event("Left", lambda: _step(-1))
    plotter.show()


if __name__ == "__main__":
    main()
