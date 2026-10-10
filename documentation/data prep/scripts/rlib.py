import sys, types, importlib.util
if importlib.util.find_spec("vtkmodules.vtkRenderingMatplotlib") is None:
    sys.modules["vtkmodules.vtkRenderingMatplotlib"] = types.ModuleType("vtkmodules.vtkRenderingMatplotlib")
import os
os.environ.setdefault('VTK_SMP_MAX_THREADS','1')
import numpy as np, pyvista as pv
from scipy.spatial import cKDTree
R='C:/dev/UQ/aneux'
OUT=f'{R}/documentation/data prep/renders'
GREY='#c9c9c9'; VESSEL='#d8b4a6'; NAVY='#104480'; RED='#c0392b'; GOLD='#e0a526'; TEAL='#2a9d8f'
pv.global_theme.font.family='times'
def load(p):
    m=pv.read(p)
    if not isinstance(m,pv.PolyData): m=m.extract_surface()
    return m.triangulate().clean()
def pca_view(pts):
    c=pts.mean(0); _,_,vt=np.linalg.svd(pts-c,full_matrices=False)
    return c, vt  # vt[2] = smallest axis (view direction), vt[0] = largest (horizontal)
def dist(a,b):
    return cKDTree(np.asarray(b.points)).query(np.asarray(a.points))[0]
def plotter(w=1400,h=1100):
    p=pv.Plotter(off_screen=True, window_size=(w,h)); p.set_background('white')
    p.enable_anti_aliasing('ssaa')
    return p
def set_cam(p, center, view, up, dist_=None, zoom=1.0, parallel=True):
    p.camera.focal_point=center
    d=dist_ or 200.0
    p.camera.position=np.asarray(center)+d*np.asarray(view)
    p.camera.up=up
    if parallel: p.enable_parallel_projection()
    p.reset_camera(); p.camera.zoom(zoom)
def mesh_kw(color, **k):
    d=dict(color=color, smooth_shading=True, specular=0.25, specular_power=20, ambient=0.15, diffuse=0.85)
    d.update(k); return d
def sdist(a,b):
    """Unsigned point-to-surface distance from a's vertices to surface b."""
    import vtk
    f=vtk.vtkImplicitPolyDataDistance(); f.SetInput(b)
    return np.abs(np.array([f.EvaluateFunction(p) for p in np.asarray(a.points)]))
