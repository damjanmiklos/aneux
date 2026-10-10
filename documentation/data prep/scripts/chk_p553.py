from rlib import *
import pyvista as pv
src=load(f'{R}/datatransform/multianeurysm/p553_FxMTDxQCFwAdGxQOHAUEABQW_RICA.vtp')
o1=pv.read(f'{R}/scratch/p553_run_fix1/surfaces/p553_FxMTDxQCFwAdGxQOHAUEABQW_RICA_1.stl')
o2=pv.read(f'{R}/scratch/p553_run_fix1/surfaces/p553_FxMTDxQCFwAdGxQOHAUEABQW_RICA_2.stl')
for n,m in (('src',src),('o1',o1),('o2',o2)):
    e=m.extract_feature_edges(boundary_edges=True,feature_edges=False,manifold_edges=False,non_manifold_edges=False)
    nm=m.extract_feature_edges(boundary_edges=False,feature_edges=False,manifold_edges=False,non_manifold_edges=True)
    c=m.connectivity('all'); k=len(np.unique(c.point_data['RegionId']))
    print(n,m.n_points,'bounds',np.round(m.bounds,0),'boundary pts',e.n_points,'NM edges',nm.n_cells,'components',k)
    pl=plotter(1100,1100); pl.add_mesh(m,**mesh_kw(VESSEL))
    ctr=np.array(src.center); pl.camera.focal_point=ctr; pl.camera.position=ctr+np.array([1,-1.2,0.8])*90; pl.camera.up=[0,0,1]
    pl.enable_parallel_projection(); pl.camera.parallel_scale=26
    pl.screenshot(f'{OUT}/_p553_{n}.png'); pl.close()
