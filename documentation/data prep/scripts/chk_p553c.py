from rlib import *
import pyvista as pv
O='p553_FxMTDxQCFwAdGxQOHAUEABQW_RICA'
for i in (1,2):
    for tag,f in (('gt',f'{R}/cleandata/uniformly_remeshed/{O}_{i}.vtp'),('tp',f'{R}/cleandata/template_mesh/{O}_{i}.vtp')):
        m=pv.read(f); pl=plotter(1100,1100); pl.add_mesh(m,**mesh_kw(VESSEL))
        ctr=np.array([6,3,0.]); pl.camera.focal_point=ctr; pl.camera.position=ctr+np.array([1,-1.2,0.8])*90; pl.camera.up=[0,0,1]
        pl.enable_parallel_projection(); pl.camera.parallel_scale=24; pl.screenshot(f'{OUT}/_p553_{tag}{i}.png'); pl.close()
