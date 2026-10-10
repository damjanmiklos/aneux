from rlib import *
import pyvista as pv
def loops(m):
    e=m.extract_feature_edges(boundary_edges=True,feature_edges=False,manifold_edges=False,non_manifold_edges=False)
    c=e.connectivity('all'); lab=c.point_data['RegionId']; out=[]
    for k in np.unique(lab):
        p=c.points[lab==k]; out.append((np.round(p.mean(0),1).tolist(),len(p),round(float(np.linalg.norm(p-p.mean(0),axis=1).mean()),2)))
    return out
src=load(f'{R}/datatransform/multianeurysm/p553_FxMTDxQCFwAdGxQOHAUEABQW_RICA.vtp')
print('src',loops(src))
for i in (1,2):
    m=pv.read(f'{R}/scratch/p553_run_fix1/surfaces/p553_FxMTDxQCFwAdGxQOHAUEABQW_RICA_{i}.stl'); print(i,loops(m))
    for j,(ctr,sc,az) in enumerate((([9,7,12],4,0),([8,10,0],5,0))):
        pl=plotter(900,900); pl.add_mesh(m,**mesh_kw(VESSEL)); pl.camera.focal_point=ctr; pl.camera.position=np.array(ctr)+np.array([1,-1.2,0.8])*80; pl.camera.up=[0,0,1]
        pl.enable_parallel_projection(); pl.camera.parallel_scale=sc; pl.screenshot(f'{OUT}/_p553_o{i}_z{j}.png'); pl.close()
pl=plotter(900,900)
for j,(ctr,sc) in enumerate((([9,7,12],4),([8,10,0],5))):
    pl=plotter(900,900); pl.add_mesh(src,**mesh_kw(VESSEL)); pl.camera.focal_point=ctr; pl.camera.position=np.array(ctr)+np.array([1,-1.2,0.8])*80; pl.camera.up=[0,0,1]
    pl.enable_parallel_projection(); pl.camera.parallel_scale=sc; pl.screenshot(f'{OUT}/_p553_src_z{j}.png'); pl.close()
