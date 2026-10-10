import sys, glob; from rlib import *
case=sys.argv[1]; zoom=float(sys.argv[2]) if len(sys.argv)>2 else 4.0
src=load(glob.glob(f'{R}/datatransform/cleaned_data/total_clean_original_mesh/{case}.*')[0])
u=load(f'{R}/cleandata/uniformly_remeshed/{case}.vtp')
sp=np.load(f'{R}/cleandata/template_mesh/{case}.spheres.npz')
fr=np.load(f'{R}/cleandata/uniformly_remeshed/{case}.ostium_frames.npz')
sac=np.asarray(u.points)[sp['sac_vertex_ids']]; neck=sac.mean(0)
cen,vt=pca_view(np.asarray(u.points))
# look at the sac from outside, along sac-mean minus nearest centerline direction approximated by sac normal
c=pv.read(f'{R}/cleandata/original_centerline/{case}.vtp'); cp=np.asarray(c.points)
near=cp[np.argmin(np.linalg.norm(cp-neck,axis=1))]
view=neck-near; view/=np.linalg.norm(view)
view=0.6*view+0.8*vt[2]*np.sign(np.dot(view,vt[2])+1e-9); view/=np.linalg.norm(view)
for tag,m in (('src',src),('uni',u)):
    p=plotter(1200,1200)
    p.add_mesh(m, **mesh_kw(VESSEL, show_edges=True, edge_color='#5a3a32', line_width=0.6))
    p.camera.focal_point=neck; p.camera.position=neck+80*view; p.camera.up=vt[0]
    p.enable_parallel_projection(); p.camera.parallel_scale=float(sys.argv[3]) if len(sys.argv)>3 else 3.0
    p.screenshot(f'{OUT}/rem_{case[:6]}_{tag}.png'); p.close()
print(src.n_points,u.n_points)
