import sys; from rlib import *
case=sys.argv[1]; clim=float(sys.argv[2]); tag=sys.argv[3]; size=int(sys.argv[4]) if len(sys.argv)>4 else 1200
parts=sys.argv[5].split(',') if len(sys.argv)>5 else ['gt','sph','dist']
uraw=pv.read(f'{R}/cleandata/uniformly_remeshed/{case}.vtp')
t=load(f'{R}/cleandata/template_mesh/{case}.vtp')
sp=np.load(f'{R}/cleandata/template_mesh/{case}.spheres.npz',allow_pickle=True)
lab=np.zeros(uraw.n_points); lab[sp['sac_vertex_ids']]=1; uraw['sac']=lab
sac=np.asarray(uraw.points)[sp['sac_vertex_ids']]; sc=sac.mean(0)
ext=np.ptp(sac,axis=0).max()
c=pv.read(f'{R}/cleandata/original_centerline/{case}.vtp'); cp=np.asarray(c.points)
near=cp[np.argmin(np.linalg.norm(cp-sc,axis=1))]; view=sc-near; view/=np.linalg.norm(view)
cen,vt=pca_view(np.asarray(uraw.points))
up=np.cross(view,vt[2]); 
if np.linalg.norm(up)<1e-6: up=vt[1]
up/=np.linalg.norm(up)
ps=float(sys.argv[6]) if len(sys.argv)>6 else 0.62*ext+1.5
def cam(p):
    p.enable_parallel_projection(); p.camera.focal_point=sc; p.camera.position=sc+80*view; p.camera.up=up; p.camera.parallel_scale=ps
if 'gt' in parts:
    p=plotter(size,size); p.add_mesh(uraw.triangulate(),scalars='sac',cmap=[VESSEL,GOLD],show_scalar_bar=False,smooth_shading=True,specular=0.25,ambient=0.15); cam(p); p.screenshot(f'{OUT}/{tag}_gt.png'); p.close()
if 'sph' in parts:
    p=pv.Plotter(off_screen=True,window_size=(size,size)); p.set_background('white'); p.enable_depth_peeling(30); p.enable_anti_aliasing('fxaa')
    p.add_mesh(t,**mesh_kw(VESSEL,opacity=0.45))
    for C,Rr,col in zip(sp['centers'],sp['radii'],[NAVY,TEAL,RED]):
        p.add_mesh(pv.Sphere(radius=float(Rr),center=C,theta_resolution=40,phi_resolution=40),color=col,opacity=0.6,smooth_shading=True)
    cam(p); p.screenshot(f'{OUT}/{tag}_sph.png'); p.close()
if 'dist' in parts:
    # distance of every GT vertex to the template surface: the stricter, GT-side direction, so a part of the sac the spheres miss shows up
    uraw['d']=sdist(uraw,t.extract_surface(algorithm='dataset_surface'))
    d=np.asarray(uraw['d']); s_=np.asarray(uraw['sac'])>0
    print('GT->template all med %.3f p95 %.3f | sac med %.3f p90 %.3f p95 %.3f max %.3f'%(np.median(d),np.percentile(d,95),np.median(d[s_]),np.percentile(d[s_],90),np.percentile(d[s_],95),d[s_].max()))
    p=plotter(size,size); p.add_mesh(uraw.triangulate(),scalars='d',cmap='magma_r',clim=[0,clim],smooth_shading=True,show_scalar_bar=False,specular=0.2); cam(p); p.screenshot(f'{OUT}/{tag}_dist.png'); p.close()
print(case,'ps',round(ps,1),'ext',round(ext,1))
