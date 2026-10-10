import sys; from rlib import *
case=sys.argv[1]; zoom=float(sys.argv[2]); ps=float(sys.argv[3])
uraw=pv.read(f'{R}/cleandata/uniformly_remeshed/{case}.vtp')
t=load(f'{R}/cleandata/template_mesh/{case}.vtp')
sp=np.load(f'{R}/cleandata/template_mesh/{case}.spheres.npz')
lab=np.zeros(uraw.n_points); lab[sp['sac_vertex_ids']]=1; uraw['sac']=lab
cen,vt=pca_view(np.asarray(uraw.points))
sac=np.asarray(uraw.points)[sp['sac_vertex_ids']]; sc=sac.mean(0)
c=pv.read(f'{R}/cleandata/original_centerline/{case}.vtp'); cp=np.asarray(c.points)
near=cp[np.argmin(np.linalg.norm(cp-sc,axis=1))]; view=sc-near; view/=np.linalg.norm(view)
up=np.cross(view,vt[2]); up/=np.linalg.norm(up)
def cam(p,whole):
    p.enable_parallel_projection()
    if whole: set_cam(p,cen,vt[2],vt[1],zoom=zoom)
    else:
        p.camera.focal_point=sc; p.camera.position=sc+80*view; p.camera.up=up; p.camera.parallel_scale=ps
for whole in (True,False):
    s='w' if whole else 'z'
    p=plotter(1400,1200); p.add_mesh(uraw.triangulate(),scalars='sac',cmap=[VESSEL,GOLD],show_scalar_bar=False,smooth_shading=True,specular=0.25,ambient=0.15); cam(p,whole); p.screenshot(f'{OUT}/tm_{case[:4]}_{s}_gt.png'); p.close()
    p=pv.Plotter(off_screen=True,window_size=(1400,1200)); p.set_background('white'); p.enable_depth_peeling(30); p.enable_anti_aliasing('fxaa'); p.add_mesh(t,**mesh_kw(VESSEL,opacity=0.45))
    for C,Rr,col in zip(sp['centers'],sp['radii'],[NAVY,TEAL,RED]):
        p.add_mesh(pv.Sphere(radius=float(Rr),center=C,theta_resolution=40,phi_resolution=40),color=col,opacity=0.6,smooth_shading=True)
    cam(p,whole); p.screenshot(f'{OUT}/tm_{case[:4]}_{s}_sph.png'); p.close()
    t['d']=sdist(t,uraw.extract_surface())
    p=plotter(1400,1200); p.add_mesh(t,scalars='d',cmap='magma_r',clim=[0,1.0],smooth_shading=True,show_edges=not whole,show_scalar_bar=False,edge_color='#555555',line_width=0.5,
        scalar_bar_args=dict(title='distance to GT [mm]',vertical=True,position_x=0.06,position_y=0.08,height=0.45,width=0.04,title_font_size=24,label_font_size=20,fmt='%.1f',color='black'))
    cam(p,whole); p.screenshot(f'{OUT}/tm_{case[:4]}_{s}_dist.png'); p.close()
d=t['d']; print('dist median',np.median(d),'p95',np.percentile(d,95),'radii',sp['radii'],'edge',sp['edge_length'],sp['sac_edge_length'],t.n_points,uraw.n_points)
