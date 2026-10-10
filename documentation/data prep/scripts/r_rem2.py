import sys, glob; from rlib import *
case=sys.argv[1]; scale=float(sys.argv[2]); oscale=float(sys.argv[3])
src=load(glob.glob(f'{R}/datatransform/cleaned_data/total_clean_original_mesh/{case}.*')[0])
u=load(f'{R}/cleandata/uniformly_remeshed/{case}.vtp')
sp=np.load(f'{R}/cleandata/template_mesh/{case}.spheres.npz')
fr=np.load(f'{R}/cleandata/uniformly_remeshed/{case}.ostium_frames.npz')
sac=np.asarray(pv.read(f'{R}/cleandata/uniformly_remeshed/{case}.vtp').points)[sp['sac_vertex_ids']]; neck=sac.mean(0)
c=pv.read(f'{R}/cleandata/original_centerline/{case}.vtp'); cp=np.asarray(c.points)
near=cp[np.argmin(np.linalg.norm(cp-neck,axis=1))]
view=neck-near; view/=np.linalg.norm(view)
up=np.cross(view,[0,0,1.0]); up/=np.linalg.norm(up)
ws=lambda m:mesh_kw(VESSEL, show_edges=True, edge_color='#1a0f0b', line_width=0.9)
def shot(m,f,pos,foc,up,ps):
    p=plotter(1200,1200); p.add_mesh(m,**ws(m))
    p.camera.focal_point=foc; p.camera.position=pos; p.camera.up=up
    p.enable_parallel_projection(); p.camera.parallel_scale=ps; p.screenshot(f); p.close()
for tag,m in (('src',src),('uni',u)):
    shot(m,f'{OUT}/rem_{case[:4]}_sac_{tag}.png',near+80*view,near,up,scale)
# ostium side view: pick the 2nd largest opening
k=int(np.argsort(fr['radius'])[-2]) if len(sys.argv)<5 else int(sys.argv[4])
o,n,r=fr['origin'][k],fr['normal'][k],float(fr['radius'][k])
side=np.cross(n,[0.3,0.5,0.8]); side/=np.linalg.norm(side)
# look slightly down into the opening so the rim is an ellipse with the whole ring in frame
elev=np.deg2rad(float(sys.argv[5]) if len(sys.argv)>5 else 32.0)
vdir=np.cos(elev)*side+np.sin(elev)*n
upv=n-np.dot(n,vdir)*vdir; upv/=np.linalg.norm(upv)
for tag,m in (('src',src),('uni',u)):
    shot(m,f'{OUT}/rem_{case[:4]}_ost_{tag}.png',(o-0.8*r*n)+80*vdir,o-0.8*r*n,upv,oscale*r)
el=lambda m:(lambda e:(e.mean(),e.std()/e.mean(),e.min()))(np.linalg.norm(np.diff(m.extract_all_edges().points[m.extract_all_edges().lines.reshape(-1,3)[:,1:]],axis=1)[:,0],axis=1))
print(case,src.n_points,u.n_points,el(src),el(u),'ostium r',r)
