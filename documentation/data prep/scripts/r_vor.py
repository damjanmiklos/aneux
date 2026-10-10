import sys; from rlib import *
W=f'{R}/scratch/p553_run_moved/_work/p553_FxMTDxQCFwAdGxQOHAUEABQW_RICA__keep1'
N='p553_FxMTDxQCFwAdGxQOHAUEABQW_RICA'
src=load(f'{R}/datatransform/multianeurysm/{N}.vtp')
out=pv.read(f'{R}/scratch/p553_run_fix1/surfaces/{N}_1.stl')
def pts(f,keep=None,seed=0):
    m=pv.read(f); r=np.asarray(m.point_data['MaximumInscribedSphereRadius']); P=np.asarray(m.points)
    ok=r<20; P,r=P[ok],r[ok]
    if keep and len(P)>keep:
        i=np.random.default_rng(seed).choice(len(P),keep,replace=False); P,r=P[i],r[i]
    return P,r
full,rf=pts(f'{W}/{N}_voronoi.vtp')
sac,rs=pts(f'{W}/{N}_voronoi_removed.vtp')
patch,rp=pts(f'{W}/1/{N}_1_parentartery.vtp',keep=int(sys.argv[2]) if len(sys.argv)>2 else 6000)
ctr=np.array([7.0,9.0,-2.0]); ZS=float(sys.argv[1]) if len(sys.argv)>1 else 6.0
d=np.array([1,-1.2,0.8]); d/=np.linalg.norm(d)
def cam(p,c,s):
    p.camera.focal_point=c; p.camera.position=c+d*120; p.camera.up=[0,0,1]; p.enable_parallel_projection(); p.camera.parallel_scale=s
def balls(p,P,color=None,scalars=None,size=4,cmap=None,clim=None):
    c=pv.PolyData(P)
    if scalars is not None: c['r']=scalars; p.add_mesh(c,scalars='r',cmap=cmap,clim=clim,point_size=size,render_points_as_spheres=True,show_scalar_bar=False,lighting=False)
    else: p.add_mesh(c,color=color,point_size=size,render_points_as_spheres=True,lighting=False)
def shot(name,fn,c,s):
    p=plotter(1200,1100); fn(p); cam(p,c,s); p.screenshot(f'{OUT}/vor_{name}.png'); p.close()
def spheres(p,P,r,color,n,opacity=0.13,seed=3):
    idx=np.random.default_rng(seed).choice(len(P),min(n,len(P)),replace=False)
    for k in idx:
        p.add_mesh(pv.Sphere(radius=float(r[k]),center=P[k],theta_resolution=24,phi_resolution=24),color=color,opacity=opacity,smooth_shading=True,specular=0.3)
def local(P,rad,*extra):
    m=np.linalg.norm(P-ctr,axis=1)<rad
    return (P[m],)+tuple(e[m] for e in extra)
def ghost(p,opacity):
    # wall clipped to the half behind the focal plane, so vessels in front do not hide the neck
    w=src.clip(normal=d,origin=ctr+1.0*d,invert=True)
    p.add_mesh(w,color=VESSEL,opacity=opacity,smooth_shading=True,specular=0.2)
RAD=6.5
# (a) whole vessel: Voronoi centres coloured by radius inside a ghosted wall
def a(p):
    p.add_mesh(src,color=VESSEL,opacity=0.18,smooth_shading=True)
    sel=np.random.default_rng(1).choice(len(full),110000,replace=False)
    balls(p,full[sel],scalars=rf[sel],size=3.2,cmap='viridis',clim=(0.2,2.0))
shot('a',a,np.array(src.center),30)
fl,frl=local(full,RAD,rf)
sl,srl=local(sac,RAD,rs)
# (b) balls of the sac (red) against the parent's (grey)
def b(p):
    ghost(p,0.20)
    balls(p,fl[::4],color='#7a7a7a',size=4)
    balls(p,sl,color=RED,size=5)
    spheres(p,sl,srl,RED,22)
shot('b',b,ctr,ZS)
# (c) sac balls discarded; balls interpolated from the parent (navy) in their place
def c(p):
    ghost(p,0.20)
    dd,_=cKDTree(sac).query(fl); keepm=dd>0.35
    balls(p,fl[keepm][::4],color='#7a7a7a',size=4)
    pl,prl=local(patch,RAD,rp)
    balls(p,pl,color=NAVY,size=5)
    spheres(p,pl,prl,NAVY,22)
shot('c',c,ctr,ZS)
def dd_(p):
    w=out.clip(normal=d,origin=ctr+1.0*d,invert=True)
    p.add_mesh(w,**mesh_kw(VESSEL))
shot('d',dd_,ctr,ZS)
