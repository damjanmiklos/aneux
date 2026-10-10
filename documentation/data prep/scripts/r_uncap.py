import sys; from rlib import *
ps=float(sys.argv[1]); elev=float(sys.argv[2]); az=float(sys.argv[3]) if len(sys.argv)>3 else 0.0; tag=sys.argv[4] if len(sys.argv)>4 else 'a'
cap=load(f'{R}/rawdata/models-v1.0/models/vessels/original/USFD_0054.vtp')
unc=load(f'{R}/datatransform/uncapped/USFD_0054.vtk')
o=np.array([17.01,30.01,38.86]); n=np.array([0.9,0.28,-0.34]); n/=np.linalg.norm(n); r=0.87
from scipy.spatial import cKDTree
_q=cKDTree(unc.points).query_ball_point(o,6.0); _m=unc.points[_q].mean(0)
if np.dot(n,o-_m)<0: n=-n
print('outward n',n)
side=np.cross(n,[0.3,0.5,0.8]); side/=np.linalg.norm(side)
side=np.cos(np.deg2rad(az))*side+np.sin(np.deg2rad(az))*np.cross(n,side)
vdir=np.cos(np.deg2rad(elev))*side+np.sin(np.deg2rad(elev))*n
upv=n-np.dot(n,vdir)*vdir; upv/=np.linalg.norm(upv)
for name,m in (('before',cap),('after',unc)):
    p=plotter(1200,1200); p.add_mesh(m,**mesh_kw(VESSEL))
    p.camera.focal_point=o-0.5*r*n; p.camera.position=o-0.5*r*n+80*vdir; p.camera.up=upv
    p.enable_parallel_projection(); p.camera.parallel_scale=ps
    p.screenshot(f'{OUT}/uncap_USFD54_{name}_{tag}.png'); p.close()
