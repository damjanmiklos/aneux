from rlib import *
from scipy.spatial import cKDTree
cap=load(f'{R}/rawdata/models-v1.0/models/vessels/original/USFD_0054.vtp').triangulate()
unc=load(f'{R}/datatransform/uncapped/USFD_0054.vtk').triangulate()
print(cap.n_points,cap.n_cells,unc.n_points,unc.n_cells)
# boundary loops of the uncapped mesh
e=unc.extract_feature_edges(boundary_edges=True,feature_edges=False,manifold_edges=False,non_manifold_edges=False)
c=e.connectivity('all'); lab=c.point_data['RegionId']
for k in np.unique(lab):
    pts=c.points[lab==k]; ctr=pts.mean(0)
    d=np.linalg.norm(pts-ctr,axis=1).mean()
    # triangles of the capped mesh that sit near this centre: flat? 
    t=cKDTree(cap.points); idx=t.query_ball_point(ctr,d*1.2)
    print(k,len(pts),np.round(ctr,2),round(d,2),'capped-mesh verts within',len(idx))
# vertices of unc not in cap and vice versa
t=cKDTree(unc.points); dd,_=t.query(cap.points)
print('capped verts absent in uncapped:',(dd>1e-4).sum())
t2=cKDTree(cap.points); dd2,_=t2.query(unc.points)
print('uncapped verts absent in capped:',(dd2>1e-4).sum())
print(cap.points[dd>1e-4][:60].mean(0), len(cap.points[dd>1e-4]))
