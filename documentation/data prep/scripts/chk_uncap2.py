from rlib import *
from scipy.spatial import cKDTree
cap=load(f'{R}/rawdata/models-v1.0/models/vessels/original/USFD_0054.vtp').triangulate()
unc=load(f'{R}/datatransform/uncapped/USFD_0054.vtk').triangulate()
dd,_=cKDTree(unc.points).query(cap.points); gone=dd>1e-4
print('removed verts',gone.sum())
# connected groups of removed verts via cells
cells=cap.faces.reshape(-1,4)[:,1:]
rem=gone[cells].all(1); anyrem=gone[cells].any(1)
print('cells fully removed',rem.sum(),'cells touching removed',anyrem.sum())
sub=cap.extract_cells(np.where(anyrem)[0]).connectivity('all')
for k in np.unique(sub.point_data['RegionId']):
    p=sub.points[sub.point_data['RegionId']==k]
    u,s,vt=np.linalg.svd(p-p.mean(0),full_matrices=False)
    print(k,len(p),np.round(p.mean(0),2),'extent',np.round(s/np.sqrt(len(p))*2,2),'normal',np.round(vt[2],2))
# cap cells area & edge lengths
