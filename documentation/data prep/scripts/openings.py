import sys, numpy as np, pyvista as pv
R='C:/dev/UQ/aneux'
def loops(m):
    m=m.extract_surface().triangulate().clean()
    e=m.extract_feature_edges(boundary_edges=True,feature_edges=False,manifold_edges=False,non_manifold_edges=False)
    if e.n_points==0: return []
    c=e.connectivity('all'); lab=np.asarray(c.point_data['RegionId'])
    out=[]
    for k in np.unique(lab):
        p=np.asarray(c.points)[lab==k]; cen=p.mean(0)
        # plane fit radius
        u,s,vt=np.linalg.svd(p-cen); nrm=vt[2]; r=np.linalg.norm((p-cen)-np.outer((p-cen)@nrm,nrm),axis=1).mean()
        out.append((r,cen,nrm,len(p)))
    return sorted(out,key=lambda t:-t[0])
for name,path in (('original(capped)',f'{R}/rawdata/models-v1.0/models/vessels/original/USFD_0054.vtp'),('uncapped',f'{R}/datatransform/uncapped/USFD_0054.vtk')):
    m=pv.read(path); print(name,m.n_points,m.n_cells,np.round(m.bounds,1))
    for i,(r,cen,nrm,n) in enumerate(loops(m)): print(' ',i+1,'r=%.2f'%r,np.round(cen,2),np.round(nrm,2),n)
fr=np.load(f'{R}/cleandata/uniformly_remeshed/USFD_0054.ostium_frames.npz'); print('GT frames r',np.round(fr['radius'],2)); print(np.round(fr['origin'],2))
