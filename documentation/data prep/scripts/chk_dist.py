import sys; from rlib import *
case=sys.argv[1]
gt=pv.read(f'{R}/cleandata/uniformly_remeshed/{case}.vtp').triangulate()
t=pv.read(f'{R}/cleandata/template_mesh/{case}.vtp').triangulate()
sp=np.load(f'{R}/cleandata/template_mesh/{case}.spheres.npz',allow_pickle=True)
print({k:(sp[k].shape if hasattr(sp[k],'shape') else sp[k]) for k in sp.files})
sac=np.zeros(gt.n_points,bool); sac[sp['sac_vertex_ids']]=True
d_gt=sdist(gt,t.extract_surface()); d_t=sdist(t,gt.extract_surface())
def st(x): return 'med %.3f p90 %.3f p95 %.3f max %.3f'%(np.median(x),np.percentile(x,90),np.percentile(x,95),x.max())
print('GT->template all   ',st(d_gt))
print('GT->template sac   ',st(d_gt[sac]),'n',sac.sum())
print('GT->template parent',st(d_gt[~sac]))
# template vertices nearest sac: those within 1.0mm of any GT sac vertex
from scipy.spatial import cKDTree
near=cKDTree(gt.points[sac]).query(t.points)[0]<0.8
print('template->GT all   ',st(d_t)); print('template->GT near sac',st(d_t[near]),near.sum())
print('sphere centres',np.round(sp['centers'],2).tolist(),'radii',np.round(sp['radii'],2).tolist())
gt['d']=d_gt; gt.save(f'{OUT}/_gt_with_d_{case[:4]}.vtp')
