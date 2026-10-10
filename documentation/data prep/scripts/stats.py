import os, glob, json, csv, sys
import numpy as np, pyvista as pv
R='C:/dev/UQ/aneux'
U=f'{R}/cleandata/uniformly_remeshed'; C=f'{R}/cleandata/original_centerline'; T=f'{R}/cleandata/template_mesh'
SRC=f'{R}/datatransform/cleaned_data/total_clean_original_mesh'
srcfiles={os.path.splitext(f)[0]:f for f in os.listdir(SRC)}
def edges(m):
    m=m.extract_surface().triangulate()
    f=m.faces.reshape(-1,4)[:,1:]
    e=np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    e=np.unique(e,axis=0)
    return np.linalg.norm(m.points[e[:,0]]-m.points[e[:,1]],axis=1)
rows=[]
ids=sorted(os.path.splitext(f)[0] for f in os.listdir(U) if f.endswith('.vtp'))
for k,i in enumerate(ids):
    r={'id':i}
    try:
        s=pv.read(f'{SRC}/{srcfiles[i]}'); el=edges(s)
        r.update(src_pts=s.n_points, src_area=s.extract_surface().area, src_emean=el.mean(), src_ecv=el.std()/el.mean(), src_emin=el.min())
        u=pv.read(f'{U}/{i}.vtp'); el=edges(u)
        r.update(u_pts=u.n_points, u_cells=u.n_cells, u_area=u.area, u_emean=el.mean(), u_ecv=el.std()/el.mean(), u_emin=el.min())
        fr=np.load(f'{U}/{i}.ostium_frames.npz'); r['n_open']=len(fr['radius']); r['r_open_min']=float(np.min(fr['radius'])); r['r_open_max']=float(np.max(fr['radius']))
        c=pv.read(f'{C}/{i}.vtp'); misr=np.asarray(c.point_data['MaximumInscribedSphereRadius'])
        L=0.0
        for ci in range(c.n_cells):
            p=c.get_cell(ci).points; L+=np.linalg.norm(np.diff(p,axis=0),axis=1).sum()
        r.update(cl_cells=c.n_cells, cl_len=L, misr_min=misr.min(), misr_med=np.median(misr), misr_max=misr.max())
        t=pv.read(f'{T}/{i}.vtp'); el=edges(t)
        sp=np.load(f'{T}/{i}.spheres.npz')
        r.update(t_pts=t.n_points, t_area=t.area, t_emean=el.mean(), t_ecv=el.std()/el.mean(), sph_r=';'.join(f'{x:.3f}' for x in sp['radii']), sac_frac=len(sp['sac_vertex_ids'])/u.n_points, t_edge=float(sp['edge_length']), t_sac_edge=float(sp['sac_edge_length']))
    except Exception as e:
        r['err']=repr(e)
    rows.append(r)
    if k%50==0: print(k, i, flush=True)
keys=sorted({k for r in rows for k in r}, key=lambda x:(x!='id',x))
with open(f'{R}/documentation/data prep/scripts/stats.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=keys); w.writeheader(); w.writerows(rows)
print('done')
