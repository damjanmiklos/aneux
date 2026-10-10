import glob, os, sys, csv
from rlib import *
import vtk
rows=[]
files=sorted(glob.glob(f'{R}/cleandata/template_mesh/*.spheres.npz'))
for k,f in enumerate(files):
    case=os.path.basename(f)[:-len('.spheres.npz')]
    try:
        sp=np.load(f,allow_pickle=True)
        gt=pv.read(f'{R}/cleandata/uniformly_remeshed/{case}.vtp'); t=pv.read(f'{R}/cleandata/template_mesh/{case}.vtp').triangulate()
        ids=sp['sac_vertex_ids']; P=np.asarray(gt.points)[ids]
        fn=vtk.vtkImplicitPolyDataDistance(); fn.SetInput(t)
        d=np.abs(np.array([fn.EvaluateFunction(p) for p in P]))
        rows.append((case,len(ids),float(np.median(d)),float(np.percentile(d,90)),float(np.percentile(d,95)),float(d.max()),float(np.mean(d>0.3)),*np.round(sp['radii'],2).tolist()))
    except Exception as e:
        print('ERR',case,e,flush=True)
    if k%100==0: print(k,len(files),flush=True)
with open(f'{R}/documentation/data prep/scripts/fit_scan.csv','w',newline='') as g:
    w=csv.writer(g); w.writerow(['case','n_sac','med','p90','p95','max','frac_gt_0.3','r1','r2','r3']); w.writerows(rows)
print('done',len(rows))
