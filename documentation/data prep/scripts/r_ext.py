import sys; from rlib import *
case=sys.argv[1]
o=load(f'{R}/rawdata/models-v1.0/models/vessels/original/{case}.stl')
c=load(f'{R}/datatransform/cleaned_data/vessels_cleaned_and_decapped/{case}.vtp')
d=dist(o,c); o['removed']=(d>0.05).astype(float)
cen,vt=pca_view(np.asarray(o.points))
for tag,view in (('a',vt[2]),('b',-vt[2])):
    p=plotter(1600,1200)
    p.add_mesh(o, scalars='removed', cmap=[VESSEL,RED], show_scalar_bar=False, smooth_shading=True, specular=0.25, ambient=0.15)
    set_cam(p,cen,view,vt[1],zoom=1.25)
    p.screenshot(f'{OUT}/ext_{case[:4]}_{tag}.png'); p.close()
print(o.n_points, c.n_points, (d>0.05).mean())
