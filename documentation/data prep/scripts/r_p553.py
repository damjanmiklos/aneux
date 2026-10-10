import json; from rlib import *
case='p553_FxMTDxQCFwAdGxQOHAUEABQW_RICA'
m=load(f'{R}/datatransform/multianeurysm/{case}.vtp')
cen,vt=pca_view(np.asarray(m.points))
p=plotter(1600,1200); p.add_mesh(m,**mesh_kw(VESSEL,opacity=0.55))
cols={'inlet':'#2e7d32','top':RED,'outlet':NAVY}
for k in (1,2):
    d=json.load(open(f'{R}/datatransform/hemoMesh/picked_points/{case}_{k}.json'))['aneurysms'][0]
    for name,c in cols.items():
        p.add_mesh(pv.Sphere(radius=0.5 if name!='top' else 0.8,center=d[name]),color=c)
        p.add_point_labels([d[name]],[f'{k}:{name}'],font_size=18,text_color='black',shape=None,always_visible=True)
set_cam(p,cen,vt[2],vt[1],zoom=1.0)
p.screenshot(f'{OUT}/p553_picks.png'); print(m.bounds)
