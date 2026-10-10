import sys, glob; from rlib import *
stem=sys.argv[1]; outs=sys.argv[2].split(','); zoom=float(sys.argv[3]) if len(sys.argv)>3 else 1.3
src=glob.glob(f'{R}/datatransform/multianeurysm/{stem}.*')[0]
i=load(src)
ks=[load(f'{R}/datatransform/hemoMesh/keep_one/surfaces/{o}.stl') for o in outs]
cen,vt=pca_view(np.asarray(i.points)); view=vt[2]*float(sys.argv[4]) if len(sys.argv)>4 else vt[2]
# input: mark vertices removed in any output
rem=np.zeros(i.n_points)
for k in ks: rem=np.maximum(rem,(sdist(i,k)>0.3).astype(float))
p=plotter(1400,1200); p.add_mesh(i,scalars=rem,cmap=[VESSEL,RED],show_scalar_bar=False,smooth_shading=True,specular=0.25,ambient=0.15)
set_cam(p,cen,view,vt[1],zoom=zoom); p.screenshot(f'{OUT}/ko_{stem[:5]}_in.png'); p.close()
for o,k in zip(outs,ks):
    k["new"]=(sdist(k,i)>0.15).astype(float)
    p=plotter(1400,1200); p.add_mesh(k,scalars='new',cmap=[VESSEL,NAVY],show_scalar_bar=False,smooth_shading=True,specular=0.25,ambient=0.15)
    set_cam(p,cen,view,vt[1],zoom=zoom); p.screenshot(f'{OUT}/ko_{o[:5]}{o[-2:]}_out.png'); p.close()
print('ok')
