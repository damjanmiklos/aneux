import sys; from rlib import *
case=sys.argv[1]; zoom=float(sys.argv[2])
u=load(f'{R}/cleandata/uniformly_remeshed/{case}.vtp')
c=pv.read(f'{R}/cleandata/original_centerline/{case}.vtp')
fr=np.load(f'{R}/cleandata/uniformly_remeshed/{case}.ostium_frames.npz')
cen,vt=pca_view(np.asarray(u.points))
p=plotter(1600,1200)
p.add_mesh(u, color='#e6e6e6', opacity=0.28, smooth_shading=True)
tube=c.tube(radius=0.12, n_sides=12)
p.add_mesh(tube, scalars='MaximumInscribedSphereRadius', cmap='viridis', smooth_shading=True,
           scalar_bar_args=dict(title='MISR [mm]', vertical=True, position_x=0.06, position_y=0.08, height=0.5, width=0.04, title_font_size=26, label_font_size=22, fmt='%.1f', color='black'))
for o,n,r in zip(fr['origin'],fr['normal'],fr['radius']):
    d=pv.Disc(center=o, inner=0, outer=r*1.15, normal=n, c_res=40)
    p.add_mesh(d, color=RED, opacity=0.85)
    p.add_mesh(pv.Arrow(start=o, direction=n, scale=max(1.5,2.2*r), tip_length=0.3, tip_radius=0.12, shaft_radius=0.04), color=RED)
set_cam(p,cen,vt[2],vt[1],zoom=zoom)
p.screenshot(f'{OUT}/cl_{case[:4]}.png'); p.close()
print(c.n_cells, c.n_points, list(c.cell_data.keys()), list(c.point_data.keys()))
