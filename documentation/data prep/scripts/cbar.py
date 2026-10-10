import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib as mpl
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
fig, ax = plt.subplots(figsize=(3.4, 0.7))
fig.subplots_adjust(bottom=0.5, top=0.9, left=0.05, right=0.95)
cb = mpl.colorbar.ColorbarBase(ax, cmap=mpl.cm.magma_r, norm=mpl.colors.Normalize(0, 1), orientation='horizontal')
cb.set_label('distance to ground truth [mm]', fontsize=10); cb.ax.tick_params(labelsize=9)
fig.savefig('C:/dev/UQ/aneux/documentation/data prep/renders/cbar_dist.pdf', bbox_inches='tight', pad_inches=0.04)
