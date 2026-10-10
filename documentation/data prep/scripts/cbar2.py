import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib as mpl
plt.rcParams.update({'font.family':'serif','font.size':10})
for vmax,name,ticks in ((0.6,'cbar_dist_06',[0,0.1,0.2,0.3,0.4,0.5,0.6]),(1.0,'cbar_dist_10',[0,0.2,0.4,0.6,0.8,1.0]),(2.0,'cbar_dist_20',[0,0.5,1.0,1.5,2.0])):
    fig,ax=plt.subplots(figsize=(3.4,0.7)); fig.subplots_adjust(bottom=0.5,top=0.9,left=0.05,right=0.95)
    cb=mpl.colorbar.ColorbarBase(ax,cmap=mpl.cm.magma_r,norm=mpl.colors.Normalize(0,vmax),orientation='horizontal',extend='max')
    cb.set_ticks(ticks); cb.set_label('distance from template to GT [mm]',fontsize=10); cb.ax.tick_params(labelsize=9)
    fig.savefig(f'C:/dev/UQ/aneux/documentation/data prep/figures/{name}.pdf',bbox_inches='tight',pad_inches=0.04)
