"""Edge-length figure + dataset numbers for the chapter; fills the @@...@@ placeholders."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

R = 'C:/dev/UQ/aneux'
TEX = f'{R}/documentation/data prep/data_preparation.tex'
d = pd.read_csv(f'{R}/documentation/data prep/scripts/stats.csv')
print('rows', len(d), 'errors', d['err'].notna().sum() if 'err' in d else 0)
if 'err' in d:
    print(d.loc[d['err'].notna(), ['id', 'err']].to_string())
    d = d[d['err'].isna()]

NAVY = '#104480'; GREY = '#9a9a9a'
plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.4))
b = np.linspace(min(d.src_emean.min(), d.u_emean.min()) * 0.95, d.src_emean.quantile(0.995), 45)
ax[0].hist(d.src_emean, bins=b, color=GREY, alpha=0.75, label='source')
ax[0].hist(d.u_emean, bins=b, color=NAVY, alpha=0.85, label='uniform GT')
ax[0].set_xlabel('mean edge length per surface [mm]'); ax[0].set_ylabel('surfaces')
b = np.linspace(0, d.src_ecv.quantile(0.99), 45)
ax[1].hist(d.src_ecv.clip(upper=b[-1]), bins=b, color=GREY, alpha=0.75, label='source')
ax[1].hist(d.u_ecv, bins=b, color=NAVY, alpha=0.85, label='uniform GT')
ax[1].set_xlabel('edge-length CV within surface [-]')
ax[1].legend(frameon=False)
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(f'{R}/documentation/data prep/figures/edge_stats.pdf')

def q(x, p): return float(np.percentile(x, p))
sph = np.concatenate([np.array(str(s).split(';'), float) for s in d.sph_r])
vals = {
    'EMEAN': f'{d.u_emean.mean():.3f}',
    'ECV_UNI': f'{d.u_ecv.median():.2f}',
    'ECV_SRC': f'{d.src_ecv.median():.2f}',
    'NPTS': f'{int(d.u_pts.median()):,}'.replace(',', '\\,'),
    'NPTS_RANGE': f'{int(q(d.u_pts,5)):,}--{int(q(d.u_pts,95)):,}'.replace(',', '\\,'),
    'NOPEN': f'a median of {int(d.n_open.median())} (range {int(d.n_open.min())}--{int(d.n_open.max())})',
    'CLLEN': f'{d.cl_len.median():.0f}',
    'CLMISR': f'{d.misr_med.median():.2f}',
    'SACFRAC': f'a median of {100*d.sac_frac.median():.1f}\\%',
    'SPHR': f'{np.median(sph):.2f}',
    'SPHR_RANGE': f'{q(sph,5):.2f}--{q(sph,95):.2f}',
}
extra = {
    'src_emean_med': d.src_emean.median(), 'src_emean_iqr': (q(d.src_emean, 5), q(d.src_emean, 95)),
    'u_emean_range': (q(d.u_emean, 5), q(d.u_emean, 95)), 'u_emin_min': d.u_emin.min(),
    'src_emin_med': d.src_emin.median(), 'u_emin_med': d.u_emin.median(),
    'area_ratio': (q(d.u_area / d.src_area, 1), d.u_area.div(d.src_area).median(), q(d.u_area / d.src_area, 99)),
    't_emean_med': d.t_emean.median(), 't_pts_med': d.t_pts.median(),
    'src_ecv_range': (q(d.src_ecv, 5), q(d.src_ecv, 95)), 'u_ecv_range': (q(d.u_ecv, 5), q(d.u_ecv, 95)),
}
for k, v in {**vals, **extra}.items():
    print(k, v)
s = open(TEX, encoding='utf-8').read()
for k, v in vals.items():
    s = s.replace(f'@@{k}@@', v)
open(TEX, 'w', encoding='utf-8').write(s)
print('left:', [w for w in s.split('@@')[1::2]])
