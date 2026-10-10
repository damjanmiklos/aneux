p='C:/dev/UQ/aneux/documentation/data prep/data_preparation.tex'
s=open(p,encoding='utf-8',newline='').read()
BS=chr(92)
# 1. Fig. 9 block
a=s.index(BS+'begin{figure}[H]\n    '+BS+'centering\n    '+BS+'includegraphics[width=0.325'+BS+'textwidth]{figures/tm_p141_z_gt.png}')
b=s.index(BS+'label{fig:dp_template}')+len(BS+'label{fig:dp_template}\n'+BS+'end{figure}\n')
def panel(img,cap): return None
new=(BS+"begin{figure}[H]\n    "+BS+"centering\n"
"    "+BS+"includegraphics[width=0.325"+BS+"textwidth]{figures/tmA_gt.png}"+BS+"hfill\n"
"    "+BS+"includegraphics[width=0.325"+BS+"textwidth]{figures/tmA_sph.png}"+BS+"hfill\n"
"    "+BS+"includegraphics[width=0.325"+BS+"textwidth]{figures/tmA_dist.png}"+BS+BS+"[1mm]\n"
"    "+BS+"hfill"+BS+"includegraphics[width=0.325"+BS+"textwidth]{figures/cbar_dist_06.pdf}\n"
"    "+BS+"caption{Template creation on the example vessel. Left: the GT with the detected aneurysm (sac and neck lip) in yellow. Middle: the template (translucent) with the three fitted spheres. Right: the GT surface coloured by the distance of each vertex to the template surface, measured to the nearest point of the template triangles. The scale ends at 0.6"+BS+",mm and saturates above it. Cream is below 0.1"+BS+",mm. On the sac the median distance is 0.07"+BS+",mm and the 95th percentile 0.23"+BS+",mm. On the parent it is larger (up to 0.9"+BS+",mm), since the 5"+BS+",mm radius knots deliberately ignore local calibre changes, which the network learns in the second stage.}\n"
"    "+BS+"label{fig:dp_template}\n"+BS+"end{figure}\n")
s=s[:a]+new+s[b:]
# 2. numbers in text
old=("Its distance to the GT has a median of 0.12"+BS+",mm and a 95th percentile of 0.42"+BS+",mm, and the largest deviations lie on the parent artery, where the 5"+BS+",mm radius knots deliberately ignore local bulges (Figure~"+BS+"ref{fig:dp_template}).")
assert s.count(old)==1, s.count(old)
newt=("We measure the distance from every GT vertex to the nearest point of the template surface, which is the stricter direction because it also sees sac wall that the template does not reach. Over the whole surface it has a median of 0.15"+BS+",mm and a 95th percentile of 0.47"+BS+",mm. On the sac alone the median is 0.07"+BS+",mm and the 95th percentile 0.23"+BS+",mm. The largest deviations lie on the parent artery, where the 5"+BS+",mm radius knots deliberately ignore local bulges (Figure~"+BS+"ref{fig:dp_template}).")
s=s.replace(old,newt)
open(p,'w',encoding='utf-8',newline='').write(s)
print('ok')
