p='C:/dev/UQ/aneux/documentation/data prep/data_preparation.tex'
s=open(p,encoding='utf-8',newline='').read()
B=chr(92); mm=B+',mm'
def rep(a,b):
    global s
    assert s.count(a)==1,(s.count(a),a[:70]); s=s.replace(a,b)
rep("Figure~"+B+"ref{fig:dp_template_limit} shows one of them. The sac is a tall, tilted horn. The sphere penalty",
    "Figure~"+B+"ref{fig:dp_template_limit} shows two of them. The upper sac is a typical moderate case. It is bilobed, with a narrow neck between its lobes and a pointed tip, and the three smooth spheres reproduce its gross form but not the waist or the tip: the sac surface of the template lies a median of 0.24"+mm+" from the GT wall (95th percentile 0.66"+mm+", maximum 1.1"+mm+"), and the GT wall a median of 0.31"+mm+" from the template (95th percentile 0.92"+mm+"). The lower sac is one of the worst cases. It is a tall, tilted horn. The sphere penalty")
a=s.index(B+"begin{figure}[H]\n    "+B+"centering\n    "+B+"includegraphics[width=0.325"+B+"textwidth]{figures/tmB_gt.png}")
b=s.index(B+"label{fig:dp_template_limit}")+len(B+"label{fig:dp_template_limit}\n"+B+"end{figure}\n")
def row(tag,bar):
    return ("    "+B+"includegraphics[width=0.325"+B+"textwidth]{figures/%s_gt.png}"%tag+B+"hfill\n"
            "    "+B+"includegraphics[width=0.325"+B+"textwidth]{figures/%s_sph.png}"%tag+B+"hfill\n"
            "    "+B+"includegraphics[width=0.325"+B+"textwidth]{figures/%s_dist.png}"%tag+B+B+"[1mm]\n"
            "    "+B+"hfill"+B+"includegraphics[width=0.325"+B+"textwidth]{figures/%s.pdf}"%bar+B+B+"[3mm]\n")
new=(B+"begin{figure}[H]\n    "+B+"centering\n"+row('tmC','cbar_dist_10')+row('tmB','cbar_dist_20')+
"    "+B+"caption{Two sacs that three spheres approximate poorly. Each row shows the GT with the detected sac in yellow (left), the template with the three fitted spheres (middle) and the finished template coloured by its distance to the GT (right). Top: a bilobed sac with a pointed tip, a moderate fit (spheres of 2.2, 1.8 and 1.0"+mm+"; scale ends at 1"+mm+"). Bottom: an elongated, tilted sac, one of the worst fits in the dataset (spheres of 8.3, 4.2 and 7.6"+mm+"; scale ends at 2"+mm+"). The bottom sac is outside the spheres over most of its broad face and at its tip, and the spheres themselves bulge away from the wall between the rings where they touch it.}\n"
"    "+B+"label{fig:dp_template_limit}\n"+B+"end{figure}\n")
s=s[:a]+new+s[b:]
open(p,'w',encoding='utf-8',newline='').write(s); print('ok')
