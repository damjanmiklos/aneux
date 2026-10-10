p='C:/dev/UQ/aneux/documentation/data prep/data_preparation.tex'
s=open(p,encoding='utf-8',newline='').read()
BS=chr(92)
anchor="Across the dataset, a median of 5.9"+BS+"% of the GT wall was detected as aneurysm, and the fitted sphere radii have a median of 1.58"+BS+",mm (5--95"+BS+"%: 0.68--3.96"+BS+",mm)."
assert s.count(anchor)==1
i=s.index(anchor)+len(anchor)
para=("\n\nThree spheres are a coarse description, and they fit the sac closely only when the sac is roughly round. Over the dataset, the median per-case median distance of the sac wall to the template is 0.13"+BS+",mm. In 10"+BS+"% of the cases the median exceeds 0.30"+BS+",mm, and in 11 cases (1.5"+BS+"%) it exceeds 0.5"+BS+",mm. The worst cases are giant sacs with calibres of several millimetres, and sacs that are elongated, flattened or multi-lobed, which three spheres of a common smooth minimum cannot follow. Figure~"+BS+"ref{fig:dp_template_limit} shows one of them. The sac is a tall, tilted horn. The sphere penalty keeps every sphere inside the GT wall, so the fit settles on two large spheres stacked along the long axis and a small one near the tip, and the broad face of the sac sits up to 2--3"+BS+",mm outside the template (median 0.61"+BS+",mm, 95th percentile 1.95"+BS+",mm). We accept this, since the template is only the starting point of the second stage, but it also means that the network has to learn the shape of such sacs from the deformation, not from the template.\n\n")
fig=(BS+"begin{figure}[H]\n    "+BS+"centering\n"
"    "+BS+"includegraphics[width=0.325"+BS+"textwidth]{figures/tmB_gt.png}"+BS+"hfill\n"
"    "+BS+"includegraphics[width=0.325"+BS+"textwidth]{figures/tmB_sph.png}"+BS+"hfill\n"
"    "+BS+"includegraphics[width=0.325"+BS+"textwidth]{figures/tmB_dist.png}"+BS+BS+"[1mm]\n"
"    "+BS+"hfill"+BS+"includegraphics[width=0.325"+BS+"textwidth]{figures/cbar_dist_20.pdf}\n"
"    "+BS+"caption{A sac that three spheres cannot approximate. Left: the GT with the detected sac in yellow. Middle: the template with the three fitted spheres (radii 8.3, 4.2 and 7.6"+BS+",mm). Right: the GT coloured by its distance to the template, on a scale that ends at 2"+BS+",mm. The elongated, tilted sac is outside the spheres over most of its broad face and at its tip.}\n"
"    "+BS+"label{fig:dp_template_limit}\n"+BS+"end{figure}\n")
s=s[:i]+para+fig+s[i:]
open(p,'w',encoding='utf-8',newline='').write(s)
print('ok')
