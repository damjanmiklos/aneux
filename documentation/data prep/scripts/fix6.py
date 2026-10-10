p='C:/dev/UQ/aneux/documentation/data prep/data_preparation.tex'
s=open(p,encoding='utf-8',newline='').read()
reps=[("into 736 training samples","into 737 training samples"),
("In the final set, 53 multi-aneurysm vessels yield 115 single-aneurysm samples, 62 of them at a bifurcation and 53 on a sidewall.","In the final set, 54 multi-aneurysm vessels yield 117 single-aneurysm samples, 62 of them at a bifurcation and 55 on a sidewall."),
("All 115 samples of the final run passed.","All 117 samples passed."),
("The assembled set contains 736 surfaces: 115 from aneurysm isolation, 388 from cap or extension removal and 233 used as published.","The assembled set contains 737 surfaces: 117 from aneurysm isolation, 387 from cap or extension removal and 233 used as published."),
("All 736 surfaces produced","All 737 surfaces produced"),
("it produces 736 training samples","it produces 737 training samples"),
("53 multi-aneurysm vessels & Voronoi","54 multi-aneurysm vessels & Voronoi"),
("& 115 samples \\\\","& 117 samples \\\\"),
("& 736 surfaces \\\\","& 737 surfaces \\\\"),
("736 $\rightarrow$ 736","737 $\rightarrow$ 737"),
("& 736 GT + frames","& 737 GT + frames"),("& 736 centrelines","& 737 centrelines"),("& 736 templates","& 737 templates")]
for a,b in reps:
    n=s.count(a); print(n,a[:50])
    s=s.replace(a,b)
open(p,'w',encoding='utf-8',newline='').write(s)
