p='C:/dev/UQ/aneux/documentation/data prep/data_preparation.tex'
s=open(p,encoding='utf-8',newline='').read()
B=chr(92); mm=B+',mm'
def rep(a,b):
    global s
    assert s.count(a)==1,(s.count(a),a[:60]); s=s.replace(a,b)
rep("Right: the GT surface coloured by the distance of each vertex to the template surface, measured to the nearest point of the template triangles. The scale ends at 0.6"+mm+" and saturates above it. Cream is below 0.1"+mm+". On the sac the median distance is 0.07"+mm+" and the 95th percentile 0.23"+mm+". On the parent it is larger (up to 0.9"+mm+"),",
    "Right: the finished template, smoothed, meshed and cut at the ostia, coloured by the distance of each of its vertices to the GT surface. The scale ends at 0.6"+mm+" and saturates above it. Cream is below 0.1"+mm+". On the sac the median distance is 0.06"+mm+" and the 95th percentile 0.20"+mm+". On the parent it is larger (up to 0.9"+mm+"),")
rep("We measure the distance from every GT vertex to the nearest point of the template surface, which is the stricter direction because it also sees sac wall that the template does not reach. Over the whole surface it has a median of 0.15"+mm+" and a 95th percentile of 0.47"+mm+". On the sac alone the median is 0.07"+mm+" and the 95th percentile 0.23"+mm+".",
    "We measure the distance from every vertex of the finished template to the nearest point of the GT surface. Over the whole template it has a median of 0.12"+mm+" and a 95th percentile of 0.42"+mm+". On the sac alone the median is 0.06"+mm+" and the 95th percentile 0.20"+mm+". Measured the other way, from the GT vertices to the template, which also sees sac wall that the template does not reach, the figures on the sac are 0.07 and 0.23"+mm+".")
rep("and the broad face of the sac sits up to 2--3"+mm+" outside the template (median 0.61"+mm+", 95th percentile 1.95"+mm+").",
    "and the broad face of the sac sits up to 2--3"+mm+" outside the template. The sac surface of the template lies a median of 0.66"+mm+" from the GT wall (95th percentile 2.6"+mm+", maximum 4.3"+mm+"), and the GT wall lies a median of 0.61"+mm+" from the template (95th percentile 1.95"+mm+").")
rep("Right: the GT coloured by its distance to the template, on a scale that ends at 2"+mm+". The elongated, tilted sac is outside the spheres over most of its broad face and at its tip.",
    "Right: the finished template coloured by its distance to the GT, on a scale that ends at 2"+mm+". The elongated, tilted sac is outside the spheres over most of its broad face and at its tip, and the spheres themselves bulge away from the wall between the rings where they touch it.")
open(p,'w',encoding='utf-8',newline='').write(s); print('ok')
