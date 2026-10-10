p = 'C:/dev/UQ/aneux/documentation/data prep/data_preparation.tex'
s = open(p, encoding='utf-8').read()
R = [
(r"This value is close to the typical edge length of the source data (about 0.13\,mm), so the remeshing changes how the surface is sampled without changing how finely it is resolved.",
 r"Across the assembled surfaces the mean edge length ranges from 0.11 to 0.33\,mm (5--95\%, median 0.20\,mm). We chose the target at the fine end of this range, so that remeshing does not noticeably coarsen any surface and the finest-resolved vessels keep their detail."),
(r"All 736 surfaces passed the gates. Across the dataset, the remeshed GT has a mean edge length of 0.124\,mm",
 r"All 736 surfaces passed the gates. Across the dataset, the remeshed GT has a mean edge length of 0.124\,mm (the remesher settles slightly below its target), with the per-surface means lying within 0.123--0.124\,mm (5--95\%), and a median shortest edge of 25\,$\mu$m instead of 6\,$\mu$m. Its"),
(r"Remeshing collapses the spread between surfaces and roughly halves the spread within them.",
 r"Remeshing collapses the spread between surfaces and reduces the spread within them from a median CV of 0.22 to 0.13."),
(r"Lastly, the surface is remeshed with VMTK at a target edge length of 0.3\,mm.",
 r"Lastly, the surface is remeshed with VMTK at a target edge length of 0.3\,mm (a realised median of 0.22\,mm)."),
]
for a, b in R:
    assert a in s, a[:70]
    s = s.replace(a, b)
open(p, 'w', encoding='utf-8').write(s)
print('ok')
