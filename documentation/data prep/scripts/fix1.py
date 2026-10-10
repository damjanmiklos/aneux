p = 'C:/dev/UQ/aneux/documentation/data prep/data_preparation.tex'
s = open(p, encoding='utf-8').read()
R = [
(r"We therefore turn every vessel with $k$ aneurysms into $k$ samples. Each sample keeps one aneurysm, and the other $k-1$ are removed and the parent artery is reconstructed in their place (Figure~\ref{fig:dp_keepone}). One patient (three vessels) was excluded after visual review. This leaves 53 multi-aneurysm vessels and 115 single-aneurysm samples, 62 of them at a bifurcation and 53 on a sidewall.",
 r"We therefore turn every vessel with $k$ aneurysms into $k$ samples. Each sample keeps one aneurysm, and the other $k-1$ are removed and the parent artery is reconstructed in their place (Figure~\ref{fig:dp_keepone}). In the final set, 53 multi-aneurysm vessels yield 115 single-aneurysm samples, 62 of them at a bifurcation and 53 on a sidewall."),
(r"For every aneurysm to be removed, the parent centrelines are cut at the neck and re-interpolated across it. All Voronoi balls inside a tube of twice the local radius around the interpolated segment are discarded. The parent's own balls on either side of the gap are then parallel-transported along the interpolated centreline to fill it \citep{ford2009}, and any ball that still protrudes towards the sac is pruned.",
 r"For every aneurysm to be removed, the parent centrelines are cut at the neck and re-interpolated across it. In the region of the neck, only the Voronoi balls lying within a tube of twice the local radius around the healthy parent centreline are kept, and the balls of the sac are discarded. The gap is then filled with the parent's own balls from either side, which are interpolated across it and parallel-transported along the patched centreline \citep{ford2009}. Any remaining ball that still protrudes towards the sac is pruned."),
(r"Every vertex that lies on the original surface is snapped back onto it.",
 r"Every vertex of the reconstruction that lies within a small distance of the original wall is snapped back onto it."),
(r"When a gate fails, the removal is retried with adjusted parameters, up to six times, for example with a shorter interpolation window or with the branch that was sealed handled explicitly.",
 r"When a gate fails, the removal is retried with adjusted clipping of the parent centrelines, up to six times."),
(r"An untouched original is used only when no processed version exists. Five pairs",
 r"An untouched original is used only when no processed version exists. One patient (three vessels) was excluded from the dataset after visual review. Five pairs"),
(r"This value matches the median edge length of the source data, so the remeshing changes how the surface is sampled without changing how finely it is resolved.",
 r"This value is close to the typical edge length of the source data (about 0.13\,mm), so the remeshing changes how the surface is sampled without changing how finely it is resolved."),
(r"Before saving, each GT has to pass four gates: an area within 0.88--1.20 times that of the source, a single closed-manifold component of genus zero, exactly one planar rim for every ostium frame, and no self-intersections.",
 r"Before it is saved, each GT has to pass two gates. Its area must lie within 0.88--1.20 times that of the source, which catches both a lost sac or branch and an extension that was not clipped off. It must also have exactly as many openings as the source, since a torn rim would show up as an additional one. An audit of the final set further confirmed that every GT is a single manifold component of genus zero with exactly one planar rim per ostium frame."),
(r"This last step follows the same intuition as neck-detection methods based on the parent tube \citep{cardenes2011}.",
 r"This is a deliberately simple definition of the neck; for a more elaborate, Voronoi-based alternative see \citet{cardenes2011}."),
(r"We therefore run four starts, each forcing a different deep medial ball to be the first sphere.",
 r"We therefore run four starts: the greedy one, and three more that each force a different deep medial ball to be the first sphere."),
(r"A short list of more conservative build plans, such as a wider gap to foreign vessel or flattening around a crowded stub, is tried only after the default build has failed.",
 r"Two fallback build plans are tried only after the default build has failed. One drops the carve away from foreign vessel, and the other flattens the walls just behind a crowded ostium plane."),
]
for a, b in R:
    assert a in s, a[:70]
    s = s.replace(a, b)
open(p, 'w', encoding='utf-8').write(s)
print('ok')
