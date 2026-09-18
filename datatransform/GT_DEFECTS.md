# Ground-truth mesh defects

A register of the defects found in the 747-case remeshing run and in the 119
keep-one ground-truth meshes that `hemoMesh/remove_other_aneurysms.py` produced.
One section per defect family: what it looks like, how many meshes it hits, what
actually causes it, and what has been done about it.

Two dataset populations are involved and they are easy to confuse:

* `cleaned_data/total_clean_original_mesh/` holds **747** files, of which only
  **119 are `.stl`**. Those 119 are exactly the keep-one outputs. The other 628
  are `.vtp`/`.vtk` from other sources and no defect below touches them.
* The 14-hour `template_creation/remeshing.py` run consumed all 747 and returned
  **109 errors and 9 hangs**.

Status key: **fixed** (change in, tests in), **in progress**, **open**.

---

## 1. A damaged mesh shipped because the guard fired too late — fixed

**29 of 119.** `_assert_opening_count_unchanged` runs inside `main_run`, which is
*after* the STL has already been written. `run_keep_one_job` caught the resulting
exception, saw a non-empty STL on disk, salvaged it, marked the job `ok` and
deleted the work directory. The guard was correct and its verdict was discarded.

26 of the 29 had simply **lost an opening** (7 to 6, 8 to 7, ...). The lost rims
are small side branches, r = 0.11-0.95 mm, mostly 7-43 mm away from the removed
sac, dropped by the Voronoi reconstruction rather than by the clip.

**Cause: the driver**, not the mesh and not the geometry code.

**Fix** (`remove_other_aneurysms.py`): `_is_mesh_verdict` splits exceptions that
are a verdict *about the surface* (`OpeningCountChangedError`, `VesselSplitError`,
`MeshQualityError`) from ones raised by steps that run *after* the surface is
final (centerlines, the geometry report). Only the latter may salvage. A verdict
sends the mesh to `<output>/rejected/` with the reason beside it, so an hour of
Voronoi work is not thrown away and the keep/discard call can be made by looking
at it. Covered by `TestSalvageOnlyForLaterSteps` and `TestRejectionIsParked`.

## 2. Fan tents in the ground truth — cause found, weld swept, rest quarantined

**3 of 119: p375_2, p376_3, p551_2.** A single vertex carries hundreds of
triangles and a sheet is stretched across the lumen. Confirmed by the user on
screenshots.

Traced sub-step by sub-step on p376_3:

| stage | pts | tris | max valence | hubs | area |
|---|---|---|---|---|---|
| reconstructed input | 249,553 | 499,146 | 18 | 0 | 2274.5 |
| after clip at opening planes | 232,580 | 464,207 | 18 | **0** | 2110.7 |
| **after vmtk remesh** | 68,915 | 137,990 | **327** | **50** | **4587.0** |
| after all 6 `_delete_outboard_leftover` | 68,915 | 137,990 | 327 | 50 | 4587.0 (no-op) |
| after `restore_wall_texture` | 68,900 | 137,947 | 327 | 50 | 4602.1 |

**Cause: `SurfaceMesh._remesh_surface_vmtk`** (`src/mesh/surface_mesh.py`). It is
the only step that creates a hub, and it **more than doubles the surface area**.
The input it is handed is sound: zero hubs, max valence 18, mean edge 0.112 mm.

Nine settings across both element-size modes were tried. **Every one of them
diverges**, and fewer iterations does not help monotonically:

| mode | size | iters | area | drift | max val | hubs | CV |
|---|---|---|---|---|---|---|---|
| area | 0.025 | 5 (shipped) | 4587.0 | 2.173x | 327 | 50 | 1.96 |
| area | 0.025 | 3 | 6320.4 | 2.994x | 199 | 30 | 2.12 |
| area | 0.025 | 1 | 5483.7 | 2.598x | 106 | 8 | 2.07 |
| area | 0.01 | 5 | 5125.5 | 2.428x | 295 | 59 | 2.24 |
| area | 0.01 | 3 | 7298.8 | 3.458x | 183 | 47 | 2.43 |
| area | 0.005 | 3 | 7458.1 | 3.533x | 206 | 62 | 2.63 |
| edgelength | 0.15 | 5 | 5052.0 | 2.394x | 370 | 56 | 2.26 |
| edgelength | 0.15 | 3 | 6986.1 | 3.310x | 178 | 45 | 2.38 |
| edgelength | 0.15 | 1 | 5870.9 | 2.782x | 98 | 6 | 2.27 |

So **no parameter choice fixes this** — on this input the remesher itself fails,
and the fix has to be to detect the failure rather than to tune around it.

Dataset-wide area drift: p375_2 2.567x, p376_3 1.577x, every other mesh <=1.011x
(median 0.957x), so this is a divergence on particular inputs, not a constant bias.

**Guard already in** (`remove_other_aneurysms.py`): `MAX_VERTEX_VALENCE = 20`
rejects a tented mesh before it is copied. Over the 119 meshes the worst sound
one peaks at 13 and the three damaged ones at 39, 273 and 336. Being a plain
neighbour count it carries no millimetre scale, so it does not need retuning for
a different vessel size. Covered by `TestMeshQualityGate`.

**Root cause.** The surface handed to the remesher is a marching-cubes
reconstruction, and it is not as sound as the valence column above suggests. It
carries **1,031 triangles under 1e-6 mm2, edges down to 0.000010 mm, 25
non-manifold edges and 5 duplicate triangles**. vmtkSurfaceRemeshing works by
collapsing and splitting edges and projecting the result back onto the input, and
a triangle that small has no usable normal to project against — so points get
thrown off the surface, which is both the area doubling and the tents.

Welding those degeneracies out first, with `vtkCleanPolyData` at an absolute
tolerance set as a fraction of the mesh's own mean edge:

| weld tolerance | area drift | edge CV | hubs | aspect > 50 | non-manifold | dup tris |
|---|---|---|---|---|---|---|
| none | 2.173x | 2.022 | 36 | 5,656 | 25 | 5 |
| 0.001 mm | 2.096x | 2.001 | 29 | — | 25 | 5 |
| **0.005 mm (shipped)** | **1.116x** | **0.593** | **16** | **1** | 25 | 5 |
| 0.02 mm | 1.364x | 1.067 | 22 | — | **61** | **43** |

0.005 mm is a twentieth of the mean edge, and it is where the curve turns: it
preserves the area itself to four figures (1.000x) while taking aspect>50 from
5,656 to 1. Past it the weld starts joining walls that merely *pass close to one
another* — non-manifold edges 25 -> 61, duplicate triangles 5 -> 43 — and the
remesh gets worse again. The tolerance is stored as
`REMESH_WELD_FRACTION_OF_EDGE = 0.05`, a fraction rather than a distance, so it
does not need retuning for a finer reconstruction or a different vessel size.

**Welding is a large improvement but not a cure.** Sweeping the settings again on
the *cleaned* surface (base area 2110.7):

| mode | size | iters | pts | area | drift | CV | max val | hubs |
|---|---|---|---|---|---|---|---|---|
| area | 0.025 | 5 (shipped) | — | 2354 | **1.116x** | 0.593 | — | 16 |
| area | 0.025 | 3 | 302,458 | 2605.7 | 1.235x | 0.852 | 29 | 15 |
| area | 0.025 | 1 | 251,646 | 2755.1 | 1.305x | 1.083 | 22 | 3 |
| edgelength | 0.15 | 3 | 394,755 | 2604.4 | 1.234x | 0.918 | 34 | 18 |
| edgelength | 0.15 | 1 | 288,718 | 2762.4 | 1.309x | 1.109 | 22 | 4 |

Every one still drifts past the 1.2x tolerance or leaves hubs. What is left after
the weld is the 25 non-manifold edges, and no amount of point merging repairs
topology — it needs a remesher that is robust to it, and none of pyacvd,
pymeshlab, open3d or trimesh is installed in either env.

**So the shipped behaviour is: weld, remesh, and insist.**
`_weld_degenerate_triangles` cleans the input, `_remesh_diagnosis` measures the
result against that cleaned input on both symptoms — `REMESH_MAX_AREA_DRIFT =
1.2` either way and `REMESH_MAX_VALENCE = 20` — and a remesh that diverged
**raises `RemeshDivergedError`**, which `_MESH_VERDICT_ERRORS` in the driver
parks as a verdict rather than a crash.

**An earlier version of this note said the diverged remesh should fall back to
its input, and that was wrong.** Measuring the fallback rather than assuming:

| | fallback surface | a sound remesh |
|---|---|---|
| edge CV | **0.415** | 0.133 |
| min edge | **0.000010 mm** | 0.0104 mm |
| triangles with aspect > 50 | **5,656** | 0 |
| triangles under 1e-6 mm2 | **1,031** | 2 |

The fallback violates the uniform-edge objective outright, and it is exactly the
kind of input section 7 records as making `remeshing.py` diverge downstream. It
also **converts a caught failure into an uncaught one**: a tent peaks at valence
327 so the driver's quality gate catches it, but the dense fallback peaks at 18
and sails straight through to be shipped. Raising is the right outcome — the case
is parked carrying the measurement that condemned it, and nothing damaged
reaches the dataset.

**Guard still in, as the backstop** (`remove_other_aneurysms.py`):
`MAX_VERTEX_VALENCE = 20` and `MAX_EDGE_LENGTH_CV = 0.25` reject a damaged mesh
before it is copied, whatever produced it. Over the 119 meshes the 116 sound ones
span CV 0.125-0.188 and peak at valence 13; the three damaged ones sit at CV
0.317 / 1.256 / 1.943 and valence 39 / 273 / 336. Being a plain neighbour count
and a dimensionless ratio, neither needs retuning for a different vessel size.
Covered by `TestMeshQualityGate`.

**Verified end to end** on the exact surface that produced the tents:

```
input        pts=232580  area=2110.7  CV=0.415  minedge=0.000010  slivers=1031
after weld   pts=223941  area=2110.7  CV=0.391  minedge=0.005604  slivers=  40
RESULT: RemeshDivergedError -- the remesh left 15 triangle-fan hub(s);
        the worst vertex carries 29 triangles
```

The weld leaves the area unchanged to four figures, lifts the minimum edge by
560x and removes 96% of the slivers; the case that cannot be built correctly is
then refused instead of shipped. Covered by `TestRemeshDiagnosis`.

**Superseded in part**: that verification used a single weld tolerance, 0.05 of
the mean edge, chosen on the reasoning that welding harder creates non-manifold
junctions and must make things worse. Section 7 measured that reasoning through
to the remesh and it is wrong — the tolerance that makes the most junctions is
the one that converges, and p379's reconstruction needs 0.25 to do it while
still diverging at 2.157x at 0.05. `_remesh_surface_vmtk` now sweeps
`REMESH_WELD_FRACTIONS` rather than using the one value, so p376_3 gets four
attempts where it got one. Whether that is enough for it is not yet known; it
will be answered by the keep-one rebuild, and until then the case stays parked
by the same raise, which is the safe direction to be wrong in.

## 3. A ball on an opening (p375_1) — fixed and verified

The user's report: one opening "has a ball at the end with parts missing from
that scattered ball, that ball shouldn't even be there".

Not global inflation: p375_1's area drift is 0.993x, so this is **local**. At the
bifurcation outlet, against a healthy opening on the same mesh:

| | this opening | a healthy one |
|---|---|---|
| rim points | 235 | 112 |
| radius | 1.753 | 1.247 |
| out-of-plane sd | **1.012 mm** | 0.122 mm |
| perimeter / circle | **2.31** | 1.04 |

plus two pinholes. p375_1 also fails the opening-count guard 5 to 7, and those
two extra openings are exactly the pinholes — so it is now caught and quarantined
rather than shipped (reproduced: `5 before, 7 after`).

**Working explanation:** the Voronoi extension stub past that opening is fatter
than the clip cylinder. `_opening_clip_radius(r) = max(1.5r, r + 0.2)` sizes that
cylinder from the *inscribed-sphere* radius, so on a wide or non-circular opening
the clip bores through the middle of the stub and leaves an annular collar — the
"ball with parts missing".

**Fix** (`removal.py`): `_rim_extents_from_surface` measures each opening's real
rim reach off the original surface and `_opening_radius` takes the larger of that
and the inscribed radius, so the clip cylinder can never be narrower than the
opening it is cutting around. It falls back to the inscribed radius when no rim
is close enough to be this opening's, since clipping conservatively beats
clipping against someone else's rim.

The mechanism is confirmed on p375: **all five** of its openings have a rim that
reaches past the inscribed radius, the worst by 52%.

| inscribed | real rim | clip radius before | after |
|---|---|---|---|
| 2.093 | 2.515 | 3.139 | 3.773 |
| 1.031 | 1.568 | 1.546 | 2.352 |
| 0.888 | 1.038 | 1.332 | 1.557 |
| 0.943 | 1.188 | 1.415 | 1.782 |
| 0.393 | 0.539 | 0.593 | 0.808 |

**Verified end to end.** p375 keep 1 previously failed the opening-count guard
5 to 7; with the clip sized from the rim it now passes, and every rim on the
mesh is round:

| | shipped (old) | clip fix |
|---|---|---|
| openings | 7 (5 real + 2 pinholes) | **5** |
| worst rim, out-of-plane sd | **1.012 mm** | **0.012 mm** |
| worst rim, perimeter / circle | **2.31** | **1.09** |
| every other rim, perimeter / circle | 0.68-1.03 | 1.00 |
| edge CV | 0.132 | 0.132 |

The ball is gone, the two pinholes with it, and edge uniformity is untouched.

## 4. A sac left behind where it should have been removed — detector in, cause open

Reported by the user on p379_1 ("did not fully delete one of the aneurysms and
left a little stub") and independently on p431_1.

Metric: distance from each **removed** aneurysm's picked dome apex to the output
surface. A correct removal pulls the wall back to the parent vessel and leaves
that point well clear. Over 156 removals the median is 2.41 mm, and the failures
separate sharply — nine below 0.12 mm, then nothing until 0.87 mm:

| mesh | apex-to-surface |
|---|---|
| p349_1 | 0.036 mm |
| SNF00000360_01_2 | 0.046 mm |
| p431_2 | 0.050 mm |
| SNF00000364_01_2 | 0.053 mm |
| SNF00000228_01_1 | 0.086 mm |
| **p379_1** | 0.087 mm |
| p379_3 | 0.087 mm |
| p439_1 | 0.100 mm |
| **p431_1** | 0.119 mm |

Both meshes the user flagged by eye are in the list, which is the check
validating against an independent observer rather than against itself.

**Guard in** (`remove_other_aneurysms.py`): `RESIDUAL_SAC_MM = 0.25` sits inside
the gap and is still about two edge lengths above the mesh's own resolution. It
measures "is the wall still here", not a size, so it holds for a small sidewall
sac as well as a big one.

**p439 belongs here, and it is the extreme case.** The user reported it as
"p439_2 leaves a ball instead of removing that aneurysm", which read at first
like a separate failure mode. Measuring both files says otherwise. For each sac,
its distance to the surface of each output and the surface area within 6 mm of
its apex:

| sac | file | role | apex-to-surface | area within 6 mm |
|---|---|---|---|---|
| #1 | `..._1.stl` | kept | 0.104 mm | 137.98 mm2 |
| #1 | `..._2.stl` | **removed** | 3.739 mm | 35.88 mm2 |
| #2 | `..._1.stl` | **removed** | **0.100 mm** | **112.35 mm2** |
| #2 | `..._2.stl` | kept | 0.038 mm | 97.76 mm2 |

Removing sac #1 took away **74% of the local surface** and left the centerline a
normal standoff nearby (median 1.086, max 1.448) with all five openings round to
within perim/circumference 1.00-1.05. That file is sound.

Removing sac #2 took away **-15%** — the mesh that is supposed to have lost that
sac carries *more* surface there than the mesh that keeps it — with the apex
0.100 mm from the wall. **The sac was not removed at all.** So the ball is sac #2
sitting intact, and it is in the file ending `_1`, not `_2`. Worth stating
plainly because the report named the other file: `p439_..._1.stl` is the damaged
one, and the sac involved is #2.

That makes it the same defect as the stubs above, at its limit — nothing came off
instead of most of it coming off — and the `RESIDUAL_SAC_MM` detector already
catches it (0.100 mm, sixth in the table).

**Cause: found — the aneurysm centerline is built from the wrong line.**

The mask that gets subtracted is a tube around the aneurysm centerline, so the
first thing to check is whether that centerline goes anywhere near the sac. On
p379 keep 1, with the two removed sacs side by side:

| | sac 2 (bifurcation) | sac 3 (sidewall) |
|---|---|---|
| full Voronoi reaches to | 0.681 mm of the apex | 0.415 mm of the apex |
| aneurysm centerline ends | **7.580 mm from the apex** | 0.000 mm from the apex |
| mask reaches to | **3.540 mm of the apex** | 0.415 mm of the apex |
| mask size | 46,320 points | 6,449 points |

The sidewall sac's centerline runs to the dome and its mask covers the sac to the
resolution of the diagram itself. The bifurcation sac's centerline stops 7.6 mm
short, its tube therefore stops 3.5 mm short, and the top of the sac is never
subtracted. That is the stub.

Why it stops short: for a bifurcation, `patchandinterpolatecenterlines.py` took
the aneurysm centerline to be **cell 0** of the first daughter centerline. But
`vmtkcenterlines` emits its lines in its own order, not in the order the targets
were supplied. On p379 daughter 1 has three cells and cell 0 is the one running
back to the *inlet*, 24.9 mm from the dome; the line that ends on the dome is
cell 2:

```
1_dau1cl.vtp  cells=3
   cell 0: n=794  starts near outlet1  ends 24.91 mm from apex
   cell 1: n=746  starts near outlet1  ends near outlet2
   cell 2: n=465  starts near outlet1  ends  0.00 mm from apex   <- the dome line
```

So the aneurysm tube was built around a stretch of parent artery and the sac it
was meant to remove survived. The sidewall path never had the bug because it
takes its centerline from `SaveAneurysm(forwardCenterlines)`, which is the branch
past the diverging point by construction.

**This accounts for eight of the nine.** Sorting the 119 meshes by how close the
nearest removed sac still is, and reading off how each was picked:

| | n | residual under 0.25 mm | median standoff |
|---|---|---|---|
| bifurcation | 63 | **8** | 2.20 mm |
| sidewall | 56 | 1 | 2.63 mm |

Every residual in the table above is a bifurcation except p431_1 at 0.119 mm, the
mildest of the nine and the one the user described as "a small stub". That one is
a sidewall and needs its own answer; it is the only part of this section still
open.

**Fix** (`patchandinterpolatecenterlines.py`, `removal.py`): `dome_cell_id` picks
the daughter cell that actually ends on the dome instead of trusting cell 0. It
uses the picked dome point, which the driver now threads through `cut_aneurysm`
as `top_point`, and falls back — when no point is supplied — to the endpoint
standing farthest off the parent artery, which is what an aneurysm is. The chosen
cell is then used for both the clipping points and the centerline itself, which
previously re-extracted cell 0 a second time. Covered by `tests/test_dome_cell.py`,
which fixes the emitted orders the selection has to survive, including p379's own.

**End-to-end verification is running.**

## 5. A ball instead of the removed aneurysm (p439) — merged into section 4

Kept as a heading so the numbering the rest of this file refers to does not
move. It looked like a second failure mode and it is not: p439 is section 4 at
its limit, the sac not removed at all rather than mostly removed, and in the
file ending `_1` rather than `_2`. The numbers are in section 4.

## 6. Centerline hangs on the SNF vessels — fixed and verified

**6 cases** (SNF00000059, 357, 535, 538_02, 592_01, 614) plus 3 others hung.
`vmtkCenterlines` tetrahedralises the whole capped surface before tracing
anything, and on these that never returns. It is a native hang, not an
exception, so no `try` could reach it — and the existing
retry-without-extensions path was correct but **unreachable**, because the call
above it never came back.

`DelaunayTolerance` is conclusively not the cause: all five values from 0.001 to
0.3 hang.

**Fix** (`template_creation/`): `centerline_worker.py` runs the trace in a child
process and `vessel_pipeline.py` puts a `CENTERLINE_TIMEOUT_S = 600` clock on it,
returning `None` instead of hanging so the existing fallback can run. All six
then trace a valid centerline off the un-extended surface in 4.2-14.5 s and
reach every target: 59 in 5.3 s 4/4, 357 in 4.7 s 7/7, 535 in 4.2 s 10/10,
538_02 in 10.0 s 8/8, 592_01 in 14.5 s 14/14, 614 in 4.4 s 9/9.

**Verified end to end**: all six run through `remeshing.py` in 13.5 minutes,
6 success / 0 error, and the outputs meet every objective — opening counts match
their targets exactly, and the edge length is uniform with no fan hubs:

| case | pts | openings | mean edge | CV | max valence | hubs |
|---|---|---|---|---|---|---|
| SNF00000059 | 111,569 | 5 | 0.1233 | 0.129 | 8 | 0 |
| SNF00000357 | 90,497 | 8 | 0.1244 | 0.120 | 7 | 0 |
| SNF00000535 | 140,584 | 11 | 0.1233 | 0.124 | 8 | 0 |
| SNF00000538_02 | 105,499 | 9 | 0.1236 | 0.125 | 8 | 0 |
| SNF00000592_01 | 139,476 | 15 | 0.1233 | 0.126 | 8 | 0 |
| SNF00000614 | 99,494 | 10 | 0.1232 | 0.124 | 8 | 0 |

## 7. Remesh divergence in the GT pipeline — fixed

`vmtkSurfaceRemeshing` would return a surface half again the area of the one it
was given, report nothing, and let it ship. p097 came out at 1.47x.

**The cause is edges far shorter than the mesh around them, and the cure is a
weld tolerance measured in that mesh's own units.** The pipeline already welded,
at `WELD_TOLERANCE_MM` = 1e-3 mm, which is the right instrument for exact
duplicates and roughly forty-five times too small for this. p097 entered the
remesh with its shortest edge at 0.000777 mm inside a mesh averaging 0.1337, and
still ran away. Welding the same surface at 0.05 of its mean edge — 0.0067 mm —
lands it at **0.999x with an edge-length CV of 0.119**.

The tolerance cannot be a constant, because the value that works is not the same
on every surface:

| weld, as a fraction of the mean edge | p097 (template) | p379 GT reconstruction (hemoMesh) |
|---|---|---|
| none | 1.476x, CV 1.137 | — |
| **0.05** | **0.999x, CV 0.119** | 2.157x, CV 1.746 |
| 0.10 | 0.999x, CV 0.119 | — |
| **0.25** | — | **0.995x, CV 0.133** |
| 0.50 | — | 10.056x, CV 1.591 |

So it is swept, gentlest first, and the first tolerance whose remesh survives
inspection is taken. Starting at no weld at all means a case that already
converges is remeshed exactly as it was before. The sweep stops at 0.25 because
past there it gets worse, not better.

**What the weld actually does is worth recording, because it is not what it
looks like.** It does not clean the surface up. Measured on p379 either side of
the change, it moves no point off the surface at all (max deviation 0.0000 mm)
and holds the area to within 0.08% — but it *creates* non-manifold edges as it
goes, none at 0.05 and 79 by 0.25. The tolerance that makes 79 of them is the
one that converges. What defeats this remesher is short edges, not junctions,
and the old comment in `surface_mesh.py` had that backwards.

Nothing else is the lever. On p379's reconstruction both element-size modes and
1, 2, 5 and 10 iterations all diverge by 2.0x–2.6x, and `MinEdgeLength` changes
the result not at all (5263.3 mm2 with and without).

Every attempt is scored against the **unwelded** input, so a tolerance that ate
geometry cannot pass by flattering itself.

**Status**: p097 rebuilds clean. p379_1 and p379_3 need their ground truth
rebuilt first (section 4), and the reconstruction that feeds them now converges
at 0.995x, so they are expected to follow.

## 8. Silent surplus openings — the count was wrong, not the mesh

p489 shipped claiming 9 openings against 5 anatomical profiles, p129 6 against
4, and nothing rejected either.

**p489 was never broken.** Counting its boundary edges by connectivity — pure
topology, which cannot split a connected rim — gives **5**, matching its 5
profiles exactly. Both inputs are clean too: p489 arrives with 5 openings and
p129 with 4, so nothing was tearing rims.

The fault was in `extract_boundary_loops`. It runs two extractors, because each
fails where the other holds, and took **whichever reported more loops**. That is
wrong in the one direction that matters: a rim counted twice is an opening that
does not exist.

| | VMTK extractor | stripper | connectivity | was chosen |
|---|---|---|---|---|
| p489 | **9** | 5 | 5 | 9 |
| p129 | **6** | 5 | 5 | 6 |
| C0010 | 7 | 7 | 7 | 7 |
| SNF00000152 | 4 | 4 | 4 | 4 |

**Fix**: `count_boundary_regions` is the referee. Whichever extractor agrees
with the number of boundary regions is the one telling the truth; if neither
does, the one that over-counts least is kept, since an invented opening is worse
than a missed rim walk. Checked across all 110 outputs of the validation run:
**every one now agrees with connectivity**, where two did not before.

**The miscount was also making the tear**, which is why p129 came out worse
than its input. `patch_wall_pinholes` caps every loop and then re-opens the one
cap nearest each rim it means to keep. Hand it a rim split in two and both
halves go looking for a cap: the first takes the right one, the second takes the
next nearest -- which on p129 was the pinhole 0.85 mm away that had just been
patched. So the patcher reported "patched 4 wall pinhole(s)" and shipped a mesh
with one of them re-opened, in the same run.

Re-run end to end with the referee in place, p129 comes out with **4 openings
against 4 profiles** and the 6-point tear is gone:

| | old run | with the referee |
|---|---|---|
| p489 | warned 9 openings vs 5 profiles | 5 openings, no warning |
| p129 | warned 6 vs 4; output carried a 6-pt r=0.197 tear | 4 openings, smallest 24 pts at r=0.327 |

Both were the same mesh going in. p489 was only ever miscounted; p129 was
miscounted **and** damaged by what the miscount told the patcher to do.

**Gate in** (`remeshing.py`): a finished mesh whose opening count differs from
its anatomical profile count now raises instead of printing a warning and
shipping. With the count repaired neither of these two trips it, which is the
point -- the old gate would have condemned p489, which was fine, and let p129
through, which was not.

**Regression**: the seven-case run (C0010, C0056, SNF00000152, SNF00000415,
ANSYS_UNIGE_09, p489, p129) is 7/7 success with every opening count matching
connectivity, so neither the referee nor the weld disturbs a healthy case.

## 9. Tiny edges in the input — not a defect of ours

The sub-micron edges are present in the acquired rawdata. Our pipeline removes a
few of them rather than creating them, and they are not what breaks the remesh.
Recorded here only so it stops being re-investigated.

## 10. LAPACK is broken in both conda envs — environment, not pipeline

`np.linalg.det(np.eye(3))` **hard-crashes the interpreter** in `vmtk_env` and in
`hemomesh` — Windows exception `0xC06D007F`, a delay-load DLL failure, exit 127
with no Python traceback. `np.linalg.norm` is fine, so only the calls that reach
LAPACK die: `det`, `svd`, `solve`, `inv`, `eigh`. numpy is 2.2.6 with its own
bundled OpenBLAS; `KMP_DUPLICATE_LIB_OK=TRUE` does not help.

It is worth ruling in or out as a cause of the 109 run errors, and it is **not**
one: the whole `datatransform` tree has only three non-test LAPACK call sites —
`hemoMesh/src/transform/transformer.py` (two `inv`) and `label/unextend.py`
(one `eigh`) — and none of them is on the remeshing or removal path.

What it does break is **testing**: any fixture built with `pv.Sphere()` takes the
interpreter down with it, because `pv.Sphere` rotates via LAPACK. That kills
`tests/test_aneurysm_removal.py` and `template_creation/test_remeshing.py`
mid-run. The suites covering the work in this register avoid LAPACK and run
clean, but the two above cannot be run until the env is repaired.

It also explains a diagnostic dead end hit while investigating section 2: a probe
using `np.linalg.svd` for a best-fit plane died at exit 127 with no output, and
had to be rewritten around a Newell normal.

Repairing it means reinstalling numpy in both envs, which would disturb a working
pipeline, so it is left for a deliberate decision rather than done in passing.

## 11. Triangle fans in the shipped meshes — fixed

Ten of the 111 meshes the validation run shipped carry a vertex with a spray of
triangles on it instead of a patch of properly sized ones. Area drift does not
notice — a fan barely moves the area — so all ten passed.

**Counting the triangles at that vertex does not identify the defect**, which is
why this went unnoticed. Across the ten, valence says nothing useful: C0010
carries 61 at an edge-length CV of 0.172 and ANSYS_UNIGE_30_614 carries 94 at
0.190, both perfectly good meshes, while p129 carries 21 at a CV of 0.437 and
p363 22 at 0.540, both bad.

What separates them is **how far the fan reaches, in units of the mesh's own
edge**:

| | worst fan reach | valence | edge CV |
|---|---|---|---|
| SNF00000607_01_2 | 3.0 edges | 20 | 0.130 |
| USFD_0035 | 4.7 edges | 55 | 0.168 |
| ANSYS_UNIGE_30_614 | 5.8 edges | 94 | 0.190 |
| C0010 | 8.0 edges | 61 | 0.172 |
| p129 | 20.7 edges | 21 | 0.437 |
| p551 | 21.4 edges | 206 | 0.875 |
| p391 | 23.2 edges | 117 | 0.524 |
| p363 | 32.5 edges | 22 | 0.540 |
| SNF00000538_01 | 37.3 edges | 36 | 0.580 |
| p399 | 38.4 edges | 104 | 0.762 |

A ring two edges out is a crowded vertex and nothing more. A ring twenty or
forty edges out is a patch thrown across ground that should be carrying a
hundred triangles — exactly the uniformity this dataset exists to provide. The
split is clean and the gate sits at 12, with room on both sides.

These are made here, not inherited: p399's input has no such vertex at all and
its output has one whose fan covers 104.8 mm2 where a flat disc of the same ring
would cover 73.9, so the vertex is pulled off the surface, not capping a hole.
Five of p551's eight sit within 0.06 mm of each other, one of them spanning a
2.16 mm ring in 0.414 mm2 of area — slivers collapsed onto a line.

**They are not pinhole patches.** `fan_fill_small_loops` does build a fan over a
hole's barycentre, and that is legitimate, but it only touches loops under
`WALL_PINHOLE_RADIUS_MM` = 0.20 mm. Nothing at 20 to 40 mean edges came from
there.

**Fix**: the same weld sweep as section 7, with the reach as the second test
beside area drift. It does not park these cases, it repairs them — p399 goes
from 38.4 edges to no fan at all, p363 and SNF00000538_01 likewise, and p391
from 23.2 edges to 2.0.

## 12. A centerline that stopped early, and nothing noticed — fixed

`extract_centerlines_for_tube` traces on the flow-extended surface and falls
back to the bare vessel when that fails. The two are not different algorithms —
both call `extract_voronoi_centerlines`; the only difference is whether the
extensions are on the surface while the Voronoi tessellation runs. They are
clipped off again before anything ships.

**The fallback is not the weak one. The test that chose between them was.** It
asked `n_cells >= n_targets` and nothing else, and a trace can stop early while
still having one polyline per outlet. On SNF00000415 it did, and the short one
shipped:

| SNF00000415, both clipped to the anatomical planes | tracts | length |
|---|---|---|
| extended surface (what shipped) | 17 | 328.21 mm |
| bare vessel | 46 | 514.46 mm |

98.6% of the extended trace lies within 0.5 mm of the bare one, so it is not a
different path — it is a **subset**. The 367 points it never visits are all
inside the lumen, sitting 0.881 mm from the wall with an inscribed radius of
0.895 mm, which is what a centerline point in a thin branch looks like. VMTK had
said so in passing and nothing was listening: *"Can't find a steepest descent
edge. Target not reached."*

On a healthy vessel the two agree: C0010 gives 627.53 mm against 577.74 mm,
covering 99.5% and 98.7% of each other, median deviation 0.015 mm.

**Fix**: compute both and keep the one that covers more anatomy. Coverage is
counted in occupied 0.5 mm voxels after clipping to the anatomical planes,
because arc length is wrong twice over — the raw trace re-walks the shared trunk
once per outlet, and the extended one carries extensions that are not anatomy.
A chosen trace that still misses a branch now raises instead of warning; a
centerline missing a branch makes a tube missing that branch. The second trace
costs seconds against the minutes the extended one can take.

