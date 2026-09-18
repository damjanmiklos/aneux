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

## 2. Fan tents — fixed

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

**Fix** (`src/mesh/surface_mesh.py`): `_remesh_diagnosis` measures the remesh
against its input on both symptoms — `REMESH_MAX_AREA_DRIFT = 1.2` either way,
and `REMESH_MAX_VALENCE = 20` — and `_remesh_surface_vmtk` returns the **input
surface unchanged** when the remesh diverged. There is no better setting to fall
back to, as the sweep above shows; the input is the only sound surface available,
and it costs nothing here because it is an intermediate that `remeshing.py`
remeshes to a uniform target downstream. Shipping a lumen-spanning tent instead
would not be recoverable.

Verified on the exact surface that produced the tents: the guard fires
(`the remesh changed the surface area by 2.173x (2110.7 -> 4587.0 mm2)`) and the
returned surface is the input, intact — 232,580 points, 2110.7 mm2, max valence
18, zero hubs. Covered by `TestRemeshDiagnosis`.

## 3. A ball on an opening (p375_1) — fix in, verification running

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

End-to-end re-run of p375 keep 1 in progress.

## 4. A residual stub where an aneurysm was removed — detector in, cause open

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

**Cause: open.** The Voronoi subtraction is leaving part of the sac behind; which
step drops it has not been traced yet.

## 5. A ball left instead of the removed aneurysm (p439_2) — open

Reported by the user. Distinct from section 4: here the apex is **3.739 mm**
clear of the surface, so the section 4 detector does not fire — the top of the
sac *was* taken off and a rounded body was left below it. Within 6 mm of the old
apex the mesh carries a 1128-point, 35.9 mm2 patch whose centroid is 4.40 mm from
the apex.

Being diagnosed by a debug re-run of p439 keep 2.

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

## 7. Remesh divergence in the GT pipeline (p097, p379_1, p379_3) — open

The same divergence as section 2 but in `template_creation/remeshing.py`. p097 is
cleared of a parameter cause: an 8-setting sweep on its step-1 surface all give
0.999x area, so the remesher is innocent there and the **step-4b clip** is what
introduces the damage — it takes min edge from 0.005032 to 0.000037 mm and
aspect-ratio>50 triangles from 0 to 12.

## 8. Silent surplus openings — open

p489 ships 9 openings against 5 expected, p129 6 against 4. Nothing rejects a
mesh for having *too many* openings that the profile matcher never claimed; the
count guard only catches a change during removal.

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
