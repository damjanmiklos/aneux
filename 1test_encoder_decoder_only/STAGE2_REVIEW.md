# Stage 2 (latent → mesh texture) — architecture, data and training-framework review, expanded edition

State reviewed: HEAD `e7fd7ce` ("unextend"), i.e. `d0fdc1b` ("template integrated in") plus `cc62209` ("general remeshing algorithm") and `e7fd7ce`. Files: `train_pipeline/{config, cleaned_io, dataset, raycast, geometry, ops, model, losses, train, aneuxai, test_architecture}.py`, `postprocess.py`, `aneux_paths.py`, the four-folder `cleandata/` layout, and the generators in `datatransform/template_creation/` (`vessel_pipeline.py`, `centerline_creation.py`, `uniform_remeshing.py`, `variable_remeshing.py`, and the new `remeshing.py` + `test_remeshing.py`). No pipeline code was changed by the review; every script written for it lives in `scratch/`.

What changed between the previous review (`d0fdc1b`) and this one:

- `remeshing.py` is new: a ground-truth generator that remeshes the *original* vessel (aneurysm kept) at a constant 0.15 mm edge length, cuts the ostia perpendicular to the centerline, and gates the result on area. It is verified below on two real cases (§2.2). This closes item 1 of the previous P0 list.
- `vessel_pipeline.py`: `apply_taubin_smoothing` default `pass_band` changed from 1.0 to 0.1 (stronger smoothing, silently affecting `build_parent_tube` and `compute_centerline_from_mesh`, measured in §2.6); new `remove_spurious_openings`, extension-length-aware `_opening_clip_height`, `clip_flow_extensions_and_uncap` now also drops islands and fills leftover rims; `compute_centerline_from_mesh` extracted from `process_centerline_dataset`.
- Nothing in `train_pipeline/` changed. `clip_centerline_at_profiles` (cell arrays dropped) and `compute_raycast_stretch_distances` (orientation-dependent) are unchanged, so P0 items 2 and 3 of the previous list are still open, and everything the previous review measured about the training side still holds and was re-verified where a fresh file made that possible.

Decisions taken (Damján, 9 Sep) that this review builds on: training reads only `cleandata/`; `uniformly_remeshed/` is the *original vessel with the aneurysm*, finely remeshed, and is the GT for everything; templates come from `variable_remeshing.py` and are the decoder's identity surface; there is no `coarse_remeshed` — mid/coarse are decimations of the template; Stage 1 will emit the centerline at **1 mm** with a texture token every **2 mm**, along the *entire* tree; output must be a single watertight manifold with open outlets (CFD), outlet position/planarity fixed in post; HPC wall-time effectively unlimited, accuracy over speed; manual per-case work acceptable; aneurysm removal (hemoMesh) is a later topic and is not treated here.

How to read this document. Each finding is written in four parts: *what the code does* (with the function names and, where it matters, the exact lines), *what was measured* (the probe, the case, the number), *why it matters for the model that will be trained*, and *what to change*. Everything that is a number was measured on real files produced by the current generators. The main case is **SNF00000100** (an ICA segment, 3 openings, one lateral aneurysm of roughly 25 mm outward extent, 22 076 original vertices, 1 320.6 mm² of wall); **C0002** (7 openings, one of which leaves the sac) is used where a second case is informative. All artifacts are under `scratch/uniform_probe/` (see Appendix A for the file list and the script that produced each number).

---

## 0. Executive summary (ranked by impact)

1. **The GT generator now exists and works.** `remeshing.py::process_gt_remesh_dataset` was run on SNF00000100 (§2.2). The output has 96 918 vertices / 193 575 triangles at a median edge of 0.126 mm (target 0.15), one component, three boundary loops, no non-manifold edges, 99.2 % of the original area. Surface-to-surface distance original → output is 0.042 mm median, 0.100 mm at p99, and **only 0.035 % of the original wall is more than 1 mm away** (the parent-tube output of `uniform_remeshing.py` had 9.2 % — that was the missing sac). The sac is in the GT. The remaining points are practical: generation takes **999 s per case single-threaded** at 0.15 mm (≈ 4 days for 682 cases with the default two workers; hours with 12–14), the GT is 12× denser than the template and 6× denser than the current `N_TRUE = 16 384` sampling budget, and the GT's ostia sit **0.15 / 0.15 / 0.58 mm** from the template's ostia with radii 0.06–0.22 mm larger, which matters for the boundary rows of the decoder (§2.2.4).
2. **The branch arrays are still lost, so the whole template path is parametrised by two coincident tracts.** `clip_centerline_at_profiles` still rebuilds the centerline copying only point arrays (verified again on a centerline regenerated with HEAD: cell arrays `[]`), so `extract_groupid_tracts` still falls back to `extract_unique_tracts`. On the SNF00000100 scaffold that means two tracts of 95.6 / 94.9 mm that are both the inlet-to-one-outlet path, zero junctions, 48 + 48 tokens on the same parent, template vertices split 3 961 / 3 948 at random between the two copies, 76–77 % of the kNN upsampling neighbours and a comparable share of mesh edges classified as "cross-tract" and down-weighted (§2.4, §3.1). Nothing about the tree — branch identity, junction position, per-branch tokens, the tract mask in the decoder's cross-attention — means what it is supposed to mean until this is fixed. The fix is a small generator change plus the validated tract rule; both are described in full in §2.4.
3. **`variable_remeshing.py` still produces an undensified template for inward-wound originals.** `compute_raycast_stretch_distances` accepts a ray hit only if the GT cell normal agrees with the ray direction (`dot > 0.2`) and takes that normal from the file's *stored winding* (`AutoOrientNormalsOff`). AneuX `.vtp` files from SNF/UPF/USFD/ANSYS are wound inward (11 of 11 sampled), so on SNF00000100 the ray-cast returns essentially nothing (1 non-zero of 17 303, `Max k = 1.35` in the regenerated log where the true value under the sac would be ≈ 10) and the "variable" template is a uniform 0.41 mm mesh identical in density to the uniform remesh. The one-line fix and the reason `StretchDistance` matters for training (it is the correct `r*`) are in §2.3.
4. **The template integration is structurally right and pays off — 0.79 s and 0.74 GiB per training step (was 4.28 s / 2.7 GiB) — but four of the supervision/parametrisation pieces attached to it are wrong or inert** (§3, §7):
   - `_template_r_star` (nearest GT *vertex*, projected on the template normal) says "stay put" under the sac: target offset 0.03 / 0.24 / 0.83 mm (p50 / p90 / max over the 304 sac vertices) where the outward ray reaches the GT at 4.1 / 11.8 / 24.9 mm. With `λ_rad = 1` and `valid = True` on all 7 909 vertices, the radial Huber pulls the sac region back onto the template while the Chamfer pulls it out.
   - `dth = du = 0`, `ring_med = r*`, `ambiguous = False` → `smoothness_edge_weights ≡ 1.0` on every edge (measured min = mean = 1.0): the crease-aware weighting of the Dirichlet and Laplacian terms is switched off without anyone having decided that.
   - `TUBE_RADIUS_MM = 2.0` still stands in for the local radius in `composed_radius`, the Chamfer weights and the head's floor, although `r_local` (0.87–2.66 mm on this vessel) is now cached at all three levels and used nowhere.
   - `template_centerline` is measurably the same curve as `original_centerline` (Hausdorff 0.12 mm, MISR within 0.06 mm); the training code reads the original and then ignores it in favour of the template's.
   Measured and good: the fine level *is* the posed template (identity displacement 0.0 at init); decimation kept 3 boundary loops, 1 component, 0 non-manifold edges at every level (7 909 / 1 980 / 638 vertices at 0.41 / 0.89 / 1.56 mm); kNN upsampling has no opposite-wall mixing (0.0 % / 0.04 %); template normals point outward on 99.8 % of vertices (by luck of winding, not by construction).
5. **The Taubin default change is benign but should be known.** Windowed-sinc smoothing at pass band 0.1 × 15 iterations moves the SNF00000100 wall by 0.041 mm median / 0.178 mm max versus 0.032 / 0.092 mm at the old pass band 1.0 (§2.6). The regenerated centerline differs from the old one by ≤ 0.14 mm and the MISR by ≤ 0.06 mm; the regenerated template by ≤ 0.29 mm (mostly remesh noise). Because the GT uses an even lighter setting (1.5 × 5 iterations), the "should the GT be smoothed like the template" question of the previous review (§15.1) is answered by measurement: the two smoothing levels differ by less than one GT edge length.
6. **Latent, decoder kernel and training hygiene findings carry over unchanged** (§4–§8): 96 × 128 = 12 288-dim near-autoencoder latent with a KL weight that contributes 0.3 % of the objective; θ-blind SplineConv pseudo-coordinates on the template (ring neighbours land at `e_th = 0.5 ± 0.035`); `aggr="add"` without normalisation; no augmentation, no resume, `print` logging, EMA horizon 47 epochs. With the 2 mm token contract now fixed, `LATENT_LEN = 96` *fixed* slots contradicts it — on this 95 mm case the spacing is 2.02–2.04 mm by coincidence, on a 50 mm vessel it would be 1 mm.

Good and worth keeping: the centerline-intrinsic design, the template as identity surface, kNN inter-level tables with `__inc__`, the decoupled radial/shear head with identity init, multi-scale point-to-plane Chamfer, cache/warm-up infrastructure and the rawdata guard, FP32/TF32, the smoke-test culture (55/55 pass), and now a GT generator with hard quality gates.

---

## 1. What the pipeline does, explained

### 1.1 The idea in one paragraph

A vessel with an aneurysm is modelled as *a healthy tube plus a residual*. The healthy tube is the **template**: a surface built around the vessel's centerline using the maximum inscribed sphere radius (MISR) at every centerline point, so it has the right calibre everywhere but no sac. The **residual** is what the decoder learns: a displacement of every template vertex, mostly along the template normal (radial, `Δr`) and a little within the tangent plane (shear, `Δs`), that moves the template onto the ground-truth wall — including the sac. The displacement field is not predicted from raw 3-D coordinates; each template vertex is described by where it sits on the centerline tree (arc-length fraction `u` along its branch, angle `θ` around the branch, branch id) and it reads the shape information from a **latent tree**: a sequence of tokens placed along the centerline at fixed spacing. The encoder produces those tokens from the GT surface; Stage 1 will later produce them from a condition vector, together with the centerline itself. Because the template's triangles are never re-connected — only moved — the output has the template's topology: one component, one boundary loop per ostium, watertight otherwise, which is what CFD needs.

### 1.2 The data flow as built (`e7fd7ce`)

```
rawdata/.../vessels/original/{id}.vtp                       (read-only)
   ├─ remeshing.py            → cleandata/uniformly_remeshed/{id}.vtp   GT: original wall, 0.15 mm remesh, sac kept (§2.2)
   ├─ centerline_creation.py  → cleandata/original_centerline/{id}.vtp  Voronoi centerline + MISR, polylines per path (§2.4)
   ├─ variable_remeshing.py   → cleandata/template_mesh/{id}.vtp        MISR polyball tube, adaptively remeshed (§2.3)
   └─ centerline_creation.py on the template
                              → cleandata/template_centerline/{id}.vtp  (≈ original_centerline, §2.5)

train_pipeline/dataset.py  (cache build, one .pt per case, CACHE_VERSION = 9)
   _build_data → build_scaffold(template_cl, vessel_mesh=GT, template_mesh=tpl)
     → extract_groupid_tracts   → falls back to extract_unique_tracts (no cell arrays on real files)
     → _choose_inlet / _orient_tracts → canonical pose (inlet tangent → +Z)
     → _fit_dense_tract: cubic spline, 0.2 mm samples, Bishop (parallel-transport) frames
     → fine   = posed template_mesh: faces, edges; per vertex u, θ, tract, r_local, u_step by nearest dense-CL sample
       mid    = pyvista decimate(keep 25 %, ≥ 256 vertices), same projection
       coarse = pyvista decimate(keep  8 %, ≥ 64 vertices),  same projection
     → knn_upsample_tables (k = 3, inverse distance) coarse → mid → fine
     → r* = R + n · (nearest GT vertex − x_tpl), valid everywhere, dth = du = 0          (§7.1)
     → x_true: 16 384 GT vertices (75 % FPS over all, 25 % FPS over far-from-CL) + GT normals
     → 96 latent token slots (per tract ∝ arc length, + junction tokens), token_attend mask

model.py
   Encoder  PointNeXt: stem(xyz) → SA 16384→1024→256→64→64 (radius 1.5/3/6/12 mm), 2 InvRes blocks per stage
            → CenterlineLatentHead: 96 positional queries (γ(u), tract emb) → μ, logσ² ∈ R^128 per token
   Latent   reparameterise → LatentTractSelfAttention (window ±2 tokens, ALiBi, gated residual)
   Decoder  per level: LatentCrossAttention (query γ(u), γ(θ); keys z + γ(u_token); tract mask)
            + coarse-only CoarsePositionalSelfAttention
            + 4 × ResidualSplineConv(128, kernel 5³, degree 2, aggr add, no norm)
            + DecoupledDisplacementHead: Δr = softplus(·) − R_MARGIN, Δs = 3 mm · tanh(·), in the vertex frame
            coarse → kNN upsample → mid → kNN upsample → fine (template vertices)

losses.py  recon = Σ_levels w_l · weighted Chamfer (point-to-plane Huber δ = 1 mm + 0.2 L2; weights 1 + d_cl/R ≤ 4)
           + λ_kl · KL (annealed 20 epochs) + 0.15 · Dirichlet(Δr, Δs) + 0.05 · Laplacian + 0.02 · normal consistency
           + 1.0 · radial Huber(r_pred vs r*) on valid nodes (+ 0.5 × at mid)

train.py   AdamW 2e-4, wd 1e-4, cosine over 200 epochs, bs 1 × accum 8, clip 1.0, EMA 0.999,
           validation every 5 epochs on EMA weights, random 85/15 split, best.pt by val recon + rad
```

The legacy Bishop-tube path (`build_scaffold` without `template_mesh`, `build_scaffold_from_centerline`) still exists as a fallback and for the synthetic tests; with `require_templates=True` (the default) it is never used on `cleandata`.

### 1.3 Glossary of the objects that appear everywhere below

| term | meaning here | where it comes from |
|---|---|---|
| **GT** (ground truth) | the vessel wall the model must reproduce, aneurysm included | `cleandata/uniformly_remeshed/{id}.vtp`, produced by `remeshing.py` |
| **template** | the healthy-tube surface the decoder starts from (its "identity"); its vertices are the decoder's fine-level nodes | `cleandata/template_mesh/{id}.vtp`, produced by `variable_remeshing.py` |
| **polyball** | the implicit surface "union of spheres centred on the centerline with radius = MISR"; the template is its marching-cubes isosurface | `generate_base_surface` |
| **MISR** | maximum inscribed sphere radius at a centerline point = local lumen radius | VMTK Voronoi centerline (`MaximumInscribedSphereRadius` point array) |
| **centerline / tracts** | the medial curve of the lumen; a *tract* is one branch of it between two nodes (inlet, junction, outlet) | `original_centerline`, `extract_groupid_tracts` |
| **dense CL** | each tract resampled at 0.2 mm on a cubic spline, with a Bishop frame (t, n, b) at every sample | `_fit_dense_tract` |
| **Bishop frame** | a moving frame along a curve whose normal does not twist about the tangent (parallel transport); gives a stable `θ = 0` direction | `_compute_parallel_transport_frames` |
| **(u, θ)** | intrinsic coordinates of a surface vertex: `u` ∈ [0, 1] arc-length fraction along its tract, `θ` ∈ (−π, π] angle of the vertex around the centerline in the Bishop frame | `_project_points_to_tracts` (nearest dense-CL sample) |
| **r_local** | distance from a template vertex to its nearest dense-CL sample = the local healthy radius at that vertex | `_project_points_to_tracts`, cached as `r_local[_mid/_coarse]` |
| **r\*** | the radial *target*: how far from the centerline the wall should be at a template vertex, expressed as `R + n·(x_GT − x_tpl)` | `_template_r_star` → `nearest_normal_offset_r_star` |
| **x_true** | the point cloud the encoder sees and the Chamfer loss compares against: 16 384 GT vertices | `_hybrid_true_points` |
| **tokens** | the latent: 96 slots, each a 128-vector, each attached to a position `u` on a tract (or to a junction) | `_build_latent_tokens`, `CenterlineLatentHead` |
| **levels** | coarse (638 vertices), mid (1 980), fine (7 909) meshes of the same template; the decoder predicts a displacement at each and upsamples | `_build_template_scaffold` |
| **Δr, Δs** | per-vertex radial displacement along the vertex normal and 2-D shear in the tangent plane | `DecoupledDisplacementHead`, `decoupled_displacement` |
| **stretch / `StretchDistance`** | in `variable_remeshing.py`: the outward ray distance from a template vertex to the GT; large under a sac; drives densification | `compute_raycast_stretch_distances` |
| **GroupIds / Blanking / TractIds / CenterlineIds** | VMTK's branch labelling of a centerline: which branch each segment belongs to, and whether it is inside the bifurcation "blanked" zone | `vmtkBranchExtractor` (cell data) |
| **profiles / ostia** | the open ends of the vessel (inlet and outlets); a profile is one boundary loop with barycentre, normal and radius | `measure_open_profiles` |
| **flow extensions** | straight tubes VMTK adds at every ostium so the Voronoi centerline reaches the actual opening; removed again before saving | `add_flow_extensions`, `clip_flow_extensions_and_uncap` |
| **pipe-section cut** | the way ostia are opened: a finite cylinder along the outward centerline tangent, giving a planar rim perpendicular to the centerline | `clip_one_opening_pipe_section` |

---

## 2. Data generation: the `cleandata` contract versus what the generators produce

### 2.1 Folder by folder

| folder | intended content | what the current generator produces (HEAD) | consumer in training code |
|---|---|---|---|
| `uniformly_remeshed` | original vessel **with aneurysm**, fine isotropic remesh (GT) | **now correct**: `remeshing.py` → 0.15 mm remesh of the original, sac kept, ostia cut perpendicular to the centerline (§2.2). `uniform_remeshing.py` still exists and still writes the *parent tube*; it must not be pointed at this folder | `vessel_file`: `x_true`, GT normals, `r*` |
| `template_mesh` | parent-tube template from `variable_remeshing.py` | parent tube, "adaptively" remeshed — **adaptivity inactive on inward-wound inputs** (§2.3) | fine identity surface; mid/coarse by decimation |
| `original_centerline` | `centerline_creation.py` on the original | polylines, one per inlet→outlet path; MISR kept; **branch cell arrays dropped** (§2.4) | read, then **unused** when a template exists |
| `template_centerline` | `centerline_creation.py` on the template | the same curve as `original_centerline` to within 0.12 mm (§2.5) | tracts, canonical pose, (u, θ), tokens |

### 2.2 The GT generator `remeshing.py` — verified

#### 2.2.1 What it does, step by step

`process_gt_remesh_dataset(dataset_id, v_file, output_dir, target_edge_length=0.15, extension_length=5.0, sample_spacing=0.1)`:

1. **`prepare_gt_surface(original)`** — `clean_triangulate` (merge duplicate points, triangulate), `drop_degenerate_triangles` (edges < 1e-5 mm), `repair_nonmanifold_triangles`. It deliberately does **not** call `sanitize_vessel_for_vmtk`, which would decimate dense originals to ~20 000 points; the GT keeps the original tessellation as the starting point for the remesh. On SNF00000100 the working surface has 22 076 points, as the original.
2. **`measure_open_profiles(gt_surface)`** — finds the boundary loops of the *original* and fits barycentre / normal / radius to each: 3 profiles, radii 2.739 / 1.192 / 0.999 mm.
3. **`_gt_centerline`** — builds a *separate* working copy (`sanitize_vessel_for_vmtk` → `apply_taubin_smoothing()` at the new strong default), adds flow extensions, runs the Voronoi centerline (`extract_centerlines_for_tube`), resamples at 0.1 mm, smooths it (`smooth_centerline_preserve_misr`), runs `extract_branches`. This centerline is used for exactly one thing: the **frames** (origin, outward tangent, radius) at which the ostia are cut. The working copy is discarded. Note that this centerline is *not* the one saved in `original_centerline` — that one is computed by `centerline_creation.py` on a non-sanitised but smoothed copy — so the two agree only to the extent the two smoothing/sanitisation paths agree (they do, to ≈ 0.1 mm here).
4. **Flow extensions on the detailed original**, then **`clip_flow_extensions_and_uncap(extended_gt, gt_profiles, extension_length, centerline=branched)`** — for every profile a pipe-section cut (finite cylinder of radius `1.5·r` and height `max(7r, extension_length + 2r)` along the outward centerline tangent) removes the extension *and* the original, usually ragged, rim, leaving a planar opening perpendicular to the centerline. The extension-length-aware height is new in `cc62209`: at 7·r a 5 mm extension on a 0.5 mm outlet was longer than the cutter and survived as a fragment. Afterwards `drop_tiny_islands` and the new `remove_spurious_openings` (cap every loop, re-open only the loops that match a profile within `max(4r, 3 mm)`) clean up leftovers. On SNF00000100: 3/3 clipped, openings r = 2.564 / 0.990 / 1.187 mm, planarity `axial_std = 0.000 mm` for all three.
5. **Light Taubin** — `apply_taubin_smoothing(opened_gt, pass_band=1.5, n_iter=5, boundary_smoothing=False)`. Pass band 1.5 is the weak end of `vtkWindowedSincPolyDataFilter`'s [0, 2] range (2 = no smoothing); five iterations remove only the highest-frequency segmentation staircase. Boundary smoothing is off so the freshly cut rims stay planar.
6. **Isotropic remesh** — `remesh_surface_isotropically(opened_gt, target_edge_length=effective_edge, n_iter=20, connectivity_iter=20)` with `ElementSizeMode = "edgelength"`, `PreserveBoundaryEdges = 1`, `MinEdgeLength = 0.01`. `effective_edge = uniform_edge_length_for_profiles(profiles, 0.15)` = `min(0.15, max(0.15, 0.55·r_min))`, which is **always 0.15** — the profile-aware rule that matters at 0.5 mm is a no-op at 0.15 mm (harmless, but the log line "R_min=…" suggests an adaptation that does not happen).
7. **`finalize_surface`** (clean, keep the largest region) → **`assert_gt_remesh_scale`**: area ratio to the original must be in [0.88, 1.20] — below 0.88 means "aneurysm or a branch lost" (the parent tube would score 0.84–0.86), above 1.20 means "extensions not clipped". **`assert_template_quality`**: non-manifold edges, minimum edge, triangle quality, number of openings = number of profiles. Then `save_polydata`.

The script sets `OMP/MKL/OPENBLAS/VTK_NUMBER_OF_THREADS = 1` and defaults to two worker processes (`add_shared_cli_args(default_workers=2)`), input `VESSELS_ORIGINAL`, output `CLEANDATA_UNIFORM`. It never writes under `rawdata/`.

#### 2.2.2 Measurements (`scratch/probe_gt_remesh.py`, 400 000 area-weighted surface samples per side)

SNF00000100, `target_edge_length = 0.15`:

| quantity | value |
|---|---|
| original | 22 076 vertices, 1 320.6 mm² |
| GT output | **96 918 vertices, 193 575 triangles**, 1 309.7 mm² (ratio **0.992**) |
| GT edge length p05 / p50 / p95 | 0.101 / **0.126** / 0.147 mm (VMTK lands ~0.85× the target) |
| topology | 3 boundary loops, 1 component, 0 non-manifold edges; `min_edge = 0.041 mm`, median triangle quality 0.995, slivers 0.00 % |
| ostia | radii 0.989 / 1.189 / 2.567 mm; planarity std 8e-7 / 9e-7 / 2e-6 mm (exactly planar) |
| original → GT distance p50 / p95 / p99 / max | **0.042 / 0.084 / 0.100 / 1.21 mm** |
| original area farther than 0.25 / 0.5 / 1 mm from GT | 0.35 % / 0.22 % / **0.035 %** |
| GT → original distance p50 / p95 / p99 / max | 0.042 / 0.084 / 0.099 / 1.07 mm |
| GT area farther than 1 mm from the original | 0.004 % |
| generation time | **999 s** (Voronoi ≈ 15 s, 20 remesh iterations ≈ 25 s each, "Final mesh improvement" ≈ 7 min), 365 MB RSS |

Read the two distance rows together with the parent-tube numbers from the previous review (original → output p95 3.41 mm, max 7.49 mm, 9.2 % of the wall > 1 mm away): the sac that was missing is now present, and the residual 0.04 mm is the light Taubin plus the remesh projecting new vertices onto the smoothed surface. The 1.2 mm maximum is at the ostia, where the pipe-section cut has replaced the original rim by a plane perpendicular to the centerline (the GT inlet normal is [0.34, 0.01, 0.94] where the original rim normal was [0, 0, −1] — a ~20° tilt — and its radius 2.567 vs 2.739 mm), which is intended.

GT versus the existing template (same case, template from `scratch/uniform_probe/template_mesh/`):

| quantity | value |
|---|---|
| template → GT distance p50 / p95 / p99 / max | 0.083 / 0.353 / 0.632 / 2.18 mm |
| template area farther than 0.25 / 0.5 / 1 mm from GT | 11.0 % / 2.1 % / 0.48 % |
| GT → template distance p50 / p95 / p99 / max | 0.103 / **3.41 / 5.95 / 7.45 mm** |
| GT area farther than 0.25 / 0.5 / 1 mm from the template | 20.6 % / 11.2 % / **9.07 %** |
| ostium centre offsets (GT vs template) | 0.152 / 0.148 / **0.576 mm** (inlet) |
| ostium radii GT vs template | 0.989 vs 0.933; 1.189 vs 1.132; 2.567 vs 2.348 mm |
| rim-to-rim distance p50 / max | 0.12 / 0.22; 0.14 / 0.25; **0.31 / 0.70 mm** (inlet) |

This is the residual the decoder must learn: 9.07 % of the GT wall (the sac) is more than 1 mm away from the identity surface, up to 7.45 mm along the shortest path and up to 24.9 mm along the template normal (§7.1); on the healthy 90 % the template is within 0.1 mm of the GT at the median but with a tail — 11 % of the template area is more than 0.25 mm off, 2 % more than 0.5 mm, where the MISR tube is rounder than the true cross-section or where the Voronoi centerline is pulled toward the sac and the polyball bulges at the neck.

C0002, same generator, same 0.15 mm target (this is the case on which `uniform_remeshing.py` also dropped a side branch: 6 of 7 openings, 9.9 % of the original wall > 1 mm from the parent-tube output, max 14.0 mm, area ratio 0.84):

| quantity | value |
|---|---|
| original | 40 241 vertices, 1 646.7 mm² |
| GT output | **121 426 vertices, 242 464 triangles**, 1 631.6 mm² (ratio **0.991**) |
| GT edge length p05 / p50 / p95 | 0.101 / **0.125** / 0.146 mm |
| topology | **7 boundary loops** (all 7 anatomical profiles clipped), 1 component, 0 non-manifold edges; `min_edge = 0.031 mm`, median triangle quality 0.996, slivers 0.01 % |
| original → GT distance p50 / p95 / p99 / max | **0.042 / 0.076 / 0.093 / 1.22 mm** |
| original area farther than 0.25 / 0.5 / 1 mm from GT | 0.28 % / 0.15 % / **0.025 %** |
| GT → original p50 / p99 / max | 0.042 / 0.091 / 1.05 mm |
| generation time | **570 s** |

The small branch that leaves the sac (Profile 3, r = 0.58 mm, the one `build_parent_tube` skipped with "no nearby tube wall (closest 13.78 mm)") is in the GT. Together with SNF00000100 this is the signature of a remesh-of-the-original: the two distance directions are nearly symmetric, the area ratio is 0.99, and the fraction of the original wall more than 1 mm from the output is 0.03 %, not 9–10 %.

#### 2.2.3 What the GT density means for training

The GT is now **12.3× denser than the template** (96 918 vs 7 909 vertices) and **5.9× denser than `N_TRUE = 16 384`**. Consequences, each with a recommendation:

- **`x_true` sampling.** `_hybrid_true_points` takes 12 288 points by farthest-point sampling over all GT *vertices* and 4 096 by FPS over the vertices farther than `TUBE_RADIUS_MM + 1 mm` from the centerline, once, at cache time. At 97 k vertices the FPS sets are well separated (fewer duplicates than the 3 099 of 16 384 seen on the 22 k-vertex stand-in), but the encoder still sees a fixed 17 % subsample per epoch. Caching the full GT (97 k × (xyz + normal) × 4 B ≈ 2.3 MB per case, 1.6 GB for 682 cases) and drawing a fresh `x_true` every epoch is cheap and is the single largest source of free augmentation on ~200–680 samples (§8, §9).
- **Chamfer resolution floor.** With 16 384 GT samples on 1 310 mm² the mean sample spacing is ≈ 0.28 mm; point-to-plane distances below that are dominated by sampling noise. With 97 k it is 0.12 mm. The reconstruction target should therefore be measured against the full GT (or a per-epoch resample of ≥ 32 k), otherwise "0.1 mm accuracy" is not observable in the loss.
- **`r*`.** The ray-cast `r*` proposed in §7.1 needs the GT as a *surface* (cell locator), which it is; the current nearest-*vertex* `r*` gets slightly better with denser vertices but remains wrong under the sac for the geometric reason given there.
- **Encoder cost.** PointNeXt's first set-abstraction does FPS 16 384 → 1 024 on the GPU; if `x_true` grows to 32–64 k the stem and first stage scale linearly (the 64 ms forward becomes ~100–150 ms) — negligible next to the 0.79 s decoder step.
- **Cache size.** Each `.pt` currently stores x_true (16 384 × 3) plus normals; storing the full GT increases the cache by ~2 MB per case. Fine.
- **Disk.** GT `.vtp` files are 5.1 MB each (SNF00000100) → ≈ 3.5 GB for 682 cases. Fine.

#### 2.2.4 Remaining concerns and recommendations for `remeshing.py`

1. **Generation cost.** 999 s single-threaded per case. With the default `--workers 2` the 682 originals take ≈ 95 h; the 5950X has 16 cores and each process uses 365 MB, so 12–14 workers bring it to ≈ 12–14 h (I/O and memory permit it). On the HPC login/compute node it is embarrassingly parallel. Two cheaper knobs if the time matters: `GT_REMESH_N_ITER = 20` gives median triangle quality 0.995 and 0.00 % slivers — 10 iterations (VMTK's default) would very likely still pass `assert_template_quality` and save ~4 min/case; and the intermediate remesher output has 486 026 points before `finalize_surface` cleans it to 96 918, so peak memory scales with 5× the final count (fine at 0.15 mm, worth knowing if anyone tries 0.1 mm).
2. **Ostium mismatch between GT and template.** The GT is cut on frames from *its own* centerline (§2.2.1 step 3), the template on frames from `build_parent_tube`'s centerline; both are pipe-section cuts perpendicular to the local tangent, but the centres differ by 0.15 / 0.15 / 0.58 mm and the GT radii are 0.06–0.22 mm larger because the polyball's MISR radius is the *inscribed* radius, smaller than the true rim of a non-circular cross-section. For the decoder this means the boundary rows of the template are 0.1–0.7 mm inside the GT rim and have to move outward and along the tangent to match — the Chamfer will ask for that, and nothing currently keeps boundary vertices in their cut plane. Either (a) accept it and add the boundary-plane constraint of §11 (rim vertices constrained to slide in the cut plane), or (b) generate GT and template in one process from one centerline and one set of frames (§2.9), which removes the centre offset; the radius difference remains and is exactly the residual the decoder is for.
3. **`effective_edge` is constant.** See step 6 — cosmetic, but the log line implies profile-dependence that does not exist at 0.15 mm.
4. **Area gate range.** [0.88, 1.20] catches the two known failure modes. A GT that loses a *small* side branch (C0002's Profile 3 is 0.58 mm radius; its ~10 mm of branch is ≈ 36 mm² of 1 647 mm², 2 %) passes the gate. The `n_openings == n_profiles` check in `assert_template_quality` is the one that catches it — keep it a hard error (it is), and make the `n_clipped < n_in` warning in the uncap step a hard error too, since a GT with a ragged un-cut rim has a non-planar outlet that CFD will reject.
5. **Two centerline computations per case, three per case across the generators.** `remeshing.py` computes a Voronoi centerline (≈ 15 s) that it throws away; `centerline_creation.py` computes another one on a slightly different surface; `variable_remeshing.py` a third. §2.9 argues for one process per case sharing one centerline.
6. **Smoothing consistency with the template** — answered by measurement in §2.6: irrelevant at these settings.
7. **`test_remeshing.py`** requires `pytest`, which is not installed in `vmtk_env`; running it through a pytest-free shim (`scratch/run_test_remeshing_nopytest.py`) gives **14/14 pass**. The tests cover the scale gates, constant edge length, planar rims after pipe-section cuts, CLI defaults, extension-stub removal and `remove_spurious_openings` on synthetic spheres/tubes; they do not (and cannot cheaply) cover the sac-preservation property — the probe above is the test for that, and it should become a fixture-based regression test on one small real case once `cleandata` is populated.

### 2.3 `variable_remeshing.py`: how the template is built, and why it is not densified under the sac

#### 2.3.1 The parent tube

`process_variable_dataset` → `build_parent_tube(vessel_mesh)`:

1. `sanitize_vessel_for_vmtk` (decimate to ~20 000 points if denser, drop degenerates) → `apply_taubin_smoothing()` (now pass band 0.1, 15 iterations, §2.6).
2. `measure_open_profiles` on the smoothed surface; `add_flow_extensions` (5 mm); `measure_open_profiles` again on the extended ends (the seeds for the centerline).
3. `extract_centerlines_for_tube` → VMTK Voronoi centerline from the inlet seed to every outlet seed, with MISR per point; `resample_centerline` (0.1 mm); `smooth_centerline_preserve_misr`; `extract_branches` (`vmtkBranchExtractor`: GroupIds / Blanking / CenterlineIds / TractIds as **cell** data).
4. `extra_opening_spheres`: constant-radius spheres continuing 2·r past each ostium so the tube does not taper at the cut.
5. **`generate_base_surface`**: `stamp_polyball_image(points, radii)` writes the implicit function max over spheres onto a voxel grid (0.27 mm spacing here, 220 × 199 × 250 voxels, at least 4.5 voxels across the smallest diameter), `vmtkMarchingCubes` extracts the isosurface, and the result is decimated to ≈ 20 000 points. **The input vessel contributes nothing here except its bounds, its profiles and its centerline.** This is why `uniform_remeshing.py`, which runs the same function, cannot output the aneurysm — a fact confirmed by measurement in the previous review (9.2 % / 9.9 % of the original wall > 1 mm from the output on SNF00000100 / C0002).
6. `clip_flow_extensions_and_uncap` with pipe-section cuts → `open_base_surface` (17 256 points on SNF00000100), openings r = 2.361 / 0.932 / 1.132 mm.

Then, specific to the variable path:

7. `compute_template_local_radii`: `R_template` per vertex by projecting onto the centerline *polyline* and interpolating the MISR (floor 0.30 mm).
8. **`compute_raycast_stretch_distances(open_base_surface, vessel_mesh, r_template)`**: for every template vertex, cast a ray along the outward normal up to 25 mm and record the distance to the first GT hit (the "stretch") — this is the field that says "the wall is far away here, put more triangles here".
9. `build_target_edge_array`: `k = 1 + stretch / R_template`, `target_edge = min(0.5, 0.55·R) / k^1.5` (floor 0.01 mm) → per-vertex `TargetEdgeLength`.
10. `remesh_surface_adaptively` (`ElementSizeMode = "edgelengtharray"`) → `finalize_surface` → scale and quality gates → save.

#### 2.3.2 The orientation dependence, line by line

```python
# vessel_pipeline.py 1743–1824 (unchanged since d0fdc1b)
template_normals_filter.AutoOrientNormalsOff()          # template normals as wound (inward on the MC surface here)
...
gt_normals_filter.ConsistencyOn(); gt_normals_filter.SplittingOff()   # no AutoOrient → GT cell normals as wound
...
outward = -template_normals / lens                      # "VTK normals are inward here" (docstring)
...
hit = locator.IntersectWithLine(p0, p_end, tol, t, x, pcoords, sub_id, cell_id)
if hit:
    d = ...
    if 0.10 < d <= (3.5 * r_local):
        cid = int(cell_id.get())
        if gt_cell_normals is not None and 0 <= cid < len(gt_cell_normals):
            if float(np.dot(n, gt_cell_normals[cid])) > 0.2:      # ← accept only if GT normal agrees with the ray
                distances[i] = d
```

Three facts combine:

- The GT normal used in the test is whatever orientation the *file's triangle winding* implies. `ConsistencyOn` makes neighbouring triangles agree with each other but does not decide inside/outside; `AutoOrientNormalsOn` would (for closed surfaces, and approximately for open tubes).
- AneuX originals are not consistently wound across sources. Measured with `scan_gt_winding.py` (the as-wound normal disagrees with the auto-oriented one): `.vtp` from SNF / UPF / USFD / ANSYS: **11 of 11 sampled inward**; C0xxx `.vtp`: 0 of 4 inward; p\* `.stl`: 1 of 15 inward.
- The test `dot(n, gt_normal) > 0.2` therefore *rejects* every legitimate outward hit on an inward-wound file (the dot product is ≈ −1).

Measured (`probe_stretch_orientation.py`, `probe_template_rstar.py`):

| | non-zero stretch | max |
|---|---|---|
| pipeline call as-is on `open_base_surface` vs SNF00000100 | **1 of 17 303** | 0.56 mm |
| same call, GT winding flipped | 5 862 | 6.46 mm |
| true outward ray from the *final* template under the sac (304 vertices) | 304 | 24.9 mm (median 4.1) |
| pipeline stretch re-evaluated on the final template | **0 of 7 909** | 0 |

The freshly regenerated log (HEAD, `scratch/uniform_probe/v2/`) shows the same symptom: `Stretch Factor k metrics: Max k=1.35, Healthy Vessel k=1.00, Stretched Zone k=1.35` — one or two vertices barely above the 0.3 mm threshold. With the sac 4–25 mm out and `R ≈ 2.7 mm`, the correct `k` under the sac is 2.5–10 and the target edge there `0.5 / k^1.5` = 0.13–0.02 mm. Instead the template is uniform: 7 909 vertices at a 0.41 mm median edge, against 7 920 for the uniform remesh of the same tube.

Why it matters beyond density: `StretchDistance` — the same quantity, exported per template vertex — is the correct radial target `r*` for the loss (§7.1). Today it would export zeros.

Fix (any one of): test `abs(dot) > 0.2`; or `gt_normals_filter.AutoOrientNormalsOn()` (98.5 % outward on SNF00000100); or, most robust for open tubes, orient both template and GT normals by the sign of `n · (x − cl_nearest)` (the radial direction from the nearest centerline point), which needs no closed-surface assumption. Then regenerate `template_mesh`. Two further notes from the same probe: the *saved* template's normals come out **outward** (99.9 %) after remeshing, opposite to `open_base_surface` — no consumer may assume a fixed convention; and `n_clipped < n_profiles` is only a warning in `build_parent_tube` (C0002 lost its Profile 3 branch: "no nearby tube wall (closest 13.78 mm)") — the GT now keeps that branch, so the template would be *missing an opening that the GT has*, which the training code has no check for. Make it a hard gate.

### 2.4 Centerlines: the lost branch arrays and what they do to the scaffold

#### 2.4.1 The mechanism

`vmtkBranchExtractor` (called in `extract_branches`) labels the centerline: `GroupIds` (which anatomical branch a segment belongs to — the parent has one id, each daughter another, and the *bifurcation zone* its own blanked group), `Blanking` (1 inside the bifurcation zone), `CenterlineIds` (which inlet→outlet path), `TractIds`. In VMTK these are **cell** arrays: the extractor splits each path polyline into one cell per (group, blanking) run.

`clip_centerline_at_profiles` (vessel_pipeline.py 1646–1691) then trims the flow-extension ends and **rebuilds** the polydata: it creates new `vtkPoints` / `vtkCellArray`, and copies arrays as follows —

```python
for ai in range(vtk_cl.GetPointData().GetNumberOfArrays()):     # point arrays only
    ...
    dst.SetTuple(i, src.GetTuple(locator.FindClosestPoint(out.GetPoint(i))))
    out.GetPointData().AddArray(dst)
# no loop over vtk_cl.GetCellData() → GroupIds, Blanking, CenterlineIds, TractIds are gone
```

Verified again on the centerline regenerated with HEAD (`scratch/uniform_probe/v2/original_centerline/SNF00000100.vtp`): cell arrays `[]`, point arrays `EdgeArray, EdgePCoordArray, MaximumInscribedSphereRadius, TCoords`. The same holds for `template_centerline`.

On the training side, `extract_groupid_tracts` (dataset.py 564–636) begins with

```python
group_pt = _point_data_array(centerline_mesh, "GroupIds")
group_cell = _cell_data_array(centerline_mesh, "GroupIds")
if group_pt is None and group_cell is None:
    return extract_unique_tracts(centerline_mesh, snap=snap)      # silent fallback
```

so on every real file it runs `extract_unique_tracts`, which builds a snapped graph of polyline segments and splits at degree ≠ 2 nodes. VMTK writes one polyline per inlet→outlet path; the two paths of SNF00000100 share the parent geometrically but *not* topologically (their points are near-coincident, ~0.01–0.1 mm apart, not identical), so the snapping at `1e-4` mm does not merge them and the result is **two independent tracts, each the whole inlet-to-outlet path, and zero junctions**.

#### 2.4.2 Measured consequences on the SNF00000100 template scaffold (`probe_template_scaffold.py`)

| quantity | value | what it should be |
|---|---|---|
| tracts | **2** (95.6 mm and 94.9 mm, both inlet → one outlet) | 3 (parent 88.8, daughters 13.3 / 12.9 mm) |
| junctions | **0** | 1 |
| tokens | 48 + 48, spacing 2.04 / 2.02 mm | ≈ 44 + 7 + 7 at 2 mm |
| fine vertices per tract | **3 961 / 3 948** | ≈ 7 000 / 450 / 450 |
| upsample neighbours on the "other" tract (coarse→mid / mid→fine) | **76.3 % / 77.1 %** | a few % (near the ostium only) |
| `_cross_tract_smooth` down-weighting (× 0.05) | applied to a comparable share of parent edges | parent–daughter coupling edges only |

Reading the table:

- **Vertex assignment.** Every template vertex gets its (u, θ, tract) from the *nearest dense-CL sample*. Along the parent, the two copies of the centerline are a few hundredths of a millimetre apart, so which copy is nearest is decided by numerical noise: 3 961 vertices land on copy 0, 3 948 on copy 1, interleaved along the whole parent. Two neighbouring vertices on the same ring are, as far as the model is concerned, on *different branches*.
- **Tokens.** `allocate_token_counts(96, [95.6, 94.9], 0)` gives 48 per copy. The daughters (13 mm each) receive the last ~6 tokens of each copy; the parent receives two redundant token sets. The `token_attend` mask (each token attends to its own tract) then lets a parent vertex on copy 0 see only copy 0's 48 tokens — half the latent is invisible to half the vertices, and the two halves must learn to encode the same wall twice.
- **Cross-attention tract mask, `LatentTractSelfAttention` per-tract windows, `CoarsePositionalSelfAttention` same-tract locality, `_cross_tract_smooth`** — all of these use `tract_id` and all of them are currently applied to a random bipartition of the parent.
- **Inlet and canonical pose.** `_choose_inlet` picks the longest arm of the highest-degree node — a rule written for a tree with a bifurcation node. With two full-length paths that meet at most at the inlet seed there is no degree-3 node, the "root" is whichever endpoint the tie-break yields, and the inlet (hence the canonical pose, `u = 0`, and the Bishop frame origin) is chosen by a rule that was never meant for this input. It happens to give the inlet on SNF00000100; nothing guarantees it elsewhere.
- **Smoothness weights and Dirichlet.** `_cross_tract_smooth` multiplies the Dirichlet/Laplacian weight by 0.05 on edges whose ends have different `tract_id`; with the random bipartition that is roughly half the parent's edges, so the smoothness terms are ~20× weaker than intended over the parent.

#### 2.4.3 The fix, both sides

Generator: in `clip_centerline_at_profiles`, carry the cell arrays — either keep the (group, blanking) cell structure and copy each kept cell's tuple, or convert cell data to point data before the rebuild (`vtkCellDataToPointData`, then the existing point-array copy handles it). One function, ~10 lines.

Training: the validated rule from the previous review (`prototype_branch_tracts.py`), which `extract_groupid_tracts` should implement instead of its current body:

1. group polyline runs by `GroupIds`; **keep the longest copy per GroupId** (the parent appears once per path);
2. **attach each blanked run to the daughter that follows it** (so daughters start at the bifurcation, not 1.6 mm after it — real files have ~1.6 mm blanks, the synthetic tests 0.4 mm);
3. snap tract endpoints at **1 mm** (`GROUPID_ENDPOINT_SNAP_MM`) to rebuild the tree.

On SNF00000100 this gives 3 tracts (88.8 / 13.3 / 12.9 mm), 1 junction, 100 % coverage, ≤ 1 % overlap. And the silent fallback must become an error for `cleandata` inputs: a real centerline without `GroupIds` is a generator bug, not a synthetic test.

### 2.5 `template_centerline` versus `original_centerline`

Same case, both from `process_centerline_dataset`: 1 922 vs 1 919 points, 6 cells each; Hausdorff 0.12 mm, p95 0.06 mm; MISR difference p50 +0.011 mm, max 0.062 mm. This is expected: the template is the polyball of the original centerline, so its Voronoi centerline is that centerline again, up to the smoothing and marching-cubes noise.

Consequences. (i) The folder buys nothing for training. (ii) `_build_data` reads `original_centerline` (it raises if it is missing) and then passes `template_cl` to `build_scaffold`, ignoring the original. (iii) At inference the natural input is the Stage-1 centerline; routing it through `build_parent_tube` → template → `centerline_creation.py` → tracts adds a VMTK pass and a second, slightly different curve on which the tokens then live. Recommendation: parametrise, tokenise and pose from `original_centerline` in training and from the Stage-1 centerline at inference; keep `template_centerline` at most as a per-case consistency check (Hausdorff < 0.5 mm), or drop the folder.

### 2.6 The Taubin default change (`pass_band` 1.0 → 0.1), measured

`apply_taubin_smoothing(surface, pass_band=0.1, n_iter=15, feature_angle=45.0, boundary_smoothing=True)` is called with defaults by `build_parent_tube` (line 2067) and `compute_centerline_from_mesh` (line 2298), so every template and every centerline generated from now on sees stronger smoothing than the files used in the previous review. `vtkWindowedSincPolyDataFilter`'s pass band is the fraction of the spatial-frequency range that passes: 0.1 attenuates everything shorter than roughly 10 edge lengths, 1.0 roughly 2 edge lengths (the filter is a windowed low-pass in the graph-Laplacian eigenbasis, so "edge lengths" is a loose but useful unit). Fifteen iterations of a non-shrinking filter on a 0.2 mm-edge original cannot move the wall far. Measured (`scratch/probe_smoothing_change.py`, SNF00000100):

| | pass band 1.0 (old) | pass band 0.1 (new) |
|---|---|---|
| vertex displacement p50 / p95 / max | 0.032 / 0.048 / 0.092 mm | **0.041 / 0.065 / 0.178 mm** |
| distance to the unsmoothed original p50 / p95 / max | 0.032 / 0.048 / 0.092 mm | 0.041 / 0.065 / 0.178 mm |
| area | 1 324.09 mm² | 1 324.01 mm² (original 1 320.63) |

Downstream, regenerating with HEAD and comparing to the `d0fdc1b`-era files:

| | old | new |
|---|---|---|
| `original_centerline` points / cells | 1 922 / 6 | 1 922 / 6 |
| centerline Hausdorff old→new / new→old | | 0.137 / 0.140 mm (p95 0.068 / 0.070) |
| MISR new − old p05 / p50 / p95 / min / max | | −0.010 / +0.001 / +0.009 / −0.057 / +0.041 mm |
| MISR range | 0.925–2.241 mm | 0.935–2.242 mm |
| `template_mesh` vertices / area | 7 909 / 1 130.9 mm² | 7 939 / 1 132.4 mm² |
| template edge p05 / p50 / p95 | 0.321 / 0.410 / 0.492 mm | 0.321 / 0.409 / 0.493 mm |
| template old → new distance p50 / p95 / max | | 0.153 / 0.229 / 0.295 mm |

The template-to-template difference (0.15 mm median) is larger than the smoothing effect itself (0.04 mm) because the remesher places vertices differently on each run; it is not a shape change. Two conclusions: the change is safe for everything built so far; and the open question "should the GT be smoothed like the template so the three products are consistent" is moot — the GT's 1.5 × 5 smoothing and the template/centerline's 0.1 × 15 smoothing differ from each other, and from the raw original, by at most 0.18 mm, i.e. less than one GT edge (0.13–0.15 mm) and far less than the 0.41 mm template edge.

### 2.7 Findings from the legacy cache that still apply

- **Coincident tracts poison everything downstream** — now on the template path (§2.4).
- **Short centerlines** (94 of 200 legacy cases covered only the aneurysm neighbourhood) are fixed by construction by re-extracting inlet → all outlets, provided `measure_open_profiles` finds every opening; `hascap.csv` remains the exclusion list for capped originals.
- **Constant 2 mm radius**: the scaffold geometry no longer depends on `TUBE_RADIUS_MM` (template), but the constant still parametrises the loss weights and the head (§7).
- **Fixed 1 000 fine rings / varying physical resolution**: resolved by the template — 0.41 mm edges everywhere, physical.
- **Voronoi centerline bends into large sacs**; the polyball then bulges at the neck (visible in the C0002 render `C0002_uniform_vs_original.png`). A cache-time diagnostic (deviation of the centerline from a spline through the healthy segments; MISR-cap hit rate) is still recommended; the hemoMesh removal option is deferred by decision.
- **No loss weight has been validated at convergence** (the only checkpoint ever written was at epoch 1).

### 2.8 The I/O layer (`cleaned_io.py`, `aneux_paths.py`)

Sound and simpler than before: four folders, completeness gating (`sample_is_complete`), rawdata rejection (`reject_rawdata_paths`), lazy VMTK import, `ensure_derived` for centerlines only, `_init_kwargs` round-trip for the spawn pool. `_build_data` requires templates by default and raises with an actionable message. Minor: `list_cleandata_samples` creates folders as a side effect; the fallback tube path is reachable when `require_templates=False`, which should never be the case on `cleandata`; `aneux_paths.CENTERLINES` (the AneuX-provided centerlines) is now explicitly marked read-only archive — good.

### 2.9 Generation cost and one-process-per-case

Per case on the 5950X, single-threaded: GT remesh 999 s (SNF00000100) / 570 s (C0002) (§2.2), template 34 s, centerline 9 s, template centerline ≈ 9 s. Three of the four run a Voronoi centerline on (slightly) different surfaces, and the GT and the template are cut with frames from two different centerlines (§2.2.4 item 2). One `process_case(v_file)` that computes the working copy once, the centerline once, cuts GT and template with the same frames, exports `R_template` and `StretchDistance` on the final template vertices (§7.1), and writes all three (or four) products, would remove the duplicated ~30 s, the frame inconsistency, and the possibility of a template with fewer openings than its GT. With `n_clipped == n_profiles` as a hard gate on both products and `hascap.csv` exclusions, the whole set is ≈ 682 × 17 min / 14 workers ≈ 14 h locally.

---

## 3. The scaffold build at cache time (`dataset.py`), step by step, with what was measured

This is the code that turns the four `cleandata` files of one case into the one `Data` object the model trains on. It runs once per case (`_write_cache`, 2.0 s per case on SNF00000100) and everything the model ever sees is decided here.

### 3.1 Tracts → tree → canonical pose (`_prepare_tracts`)

1. `extract_groupid_tracts(centerline_mesh)` → list of tracts (N × 3 polylines), their endpoint node ids, junction node ids. On real files: the fallback path of §2.4.
2. If more than `MAX_TRACTS = 16`, keep the 16 longest.
3. `_choose_inlet`: root = node with the most incident tracts; parent tract = the longest arm at the root; inlet = the far end of that arm.
4. `_orient_tracts`: BFS from the inlet, reversing each tract so that all run away from the inlet; junction incidence lists and junction coordinates.
5. `origin` = mean of all tract points; `R = _canonical_pose(inlet_tract)` = rotation sending the inlet tangent (first five samples) to +Z and a derived normal to +X. Every later coordinate — template, GT, tokens — is expressed in this frame (`transform_vessel_mesh(mesh, origin, R)`), so the model never sees the scanner frame. The canonical pose depends on the inlet choice, i.e. on §2.4.

### 3.2 Dense tracts and Bishop frames (`_fit_dense_tract`)

Each tract is deduplicated, fitted with a cubic B-spline (`_fit_centerline_spline`), sampled at `DENSE_CL_SPACING_MM = 0.2` (≥ 32 samples), the tangent is the spline derivative, and the normal/binormal come from `_compute_parallel_transport_frames`: starting from an arbitrary normal at the inlet, each successive normal is the previous one rotated by the minimal rotation taking the previous tangent to the current one. This is the Bishop (rotation-minimising) frame; unlike the Frenet frame it is defined on straight segments and does not flip at inflection points. `u` is the normalised arc length along the dense samples; `arc` is the total length. The frame is used for two things: to define `θ` of every surface vertex, and to build the per-vertex frame the head displaces in.

### 3.3 The fine level = the posed template (`_level_from_surface`)

```python
# dataset.py 1541–1579
proj = self._project_points_to_tracts(pts, dense_tracts)    # nearest dense-CL sample → u, θ, tract_id, r_local, frame
nrm  = self._surface_normals(mesh, n)                       # pyvista point normals (as wound)
n_v, t_v, b_v = self._vertex_frames_from_mesh(nrm, proj["t"], proj["n_cl"], proj["b_cl"], proj["theta"])
edges = self._edges_from_faces(faces, n)                    # bidirectional, unique
u_step = self._u_step_from_edges(proj["u"], proj["tract_id"], edges)   # mean |Δu| over same-tract edges per vertex
```

`_project_points_to_tracts` does one kd-tree query per vertex against the concatenated dense samples of all tracts and returns, for the nearest sample: its `u`, its tract id, the local frame, and `θ = atan2((x − cl)·b, (x − cl)·n)` — the vertex's angle around the centerline in the Bishop frame. `r_local = |x − cl|` is the vertex's distance to the centerline, i.e. the local healthy radius (measured 0.87 / 1.88 / 2.66 mm min / p50 / max on SNF00000100, tracking the MISR to within 0.05 mm at the median).

`_vertex_frames_from_mesh` uses the **mesh normal** as the vertex normal `n_v` (falling back to the Bishop radial direction `cos θ n + sin θ b` only where the mesh normal is degenerate), then builds `t_v` (tangent projected orthogonal to `n_v`) and `b_v = n_v × t_v`. This is the frame in which the head's `Δr` (along `n_v`) and `Δs` (in the `t_v`, `b_v` plane) are applied. **The sign of `n_v` is whatever the file's winding gives** — measured 99.76 % outward on SNF00000100 (`fine_normals_outward_frac`), which is a property of how VMTK's remesher happened to wind that file, not of the code. Everything radial (the sign of `r*`, the head floor, the Chamfer weights) assumes outward; §10 asks for one dot product with the radial direction to enforce it.

Faces are kept (`face`), so the Laplacian and normal-consistency losses operate on the template's real triangles. `branch_nl = [n]`, `n_radial = 1`: the grid bookkeeping of the legacy tube is neutralised — one "branch" with `n` "rings" of one vertex each — so that `upsample_branch_concat` is never used when the kNN tables are present.

Measured fine level (SNF00000100): 7 909 vertices, 15 699 faces, 47 218 directed edges (mean degree 6.0), 3 boundary loops (121 boundary edges), 1 component, 0 non-manifold edges, edge length p05 / p50 / p95 = 0.32 / 0.41 / 0.49 mm.

### 3.4 Mid and coarse by decimation; kNN upsampling (`_decimate_keep`, `knn_upsample_tables`)

`_decimate_keep(mesh, keep_frac, min_points)` tries `pyvista.decimate(reduction)` (quadric), then `decimate_pro`, then an FPS vertex subset with adjacent cells as a last resort. Measured: mid 1 980 vertices (25 %), 3 924 faces, edges 0.43 / 0.89 / 1.71 mm; coarse 638 vertices (8 %), 1 255 faces, edges 0.81 / 1.56 / 2.96 mm; **all three levels keep 3 boundary loops, 1 component, 0 non-manifold edges**. Both coarser levels are re-projected onto the dense CL with the same routine, so they have their own (u, θ, tract, r_local, frame).

`knn_upsample_tables(src_pts, dst_pts, k=3)`: for each destination vertex, the 3 nearest source vertices and inverse-distance weights. `knn_weighted_upsample` then interpolates the coarser level's displacement and hidden features onto the finer level. The `__inc__` override on `AneurysmData` shifts `upsample_idx_*` by the source node count when PyG batches graphs, verified in `test_template_scaffold_starts_from_mesh` with a 2-graph batch. Measured neighbour distances 0.90 / 1.44 mm (p50 / p95, coarse → mid) and 0.53 / 0.84 mm (mid → fine); **0.0 % / 0.04 % of destination vertices have a neighbour on the opposite wall** (test: `n_i · n_j < 0`), because the lumen (diameter ≥ 1.7 mm) is wider than the neighbour search. What is *not* checked and should be: same-tract neighbours (76 % "other tract" today is the §2.4 artefact, but after the fix a daughter's ostium vertices could still pick parent neighbours), and that decimation kept `n_profiles` loops (it did here; `decimate_pro` and the FPS fallback give no such guarantee).

### 3.5 Tokens (`allocate_token_counts`, `_build_latent_tokens`)

`allocate_token_counts(LATENT_LEN=96, arc_lengths, n_junctions)`: reserve up to one slot per junction (never fewer than `MIN_TOKENS_PER_TRACT = 2` per tract), spread the rest over the tracts in proportion to arc length (`allocate_ring_counts`), rebalance to integers. Then per tract, `n_tok` positions `u_q = linspace(0, 1, n_tok)` (so the first token is at the tract start and the last at its end), each with `token_pos` interpolated on the dense CL; one token per junction at `u = 1` with `tract_id = −1` attending to all incident tracts; padding slots repeat the last token. `token_attend[slot, tract]` is the boolean mask the decoder uses.

Measured: 2 tracts → 48 + 48 tokens, 0 junction tokens, spacing 95.6 / 47 = 2.04 mm and 94.9 / 47 = 2.02 mm. That this equals the 2 mm contract is a coincidence of this vessel's 95 mm length; the same code gives 1.0 mm tokens on a 48 mm vessel and 3 mm tokens on a 145 mm one. §5.4 gives the arc-length-driven replacement.

### 3.6 The GT point cloud (`_hybrid_true_points`)

Inputs: the posed GT *vertices* and the dense CL. `d_all` = distance of each vertex to the centerline polyline. 12 288 points (75 %) by farthest-point sampling over all vertices; 4 096 (25 %) by FPS over the vertices with `d_all > TUBE_RADIUS_MM + FAR_CL_MARGIN_MM = 3 mm` (the "far" set — on a 2.7 mm-radius vessel this is the sac and a little of the wall; the threshold is another place where the constant 2 mm radius stands in for `r_local`). If the far set is smaller than 4 096 it is padded with FPS over everything. Concatenate, truncate to 16 384. Measured on the 22 k-vertex stand-in GT: 13 285 unique of 16 384 (3 099 duplicates where the two FPS sets picked the same vertex), `cl_dist` p50 2.04 mm, max 9.38 mm, 21 % farther than R + 1 mm. GT normals per sample come from `closest_cell_normals` on the posed GT mesh (face normals at the nearest cell), which is what the point-to-plane Chamfer uses. On the 97 k-vertex real GT the duplicate count will drop and the far set will be pure sac; but the sampling is still done once and frozen in the cache (§8, §9).

### 3.7 The radial target `r*` and the smoothness statistics (`_template_r_star`)

```python
# raycast.py 207–227
_, idx = cKDTree(gt_pts).query(pos, k=1)                      # nearest GT vertex to each template vertex
offset = ((gt_pts[idx] - pos) * normal).sum(axis=1)           # projected on the template normal
r_star = tube_radius + offset                                 # R + n·(x_GT − x_tpl); "matches composed_radius at identity"
out["valid"] = ones; out["ring_med"] = r_star.copy()          # dth, du stay zero (empty_r_star)
```

What this computes is "how far, along my normal, is the *nearest* GT vertex". On the healthy wall the nearest GT vertex is right there (offset p50 0.06 mm, p95 0.32 mm — fine). Under the sac the nearest GT vertex is *not* the dome 4–25 mm out along the normal; it is the neck rim or the sac wall a fraction of a millimetre beside the template vertex, so the offset is tiny. §7.1 has the full numbers and the fix. `dth`, `du` (angular and longitudinal gradients of `r*`) and `ring_med` (ring median of `r*`) were the grid statistics of the tube path (`compute_level_r_star`) and are simply not computed here — `dth = du = 0`, `ring_med = r*`, `ambiguous = False`.

### 3.8 The assembled sample, in numbers (SNF00000100)

| field | value |
|---|---|
| levels | fine 7 909 / mid 1 980 / coarse 638 vertices; 3 loops, 1 component, manifold at every level |
| tracts / junctions / tokens | 2 / 0 / 48 + 48 (should be 3 / 1 / ≈ 44 + 7 + 7) |
| `r_local` | 0.87 / 1.88 / 2.66 mm |
| normals outward | 99.76 % |
| `r*` valid / ambiguous | 100 % / 0 % ; `dth_max = du_max = 0`; `ring_med == r_star` |
| `r*` offset under the sac (304 vertices) | p50 0.029 / p90 0.240 / max 0.828 mm — true outward ray 4.09 / 11.80 / 24.88 mm |
| `smoothness_edge_weights` | min = mean = 1.0 on every edge |
| upsampling | opposite wall 0.0 % / 0.04 %; other tract 76 % / 77 % |
| `x_true` | 16 384 (13 285 unique), 21 % far |
| losses at init | recon 1.315, kl 7.134, disp 0.0, lap 0.190, norm 0.0091, rad 0.0157 |
| identity displacement at init | p50 = max = 0.0 mm (the fine level *is* the template) |
| train step (fwd + bwd, bs 1) / peak VRAM | 790 ms / 0.74 GiB (3080 Ti) |
| cache build | 2.0 s |

---

## 4. Encoder (PointNeXt)

### 4.1 What it does

`PointNeXtEncoder.forward(data)`: `stem` (Linear 3 → 32, LayerNorm, LeakyReLU) on the raw `x_true` coordinates, then four `SetAbstraction` stages — FPS to `n_out` centres, radius grouping (`radius_mm`, up to 256 neighbours), a shared MLP on `[h_neighbour, (Δxyz / radius)]`, max-pool per centre — each followed by two `InvResMLP` blocks (inverted-residual MLP with radius grouping, gated residual `α = 0.1` at init):

| stage | centres | radius | hidden |
|---|---|---|---|
| 1 | 16 384 → 1 024 | 1.5 mm | 64 |
| 2 | 1 024 → 256 | 3 mm | 128 |
| 3 | 256 → 64 | 6 mm | 256 |
| 4 | 64 → 64 | 12 mm | 512 |

After stage 4 there are 64 tokens of 512 features, one per ~1.5 mm of centerline on a 95 mm vessel... no — 64 centres spread by FPS over the *surface*, so roughly one per 20 mm² of wall, each summarising a 12 mm neighbourhood. `nearest_centerline_attr` assigns each centre its nearest dense-CL `u` and tract. Then `CenterlineLatentHead`: 96 queries, one per latent slot, each `w_q([γ(u_token), tract_emb])`; keys `w_k([h, γ(u_centre), tract_emb])`, values `w_v(h)`; a single softmax attention; `mu_head`, `logvar_head` (clamped to [−8, 2]). Parameters: 18.5 M (12.6 M in stage 4's 512-wide InvRes blocks acting on 64 points; 1.2 M in the head); forward 64 ms.

### 4.2 Findings

1. **The input is coordinates only.** GT normals are cached (`x_true_normal`) and not fed to the encoder; nor is the one feature that would make the encoder's job explicit now that a template exists — the **signed distance of each GT point to the template** (positive outside the healthy tube = sac). Both are free: concatenate to the stem input (3 → 7 channels). The encoder's task is precisely "describe the residual"; giving it the residual as a feature removes the need to infer the tube from 16 k points.
2. **The latent head is thin.** One head, one layer, no feed-forward block, no LayerNorm, positional queries only (the query knows *where* it is, `γ(u)` and tract, but not *what* is there until the softmax has mixed the 64 values). Angular information (θ) is not in the query or key at all — it can only arrive through the content of `h`. On a 2 mm-token contract with ~50 tokens per case, the head should be a small transformer: assign each of the 64 (or more) centres to its nearest token along the tree, pool per token (attention or max), then 2–4 self-attention layers over the token *tree* with arc-length and branch-depth positional encodings (§5.4).
3. **Capacity is in the wrong place.** 12.6 M parameters process 64 points at stage 4; the head that has to produce 96 × 128 latent values has 1.2 M. Halve stage 4 (256 wide) and give the head the difference.
4. **`logvar.clamp(-8, 2)`** is a hard clamp: outside the range the gradient to `logvar_head` is exactly zero, so a token whose log-variance drifts past −8 (posterior collapsed to a point) or 2 stops learning its variance. A soft bound (`-8 + 10·sigmoid(·)`) or simply a wider range with the free-bits KL of §5.3 avoids it.
5. **Batching.** With `n_graphs > 1` the head loops over graphs in Python (`for g in range(n_graphs)`); fine at batch 4–8, but the per-graph GEMM pattern means batching does not buy throughput in this module.

---

## 5. Latent space and VAE

### 5.1 As built

96 tokens × 128 dims = **12 288** latent dimensions per case. `vae_kl_loss` sums the KL over the 128 dims and averages over tokens (and batch); at init on the real sample KL = 7.13 → weighted by `LAMBDA_KL = 5e-4` (after a 20-epoch linear warm-up) → **0.0036** in the objective against recon 1.315. That is 0.3 % of the loss: the posterior is free to be arbitrarily sharp and the latent to have any scale — functionally an autoencoder with a noise term that vanishes as training proceeds. `LatentTractSelfAttention` (window ±2 tokens along `u`, ALiBi penalty `8·|Δu|`, gate ≤ 0.5, zero-initialised output) mixes neighbouring tokens after sampling — a mild smoothing of the code along the tree, but on the training path only (it is also inside `decode`, so it does apply at inference; it is the *encoder's* posterior that is not regularised toward smoothness).

### 5.2 What Stage 1 needs from the latent

Stage 1 emits, per centerline sample at 1 mm, `[x, y, z, r]` and, on every second sample, a texture token `z_1 … z_D`, along the whole tree. For that to be learnable and for Stage 2 to decode it faithfully, the latent should be: a tree-structured sequence of **local** tokens (each token describes the wall near its position, so Stage 1 can predict it from local conditions); of **fixed per-dimension scale** (so Stage 1's regression loss is meaningful); **smooth** (a Stage-1 prediction near the training codes should decode to a sensible wall, which requires the decoder to have seen a neighbourhood around each code — that is what the posterior noise provides); with a **known null code for healthy wall** (so Stage 1 only has to model where the sac is and what it looks like, and the rest of the tree can emit the prior mean); and the decoder should be **robust to Stage 1 error** (the noise-robustness curve below is the contract).

### 5.3 VAE vs autoencoder + noise — trade-off and recommendation

| | β-VAE with a meaningful β | AE + fixed-σ noise + standardisation | **Light per-token KL-VAE with free bits (recommended)** |
|---|---|---|---|
| reconstruction accuracy | costs accuracy; the trade-off is steep with ~200–680 samples | maximal | near-AE: the free-bits floor means KL only bites on dimensions that carry no information |
| latent scale | unit by construction | must be standardised explicitly after training | near unit; a final standardisation pass is cheap insurance |
| healthy segments | collapse to the prior — **desired**: the template already explains them | no pressure toward a null code; healthy tokens are arbitrary points in R^128 | same desirable collapse; the per-token KL along the tree becomes a diagnostic of where the template fails |
| robustness to Stage 1 error | built in (decoder trained on the posterior spread) | σ controls it directly; must be tuned by hand | built in at the learned posterior width; add extra noise if it narrows |
| sampling without Stage 1 | yes | no | healthy tokens yes; sac tokens from the aggregate posterior |
| failure mode | global posterior collapse → blurry sacs | σ mis-set → brittle (too small) or blurry (too large) | two knobs (β, free bits) with a wide safe range |

Recommendation, concretely: per-token KL to N(0, I) with **free bits** ≈ 0.1–0.25 nats per dimension (KL below the floor costs nothing, so the model is not punished for using a dimension a little); β such that the total KL is **2–5 % of the reconstruction term at convergence** (start around 1e-2 on a per-dimension-averaged KL and adjust); the existing 20-epoch warm-up. Log three things: **active units** (dimensions whose KL exceeds the floor), separately for healthy and sac tokens; the **per-token KL profile along the tree** (should be ≈ 0 on healthy wall, large under the sac — if it is flat, the latent is not local); and a **noise-robustness curve** — reconstruction error when decoding `μ + σ·ε` for `σ ∈ {0, 0.25, 0.5, 1}`, as the published contract with Stage 1. Fall back to AE + fixed σ ≈ 0.2–0.3 only if β proves untunable on this data. Once Stage 1 exists, fine-tune the decoder on Stage 1 samples.

### 5.4 Token geometry under the 1 mm / 2 mm decision

| `ds_tok` | tokens per sample (80–130 mm tree) | tokens on a sac (neck 3–6 mm, height 3–15 mm) | comment |
|---|---|---|---|
| 1 mm | 80–130 | 5–15 | every token sees half a sac; Stage 1 must model 100+ tokens |
| **2 mm** | **40–70** | **3–8** | a sac is 3–8 tokens: neck, body, dome are separable |
| 3–4 mm | 25–40 | 1–4 | a sac becomes one blob token |

Given the decision, the changes are:

- `TOKEN_SPACING_MM = 2.0` and `CL_SAMPLE_MM = 1.0` as shared constants of both stages; per branch `n_tok = floor(L / 2) + 1` tokens at arc positions `k · 2 mm` (first token at the branch start). The daughter's first token *is* the junction — no separate junction tokens; the tree gives adjacency. `LATENT_LEN` becomes the padding maximum (e.g. 128) with a validity mask, not the count. `allocate_token_counts` and the fixed-slot logic go.
- Tokens on the **original / Stage-1 centerline** (§2.5), not on `template_centerline`.
- Encoder head as in §4.2 item 2.
- Decoder cross-attention (§6.4) restricted to the ~5 nearest tokens along the vertex's branch (plus the neighbouring branch's tokens within ~4 mm of an ostium), instead of softmax over all of a tract's slots — with 40–70 tokens and a positional query this is what the softmax converges to anyway, and the restriction makes the latent provably local.

---

## 6. Decoder (`ProgressiveSplineDecoder`)

### 6.1 What happens at one level

Take the coarse level (638 vertices). `LatentCrossAttention`: every vertex forms a query from its intrinsic position only, `w_q([γ(u), γ(θ)])` (16 + 12 Fourier features); keys are `w_k([z_token, γ(u_token)])`, values `w_v(z_token)`; the softmax is masked to the tokens whose `token_attend` row includes the vertex's tract; the output is `out([attended, γ(u), γ(θ)])` → 128 hidden features per vertex. Then `CoarsePositionalSelfAttention` (coarse only): multi-head self-attention among vertices of the same tract within `COARSE_ATTN_RINGS = 4` steps of `u` (and across tracts near ostia), gated residual. Then four `ResidualSplineConv` layers: `h ← h + ELU(SplineConv(h, edge_index, pseudo))` where the kernel weight for each edge is a degree-2 B-spline evaluated at the edge's 3-D pseudo-coordinate (`e_u, e_th, e_kind`) on a 5 × 5 × 5 grid of control weights (125 × 128 × 128 parameters per layer). Finally `DecoupledDisplacementHead`: `Δr = softplus(W_r h + b_r) − R_MARGIN`, `Δs = 3 mm · tanh(W_s h)`; `decoupled_displacement` turns them into `Δx = Δr n_v + Δs_1 t_v + Δs_2 b_v`. The head's weights are zero and its bias is set so that `Δr = 0` at init — the decoder starts as the identity on the template (measured 0.0 mm).

The mid level receives `h_mid = cross_attention + mid_init(upsampled Δx_coarse) + σ(α_c)·upsampled h_coarse`, runs its four convs, predicts a *residual* `Δr_mid` (clamped by `clamp_residual_radial` so the composed radial offset stays ≥ −R_MARGIN) and `Δs_mid`, and adds them to the upsampled coarse displacement. The fine level does the same on top of mid. Output: `x_pred = pos_fine + Δx_total`, plus the intermediate levels for the multi-scale Chamfer.

### 6.2 Pseudo-coordinates — still θ-blind on the template

```python
# geometry.py 37–63
e_u  = 0.5 + 0.5 * (du / step).clamp(-1, 1)      # step = max(u_step[src], u_step[dst]) — per-vertex mean |Δu| over same-tract edges
e_th = 0.5 + 0.5 * (dth / π).clamp(-1, 1)        # Δθ wrapped to (−π, π], mapped to [0, 1]
e_kind = 0.5 * same_tract
```

`e_u` is right: `u_step` is a physical, per-vertex step, so a neighbour one edge along the vessel lands at `e_u ≈ 0` or `1` — the kernel's extreme knots — and a ring neighbour at `0.5`. `e_th` is not: on the template a 0.41 mm edge on a 1.88 mm radius spans `Δθ = 0.41 / 1.88 ≈ 0.22 rad`, so `e_th = 0.5 ± 0.5 · 0.22 / π = 0.5 ± 0.035`. A degree-2 open B-spline with 5 control points has 3 knot intervals of width 1/3; a shift of 0.035 is a tenth of an interval, i.e. the basis functions at `e_th = 0.465`, `0.5` and `0.535` are almost identical. **The kernel cannot tell the left ring neighbour from the right one or from an axial neighbour.** The θ axis of the 5 × 5 × 5 kernel carries no information, and since `e_kind` takes only two values, of the 125 weight matrices per layer only the ~10 combinations along `u` and `kind` are actually distinguishable; the decoder is effectively a 1-D convolution along `u` with an isotropic in-ring average. On the daughter branches (radius 0.9 mm) `Δθ ≈ 0.45 rad`, `e_th = 0.5 ± 0.07` — still within a quarter of an interval.

Fix: normalise `Δθ` by the local circumferential step, `e_th = 0.5 + 0.5 · clamp(Δθ · r_local / edge_len_ref, −1, 1)` with `edge_len_ref` the vertex's mean edge length (both are available at cache time), so that ring neighbours land at 0 / 1 as `u` neighbours do. Separately, `e_kind` (0 or 0.5) occupies a full 5-knot axis to encode one bit: use kernel `(5, 5, 2)` with degree 1 on the last axis (saves 60 % of the SplineConv parameters and time), or a separate small conv for cross-branch edges.

### 6.3 Compute profile on the real template scaffold (3080 Ti, SNF00000100)

| | tube path (legacy, 64 000 fine nodes) | **template path (7 909 / 1 980 / 638)** |
|---|---|---|
| fwd + bwd, one sample | 4.28 s (with AdamW step) | **0.79 s** |
| peak VRAM | 2.7 GiB | **0.74 GiB** |
| cache build per case | — | 2.0 s (no ray-casting) |

The 5 × 5 × 5 SplineConv (24.6 M of 43.4 M parameters) is still the bulk of the time, but the template's physical resolution removed the 8× node excess of the tube. On the A100s this is comfortably a batch of 8–16 per GPU; the point is not speed but being able to afford 2–5 · 10⁴ optimisation steps, deeper decoders and k-fold sweeps (§9). Kernel `(5, 5, 2)` and 64-channel fine-level convs are still worth taking.

### 6.4 Aggregation and normalisation

`make_spline_conv(..., aggr="add", root_weight=False)` and no normalisation layer anywhere in the decoder convs. `aggr="add"` sums over neighbours, so a vertex's update scales with its degree; on the template the degree is regular (mean 6.0 fine, 6.0 mid, 5.9 coarse; range 5–8) so this is less harmful than on the junction-coupled tube, but boundary vertices (degree 3–4) still receive systematically smaller updates. `root_weight=False` means a vertex's own feature enters only through the residual skip, never through a learned self-weight. Use `aggr="mean"` (or degree normalisation), `root_weight=True`, and pre-norm (LayerNorm on `h` before each conv) — standard, cheap, and they make the six-to-eight-layer decoders of §9 trainable.

### 6.5 Cross-attention from vertices to the latent

Content-keyed positional interpolation of the latent along `u`; single head; no feed-forward block; θ enters only through `out`, so the *choice* of token cannot depend on θ (correct — tokens are per-position — but the *read-out* per θ is a single linear map of the attended vector). Two heads and an FFN cost nothing at these sizes. With fixed 2 mm tokens, restrict attention to the nearest tokens (§5.4). The tract mask is today applied to the coincident tracts of §2.4 — after that fix the mask means what it should.

### 6.6 Hierarchy and upsampling — done, with two refinements

`knn_upsample_tables` (k = 3, inverse distance, `__inc__` correct under batching) replaces the bilinear cylindrical upsampling. Two refinements: restrict candidates to the same branch (after §2.4) and to `n_i · n_j > 0`, so thin daughters near an ostium cannot pick parent neighbours through the wall; and assert at cache time that decimation kept `n_profiles` loops and one component (it did here, but `decimate_pro` and the FPS fallback have different guarantees). Also, the decoder currently sees **no centerline geometry as node features** — curvature, torsion, `r_local`, distance to the nearest ostium/junction. `r_local` is already cached; concatenating `[r_local, κ, τ, d_ostium]` to the cross-attention output is one line and gives the convs what they need to place a dome on a bend.

### 6.7 Displacement head bounds

`Δr = softplus(·) − R_MARGIN_MM` with `R_MARGIN_MM = TUBE_RADIUS_MM = 2.0` constant; `Δs = SHEAR_MAX_MM · tanh(·) = 3 mm · tanh`. On a 0.87 mm daughter branch the floor lets a vertex move 2 mm *inward* — through the centerline and out the other side. On the trunk, 3 mm of shear cannot carry a vertex around an overhanging dome (the sac on SNF00000100 has vertices whose GT counterpart along the normal is 25 mm out; a dome that overhangs needs vertices to slide several millimetres tangentially). Both bounds should be `r_local`-relative — floor ≈ −0.8 · r_local (a vertex may approach but not cross the centerline), shear cap ∝ max(3 mm, k · r_local) — and the **coarse level should be free 3-D** (638 vertices; its job is to place the dome, not to respect a tube frame).

### 6.8 Output topology

Inherited from the template: one component, `n_profiles` boundary loops, manifold — the CFD prerequisite is structural. What keeps the *deformed* surface fold-free and self-intersection-free is the subject of §11.

---

## 7. Losses on the template path

Measured at identity init on the SNF00000100 template scaffold: recon 1.315, kl 7.134, disp 0.0, lap 0.190, norm 0.0091, rad 0.0157. The identity displacement is 0.0 mm, so every non-zero term at init is the cost of the *template itself* against the GT (Chamfer, Laplacian of the template's curvature, radial Huber of the nearest-vertex `r*`) plus the untrained KL. `weighted_total` then multiplies by `λ_recon = 1`, `λ_kl = 5e-4` (after a 20-epoch anneal), `λ_disp = 0.15`, `λ_lap = 0.05`, `λ_norm = 0.02`, `λ_rad = 1.0`. At init the objective is therefore dominated by recon (1.315) and rad (0.016); the weighted KL is 0.0036. None of these weights has been validated at convergence — the only checkpoint ever written was at epoch 1.

### 7.1 Radial Huber versus `r*` — actively opposes the sac

`composed_radius(x, x_tube, normal, tube_radius) = R + n · (x − x_tube)` with `R = TUBE_RADIUS_MM = 2.0`. The predicted radius is this quantity at the displaced vertex; the target is `r*` from `_template_r_star` (§3.7); `radial_huber_loss` is Huber(δ = 1 mm) of the difference, averaged over `valid` nodes, with a 0.5× copy at mid (`LAMBDA_RAD_MID`). `λ_rad = 1` so this term has the same weight as the Chamfer.

What `r*` actually is, measured on the 304 template vertices whose true outward ray to the GT exceeds 1 mm:

| | nearest-vertex offset (`r*` − R) | true outward ray |
|---|---|---|
| p50 | **0.029 mm** | **4.09 mm** |
| p90 | 0.240 mm | 11.80 mm |
| max | 0.828 mm | 24.88 mm |
| ratio offset / ray, p50 / max | **0.0045 / 0.255** | |

The nearest GT *vertex* under the sac is the neck rim or the sac wall a fraction of a millimetre beside the template vertex, not the dome 4–25 mm out along the normal. The projection `n · (x_nn − x_tpl)` is therefore almost zero. `valid = True` on all 7 909 vertices and `ambiguous = False` everywhere, so the Huber is applied in full under the sac. At identity the term is small (0.016) because the target *is* "stay put"; as soon as the Chamfer starts to pull a dome out, this term pulls it back. Healthy wall is fine (offset p50 0.065 mm, p95 0.315 mm).

The quantity that *would* be the right target is the outward ray along the template normal — exactly `compute_raycast_stretch_distances` after the orientation fix of §2.3, or a cache-time ray-cast with misses/grazes marked `valid = False` and double hits `ambiguous = True`. The previous review already measured that ray (`raw_ray_under_sac`: median 4.09 mm, 3 302 raw misses on the whole template, which must stay invalid). Exporting `StretchDistance` on the final template vertices and reading it as `r*` removes the nearest-vertex approximation entirely.

### 7.2 Dirichlet / Laplacian weights — silently off

`smoothness_edge_weights` was written for the tube grid: it down-weights an edge when `dth` or `du` (angular / longitudinal gradient of `r*`) is large, or when `|r* − ring_med|` exceeds 1 mm, or when either end is `ambiguous`. The formula is `w = 1 / (1 + β_θ dθ + β_u du + β_r rdev)`, then `w_ambiguous = 0.05` on crease vertices. On the template path `dth = du = 0`, `ring_med = r*`, `ambiguous = False` (§3.7), so `w_on = 1` on every valid-valid edge. Measured: **min = mean = 1.0, fraction < 1 = 0**. The crease-aware weighting is switched off. `_cross_tract_smooth` still multiplies by 0.05 on edges whose ends have different `tract_id` — which, until §2.4 is fixed, is a random bipartition of the parent, not parent–daughter coupling.

Replace the grid statistics with edge-based ones that exist on any mesh: `|r*_i − r*_j| / edge_len` and the 1-ring median of `r*`. Better still, drive the weight from the GT distance itself (a large gradient of `StretchDistance` *is* the neck). Until then, Dirichlet is a uniform smoothness prior on `(Δr, Δs)` and the Laplacian is a uniform smoothness prior on absolute positions.

### 7.3 Chamfer — weights and sampling

The reconstruction term is a symmetric point-to-plane Chamfer mixed with 0.2 L2, Huber δ = 1 mm, at three levels (`λ_cd_mid = 0.5`, `λ_cd_coarse = 0.05`). Weights are `1 + d / R` capped at 4, so a point 6 mm from the centerline is weighted 4× a point on the wall.

- The **GT side** uses the true distance to the dense centerline (`x_true_cl_dist`). Correct.
- The **predicted side** uses `composed_radius` with the constant `R = 2.0` as a stand-in for the distance from the centerline: `|R + n · Δ|`. On a 0.87 mm daughter this overstates the radius; under a 25 mm sac it saturates at the cap of 4 anyway. Use `r_local + n · Δ` — `r_local` is already cached.
- Predicted points are the **vertices** of the current level. A stretched dome is under-sampled relative to the healthy wall (the template has uniform 0.41 mm edges; after a 10× radial stretch those triangles are long and skinny and their vertices do not cover the dome). Sample on predicted faces (uniform or area-weighted) so the Chamfer sees the surface, not the tessellation. Combined with caching the full GT (§2.2.3), this removes the 16 384-point floor.

### 7.4 Laplacian on absolute positions

`_uniform_laplacian_smoothing` is `||LV||` on `x_pred`, i.e. it penalises the template's own curvature (the siphon, the ostium flare) as well as folds of the displacement. At init it is 0.190 — the cost of the template looking like a tube with bends. A Laplacian on `Δx` or on `(Δr, Δs)` would be zero at identity and would only punish deformation. Low priority next to §7.1–§7.3.

Normal consistency (`1 − cos(n_i, n_j)` on adjacent faces) is 0.009 at init and is the right term for keeping the surface smooth; it does not know about *flips* relative to the template (a triangle that has gone through itself has a normal that still agrees with its neighbours). That is §11.

### 7.5 What is missing

A **triangle-stretch / edge-length-ratio** regulariser and a **normal-flip penalty** (`n_pred · n_template < 0` per triangle). These are the real failure modes of a fixed-topology deformer on overhanging sacs: triangles invert, edges stretch by 10×, the surface self-intersects. They belong with the CFD requirements of §11, not as an afterthought to the Chamfer. KL is §5.

---

## 8. Training framework

`aneuxai.py` sets `BATCH_SIZE = 1`, `ACCUM_STEPS = 8`, `EPOCHS = 200`, `VAL_SPLIT = 0.15`, `VAL_EVERY = 5`, `SEED = 31`, `NUM_WORKERS = 3`. `train_model` then runs AdamW at 2e-4 with weight decay 1e-4 on *every* parameter, cosine annealing over 200 epochs with no warm-up, gradient clip 1.0, EMA 0.999 updated every optimiser step, validation on EMA weights every 5 epochs, `best.pt` selected by `val_recon + λ_rad · val_rad`. `last.pt` stores model / EMA / optimiser / scheduler / epoch / metrics, but nothing ever *loads* it — there is no resume. Logging is `print` plus a history list returned at the end.

| item | current | issue / suggestion |
|---|---|---|
| batch | bs 1 × accum 8 → ~21 steps/epoch on ~170 train cases, **4.3 k steps in 200 epochs** | plan 2–5 · 10⁴ steps; real bs 4–8 now fits easily (0.74 GiB/sample) |
| optimizer / schedule | AdamW 2e-4, wd 1e-4 on everything, cosine, no warm-up | exclude LayerNorms, biases and gates from wd; 200–500-step LR warm-up |
| EMA | 0.999 per step | horizon `1 / (1 − 0.999) = 1 000` steps ≈ **47 epochs** at 21 steps/epoch: early `best.pt` selection uses near-initial weights. 0.99–0.995 with a warm-up `min(d, (1 + n) / (10 + n))` |
| validation | every 5 epochs, EMA only, ~15 % random split | also log non-EMA val; k-fold (§9) |
| augmentation | none | L/R mirror (before scaffold build — flips Bishop handedness consistently), per-epoch resampling of `x_true` from the full GT (cache all vertices, not 16 384 — 3 099 of the 16 384 are duplicates on the 22 k-vertex stand-in), θ-phase and ±5° pose jitter |
| resume | `last.pt` is written, never read | load model / EMA / opt / sched / epoch / RNG |
| logging | `print`, history at the end | per-epoch CSV / TensorBoard (the KL profile and noise-robustness curve of §5.3 live here) |
| device | `gpu_index = 1 if n_gpu > 1 else 0` | `CUDA_VISIBLE_DEVICES` or an argument |
| multi-GPU | none | 4 × A100 as 4 independent configs / folds |
| precision | FP32 tensors, TF32 matmuls | correct; keep it |
| split | random 85/15, no test set | 5-fold CV + a held-out test fold |

The EMA horizon is the one that silently wastes early validation: at decay 0.999 the shadow is a 1 000-step box filter, so the "best" checkpoint of the first 50 epochs is still mostly the initial weights. Combined with `val_every = 5` and a 4.3 k-step budget, the training loop as written cannot tell a working model from an identity decoder.

---

## 9. Spending compute for accuracy, not speed

HPC wall-time is effectively unlimited; VRAM on the 3080 Ti already has headroom (0.74 GiB of 12) and an A100 40 GB is not the constraint. The levers below are ordered by expected accuracy gain per unit of extra compute, using only things this review has already argued for.

| lever | expected effect | cost |
|---|---|---|
| full GT point set + face-sampled predicted points, resampled every epoch | removes the 16 k sampling floor and the dome vertex-density bias (§2.2.3, §7.3) | memory only |
| augmentation set of §8 | largest single generalisation gain on ~200–680 samples | free |
| 5–10× more optimisation steps | nowhere near convergence at 4.3 k steps | time, now cheap (0.79 s/step → 2–5 · 10⁴ steps is 4–11 GPU-hours per fold) |
| decoder depth (6–8 residual convs/level, pre-norm) and multi-head cross-attention + FFN instead of the 5³ kernel | capacity where it is used (§6.2, §6.4) | modest |
| local-pooling transformer latent head (§5.4) | better latent, less overfitting | small |
| normals + template signed distance into the encoder (§4.2) | cheap accuracy | free |
| densified templates under the sac (§2.3 fix) | fine resolution where the residual lives | regenerate templates |
| curriculum: radial-only warm-start on the parent, then unlock sac terms | stable convergence against the currently opposing radial Huber | free |
| 5-fold CV | error bars on every design decision | 5× per config — what the A100s are for |

---

## 10. Template scaffold: what is done, what is wrong, what remains

### 10.1 Status of the migration items

| component | previous review | now (`e7fd7ce`) |
|---|---|---|
| fine scaffold = template | to do | **done**: posed `template_mesh`, exact identity at init |
| hierarchy | coarse templates needed | **done differently**: `decimate` 25 % / 8 % of the template; openings and manifoldness preserved on the probe (assert it) |
| inter-level upsampling | barycentric / kNN tables | **done**: kNN k = 3 inverse distance, batched correctly, no cross-wall mixing |
| grid bookkeeping (`branch_nl`, `n_radial`) | to neutralise | **done**: `n_radial = 1`, one "branch" per level |
| (u, θ, branch) per vertex | export or project | **done by projection** to the nearest dense-CL sample — onto *coincident* tracts until §2.4 is fixed |
| `r*` | `StretchDistance` along template normal | **wrong**: nearest-vertex normal offset (§7.1) |
| smoothness weights | edge-based | **disabled**: dth/du zero (§7.2) |
| `R_local` per vertex | MISR interpolation | **cached** (`r_local`, `r_local_mid`, `r_local_coarse`) but **unused** by model and losses |
| tokens at 2 mm on the original centerline | to do | fixed 96 slots on the template centerline (§5.4, §2.5) |
| pseudo-coords in physical units | to do | u yes (`u_step`), θ no (§6.2) |
| displacement bounds `r_local`-relative, free 3-D at coarse | to do | constants (§6.7) |
| GT generator | parent tube | **done**: `remeshing.py`, verified on SNF00000100 and C0002 (§2.2) |
| `postprocess.tensor_to_vtp` | template faces | not checked here |

### 10.2 Remaining design (ordered)

1. **Data first.** Cell arrays through the clip + `extract_groupid_tracts` rule (§2.4); orientation-agnostic stretch test and regenerated templates (§2.3); parametrise from `original_centerline` (§2.5). Generate the three products of one case in one process so they share one centerline and one set of ostium frames (§2.9); export `R_template` and `StretchDistance` on the final template vertices.
2. **Normals.** Orient every level's normals outward by the radial direction from the nearest centerline sample (one dot product; the sign of `r*`, the head floor and the Chamfer weights all depend on it). Today it is correct on 99.8 % of SNF00000100's vertices only because of how that file happens to be wound.
3. **`r*` and weights** as in §7; **`r_local`** in `composed_radius`, Chamfer weights, head bounds; θ pseudo-coordinate in physical units.
4. **Tokens** per §5.4 with arc-length-driven counts and a mask.
5. **Cache the full GT** (vertices + normals) and a mirrored copy; cache-time gates: openings = `n_profiles` at every level, one component, no non-manifold edges, GT→template p99 *outside the sac* ≤ 0.5 · `r_local`, every GT vertex within a few millimetres of the centerline.
6. **Inference path** = Stage-1 centerline (1 mm, MISR) → `generate_base_surface` + uncap → remesh → Stage 2 decode. Variable remeshing needs the GT for its stretch field; at inference only a uniform remesh (or a stretch predicted from the tokens) is available. Train with both densities, or train on uniform templates and let the final remesh (§11) handle density — decide before regenerating.

---

## 11. Watertight single manifold for CFD

Topology is now inherited from the template (§6.8): one component, `n_profiles` boundary loops, manifold. That is the prerequisite, not the guarantee. A vertex-displacement decoder on a 25 mm overhang can fold triangles, invert normals and put the surface through itself while still having the right connectivity. Mandatory to keep the output CFD-ready:

1. **Fold prevention during training.** A hinge penalty on `n_pred · n_template < 0` per triangle, plus a stretch regulariser (per-triangle singular values of the deformation gradient, or edge-length ratios against the template). These are the missing terms of §7.5.
2. **Displacement bounds that can reach the dome** without crossing the lumen (§6.7): `r_local`-relative floor and shear, free 3-D at coarse.
3. **Verification in `postprocess.py`**, as validation metrics, not as a silent repair: self-intersection count, boundary loops = `n_profiles`, one component, minimum triangle angle. Model selection of §12 uses these; a folded mesh with a good Chamfer is a failed sample.
4. **A final isotropic remesh** of the decoded surface before CFD (the same `remesh_surface_isotropically` already used for the GT, at a CFD-appropriate edge — not 0.15 mm). This is the step that restores triangle quality after a large deformation; it is not a substitute for (1).

Optional, cheap, and matching the GT/template ostium mismatch of §2.2.4: keep boundary-loop vertices in their profile plane during training (they may slide in-plane, they may not leave it). Outlet position and planarity are already specified as a post-process; constraining them in the loss removes a degree of freedom the Chamfer would otherwise spend.

---

## 12. Evaluation protocol

Held-out folds, millimetre units, non-Huber (the training Chamfer's Huber δ = 1 mm hides the tail that CFD cares about):

- **Global:** symmetric Chamfer mean and p95, Hausdorff, normal-angle error.
- **Sac-specific** versions of the same, using a sac mask. Two equivalent definitions, both already measurable: template vertices whose outward ray to the GT exceeds 1 mm (304 of 7 909 on SNF00000100), or GT points farther than `r_local + 1 mm` from the centerline. Report both so a model that grows the sac in the wrong place cannot hide in the global Chamfer.
- **Neck-plane error and sac volume error** — the two numbers a clinician actually looks at.
- **The mesh-validity counts of §11** (self-intersections, loops, components, min angle).
- **The noise-robustness curve of §5.3** (`μ + σ·ε` for σ ∈ {0, 0.25, 0.5, 1}) — the interface contract with Stage 1.

The current validation score (`val_recon + λ_rad · val_rad`) stays for model selection only; it is not the number that is reported.

---

## 13. Tests

`test_architecture.py`: 55/55 pass (~4.5 min). `test_template_scaffold_starts_from_mesh` is a good addition (posed template = fine level, kNN tables, `__inc__` under a 2-graph batch, finite output). `test_remeshing.py`: 14/14 pass through the pytest-free shim (vmtk_env has no `pytest`). What the architecture suite cannot see, and should:

- **Coaxial synthetic cylinders** (template r = 2.0, GT r = 2.4, no sac): the nearest-vertex `r*` is exact there, so its failure under a sac is invisible. Add a cylinder with a lateral bulge and assert `r*` ≈ ray distance under it.
- Both synthetic meshes are wound **outward**; nothing asserts that scaffold normals point *away* from the centerline.
- Every centerline test uses point-data arrays with identical copies and a 0.4 mm blank. Real files have cell arrays (dropped), near-coincident copies and ~1.6 mm blanks. A fixture from `scratch/uniform_probe/original_centerline/SNF00000100.vtp` (or `scratch/cl_probe/SNF00000100_branched_unclipped.vtp`) asserting 3 tracts / 1 junction would have caught §2.4 twice.
- `smoothness_edge_weights` returning all ones; token spacing ≈ 2 mm independently of vessel length; `e_th` of a ring neighbour ≈ 0 or 1; decimation keeping `n_profiles` loops at mid and coarse.
- Sac preservation on a real GT remesh (the probe of §2.2): area ratio ∈ [0.88, 1.20], openings = profiles, original→output fraction > 1 mm below a small threshold. The synthetic scale gates already exist; they do not replace one fixture case.

Earlier scratch probes still relevant: `analyze_stage2_pipeline.py`, `scan_tract_quality.py`, `probe_variable_templates.py`, `time_stage2_step.py`, `profile_stage2_step.py`, `probe_branch_arrays.py`, `probe_groupid_tracts.py`, `prototype_branch_tracts.py`, and from this pass `probe_gt_remesh.py`, `probe_smoothing_change.py`, `probe_template_scaffold.py`, `probe_template_rstar.py`, `probe_stretch_orientation.py`, `scan_gt_winding.py`.

---

## 14. Prioritised change list

**P0 — data generation (nothing should be trained before these; the last three change files in `cleandata/`)**

1. ~~GT generator that remeshes the *original* vessel~~ — **done**: `remeshing.py`, verified on SNF00000100 and C0002 (§2.2). Remaining: point `uniformly_remeshed/` at this script, not at `uniform_remeshing.py`; decide workers / `n_iter` for the 682-case run (§2.2.4).
2. Orientation-agnostic (or radially oriented) stretch test in `compute_raycast_stretch_distances` (§2.3); regenerate `template_mesh`; export `R_template` / `StretchDistance` on the final vertices.
3. Carry `GroupIds` / `Blanking` / `CenterlineIds` / `TractIds` through `clip_centerline_at_profiles`; replace the body of `extract_groupid_tracts` with the validated rule; error instead of fallback on `cleandata` (§2.4).
4. One process per case for centerline + GT remesh + template; `n_clipped == n_profiles` as a hard gate on both products; `hascap.csv` exclusions (§2.9).

**P0 — training-side consumption of the template (small, local edits)**

5. Parametrise / pose / tokenise from `original_centerline`; drop or demote `template_centerline` (§2.5).
6. Outward normal orientation by radial direction at every level (§10.2).
7. `r*` from an outward ray-cast (or the exported `StretchDistance`) with validity / ambiguity; edge-based `dth` / `du` / `ring_med` (§7.1, §7.2).
8. `r_local` in `composed_radius`, Chamfer weights and head bounds (§6.7, §7.3).

**P1 — model**

9. Tokens at 2 mm with arc-length-driven counts and a mask; local-pooling transformer latent head; light per-token KL with free bits; noise-robustness curve at validation (§5).
10. θ pseudo-coordinate in physical units; kernel `(5, 5, 2)` or a separate cross-branch conv; `aggr="mean"`, pre-norm, `root_weight=True`; 6–8 convs/level (§6).
11. Free 3-D displacement at coarse; fold and stretch penalties (§6.7, §11); face-sampled Chamfer on the full GT; robust Dirichlet with GT-driven weights (§7).
12. Encoder: normals + template signed distance in; stage-3/4 width down; node features `r_local`, curvature, torsion, ostium distance (§4, §6.6).

**P2 — training framework and evaluation**

13. Resume; per-epoch logging; EMA 0.99–0.995 with warm-up; LR warm-up; wd exclusions; device via env; augmentation set; curriculum; 5-fold harness on the A100s (§8, §9).
14. Metrics of §12 and validity checks of §11 in `postprocess.py`; final isotropic remesh step.
15. Tests of §13.

---

## 15. Open questions

1. **GT smoothing.** Measured in §2.6 and no longer open: the GT's light Taubin (pass band 1.5 × 5) and the template/centerline's strong Taubin (0.1 × 15) differ from each other, and from the raw original, by at most 0.18 mm — less than one GT edge. Either setting is consistent enough; do not match them for its own sake.
2. **Template density at inference.** Variable remeshing needs the GT's stretch field, which Stage 1 cannot provide. Train on uniform templates only (simplest, matches inference exactly), or on variable templates and accept a train/inference density mismatch, or predict a stretch field from the tokens and remesh with it? Decide before regenerating templates after the §2.3 fix.
3. **`template_centerline`.** Drop the folder, or keep it as a per-case consistency check against `original_centerline` (Hausdorff < 0.5 mm)?
4. **GT remesh target edge.** 0.15 mm on SNF00000100 gives 96 918 vertices (C0002: 121 426), 12× the template, 6× `N_TRUE`, 999 s / 570 s per case. Anything finer mostly costs cache size, encoder FPS time and generation wall-time; 0.25 mm was the previous estimate (~30–40 k vertices) and would still be denser than the template. Confirm 0.15 mm is the number so `N_TRUE` and the encoder input can be sized to it (§2.2.3). A 10-iteration remesh (VMTK default) is the obvious time/quality trade if 0.15 mm is kept.

---

## Appendix A. Probe artifacts

Everything that is a number in this review was produced by a script in `scratch/` writing into `scratch/uniform_probe/` (gitignored). Rawdata was read, never written.

| artifact | produced by | used in |
|---|---|---|
| `gt_remesh/SNF00000100.vtp` + `_gt_probe.json` | `probe_gt_remesh.py` (vmtk_env) | §2.2 |
| `gt_remesh/C0002.vtp` + log | same | §2.2.2 |
| `v2/SNF00000100_smoothing_change.json`, `v2/original_centerline/`, `v2/template_mesh/` | `probe_smoothing_change.py` | §2.6 |
| `SNF00000100_scaffold_probe.json` | `probe_template_scaffold.py` (aneurysmgnn) | §3, §6, §7 |
| `SNF00000100_rstar.json` | `probe_template_rstar.py` | §2.3, §7.1 |
| `template_mesh/SNF00000100.vtp`, `original_centerline/`, `template_centerline/` | `process_variable_dataset` / `process_centerline_dataset` | §2.3–§2.5 |
| `SNF00000100.vtp`, `C0002.vtp` (parent-tube uniform remesh) + `*_distance.json` + `*_uniform_vs_original.png` | `probe_uniform_gt.py`, `render_uniform_probe.py` | §2.2 contrast |
| `test_remeshing_log.txt` | `run_test_remeshing_nopytest.py` | §2.2.4, §13 |

