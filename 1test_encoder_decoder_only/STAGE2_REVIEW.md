# Stage 2 (latent → mesh texture) — architecture, data and training-framework review, expanded edition

State reviewed: HEAD `e7fd7ce` ("unextend"), i.e. `d0fdc1b` ("template integrated in") plus `cc62209` ("general remeshing algorithm") and `e7fd7ce`. Files: `train_pipeline/{config, cleaned_io, dataset, raycast, geometry, ops, model, losses, train, aneuxai, test_architecture}.py`, `postprocess.py`, `aneux_paths.py`, the `cleandata/` layout (four folders, reduced to three by §15 item 3), and the generators in `datatransform/template_creation/` (`vessel_pipeline.py`, `centerline_creation.py`, `uniform_remeshing.py`, `variable_remeshing.py`, and the new `remeshing.py` + `test_remeshing.py`). No pipeline code was changed by the review; every script written for it lives in `scratch/`.

What changed between the previous review (`d0fdc1b`) and this one:

- `remeshing.py` is new: a ground-truth generator that remeshes the *original* vessel (aneurysm kept) at a constant 0.15 mm edge length, cuts the ostia perpendicular to the centerline, and gates the result on area. It is verified below on two real cases (§2.2). This closes item 1 of the previous P0 list.
- `vessel_pipeline.py`: `apply_taubin_smoothing` default `pass_band` changed from 1.0 to 0.1 (stronger smoothing, silently affecting `build_parent_tube` and `compute_centerline_from_mesh`, measured in §2.6); new `remove_spurious_openings`, extension-length-aware `_opening_clip_height`, `clip_flow_extensions_and_uncap` now also drops islands and fills leftover rims; `compute_centerline_from_mesh` extracted from `process_centerline_dataset`.
- Nothing in `train_pipeline/` changed. `clip_centerline_at_profiles` (cell arrays dropped) and `compute_raycast_stretch_distances` (orientation-dependent) are unchanged, so P0 items 2 and 3 of the previous list are still open, and everything the previous review measured about the training side still holds and was re-verified where a fresh file made that possible.
- §5 (latent) is rewritten: the encoding decision is examined against the literature and taken (§5.3) — a stochastic encoder kept as a rate-controlled channel, with the specification in §5.3.6 and the measurements that set its free parameters in §5.3.7. The rest of the document is aligned to it.
- `cleandata/` was regenerated (709 GT meshes and centerlines), which allowed the last guessed number in §5 to be measured instead: the latent width. §5.3.3a estimates the degrees of freedom of one 2 mm token from 38 689 measured wall patches by four independent methods and fixes `LATENT_DIM = 16`; §5.3.5 replaces the single-case heterogeneity argument with the population table; §5.3.8 closes option F as a replacement. The same pass measured the centerline duplication of §2.4 directly on all 709 delivered files (§2.4.2a).

Decisions taken (Damján, 9 Sep) that this review builds on: training reads only `cleandata/`; `uniformly_remeshed/` is the *original vessel with the aneurysm*, finely remeshed, and is the GT for everything; templates come from `variable_remeshing.py` and are the decoder's identity surface; there is no `coarse_remeshed` — mid/coarse are decimations of the template; Stage 1 will emit the centerline at **1 mm** with a texture token every **2 mm**, along the *entire* tree; output must be a single watertight manifold with open outlets (CFD), outlet position/planarity fixed in post; HPC wall-time effectively unlimited, accuracy over speed; manual per-case work acceptable; aneurysm removal (hemoMesh) is a later topic and is not treated here. Encoding (13 Sep): keep a stochastic encoder, but as a rate-controlled channel with explicit standardisation rather than as a generative prior (option C of §5.3.4); strong regularisation toward N(0, I) is not a goal. Latent width (20 Sep): `LATENT_DIM = 16`, set by the measurement of §5.3.3a rather than by the sweep alone; the sweep now confirms it over D ∈ {4, 8, 16, 32}. Also 20 Sep, closing §15: templates keep **variable** density and Stage 1 predicts the stretch factor (so the §2.3 fix is load-bearing and `StretchDistance` must be exported); the template's ostia are cut in the **GT's** planes and rim vertices are constrained to those planes in the decoder (§2.2.4, §11); `template_centerline` is dropped; the GT edge target is 0.15 mm (VMTK delivers ≈ 0.12); no Fourier hybrid; the case-to-case interpolation landmark is deferred.

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
   - `template_centerline` is measurably the same curve as `original_centerline` (Hausdorff 0.12 mm, MISR within 0.06 mm); the training code reads the original and then ignores it in favour of the template's. The folder is now dropped and the parametrisation moves to the original (§2.5, §15 item 3).
   Measured and good: the fine level *is* the posed template (identity displacement 0.0 at init); decimation kept 3 boundary loops, 1 component, 0 non-manifold edges at every level (7 909 / 1 980 / 638 vertices at 0.41 / 0.89 / 1.56 mm); kNN upsampling has no opposite-wall mixing (0.0 % / 0.04 %); template normals point outward on 99.8 % of vertices (by luck of winding, not by construction).
5. **The Taubin default change is benign but should be known.** Windowed-sinc smoothing at pass band 0.1 × 15 iterations moves the SNF00000100 wall by 0.041 mm median / 0.178 mm max versus 0.032 / 0.092 mm at the old pass band 1.0 (§2.6). The regenerated centerline differs from the old one by ≤ 0.14 mm and the MISR by ≤ 0.06 mm; the regenerated template by ≤ 0.29 mm (mostly remesh noise). Because the GT uses an even lighter setting (1.5 × 5 iterations), the "should the GT be smoothed like the template" question of the previous review (§15.1) is answered by measurement: the two smoothing levels differ by less than one GT edge length.
6. **Latent: decision taken; decoder kernel and training hygiene findings carry over unchanged** (§4–§8). The 96 × 128 = 12 288-dim latent is a near-autoencoder with a KL weight that contributes 0.3 % of the objective — but the weight is not the main defect: validation never samples, the log-variance clamp allows σ = 0.018, the latent mixer runs after sampling and can average the noise away, and tokens are not local (§5.1). Neither Stage-1 diffusion nor latent interpolation needs a latent pushed to N(0, I) (§5.3.1–5.3.2); the real regularisation problem is ~20 latent dimensions per training case (§5.3.3). Decision: a light, rate-controlled VAE — `LATENT_DIM` 128 → **16**, soft σ floor 0.1, per-token rate target with an adaptive β, mixer before sampling, sampling at evaluation, post-training standardisation with the inactive dimensions dropped first, and a noise-robustness curve as the Stage-1 contract (§5.3.6–5.3.7). The width is measured, not guessed: over 38 689 wall patches from all 709 cases, one 2 mm token's residual field has a calibrated intrinsic dimension of 8–9 under a sac and 11–13 on healthy wall, needing 9–14 and 4–6 dimensions respectively to reach 0.1 mm, while one extra token costs ≈ 1 dimension (§5.3.3a). The same patches show the heterogeneity that justifies the stochastic encoder — a sac token's residual is 12× wider in σ and 30× larger in amplitude than a healthy one, on ≈ 5 of ≈ 56 tokens per case (§5.3.5) — and rule out the analytic Fourier alternative on sacs (§5.3.8). Other carried-over findings: θ-blind SplineConv pseudo-coordinates on the template (ring neighbours land at `e_th = 0.5 ± 0.035`); `aggr="add"` without normalisation; no augmentation, no resume, `print` logging, EMA horizon 47 epochs. With the 2 mm token contract now fixed, `LATENT_LEN = 96` *fixed* slots contradicts it — on this 95 mm case the spacing is 2.02–2.04 mm by coincidence, on a 50 mm vessel it would be 1 mm.

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
                              → cleandata/template_centerline/{id}.vtp  (≈ original_centerline — folder dropped, §15 item 3)

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
| `template_mesh` | parent-tube template from `variable_remeshing.py`, **variable density** (§15 item 2), ostia cut in the GT's planes (§2.2.4) | parent tube, "adaptively" remeshed — **adaptivity inactive on inward-wound inputs** (§2.3), so the delivered templates are uniform | fine identity surface; mid/coarse by decimation |
| `original_centerline` | `centerline_creation.py` on the original | polylines, one per inlet→outlet path; MISR kept; **branch cell arrays dropped** (§2.4) | read, then **unused** when a template exists |
| ~~`template_centerline`~~ **dropped** (§15 item 3) | `centerline_creation.py` on the template | the same curve as `original_centerline` to within 0.12 mm (§2.5) | *was*: tracts, canonical pose, (u, θ), tokens — these move to `original_centerline` |

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
2. **Ostium mismatch between GT and template.** The GT is cut on frames from *its own* centerline (§2.2.1 step 3), the template on frames from `build_parent_tube`'s centerline; both are pipe-section cuts perpendicular to the local tangent, but the centres differ by 0.15 / 0.15 / 0.58 mm and the GT radii are 0.06–0.22 mm larger because the polyball's MISR radius is the *inscribed* radius, smaller than the true rim of a non-circular cross-section. For the decoder this means the boundary rows of the template are 0.1–0.7 mm inside the GT rim and have to move outward and along the tangent to match — the Chamfer will ask for that, and nothing currently keeps boundary vertices in their cut plane.

**Decision (Damján, 20 Sep): do both halves.**

- *Generator side — the template's ostia are cut in the GT's planes, exactly.* `remeshing.py` already computes the cut frames from its own centerline (§2.2.1 step 3); those frames (origin + normal + radius per profile) become an output of the GT step and the **input** to `clip_flow_extensions_and_uncap` on the template, instead of the template pipeline measuring its own profiles. The centre offset (0.15 / 0.15 / 0.58 mm) and the plane-normal difference then vanish by construction; the radius difference (GT rim 0.06–0.22 mm wider than the polyball's inscribed MISR) remains and is exactly the residual the decoder is for. Performance: this must *not* cost a second VMTK pass or a second centerline. Both products are already to be generated in one process per case (§2.9, §10.2 step 1), so the frames are passed in memory; the template's own `measure_open_profiles` call on the clipped ends is then replaced by, not added to, the shared frames. Persist the frames per case (three planes × 7 floats) so a template can be regenerated without re-running the GT.
- *Decoder side — rim vertices stay in their plane.* The boundary-plane constraint of §11 becomes mandatory rather than optional: a vertex on a boundary loop may slide within its cut plane but may not leave it. With the planes now shared with the GT this is a constraint toward the *correct* plane, which is what makes the two halves worth doing together — constraining rim vertices to a plane that is 0.58 mm from the GT's would fight the Chamfer instead of helping it.
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

Since the decision of §15 item 2 is **variable templates with a Stage-1-predicted stretch**, step 8 is not an optional refinement: it is both the reason the templates have their density and the source of the field Stage 1 has to learn. The orientation defect below therefore blocks template regeneration, and `StretchDistance` / `TargetEdgeLength` must be exported on the *final* template vertices, after remeshing, so Stage 1 has a target defined where it will have to predict one.

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

#### 2.4.2a The same duplication measured on every `cleandata` centerline (`scratch/latent_dim/`)

The scaffold probe above reads the defect through `dataset.py`. It is also directly visible in the delivered files: `cleandata/original_centerline/` stores raw VMTK paths, one per outlet, so the trunk is repeated once per path. Over all **709 files**:

| quantity | p5 | p50 | p95 |
|---|---|---|---|
| polyline cells as stored | 6 | 22 | 68 |
| total stored length (mm) | 138 | 335 | 739 |
| **disjoint tracts after dedup** | **3** | **6** | **10** |
| **actual tree length (mm)** | **60** | **120** | **216** |

The stored length exceeds the true tree length by a factor of 2.8 at the median and 4.4 at p95, and **704 of 709 files are affected** (the five that are not are single-outlet segments). Any per-tract or per-token quantity computed from these files without dedup is counted three to ten times over, and any nearest-centerline-point lookup is decided by numerical noise between coincident copies — which is what §2.4.2 sees from inside `dataset.py`. The measurement in §5.3.3a therefore deduped first, by walking the cells longest-first and keeping from each only the longest contiguous run not already covered by an accepted tract. That stopgap recovers the tree geometry but not `GroupIds`, `Blanking` or the junction semantics, which is why §2.4.3 remains the fix.

#### 2.4.3 The fix, both sides

Generator: in `clip_centerline_at_profiles`, carry the cell arrays — either keep the (group, blanking) cell structure and copy each kept cell's tuple, or convert cell data to point data before the rebuild (`vtkCellDataToPointData`, then the existing point-array copy handles it). One function, ~10 lines.

Training: the validated rule from the previous review (`prototype_branch_tracts.py`), which `extract_groupid_tracts` should implement instead of its current body:

1. group polyline runs by `GroupIds`; **keep the longest copy per GroupId** (the parent appears once per path);
2. **attach each blanked run to the daughter that follows it** (so daughters start at the bifurcation, not 1.6 mm after it — real files have ~1.6 mm blanks, the synthetic tests 0.4 mm);
3. snap tract endpoints at **1 mm** (`GROUPID_ENDPOINT_SNAP_MM`) to rebuild the tree.

On SNF00000100 this gives 3 tracts (88.8 / 13.3 / 12.9 mm), 1 junction, 100 % coverage, ≤ 1 % overlap. And the silent fallback must become an error for `cleandata` inputs: a real centerline without `GroupIds` is a generator bug, not a synthetic test.

### 2.5 `template_centerline` versus `original_centerline`

Same case, both from `process_centerline_dataset`: 1 922 vs 1 919 points, 6 cells each; Hausdorff 0.12 mm, p95 0.06 mm; MISR difference p50 +0.011 mm, max 0.062 mm. This is expected: the template is the polyball of the original centerline, so its Voronoi centerline is that centerline again, up to the smoothing and marching-cubes noise.

Consequences. (i) The folder buys nothing for training. (ii) `_build_data` reads `original_centerline` (it raises if it is missing) and then passes `template_cl` to `build_scaffold`, ignoring the original. (iii) At inference the natural input is the Stage-1 centerline; routing it through `build_parent_tube` → template → `centerline_creation.py` → tracts adds a VMTK pass and a second, slightly different curve on which the tokens then live. Decision (20 Sep): parametrise, tokenise and pose from `original_centerline` in training and from the Stage-1 centerline at inference, and **drop the `template_centerline` folder** — `process_centerline_dataset` stops writing it, `aneux_paths.py` / `cleaned_io.py` stop expecting it, and `cleandata/` becomes three folders (§15 item 3).

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
2. **The latent head is thin.** One head, one layer, no feed-forward block, no LayerNorm, positional queries only (the query knows *where* it is, `γ(u)` and tract, but not *what* is there until the softmax has mixed the 64 values). Angular information (θ) is not in the query or key at all — it can only arrive through the content of `h`. On a 2 mm-token contract with ~50 tokens per case, the head should be a small transformer: assign each of the 64 (or more) centres to its nearest token along the tree, pool per token (attention or max), then 2–4 self-attention layers over the token *tree* with arc-length and branch-depth positional encodings (§5.4). This is what makes tokens local, which §5.2 requires and no KL setting provides (§5.1 item 4).
3. **Capacity is in the wrong place.** 12.6 M parameters process 64 points at stage 4; the head that has to produce 96 × 128 latent values has 1.2 M (≈ 56 × 16 under §5.4 and §5.3.6). Halve stage 4 (256 wide) and give the head the difference.
4. **`logvar.clamp(-8, 2)`** is a hard clamp: outside the range the gradient to `logvar_head` is exactly zero, so a token whose log-variance drifts past −8 (posterior collapsed to a point) or 2 stops learning its variance. Replace it with the soft bound of §5.3.6 item 2, `log σ²_min + (log σ²_max − log σ²_min) · sigmoid(·)` with σ_min = 0.1 and σ_max = e: the gradient never dies, and the floor stops the posterior collapsing to a deterministic code (§5.1 item 2).
5. **Batching.** With `n_graphs > 1` the head loops over graphs in Python (`for g in range(n_graphs)`); fine at batch 4–8, but the per-graph GEMM pattern means batching does not buy throughput in this module.

---

## 5. Latent space and VAE

### 5.1 As built

96 tokens × 128 dims = **12 288** latent dimensions per case (`config.py` 67–69: `LATENT_LEN = 96`, `LATENT_DIM = 128`, `LOGVAR_CLAMP = (-8.0, 2.0)`). `vae_kl_loss` (losses.py 35–43) sums the KL over the 128 dims and averages over tokens (and batch); at init on the real sample KL = 7.13 → weighted by `LAMBDA_KL = 5e-4` (after a 20-epoch linear warm-up, `kl_anneal_weight`, train.py 90) → **0.0036** in the objective against recon 1.315. That is 0.3 % of the loss: the posterior is free to be arbitrarily sharp and the latent to have any scale — functionally an autoencoder with a noise term whose strength nobody chose and nothing measures.

The weight is not the main problem. Five properties of the code decide what the latent becomes, and none of them is a hyperparameter:

1. **Validation never samples.** `reparameterize` (model.py 965–969) returns `mu` whenever `self.training` is False, and `evaluate_epoch` (train.py 308) calls `model.eval()`. Validation recon, `best.pt` selection (val recon + rad) and the EMA evaluation therefore all run on the deterministic path. The one property the posterior noise exists to buy — that a *neighbourhood* of each code decodes to a sensible wall, i.e. robustness to Stage-1 error — is never observed, so no setting of λ can be validated against it.
2. **The log-variance clamp permits a deterministic code.** `LOGVAR_CLAMP[0] = −8` means σ can reach e⁻⁴ = 0.018. At small σ the per-dimension KL grows by ≈ 1 nat per nat of `log σ`; summed over 128 dims and weighted by 5e-4 that is **0.064 of objective per nat of `log σ`**. Sharpening every dimension from σ = 1 to the clamp (Δ`log σ` = −4) costs ≈ 0.26 against a recon of 1.315 — a price the reconstruction gradient will pay on ~620 cases with 12 288 latent dimensions. Preventing it by λ alone needs λ ≈ 2e-3, at which point merely *using* the latent at unit scale (mean μ² = 1) costs 2e-3 × 128 × ½ = 0.13, ~10 % of recon: the steep β-VAE trade. The clamp is also hard, so the gradient to `logvar_head` is exactly zero outside it (§4.2 item 4).
3. **The latent mixer runs after sampling.** `forward` (model.py 980–983) is `mu, logvar = encode(data); z = reparameterize(mu, logvar); z_dec = z_attn(z, data)`, and `decode` (974–978) applies `z_attn` too. `LatentTractSelfAttention` (model.py 560) mixes ±2 tokens (`Z_ATTN_RADIUS = 2`) with an ALiBi penalty `8·|Δu|` on the *normalised* arc fraction — on a 48-token tract that is 0.17 for adjacent and 0.34 for ±2 tokens, i.e. a near-uniform five-token window. Over a signal that is smooth in `u` and noise that is i.i.d. per token, that window is a denoiser. The gate (`Z_ATTN_GATE_MAX = 0.5`, zero-initialised output) bounds how much of it the model can use but does not remove the path: **the σ the KL prices is not the σ the decoder faces.**
4. **Tokens are not local by construction.** `CenterlineLatentHead` (model.py 272–317) computes each token as `softmax(q·kᵀ/√d)·v` over all 64 encoder centres of the case, with a query built from `γ(u)` and the tract embedding only (line 301). Nothing stops the token at `u = 0.2` from summarising the wall at `u = 0.9`. A KL shapes each token's marginal; it says nothing about *which part of the surface* the token describes. Locality is an architecture property (§4.2 item 2, §5.4).
5. **The logged rate is per token, not per case.** Because the KL is averaged over tokens, 7.13 nats is one token's share; the per-case rate at init is 7.13 × 96 = 685 nats ≈ **988 bits**, and nobody logs it.

On top of these, the coincident-tract layout of §2.4 splits the 96 tokens 48/48 over two copies of the same parent and hides each half from half the vertices, so the latent currently encodes the same wall twice: any rate measured today is ≈ 2× inflated, and any λ calibrated today is calibrated against a token layout that has to change.

### 5.2 What Stage 1 needs from the latent

Stage 1 emits, per centerline sample at 1 mm, `[x, y, z, r]` and, on every second sample, a texture token `z_1 … z_D`, along the whole tree, generated by a conditional diffusion model. Two downstream uses are planned for the latent: that generation, and sweeping the latent — in particular interpolating between a healthy and an aneurysmal wall. For both to work and for Stage 2 to decode the result faithfully, the latent should be:

- a tree-structured sequence of **local** tokens — each token describes the wall near its position, so Stage 1 can predict it from local conditions;
- of **fixed, known per-dimension scale** — so Stage 1's diffusion/regression loss is well conditioned and isotropic;
- **low-dimensional relative to the data** — so a diffusion model trained on ~620 cases generalises instead of retrieving training cases;
- **smooth in the decoder's sense** — codes near the training codes decode to sensible walls, which requires the decoder to have been trained on a neighbourhood around each code;
- with a **known null code for healthy wall** — so Stage 1 only models where the sac is and what it looks like, the rest of the tree emits the null code, and "healthy" is a fixed endpoint for interpolation;
- **robust to Stage-1 error**, with that robustness *measured* (the noise-robustness curve of §5.3.7 is the contract).

Note what is *not* on the list: that the aggregate posterior equal N(0, I). §5.3 argues that neither diffusion nor interpolation needs it, and that paying for it costs reconstruction and diffusion capacity.

### 5.3 Encoding choice: VAE, autoencoder, or something else — trade-off and recommendation

The VAE was chosen for two reasons: with this few samples a diffusion model in Stage 1 should find a regularised latent easier to model, and a well-regularised latent was thought necessary for latent sweeps (healthy ↔ aneurysmal interpolation). Both reasons are examined below, because the encoding is one of the decisions everything downstream inherits. The conclusion is to **keep a stochastic encoder, but as a rate-controlled channel rather than as a generative prior** (option C) — for a reason specific to this architecture (§5.3.5), not for either of the two original ones.

#### 5.3.1 Does a diffusion model need a strongly regularised latent?

No — and taken literally the argument works against itself. If the KL really made the aggregate posterior N(0, I), Stage 1 would have nothing to learn: sampling the prior would already be the generator. Every nat of structure the KL removes is structure the diffusion model exists to model, bought with reconstruction error.

This is also where latent-diffusion practice has converged. Rombach et al. (LDM) regularise their autoencoder with a "slight KL-penalty towards a standard normal", with a weight of 1e-6 in the released configurations, explicitly *to avoid arbitrarily high-variance latent spaces* — not to make the latent Gaussian — and then rescale the latent by a fixed factor so the diffusion model sees roughly unit variance. At such weights the KL constrains the *scale* of the latent and has no meaningful effect on its *shape*; an unscaled KL severely limits capacity and degrades reconstruction (Dieleman 2025). What a diffusion model needs from a latent is bounded and standardised scale, smoothness/structure along the latent's own index, and a dimensionality it can model with the data it has.

For this project those translate into:

- **scale** — an explicit post-training standardisation pass (§5.3.6 item 7). Cheap, exact, and independent of the KL weight;
- **structure** — the tokens sit at 2 mm along a centerline and describe overlapping wall, so neighbouring tokens are correlated by construction; locality is enforced by the encoder head and decoder cross-attention (§4.2 item 2, §5.4), not by the KL;
- **dimensionality** — §5.3.3.

The first original reason therefore reduces to requirements a *light* KL plus standardisation satisfies; a strong KL adds nothing to them and removes capacity.

#### 5.3.2 Does interpolation need a strongly regularised latent?

Also no, for three reasons — and the architecture offers a better instrument for the specific sweep that motivated it.

**(a) Linear interpolation leaves a high-dimensional Gaussian shell no matter how well the latent is regularised.** For z₁, z₂ ~ N(0, I_D) the midpoint has variance ½ per dimension, so its norm concentrates at ≈ √(D/2) while typical codes concentrate at ≈ √D, with a shell thickness of ≈ 0.71 (arithmetic, not measured):

| D | E‖z‖ | ‖midpoint‖ | midpoint distance below the shell |
|---|---|---|---|
| 8 | 2.74 | 1.87 | 1.2 σ |
| 16 | 3.94 | 2.74 | 1.7 σ |
| 24 | 4.85 | 3.39 | 2.1 σ |
| 32 | 5.61 | 3.94 | 2.4 σ |
| **128 (as built)** | **11.29** | **7.97** | **4.7 σ** |

At D = 128 the midpoint of a lerp between two *perfectly* Gaussian codes lies 4.7 shell widths inside the region the decoder was trained on. A stronger KL does not fix this; spherical interpolation (slerp, White 2016) and a smaller D do.

**(b) Interpolation quality is a property of the decoder.** A sweep is meaningful when the latent support is connected and the decoder varies smoothly over it. The second condition comes from training the decoder on a neighbourhood of each code — the posterior noise — and is the same property as robustness to Stage-1 error. It is set by σ, not by KL(q(z) ‖ p(z)).

**(c) Healthy wall is already a fixed point of the model.** The template is the healthy vessel; the healthy wall corresponds to the identity displacement, which a null code must produce. "Healthy → aneurysmal" is therefore the ray `z(t) = t · z_case`, t ∈ [0, 1], anchored at a known endpoint. That requires the **null-code** property (a rate property: tokens that carry nothing sit at the prior mean) and decoder smoothness — not a Gaussian aggregate posterior. For figures of a sac "growing" out of a healthy ICA there is a more direct instrument still: interpolate the decoded displacement field, `t · (Δr, Δs)`, which is monotone, physically interpretable and watertight by construction because the template connectivity never changes.

**(d) Case-to-case interpolation needs a shared scaffold.** Two vessels have different trees and different token counts, so `z_A → z_B` is defined only after the tokens of B are expressed on A's scaffold — resampled by arc position relative to a common landmark (the bifurcation, or the neck centre). This is a correspondence problem, not a regularisation problem, and it must be solved whatever the KL.

The second original reason therefore survives in a reduced form: it requires a null code, decoder smoothness, and a moderate D — all of which option C provides.

#### 5.3.3 The real regularisation problem: dimensionality

The regularisation concern behind the VAE choice is legitimate; the KL is the wrong instrument for it. With ~620 usable cases (from `clinical.csv` and the vessel models: every anterior-circulation location tag has the ICA in the model, while `BA`, `BA tip`, `SCA`, `PICA`, `VA V4` and the single `ACA dist` case do not, and `PCA P1-P2` / `MCA M2` are mixed — ~686 of 750 rows on ~620 unique meshes):

| latent layout | dims per case | dims per training case |
|---|---|---|
| **96 × 128 (as built)** | **12 288** | **19.8** |
| 56 × 32 | 1 792 | 2.9 |
| 56 × 24 | 1 344 | 2.2 |
| **56 × 16 (decided, §5.3.3a)** | **896** | **1.4** |
| 56 × 8 | 448 | 0.7 |

(56 tokens per case is the measured 2 mm token count over the whole tree, §5.3.3a, not an estimate.)

Diffusion models trained on few examples relative to their target dimensionality reproduce training examples (Carlini et al. 2023; Somepalli et al. 2023), and here the conditioning — a full centerline and a condition vector — is so informative that nearest-training-case retrieval is close to optimal for most of the tree. A 12 288-dimensional Stage-1 target invites exactly that. The instruments that regularise, in order of effect:

1. **reduce `LATENT_DIM`** (§5.3.6 item 1) — it improves reconstruction per parameter, interpolation (table above), diffusion tractability and memorisation resistance at once;
2. **augmentation** — absent today (§8); left/right mirroring (the `side` column), rigid pose jitter, mild centerline perturbation;
3. **the template-plus-residual decomposition itself** — by far the strongest regulariser in the design, since calibre, course and topology never enter the learning problem;
4. **the geometric losses** once they are actually active (§7.1, §7.2).

Rate control (option C) is the fifth item, and its value is less "regularisation" than *making the channel's capacity a chosen, logged quantity*. No achievable rate prevents a latent from carrying enough information to identify a training case (that needs only log₂ 620 ≈ 9.3 bits); memorisation is guarded by D, augmentation, and held-out evaluation (§5.3.7), not by the KL.

#### 5.3.3a How wide a token has to be, measured on all 709 cases (`scratch/latent_dim/`)

`LATENT_DIM` was the one number in this section that was a guess. It is now measured, without training anything, on the field the decoder actually has to produce.

**Method.** For every case in `cleandata/` (709 GT meshes + centerlines; `template_mesh/` is *not* needed, because the template's radius *is* the centerline MISR): dedupe the centerline into disjoint tracts (§2.4.2a), cut each tract into 2 mm tokens, and sample each token's wall on a 4 × 16 grid — 0.5 mm in `u` × 22.5° in θ — taking the outward distance from the centerline to the GT surface per bin and subtracting MISR(u). A token is dropped if it touches a mesh opening (any vertex within 1.5 mm of a boundary loop) or if more than 10 % of its bins are empty. Each token is labelled **sac** (≥ 5 % of its vertices within 0.3 mm of the AneuX dome mesh, `aneurysms/original/{id}_dome`), **peri-sac neck** (within 2 tokens of any dome contact), **near-junction** (within 3 mm of an interior tract end) or **healthy**. Cases whose `location` tag does not carry the ICA (`BA`, `BA tip`, `SCA`, `PICA`, `VA V4`, `ACA dist`, and the mixed `PCA P1-P2` / `MCA M2`) are excluded from the statistics. Result: **38 689 token patches from 708 cases**, 35 991 of them on 644 ICA-bearing cases, ≈ 56 tokens per case.

This residual is the single-valued radial field seen from the centerline (max per bin), which is exactly the class of shape a template + Δr-along-normal decoder can express (§6.7); overhanging domes are truncated the same way the decoder truncates them. Tangential shear Δs is not measured. MISR is itself inflated inside a sac, so sac residuals are conservative.

**Four estimators, chosen because they fail in different ways** (`analyze2.py`–`analyze4.py`):

| estimator | healthy token | sac token |
|---|---|---|
| linear PCA, held-out, case-level 5-fold: K for RMS ≤ 0.1 mm | 12–16 | **> 32** (K = 32 → 0.364 mm) |
| nonlinear parametric fit: params → median RMS | 4 → 0.074 mm; 9 → 0.036 mm | 9 → 0.389 mm; 14 → 0.221 mm |
| intrinsic dimension (TwoNN / Levina–Bickel MLE), calibration-corrected | **11–13** | **8–9** |
| marginal cost of one more 2 mm token (dimension growth over windows of 1–7 tokens) | ≈ 0 | ≈ +1 |

Three things make that table readable:

1. **The linear number is an artifact, not a result.** A sac is a bump whose angular position, width and height vary from case to case — a translation manifold, which no linear basis compresses. The θ-Fourier spectrum says the same from the other side: reproducing a sac token to 0.17 mm RMS needs harmonic order 7 of the 8 available. Since the decoder is nonlinear, the PCA figure is only a ceiling. It is, however, decisive against option F (§5.3.4, §5.3.8).
2. **The intrinsic-dimension estimators were calibrated** on synthetic smooth manifolds of known dimension at matched sample sizes, because both saturate: this implementation of TwoNN runs ≈ 30 % high (true 8 → 11.3, true 16 → 20.3, true 32 → 30.2) and the Levina–Bickel MLE at k = 10 is near-unbiased to about 12, then compresses (true 16 → 14.0, true 32 → 20.0). The raw sac values (TwoNN 11.5, MLE 7.8) both map back to **≈ 8–9**; healthy (17.5 / 10.5) to **≈ 11–13**.
3. **The degrees of freedom are intrinsic, not an artifact of the grid.** Re-extracting 178 cases at 32 angular bins (patch dimension 64 → 128) moves the estimate by ≤ 1.5 dimensions in both groups. The field is a smooth low-dimensional family sampled more finely, not a richer one.

Healthy tokens have the *higher* intrinsic dimension yet need far fewer dimensions at the accuracy that matters: 4 nonlinear parameters — calibre, axial taper and two eccentricity terms — already reach 0.074 mm median RMS, and the remaining degrees of freedom live below 0.05 mm. Dimension counting has to be tolerance-aware. At a 0.1 mm target: **healthy ≈ 4–6, sac ≈ 9–14**.

**Decision: `LATENT_DIM = 16`.** It covers the calibrated estimate for both groups (8–13) with headroom for the irregular sac tail, which is where a tight D fails first. D = 8 sits exactly on the sac estimate with no margin and below the healthy intrinsic dimension — defensible, and the value most likely to force a retrain. D = 24 is above everything measured; D = 128 is off by an order of magnitude. Because one additional token costs only ≈ 1 dimension, the per-token width is genuinely a *ceiling* rather than a budget: choose 16, let the rate target (§5.3.6 item 3) do the squeezing, and read the width that was actually used off the active-unit count on sac tokens (§5.3.6 item 7).

One limit of the measurement constrains the *decoder*, not the latent: the sac field has angular gradients steeper than one 22.5° bin (several mm between adjacent bins at the neck rim). That is a statement about output mesh density, not about latent width.

#### 5.3.4 The options compared

| | A. as built (vestigial KL) | B. β-VAE, meaningful β | **C. light, rate-controlled VAE + standardisation (recommended)** | D. AE + fixed σ + standardisation | E. VQ latent + discrete prior | F. analytic cross-section descriptors |
|---|---|---|---|---|---|---|
| reconstruction | maximal | costs accuracy; steep with ~620 cases | near-maximal: rate goes where recon needs it | maximal | good; quantisation error | limited by basis order |
| latent scale | uncontrolled, unlogged | unit by construction | unit after the standardisation pass | unit after the pass | n/a (indices) | exact, physical |
| null code for healthy wall | none | yes | yes, and the per-token KL profile localises the sac | no pressure toward one | one codebook entry, if learned | exact: all coefficients 0 |
| robustness to Stage-1 error | unmeasured; collapses toward σ = 0.018 | built in | learned per token, floored, **measured** | one hand-tuned σ for all tokens | discrete: robust to small errors, brittle to wrong indices | analytic: decoder is the basis |
| fit for Stage-1 diffusion | poorly conditioned, 12 288 dims | wastes diffusion capacity | well conditioned, structure preserved | well conditioned | needs discrete diffusion / autoregression | best conditioned |
| interpolation | undefined scale | good (slerp) | good (slerp; ray from null code) | σ-dependent | poor (discrete) | exact, interpretable |
| tuning burden | — | β (steep) | one rate target (dual variable adapts β) + σ floor | σ | codebook size, commitment, dead codes | basis order only |
| main failure mode | silent collapse to an AE | blurry sacs | rate target set too low → blurry sacs (visible in the sweep) | σ too small → brittle; too large → blurry | codebook collapse | non-star-shaped sacs not representable |

Option F — each 2 mm token a truncated Fourier series of the local cross-section radius in θ (≈ 17 numbers at order 8) — is attractive for a dataset this size: exactly local, fixed scale, exact null code, interpretable, and no learned encoder to memorise. Its limit is that a large sac with a narrow neck is not single-valued in `r(θ)` about the centerline. The current decoder makes a related commitment (Δr along the template normal plus ≤ 3 mm shear, §6.7), so F is only moderately more restrictive. **That measurement has now been made** (§5.3.3a): truncating the measured sac residual at harmonic order 8 leaves 0.39 mm RMS and 2.9 mm at the p95 of per-token maximum error, and order 12 still leaves 0.22 mm RMS. A truncated Fourier series is a linear basis and a sac is a localised bump — the wrong basis for it. F is ruled out as a *replacement* at any order small enough to be worth having, and survives only as the hybrid of §5.3.8.

#### 5.3.5 Why a stochastic encoder is still the right choice here

The decisive argument is not generic. In a template-plus-residual model **the information carried by a token varies by an order of magnitude along the token index**:

- over healthy wall the template already explains the geometry; the token carries essentially nothing, and the right posterior is *wide* (σ ≈ 1, centred at the prior mean) — a large region of codes all decoding to "template, unchanged", which is what makes the null code and Stage 1's job on most of the tree easy;
- under the sac the token carries the outward displacement — 4.1 / 11.8 / 24.9 mm (p50 / p90 / max of the outward ray to the GT over the 304 sac vertices of SNF00000100, §7.1) — and the posterior must be *narrow* or the sac blurs.

The imbalance is now measured across the dataset rather than on one case (§5.3.3a; 2 mm tokens, residual against the MISR template, 644 ICA-bearing cases):

| token group | n | per case | residual σ (p50 / p95) | max residual (p50 / p95) |
|---|---|---|---|---|
| healthy | 24 108 | 37.4 | 0.092 / 0.315 mm | 0.35 / 1.14 mm |
| healthy at a junction | 6 768 | 10.6 | 0.093 / 0.611 mm | 0.36 / 2.44 mm |
| peri-sac neck | 2 552 | 4.8 | 0.089 / 0.565 mm | 0.33 / 2.97 mm |
| **sac** | **2 563** | **4.7** | **1.114 / 2.856 mm** | **4.24 / 10.25 mm** |

A sac token's residual is **12× wider in σ and 30× larger in amplitude** than a healthy one, and only ≈ 5 of ≈ 56 tokens per case are sac tokens. One global σ has to serve both ends of that range; a learned per-token σ does not.

A fixed σ (option D) cannot serve both: σ = 0.3 blurs sacs, σ = 0.05 gives healthy tokens no coverage and no null code. A learned per-token, per-dimension σ with a floor is the right instrument for a latent whose informativeness is this heterogeneous. As a by-product the per-token KL along the tree becomes a diagnostic of where the template fails — ≈ 0 on healthy wall, large under the sac — which is independently useful for the private patient data, where dome annotations will not exist.

#### 5.3.6 Option C, specified

The recipe is the latent-diffusion one — weak KL for scale, explicit standardisation for the diffusion model — plus what this architecture needs: a σ floor, a rate that is controlled rather than inherited, a null code, and a mixer that cannot launder the noise.

1. **`LATENT_DIM` 128 → 16**, from the measurement in §5.3.3a (calibrated intrinsic dimension 8–13 per 2 mm token; 4–6 dimensions reach 0.1 mm on healthy wall, 9–14 on a sac), confirmed by the rate–distortion sweep of §5.3.7 over D ∈ {4, 8, 16, 32}. With the measured ≈ 56 tokens per case under the 2 mm contract (§5.4) the Stage-1 target becomes ≈ 900 dimensions.
2. **Soft σ floor.** Replace the hard `clamp(-8, 2)` in `CenterlineLatentHead` with a smooth bound, `logvar = log σ²_min + (log σ²_max − log σ²_min) · sigmoid(raw)`, with σ_min = 0.1 (log σ² = −4.61) and σ_max = e (log σ² = 2). The floor keeps noise in the channel however strong the reconstruction pull, which decouples "keep a neighbourhood around every code" from "squeeze μ"; the smooth bound removes the dead gradient of §4.2 item 4. The KL of a dimension sitting at the floor with μ = 0 is ½(0.01 − 1 + 4.61) = 1.81 nats — the model will only go there on dimensions that pay for it in reconstruction.
3. **Controlled rate instead of a fixed λ.** Keep the per-token KL sum over dimensions, and control its mean over valid tokens with a Lagrangian multiplier (GECO-style, Rezende & Viola 2018, with the constraint on rate instead of distortion): `β ← clip(β · exp(η · (KL̄_raw − R*)), β_min, β_max)`, updated once per optimiser step, with `R*` the target rate in nats per token. Because the constraint is on the *mean*, the model is free to spend ≈ 0 on healthy tokens and much more on sac tokens — the distribution §5.3.5 asks for. Starting target: `R* ≈ 8–16` nats/token at D = 16, set finally by the sweep. The existing 20-epoch warm-up is kept (it now ramps β_max).
   *Why not per-dimension free bits, as the previous edition of this section recommended.* A per-dimension floor λ lets every dimension sit anywhere with KL < λ at zero cost: at λ = 0.25 nats that is |μ| ≤ 0.71 at σ = 1, or σ ∈ [0.55, 1.54] at μ = 0, and 128 × 0.25 = 32 nats ≈ 46 bits per token for free. The null code becomes a ball rather than a point, and the per-token KL profile is clamped from below at D·λ on every token, so the "≈ 0 on healthy wall" diagnostic reads flat by construction. A floor is still useful as insurance against posterior collapse during warm-up; if used, apply it **per token** and small, `max(λ_tok, Σ_j KL_j)` with λ_tok ≈ 0.5 nats, and apply the `max` to the accumulated batch (`bs 1 × accum 8`), not per step. Posterior collapse is a lesser risk here than in autoregressive VAEs: the decoder cannot place the sac without the latent.
4. **Mask and normalise the KL by valid tokens** (`latent_valid` from §5.4), so β and `R*` mean the same thing on a 50 mm and a 130 mm tree.
5. **Move the mixer to the encoder side.** Apply `LatentTractSelfAttention` to μ (and, if wanted, to `logvar`) *before* reparameterisation, compute the KL on the mixed posterior, and feed the sampled code directly to the decoder. The smoothing along the tree is kept, the noise the KL prices is the noise the decoder sees, and Stage 1 generates exactly the codes the decoder consumes. If the mixer is kept after sampling for any reason, log the code scale before/after it and the gate value, and read the noise-robustness curve with that path in mind.
6. **Sample at evaluation, report both paths.** `reparameterize` gains an explicit `sample` flag; `evaluate_epoch` reports recon at σ = 0 (the μ path, kept for `best.pt` selection and comparability) *and* at the learned σ. The difference between the two is the first indicator of whether the decoder has learned a neighbourhood.
7. **Standardise after training — and drop the dead dimensions first.** Run the encoder over the training split, compute the per-dimension mean and standard deviation of μ over valid tokens, store both in the checkpoint. Dimensions the rate controller left inactive (mean raw KL < 0.01 nats, §5.3.7) must be **removed before standardising**, not standardised: an unused dimension has μ ≈ 0 on every token and a near-zero spread, so dividing by its standard deviation amplifies numerical noise into the Stage-1 target. The surviving count is the width the model actually used and is what Stage 1 should be given; record it in the checkpoint beside the statistics. This is also what makes a slightly generous D cheap and a too-small D expensive — the asymmetry behind choosing 16 over 8 in §5.3.3a. Stage 1 is trained on standardised codes (preferably on posterior samples `μ + σ·ε`, which is free augmentation for Stage 1); Stage 2 de-standardises before decoding. Report the statistics separately for healthy and sac tokens: healthy tokens dominate the count and will pull the mean toward the null code, which is correct for the diffusion target but should be visible.
8. **Define the null code and the sweeps explicitly.** The null code is the prior mean in raw (un-standardised) space, 0 per dimension. Healthy → aneurysmal sweeps use `t · z_case` on the case's own scaffold; case → case sweeps use slerp per token after resampling B's tokens onto A's scaffold by arc position (§5.3.2 d). For publication-grade "growth" figures prefer the displacement-field interpolation of §5.3.2 (c).
9. **Once Stage 1 exists, fine-tune the decoder on Stage-1 samples**, with the Stage-1 error distribution replacing the Gaussian posterior noise.

#### 5.3.7 Instrumentation and acceptance

These are logged per epoch as structured records (CSV/TensorBoard, §8), because without them a well-tuned latent cannot be told apart from a collapsed one:

- **raw per-token KL** (unclamped, before β), its mean over valid tokens, and **bits per case** (`Σ_tokens KL / ln 2`);
- **β** (the dual variable) and the gap `KL̄_raw − R*`;
- **active units** — dimensions whose mean raw KL exceeds 0.01 nats — separately for tokens over healthy wall and tokens over the sac (sac tokens: those whose wall neighbourhood intersects the AneuX dome segmentation, `aneurysms/original/{id}_dome`);
- **per-token KL profile along the tree** for a fixed set of validation cases — expected ≈ 0 on healthy wall with a peak under the sac; a flat profile means the latent is not local, and no rate setting fixes that;
- **noise-robustness curve** — reconstruction error (Chamfer and the §12 surface metrics) when decoding `μ + s·ε` in *standardised* units for s ∈ {0, 0.25, 0.5, 1}. This is the published interface contract with Stage 1: Stage 1's expected error in standardised units must lie on the flat part of this curve;
- **train/validation gap** on both the μ path and the sampled path, and, once Stage 1 exists, a nearest-training-case retrieval check on generated codes (memorisation test).

Two experiments set the free parameters before the full HPC run, both on a 50-case subset on the local 3080 Ti:

1. **Rate–distortion sweep**: D ∈ {4, 8, 16, 32} × `R*` ∈ {2, 8, 16, 32} nats/token. The grid brackets the estimate of §5.3.3a from both sides; 24 adds nothing between 16 and 32. Plot sac-region and healthy-region reconstruction against measured rate (Alemi et al. 2018), and read the knee from the **sac-region** error, not the global Chamfer — the global number is dominated by healthy wall and saturates by D ≈ 4 (§5.3.3a). Choose the smallest D whose held-out sac p95 error is within ~10 % of the D = 32 result, and inspect the worst 10 % of sacs separately, because a too-small D fails on the irregular tail first. Then check **saturation**: if nearly all D dimensions are active on sac tokens at the chosen D, the latent is at capacity and D goes up one step.
2. **Noise-robustness curve** at the chosen point, recorded in the repository as the Stage-1 contract.

Acceptance for the chosen configuration: sampled-path recon within a few percent of μ-path recon; the KL profile peaked at the sac on the validation cases; decoding the null code on a case's scaffold reproduces the template to within remesh noise; every mesh along `t · z_case`, t ∈ {0, 0.25, 0.5, 0.75, 1}, has one component, three boundary loops and no non-manifold edges.

Because the §2.4 tract fix, the §5.4 token layout and the §7.1 `r*` fix all change how much the latent must carry, the sweep is run after the first two and re-checked after the third.

#### 5.3.8 What would change this decision

- **Option F as a replacement for C is closed.** The test — a per-token truncated Fourier fit of the GT cross-section radius about the centerline reaching ≲ 0.2 mm on the sac region at order 8–12 — was run on all 709 cases (§5.3.3a) and fails: order 8 leaves 0.39 mm RMS / 2.9 mm p95-max on sac tokens, order 12 leaves 0.22 mm RMS. On healthy wall it passes easily (order 8 → 0.053 mm RMS, 0.063 mm p95-max), which is the useful half of the result. What survives is the **hybrid**: analytic low orders for the healthy part (the 4-parameter calibre/taper/eccentricity fit already reaches 0.074 mm median RMS) plus a small learned residual code under the same rate control. It would keep the exact null code and interpretability on the ≈ 51 of 56 tokens per case that are not sac tokens. It is a refinement of C, not an alternative, and it is not on the critical path.
- **Option D replaces C** only if the rate controller proves unstable on this data (β oscillating or pinned at a bound across the sweep); then use a fixed σ ≈ 0.2–0.3 in standardised units, accept the loss of the per-token σ and the KL profile, and tune σ against the noise-robustness curve.
- **Option B** is not recommended at this sample size; it would be reconsidered only if unconditional sampling from the prior became a requirement.

References for §5.3: Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022 ([arXiv:2112.10752](https://arxiv.org/abs/2112.10752)); S. Dieleman, *Generative modelling in latent space* (2025, [sander.ai](https://sander.ai/2025/04/15/latents.html)); Kingma et al., *Improved Variational Inference with Inverse Autoregressive Flow* — free bits ([arXiv:1606.04934](https://arxiv.org/abs/1606.04934)); Chen et al., *Variational Lossy Autoencoder* ([arXiv:1611.02731](https://arxiv.org/abs/1611.02731)); Hoffman & Johnson, *ELBO surgery* (NIPS 2016 Workshop on Advances in Approximate Bayesian Inference) — the average KL decomposes into I(x; z) plus KL(q(z) ‖ p(z)); Alemi et al., *Fixing a Broken ELBO* ([arXiv:1711.00464](https://arxiv.org/abs/1711.00464)); Rezende & Viola, *Taming VAEs* — GECO ([arXiv:1810.00597](https://arxiv.org/abs/1810.00597)); White, *Sampling Generative Networks* — slerp ([arXiv:1609.04468](https://arxiv.org/abs/1609.04468)); Carlini et al., *Extracting Training Data from Diffusion Models* ([arXiv:2301.13188](https://arxiv.org/abs/2301.13188)); Somepalli et al., *Diffusion Art or Digital Forgery?* ([arXiv:2212.03860](https://arxiv.org/abs/2212.03860)).

### 5.4 Token geometry under the 1 mm / 2 mm decision

The tree length is now known for the whole dataset rather than assumed: after deduplication (§2.4.2a) it is **60 / 120 / 216 mm** at p5 / p50 / p95 over the 709 `cleandata` centerlines, and the token counts below follow from it (the measured count in §5.3.3a, ≈ 56 per case, is after dropping tokens that touch an opening).

| `ds_tok` | tokens per sample (p5 / p50 / p95) | tokens on a sac (neck 3–6 mm, height 3–15 mm) | comment |
|---|---|---|---|
| 1 mm | 60 / 120 / 216 | 5–15 | every token sees half a sac; Stage 1 must model 100+ tokens |
| **2 mm** | **30 / 56 / 108** | **3–8** (measured: 4.7 sac tokens per case) | a sac is 3–8 tokens: neck, body, dome are separable |
| 3–4 mm | 15 / 30 / 54 | 1–4 | a sac becomes one blob token |

Given the decision, the changes are:

- `TOKEN_SPACING_MM = 2.0` and `CL_SAMPLE_MM = 1.0` as shared constants of both stages; per branch `n_tok = floor(L / 2) + 1` tokens at arc positions `k · 2 mm` (first token at the branch start). The daughter's first token *is* the junction — no separate junction tokens; the tree gives adjacency. `LATENT_LEN` becomes the padding maximum (e.g. 128) with a validity mask (`latent_valid`, used by the head, the mixer, the decoder cross-attention and the KL, §5.3.6), not the count. `allocate_token_counts` and the fixed-slot logic go.
- Tokens on the **original / Stage-1 centerline** (§2.5), not on `template_centerline`.
- Encoder head as in §4.2 item 2.
- Decoder cross-attention (§6.4) restricted to the ~5 nearest tokens along the vertex's branch (plus the neighbouring branch's tokens within ~4 mm of an ostium), instead of softmax over all of a tract's slots — with ≈ 56 tokens (30–108 at p5–p95) and a positional query this is what the softmax converges to anyway, and the restriction makes the latent provably local.

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

The 5 × 5 × 5 SplineConv (24.6 M of 43.4 M parameters) is still the bulk of the time, but the template's physical resolution removed the 8× node excess of the tube. On the A100s this is comfortably a batch of 8–16 per GPU; the point is not speed but being able to afford 2–5 · 10⁴ optimisation steps and deeper decoders (§9). Kernel `(5, 5, 2)` and 64-channel fine-level convs are still worth taking.

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
| validation | every 5 epochs, EMA only, ~15 % random split, deterministic μ path only (`reparameterize` returns μ in eval) | also log non-EMA val; report recon at σ = 0 *and* with posterior sampling (§5.3.6 item 6) |
| augmentation | none | L/R mirror (before scaffold build — flips Bishop handedness consistently), per-epoch resampling of `x_true` from the full GT (cache all vertices, not 16 384 — 3 099 of the 16 384 are duplicates on the 22 k-vertex stand-in), θ-phase and ±5° pose jitter |
| resume | `last.pt` is written, never read | load model / EMA / opt / sched / epoch / RNG |
| logging | `print`, history at the end | per-epoch CSV / TensorBoard: raw per-token KL, bits per case, β and rate gap, active units (healthy / sac), KL profile along the tree, noise-robustness curve (§5.3.7) |
| device | `gpu_index = 1 if n_gpu > 1 else 0` | `CUDA_VISIBLE_DEVICES` or an argument |
| multi-GPU | none | 4 × A100 as 4 independent configs |
| precision | FP32 tensors, TF32 matmuls | correct; keep it |
| split | random 85/15, no test set | one fixed train / validation / test split, seeded and stored, so every configuration is compared on the same cases. Cross-validation is dropped for now (Damján, 20 Sep) — it multiplies every experiment by k and the design decisions are still large enough to read off a single split |

The EMA horizon is the one that silently wastes early validation: at decay 0.999 the shadow is a 1 000-step box filter, so the "best" checkpoint of the first 50 epochs is still mostly the initial weights. Combined with `val_every = 5` and a 4.3 k-step budget, the training loop as written cannot tell a working model from an identity decoder.

---

## 9. Spending compute for accuracy, not speed

HPC wall-time is effectively unlimited; VRAM on the 3080 Ti already has headroom (0.74 GiB of 12) and an A100 40 GB is not the constraint. The levers below are ordered by expected accuracy gain per unit of extra compute, using only things this review has already argued for.

| lever | expected effect | cost |
|---|---|---|
| full GT point set + face-sampled predicted points, resampled every epoch | removes the 16 k sampling floor and the dome vertex-density bias (§2.2.3, §7.3) | memory only |
| augmentation set of §8 | largest single generalisation gain on ~200–680 samples | free |
| 5–10× more optimisation steps | nowhere near convergence at 4.3 k steps | time, now cheap (0.79 s/step → 2–5 · 10⁴ steps is 4–11 GPU-hours per run) |
| decoder depth (6–8 residual convs/level, pre-norm) and multi-head cross-attention + FFN instead of the 5³ kernel | capacity where it is used (§6.2, §6.4) | modest |
| rate–distortion sweep, D ∈ {4, 8, 16, 32} × R* ∈ {2, 8, 16, 32} nats/token on 50 cases (§5.3.7) | confirms `LATENT_DIM = 16` (measured in §5.3.3a) and fixes the rate before the full run; the decision everything in Stage 1 inherits | 16 short runs, local GPU |
| local-pooling transformer latent head (§4.2, §5.4) | local tokens, better latent, less overfitting | small |
| normals + template signed distance into the encoder (§4.2) | cheap accuracy | free |
| densified templates under the sac (§2.3 fix) | fine resolution where the residual lives | regenerate templates |
| curriculum: radial-only warm-start on the parent, then unlock sac terms | stable convergence against the currently opposing radial Huber | free |

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
| latent channel | VAE vs AE open | **decided, not implemented**: rate-controlled VAE (§5.3.6); code is still the near-autoencoder of §5.1 |
| pseudo-coords in physical units | to do | u yes (`u_step`), θ no (§6.2) |
| displacement bounds `r_local`-relative, free 3-D at coarse | to do | constants (§6.7) |
| GT generator | parent tube | **done**: `remeshing.py`, verified on SNF00000100 and C0002 (§2.2) |
| `postprocess.tensor_to_vtp` | template faces | not checked here |

### 10.2 Remaining design (ordered)

1. **Data first.** Cell arrays through the clip + `extract_groupid_tracts` rule (§2.4); orientation-agnostic stretch test and regenerated templates (§2.3 — now blocking, since the templates are variable-density by decision §15 item 2); parametrise from `original_centerline` and stop writing `template_centerline` (§2.5, §15 item 3). Generate both products of one case in one process so they share one centerline and one set of ostium frames, with the **GT's cut frames driving the template's ostia** (§2.2.4, §2.9); export `R_template`, `StretchDistance` and `TargetEdgeLength` on the final template vertices — the last two are Stage 1's supervision for the stretch head.
2. **Normals.** Orient every level's normals outward by the radial direction from the nearest centerline sample (one dot product; the sign of `r*`, the head floor and the Chamfer weights all depend on it). Today it is correct on 99.8 % of SNF00000100's vertices only because of how that file happens to be wound.
3. **`r*` and weights** as in §7; **`r_local`** in `composed_radius`, Chamfer weights, head bounds; θ pseudo-coordinate in physical units.
4. **Tokens** per §5.4 with arc-length-driven counts and a mask; then the **latent channel** of §5.3.6 and the rate–distortion sweep of §5.3.7 (after the tract fix and the tokens, since both change how much the latent carries; re-check after the `r*` fix of step 3).
5. **Cache the full GT** (vertices + normals) and a mirrored copy; cache-time gates: openings = `n_profiles` at every level, one component, no non-manifold edges, GT→template p99 *outside the sac* ≤ 0.5 · `r_local`, every GT vertex within a few millimetres of the centerline.
6. **Inference path** = Stage-1 centerline (1 mm, MISR) + Stage-1 stretch field → `generate_base_surface` + uncap → adaptive remesh on the predicted `TargetEdgeLength` → Stage 2 decode → fold checks and final isotropic remesh (§11). Per §15 item 2 the training and inference densities are the same by construction, so no density mismatch has to be absorbed; what this costs is a Stage-1 output head and the export in step 1.

---

## 11. Watertight single manifold for CFD

Topology is now inherited from the template (§6.8): one component, `n_profiles` boundary loops, manifold. That is the prerequisite, not the guarantee. A vertex-displacement decoder on a 25 mm overhang can fold triangles, invert normals and put the surface through itself while still having the right connectivity. Mandatory to keep the output CFD-ready:

1. **Fold prevention during training.** A hinge penalty on `n_pred · n_template < 0` per triangle, plus a stretch regulariser (per-triangle singular values of the deformation gradient, or edge-length ratios against the template). These are the missing terms of §7.5.
2. **Displacement bounds that can reach the dome** without crossing the lumen (§6.7): `r_local`-relative floor and shear, free 3-D at coarse.
3. **Verification in `postprocess.py`**, as validation metrics, not as a silent repair: self-intersection count, boundary loops = `n_profiles`, one component, minimum triangle angle. Model selection of §12 uses these; a folded mesh with a good Chamfer is a failed sample.
4. **A final isotropic remesh** of the decoded surface before CFD (the same `remesh_surface_isotropically` already used for the GT, at a CFD-appropriate edge — not 0.15 mm). This is the step that restores triangle quality after a large deformation; it is not a substitute for (1).

5. **Boundary-plane constraint, mandatory (decision of §2.2.4).** Boundary-loop vertices stay in their profile plane during training: they may slide in-plane, they may not leave it. Outlet position and planarity are already specified as a post-process; constraining them in the loss removes a degree of freedom the Chamfer would otherwise spend. This is only correct because the template's cut planes are now the GT's cut planes (§2.2.4) — against a plane offset by 0.58 mm the constraint would oppose the data. Implement it as a projection after the displacement head (project the predicted offset onto the plane for rim vertices), not as a penalty, so it cannot be traded away against the Chamfer.

---

## 12. Evaluation protocol

On the held-out cases of the fixed split (§8), millimetre units, non-Huber (the training Chamfer's Huber δ = 1 mm hides the tail that CFD cares about):

- **Global:** symmetric Chamfer mean and p95, Hausdorff, normal-angle error.
- **Sac-specific** versions of the same, using a sac mask. Two equivalent definitions, both already measurable: template vertices whose outward ray to the GT exceeds 1 mm (304 of 7 909 on SNF00000100), or GT points farther than `r_local + 1 mm` from the centerline. Report both so a model that grows the sac in the wrong place cannot hide in the global Chamfer.
- **Neck-plane error and sac volume error** — the two numbers a clinician actually looks at.
- **The mesh-validity counts of §11** (self-intersections, loops, components, min angle).
- **The noise-robustness curve of §5.3.7** (decode `μ + s·ε` in standardised units for s ∈ {0, 0.25, 0.5, 1}) — the interface contract with Stage 1.
- **Latent sanity (§5.3.7)**: the null code decodes to the template within remesh noise; every mesh along `t · z_case`, t ∈ {0, 0.25, 0.5, 0.75, 1}, passes the validity counts of §11; the per-token KL profile peaks under the sac.

The current validation score (`val_recon + λ_rad · val_rad`) stays for model selection only; it is not the number that is reported.

---

## 13. Tests

`test_architecture.py`: 55/55 pass (~4.5 min). `test_template_scaffold_starts_from_mesh` is a good addition (posed template = fine level, kNN tables, `__inc__` under a 2-graph batch, finite output). `test_remeshing.py`: 14/14 pass through the pytest-free shim (vmtk_env has no `pytest`). What the architecture suite cannot see, and should:

- **Coaxial synthetic cylinders** (template r = 2.0, GT r = 2.4, no sac): the nearest-vertex `r*` is exact there, so its failure under a sac is invisible. Add a cylinder with a lateral bulge and assert `r*` ≈ ray distance under it.
- Both synthetic meshes are wound **outward**; nothing asserts that scaffold normals point *away* from the centerline.
- Every centerline test uses point-data arrays with identical copies and a 0.4 mm blank. Real files have cell arrays (dropped), near-coincident copies and ~1.6 mm blanks. A fixture from `scratch/uniform_probe/original_centerline/SNF00000100.vtp` (or `scratch/cl_probe/SNF00000100_branched_unclipped.vtp`) asserting 3 tracts / 1 junction would have caught §2.4 twice.
- `smoothness_edge_weights` returning all ones; token spacing ≈ 2 mm independently of vessel length; `e_th` of a ring neighbour ≈ 0 or 1; decimation keeping `n_profiles` loops at mid and coarse.
- Latent (§5.3.6): KL masked to valid tokens and invariant to padding; `reparameterize(sample=True)` in eval mode actually samples; log-variance never below log 0.01; the mixer is applied before sampling; decoding the null code returns the template.
- Sac preservation on a real GT remesh (the probe of §2.2): area ratio ∈ [0.88, 1.20], openings = profiles, original→output fraction > 1 mm below a small threshold. The synthetic scale gates already exist; they do not replace one fixture case.

Earlier scratch probes still relevant: `analyze_stage2_pipeline.py`, `scan_tract_quality.py`, `probe_variable_templates.py`, `time_stage2_step.py`, `profile_stage2_step.py`, `probe_branch_arrays.py`, `probe_groupid_tracts.py`, `prototype_branch_tracts.py`, and from this pass `probe_gt_remesh.py`, `probe_smoothing_change.py`, `probe_template_scaffold.py`, `probe_template_rstar.py`, `probe_stretch_orientation.py`, `scan_gt_winding.py`.

---

## 14. Prioritised change list

**P0 — data generation (nothing should be trained before these; the last three change files in `cleandata/`)**

1. ~~GT generator that remeshes the *original* vessel~~ — **done**: `remeshing.py`, verified on SNF00000100 and C0002 (§2.2). Remaining: point `uniformly_remeshed/` at this script, not at `uniform_remeshing.py`; decide workers / `n_iter` for the 682-case run (§2.2.4).
2. Orientation-agnostic (or radially oriented) stretch test in `compute_raycast_stretch_distances` (§2.3); regenerate `template_mesh`; export `R_template` / `StretchDistance` / `TargetEdgeLength` on the final vertices. **Blocking, not cosmetic**: templates are variable-density by decision (§15 item 2), so without this fix they are uniform, and the exported stretch is Stage 1's supervision target.
3. Carry `GroupIds` / `Blanking` / `CenterlineIds` / `TractIds` through `clip_centerline_at_profiles`; replace the body of `extract_groupid_tracts` with the validated rule; error instead of fallback on `cleandata` (§2.4).
4. One process per case for centerline + GT remesh + template, with the **GT's cut frames driving the template's ostia** so both products share the same boundary planes (§2.2.4); persist the frames per case; `n_clipped == n_profiles` as a hard gate on both products; `hascap.csv` exclusions (§2.9).

**P0 — training-side consumption of the template (small, local edits)**

5. Parametrise / pose / tokenise from `original_centerline`; **drop** `template_centerline` — stop writing it and stop expecting it in `aneux_paths.py` / `cleaned_io.py` (§2.5, §15 item 3).
6. Outward normal orientation by radial direction at every level (§10.2).
7. `r*` from an outward ray-cast (or the exported `StretchDistance`) with validity / ambiguity; edge-based `dth` / `du` / `ring_med` (§7.1, §7.2).
8. `r_local` in `composed_radius`, Chamfer weights and head bounds (§6.7, §7.3).

**P1 — model**

9. Tokens at 2 mm with arc-length-driven counts and a mask; local-pooling transformer latent head (§4.2, §5.4). Latent as a rate-controlled channel (§5.3.6): `LATENT_DIM` 128 → 16 (measured, §5.3.3a; confirmed by the sweep), soft σ floor 0.1, per-token rate target with an adaptive β (a small per-token floor only as warm-up insurance — no per-dimension free bits), mixer before sampling, sampling at evaluation, post-training standardisation with inactive dimensions dropped first; instrumentation and noise-robustness curve (§5.3.7).
10. θ pseudo-coordinate in physical units; kernel `(5, 5, 2)` or a separate cross-branch conv; `aggr="mean"`, pre-norm, `root_weight=True`; 6–8 convs/level (§6).
11. Free 3-D displacement at coarse; fold and stretch penalties (§6.7, §11); **boundary-plane projection for rim vertices, mandatory** (§11 item 5, §2.2.4); face-sampled Chamfer on the full GT; robust Dirichlet with GT-driven weights (§7).
12. Encoder: normals + template signed distance in; stage-3/4 width down; node features `r_local`, curvature, torsion, ostium distance (§4, §6.6).

**P2 — training framework and evaluation**

13. Resume; per-epoch logging; EMA 0.99–0.995 with warm-up; LR warm-up; wd exclusions; device via env; augmentation set; curriculum; one fixed seeded train / validation / test split (§8, §9).
14. Metrics of §12 and validity checks of §11 in `postprocess.py`; final isotropic remesh step.
15. Tests of §13.

---

## 15. Previously open questions, now decided

Nothing in this list is a fork any more. Each entry records the decision and what it obliges.

1. **GT smoothing — settled by measurement (§2.6).** The GT's light Taubin (pass band 1.5 × 5) and the template/centerline's strong Taubin (0.1 × 15) differ from each other, and from the raw original, by at most 0.18 mm — less than one GT edge. Either setting is consistent enough; do not match them for its own sake.
2. **Template density — variable, with Stage 1 predicting the stretch (Damján, 20 Sep).** Templates keep the variable density of `variable_remeshing.py`; Stage 1 will predict the stretch factor along the centerline, so the density available at inference is the density trained on and there is no mismatch to absorb. Stage 2 needs **no change** for this. Two consequences follow, and neither is optional: the §2.3 orientation fix stops being a quality improvement and becomes load-bearing, since without it `compute_raycast_stretch_distances` returns nothing and the "variable" template is uniform (`Max k = 1.35` where the true value under the sac is ≈ 10); and `StretchDistance` / `TargetEdgeLength` must be exported on the final template vertices (§10.2 step 1) as the supervision target for that Stage-1 head. Because edge length then varies by an order of magnitude along the surface, the physical-unit pseudo-coordinates of §6.2 and the `r_local`-relative bounds of §6.7 matter more, not less.
3. **`template_centerline` — drop the folder (Damján, 20 Sep).** It is the same curve as `original_centerline` to within 0.12 mm Hausdorff and 0.06 mm MISR (§2.5), and §5.4 already tokenises from the original. `cleandata/` becomes three folders; `process_centerline_dataset` should stop writing it and `aneux_paths.py` / `cleaned_io.py` stop expecting it.
4. **GT remesh target edge — 0.15 mm, confirmed by the completed run (Damján, 20 Sep).** `cleandata/uniformly_remeshed/` now holds 709 meshes generated at this target. VMTK undershoots the requested edge, so the delivered median is ≈ 0.12 mm rather than 0.15 — a known behaviour of the remesher, not a pipeline defect, and the reason the measured GT is denser than the nominal target suggests. `N_TRUE` and the encoder input are sized against the delivered density: cache the full GT and resample `x_true` per epoch (§2.2.3).
5. **Analytic cross-section descriptors — closed (Damján, 20 Sep).** Option F cannot replace the learned code: the per-token truncated Fourier fit reaches 0.053 mm RMS on healthy wall at order 8 but only 0.39 mm on sac tokens, 0.22 mm at order 12, measured on all 709 cases (§5.3.3a). The hybrid (analytic low orders on the non-sac tokens plus a learned residual code) is **not** being built; the latent stays a single learned, rate-controlled channel.
6. **Case-to-case interpolation landmark — deferred (Damján, 20 Sep).** Anchoring B's tokens on A's scaffold (bifurcation, neck centre or both, §5.3.2 d) is not decided and does not need to be: it affects only the latent-sweep tooling, which comes after Stage 1. Healthy → aneurysmal sweeps on a case's *own* scaffold (`t · z_case`) need no landmark and are unaffected.

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
| `latent_dim/patches_all/*.npz` (38 689 token patches, 708 cases), `latent_dim/patches32/` (178 cases at 32 θ bins), `latent_dim/report.txt` | `latent_dim/extract.py` + `run_all.py` (extraction), `analyze.py`–`analyze4.py` (PCA / parametric fit / intrinsic dimension / calibration) | §2.4.2a, §5.3.3a, §5.3.5, §5.3.8 |
---

## Appendix B. What needs a code change, and what does not

Groups **A**, **B** and **C** (items 1–32) are code. Group **D** (items 33–45) is explicitly *not* — it exists so that a number or a finding elsewhere in the review does not get mistaken for a task.

### A. Broken today

**`datatransform/template_creation/`**

1. `vessel_pipeline.py::clip_centerline_at_profiles` — carry the cell arrays (`GroupIds`, `Blanking`, `CenterlineIds`, `TractIds`) through the rebuild (§2.4.3).
2. `vessel_pipeline.py::compute_raycast_stretch_distances` — orientation-agnostic hit test (radial sign), then regenerate `template_mesh` (§2.3). **Blocking.**
3. `vessel_pipeline.py::build_parent_tube` — `n_clipped == n_profiles` as a hard gate, not a warning (§2.3.2).

**`1test_encoder_decoder_only/train_pipeline/`**

4. `dataset.py::extract_groupid_tracts` — replace the body with the validated GroupId rule; raise instead of falling back on `cleandata` inputs (§2.4.3).
5. `dataset.py::_template_r_star` / `raycast.py` — `r*` from an outward ray-cast or the exported `StretchDistance`, with validity and ambiguity flags (§7.1).
6. `dataset.py` (§3.7) — real `dth`, `du`, `ring_med` so `smoothness_edge_weights` is not identically 1.0 (§7.2).
7. `composed_radius`, Chamfer weights, head floor — use the cached `r_local`, not `TUBE_RADIUS_MM = 2.0` (§6.7, §7.3).
8. `dataset.py` level build — orient normals outward by the radial direction at every level (§10.2 step 2).
9. `model.py::reparameterize` + `train.py::evaluate_epoch` — explicit `sample` flag; sample at evaluation and report both paths (§5.1, §5.3.6 item 6).
10. `geometry.py` — θ pseudo-coordinate in physical units (§6.2).
11. `aneuxai.py` / `train.py` — resume from `last.pt`; EMA 0.99–0.995 with warm-up; weight-decay exclusions; LR warm-up; device from the environment (§8).

### B. Decided, still to be written

**Generators**

12. `remeshing.py` exports its ostium cut frames; the template's `clip_flow_extensions_and_uncap` consumes them instead of `measure_open_profiles`; frames persisted per case (§2.2.4).
13. Export `R_template`, `StretchDistance`, `TargetEdgeLength` on the final template vertices (§14 item 2).
14. One process per case for centerline + GT + template; `hascap.csv` exclusions (§2.9).
15. `centerline_creation.py::process_centerline_dataset` stops writing `template_centerline`; `aneux_paths.py` / `cleaned_io.py` stop expecting it (§15 item 3).

**Latent (§5.3.6)**

16. `config.py` — `LATENT_DIM` 128 → 16; `LATENT_LEN` becomes a padding maximum; `TOKEN_SPACING_MM = 2.0`, `CL_SAMPLE_MM = 1.0`.
17. `model.py::CenterlineLatentHead` — local pooling instead of softmax over all encoder centres; soft σ bound (σ_min = 0.1) replacing `LOGVAR_CLAMP`.
18. `model.py::forward` / `decode` — `z_attn` before reparameterisation.
19. `losses.py::vae_kl_loss` — mask and normalise by `latent_valid`; per-token rate target with a GECO-style adaptive β replacing the fixed `LAMBDA_KL`.
20. Post-training standardisation pass: drop inactive dimensions, store statistics and the surviving count in the checkpoint (§5.3.6 item 7).
21. The §5.3.7 instrumentation: raw per-token KL, bits per case, β and rate gap, active units split healthy / sac, KL profile along the tree, noise-robustness curve.

**Tokens (§5.4)**

22. Remove `allocate_token_counts` and the fixed-slot logic; remove padding-by-duplication in `_build_latent_tokens`; thread `latent_valid` through head, mixer, cross-attention and KL; restrict decoder cross-attention to the ~5 nearest tokens on the branch.

**Geometry and CFD (§11)**

23. Boundary-plane projection for rim vertices, applied after the displacement head (§11 item 5).
24. Fold penalty (`n_pred · n_template < 0`) and triangle-stretch regulariser (§7.5).
25. `r_local`-relative displacement bounds; free 3-D at the coarse level (§6.7).
26. `postprocess.py` — self-intersection count, boundary loops, component count, minimum angle as metrics; final isotropic remesh (§11, §12).

**Data plumbing**

27. Cache the full GT (vertices + normals); resample `x_true` every epoch; face-sampled Chamfer against it (§2.2.3, §8).

### C. Improvements, measured but optional

28. SplineConv kernel `(5, 5, 2)`; `aggr="mean"`, pre-norm, `root_weight=True`; 6–8 convs per level (§6).
29. Encoder: normals and template signed distance as inputs; stage-3/4 width down, the parameters given to the latent head; node features `r_local`, curvature, torsion, ostium distance (§4, §6.6).
30. Augmentation: L/R mirror, θ-phase, ±5° pose jitter (§8).
31. Batch size above 1 × 8; one fixed seeded train / validation / test split (§8).
32. The tests of §13.

### D. No code — nothing to implement

These are measurements, decisions or closed questions. **Do not write code for anything in this list.** They are here so that reading a section and finding a number does not turn into a task.

33. §2.2 — the GT generator is verified; every density, distance and topology number in it. Nothing to do.
34. §2.2.3 — the GT/template/`N_TRUE` density ratios. The only action they imply is item 27.
35. §2.4.2a — the centerline duplication measured on all 709 files. The dedup script is analysis scaffolding; the production fix is items 1 and 4.
36. §2.6 — the Taubin `pass_band` 1.0 → 0.1 change is measured benign (≤ 0.18 mm, under one GT edge). Do **not** revert it and do **not** match the GT's smoothing to the template's.
37. §3.8, §6.3, §10.1 — the scaffold numbers, the compute profile and the migration status table.
38. §5.3.1, §5.3.2 — the literature case that neither diffusion nor interpolation needs a latent pushed to N(0, I), and the Gaussian norm-shell table.
39. §5.3.3a, §5.3.5 — the latent-width and heterogeneity measurements. They set `LATENT_DIM = 16` and justify the stochastic encoder; the code they imply is items 16–21, nothing more.
40. §5.3.4, §5.3.8, §15 item 5 — option F and the Fourier hybrid are closed. Do not build them.
41. §15 item 1 — GT smoothing, settled.
42. §15 item 2 — the variable-density decision needs **no Stage-2 change**. What it does oblige is items 2 and 13.
43. §15 item 4 — the 0.15 mm GT edge target is confirmed and the run is done; VMTK's undershoot to ≈ 0.12 mm is expected behaviour, not a defect to fix.
44. §15 item 6 — the case-to-case interpolation landmark is deferred. Not now.
45. Cross-validation — dropped for now (§8). The 5-fold in §5.3.3a is how the PCA measurement was run, not a training plan; do not build a fold harness.
