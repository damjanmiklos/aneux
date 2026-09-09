# Stage 2 (latent → mesh texture) — architecture, data and training-framework review

State reviewed: commit `d0fdc1b` ("template integrated in") — `train_pipeline/{config, cleaned_io, dataset, raycast, geometry, ops, model, losses, train, aneuxai, test_architecture}.py`, `postprocess.py`, `aneux_paths.py`, the (still empty) four-folder `cleandata/` layout, and the generators in `datatransform/template_creation/` (`vessel_pipeline.py`, `centerline_creation.py`, `uniform_remeshing.py`, `variable_remeshing.py`). No pipeline code was changed by the review.

Decisions taken (Damján, 9 Sep) that this review builds on: training reads only `cleandata/`; `uniformly_remeshed/` is meant to be the *original vessel with the aneurysm*, finely remeshed, and is the GT for everything; templates come from `variable_remeshing.py` and are now the decoder's identity surface; `coarse_remeshed` is gone — mid/coarse are decimations of the template; Stage 1 will emit the centerline at **1 mm** with a texture token every **2 mm**, along the *entire* tree; output must be a single watertight manifold with open outlets (CFD), outlet position/planarity fixed in post; HPC wall-time effectively unlimited, accuracy over speed; manual per-case work acceptable; aneurysm removal (hemoMesh) is a later topic and is not treated here.

Everything below that is a number was measured. The new template path was exercised end-to-end on **one real case, SNF00000100**, with files produced by the current generators into `scratch/uniform_probe/` (`process_uniform_dataset`, `process_variable_dataset`, `process_centerline_dataset` on both the original and the template) plus a copy of the original vessel standing in for the intended GT. A second case, C0002, was run through `process_uniform_dataset` only. Probing scripts (scratch only): `probe_uniform_gt.py`, `render_uniform_probe.py`, `make_probe_template.py`, `probe_template_rstar.py`, `probe_stretch_orientation.py`, `scan_gt_winding.py` (vmtk_env); `probe_template_scaffold.py`, `probe_cl_pair.py` (aneurysmgnn); earlier ones listed in §12.

---

## 0. Executive summary (ranked by impact)

1. **`uniform_remeshing.py` does not remesh the vessel it is given; it remeshes the parent tube of that vessel, and the aneurysm is not in the output.** This is not a reading of the code but a measurement (§2.2): on SNF00000100 the output surface lies within 0.64 mm (p99) of the input everywhere, yet **9.2 % of the input surface is > 1 mm from the output (max 7.5 mm)** and the output has 14 % less area — exactly the sac. On C0002: 9.9 % of the input > 1 mm away, max 14.0 mm, 16 % less area, one small side branch lost as well. The mechanism: `process_uniform_dataset` → `build_parent_tube` → `generate_base_surface` stamps a **polyball image from the centerline's MISR spheres and runs marching cubes**; the input surface contributes only its centerline, its bounds and the opening profiles. `remesh_surface_isotropically` is then applied to that polyball surface, never to the input. If `uniformly_remeshed/` is filled with this, GT ≈ template and Stage 2 learns the identity. A remesh-of-the-original entry point is needed (§2.2 gives it in five lines of existing helpers).
2. **The branch arrays are still lost, so the template path is parametrised by two coincident tracts.** `centerline_creation.py` unchanged since the last review → no `GroupIds`/`Blanking` on either `original_centerline` or `template_centerline` (verified on the fresh files: cell arrays `[]`) → `extract_groupid_tracts` falls back to `extract_unique_tracts`. Measured on the new scaffold: **2 tracts of 95.6 / 94.9 mm, 0 junctions, 48 + 48 tokens**, template vertices split **3961 / 3948 at random** between the two copies along the whole parent, 76 % of upsampling neighbours and a similar share of mesh edges classified "cross-tract" (and down-weighted). The fixes from the last review (§2.3) are unchanged and still P0.
3. **`variable_remeshing.py`'s stretch ray-cast is orientation-dependent and returns zero for AneuX `.vtp` originals**, so those templates are *not* densified under the sac. `compute_raycast_stretch_distances` accepts a hit only if `n · gt_cell_normal > 0.2` with normals from `vtkPolyDataNormals(AutoOrientNormalsOff)`; SNF/UPF/USFD/ANSYS `.vtp` files carry inward-consistent winding (11 of 11 sampled), the C0xxx `.vtp` (4 of 4) and p* `.stl` files (14 of 15) outward. On SNF00000100 the pipeline call yields **1 non-zero distance (0.56 mm)** while the true outward ray under the sac is 1–25 mm (median 4.1 mm, 304 vertices); with the GT winding flipped it yields 5 862 hits, max 6.5 mm. Result: the "variable" template of SNF00000100 has 7 909 vertices at a uniform 0.41 mm — identical to the uniform remesh (7 920). Fix is one line (orientation-agnostic test, or orient by the centerline radial direction); the templates have to be regenerated afterwards. `StretchDistance`, which should be exported as the fine `r*`, is zero for the same reason.
4. **The template integration is structurally right and already pays off — 0.79 s and 0.74 GiB per train step (was 4.28 s, 2.7 GiB) — but four of its new supervision/parametrisation pieces are wrong or disabled** (§9):
   - `_template_r_star` (nearest GT *vertex*, projected on the normal) says "stay put" under the sac: offset **0.03 / 0.24 / 0.83 mm (p50/p90/max)** where the true outward distance is **4.1 / 11.8 / 24.9 mm**; `λ_rad = 1.0` and `valid = True` everywhere, so the radial Huber actively opposes sac formation.
   - `dth = du = 0`, `ring_med = r*`, `ambiguous = False` → `smoothness_edge_weights ≡ 1.0` on every edge (measured min = mean = 1.0): the crease-aware Dirichlet/Laplacian weighting is silently off.
   - `TUBE_RADIUS_MM = 2.0` and `R_MARGIN_MM` are still the base radius in `composed_radius`, the Chamfer weights and the head's floor although `r_local` (0.87–2.66 mm on this vessel) is now cached per vertex at every level.
   - `template_centerline` is measurably the same curve as `original_centerline` (≤ 0.12 mm, MISR within 0.06 mm) — a redundant VMTK pass that also creates a second centerline at inference; use the original for both stages.
   Good and measured: fine level = posed template exactly (identity displacement 0.0 at init); decimation kept **3 boundary loops, 1 component, 0 non-manifold edges** at all three levels (7 909 / 1 980 / 638 vertices, 0.41 / 0.89 / 1.56 mm edges); kNN upsampling has **no opposite-wall mixing** (0.0 % / 0.04 %); normals outward on 99.8 % of vertices (by luck of winding — not enforced).
5. **Latent, decoder kernel and training hygiene** findings carry over unchanged (§3–§8): 12 288-dim near-AE latent; θ-blind SplineConv (on the template Δθ ≈ 0.22 rad per edge → `e_th = 0.5 ± 0.035`); `aggr="add"` without normalisation; no augmentation/resume/logging, EMA horizon 47 epochs. With the token contract now fixed at 2 mm, `LATENT_LEN = 96` fixed slots contradicts it (on this case 2.02–2.04 mm by coincidence; on a 50 mm vessel it would be 1 mm) — token count must follow arc length.

Good and worth keeping: the centerline-intrinsic design, the template as identity surface, kNN inter-level tables with `__inc__`, the decoupled radial/shear head with identity init, multi-scale point-to-plane Chamfer, cache/warm-up infrastructure and rawdata guard, FP32/TF32, the smoke-test culture (55/55 pass).

---

## 1. What the pipeline does today (as built, commit `d0fdc1b`)

```
cleandata/uniformly_remeshed/{id}.vtp  (GT)        cleandata/template_mesh/{id}.vtp   (identity surface)
cleandata/original_centerline/{id}.vtp (fallback)  cleandata/template_centerline/{id}.vtp (tracts, tokens, u, θ)
   └─ dataset._build_data → build_scaffold(template_cl, vessel_mesh=GT, template_mesh=tpl)
      → extract_groupid_tracts (→ falls back to extract_unique_tracts: no cell arrays on real files)
      → _orient_tracts / canonical pose (from the tracts) → dense Bishop tracts
      → fine  = posed template_mesh: faces, edges, u/θ/tract by nearest dense-CL point, r_local, u_step
        mid   = pyvista decimate(keep 25 %, ≥ 256), coarse = decimate(keep 8 %, ≥ 64), same projection
        n_radial = 1, branch_nl = [n]  (grid bookkeeping neutralised)
      → kNN(k=3) inverse-distance tables coarse→mid→fine (+ __inc__ for batching)
      → r* = R + n·(nearest GT vertex − x_tpl), valid everywhere, dth = du = 0   ← §9.3
      → x_true: 16 384 GT points (75 % FPS + 25 % far-from-CL) + GT normals
      → 96 latent token slots (tract-proportional + junction tokens), attend masks
      → one .pt per sample (CACHE_VERSION = 9)

model.py   Encoder  PointNeXt: stem(xyz) → SA 16384→1024→256→64→64 (r = 1.5/3/6/12 mm), InvRes blocks
                    → CenterlineLatentHead: 96 positional queries → μ, logσ² (128)
           Latent   reparam → LatentTractSelfAttention (window ±2, ALiBi, gated)
           Decoder  per level: LatentCrossAttention (γ(u), γ(θ) queries → z tokens, tract-masked)
                    + coarse-only positional self-attention
                    + 4 × ResidualSplineConv(128, kernel 5³, degree 2, aggr add, no norm)
                    + DecoupledDisplacementHead → Δr = softplus − R_MARGIN, Δs = 3·tanh (vertex frame)
                    coarse → kNN upsample → mid → kNN upsample → fine (template vertices)

losses.py  recon = Σ_levels w_l · weighted Chamfer (pt-to-plane Huber δ = 1 mm + 0.2 L2, weights 1 + d_cl/R ≤ 4)
           + λ_kl·KL (annealed 20 ep) + 0.15·Dirichlet(Δr,Δs) + 0.05·Laplacian + 0.02·normal consistency
           + 1.0·radial Huber(r_pred vs r*) on valid nodes

train.py   AdamW 2e-4, wd 1e-4, cosine 200 ep, bs 1 × accum 8, clip 1.0, EMA 0.999, val every 5 ep (EMA), random 85/15 split
```

The Bishop-tube path (`build_scaffold` without `template_mesh`, `build_scaffold_from_centerline`) still exists as a fallback and for the synthetic tests; with `require_templates=True` (default) it is never used on `cleandata`.

---

## 2. Data: the `cleandata` contract versus what the generators produce

### 2.1 Folder by folder

| folder | intended content | what the current generator produces | consumer in training code |
|---|---|---|---|
| `uniformly_remeshed` | original vessel **with aneurysm**, fine isotropic remesh (GT) | **parent tube** of the original (polyball of its centerline, remeshed) — §2.2 | `vessel_file`: `x_true`, GT normals, `r*` |
| `template_mesh` | parent template from `variable_remeshing.py` | parent tube, adaptively remeshed — **adaptivity inactive on inward-wound inputs** (§2.4) | fine identity surface; mid/coarse by decimation |
| `original_centerline` | `centerline_creation.py` on the GT | polylines per source→target path; **branch cell arrays dropped** (§2.3); MISR kept | read, then **unused** when a template exists |
| `template_centerline` | `centerline_creation.py` on the template | same as the original centerline to within 0.12 mm (§2.5) | tracts, canonical pose, (u, θ), tokens |

### 2.2 `uniform_remeshing.py` produces the parent tube — measurement

`process_uniform_dataset(v_file)` → `pv.read(v_file)` → `build_parent_tube(vessel)` → `open_base_surface` → `remesh_surface_isotropically(open_base_surface)`. Inside `build_parent_tube`: Taubin → open profiles → flow extensions → Voronoi centerline + MISR → resample/smooth → `vmtkBranchExtractor` → **`generate_base_surface`: `stamp_polyball_image(pts, radii)` from the centerline points and their (clamped) MISR, then `vmtkMarchingCubes`** → uncap at the profiles. The vessel surface enters only through `reference_bounds`, the profiles and the centerline. The script does accept any vessel; what it writes is the MISR tube of that vessel's centerline.

Surface-to-surface distances (400 k area-weighted samples per side, `probe_uniform_gt.py`):

| case | input → output p50 / p95 / max | input area > 1 mm from output | output → input p99 / max | area in / out |
|---|---|---|---|---|
| SNF00000100 | 0.09 / 3.41 / **7.49 mm** | **9.2 %** | 0.64 / 2.19 mm | 1 321 / 1 131 mm² (0.86) |
| C0002 | 0.09 / 4.19 / **13.98 mm** | **9.9 %** | 0.62 / 2.13 mm | 1 647 / 1 381 mm² (0.84) |

The asymmetry is the signature of "output ⊂ input": every output point sits on the input wall, a tenth of the input wall (the sac) has no counterpart. Renders in `scratch/uniform_probe/{SNF00000100,C0002}_uniform_vs_original.png` (input coloured by distance, output with edges, overlay). C0002 also shows the polyball bulging at the neck where the Voronoi MISR inflates — the sac-attraction effect noted last time — and drops the small vessel that leaves the sac.

What a GT generator needs (all helpers exist in `vessel_pipeline.py`):

```python
def process_gt_remesh_dataset(dataset_id, v_file, output_dir, target_edge_length):
    vessel = sanitize_vessel_for_vmtk(pv.read(v_file))
    vessel = apply_taubin_smoothing(vessel)          # same smoothing as the centerline/template path, or omit (decision, §14)
    remeshed = remesh_surface_isotropically(vessel, target_edge_length)   # PreserveBoundaryEdges=1 keeps the open ends
    final, _ = finalize_surface(remeshed)
    save_polydata(final, os.path.join(output_dir, f"{dataset_id}.vtp"))
```

plus the scale/opening assertions already used by the other two entry points. Openings coincide with the template's by construction (the template is uncapped on the profiles measured on this same smoothed surface). Whether to smooth the GT is a decision (§14): smoothing makes GT, centerline and template consistent and removes segmentation staircase; not smoothing keeps the GT faithful.

### 2.3 Branch arrays and tract extraction (unchanged since the last review, re-verified on the fresh files)

`vmtkBranchExtractor` writes `GroupIds`, `Blanking`, `CenterlineIds`, `TractIds` as **cell** data; `clip_centerline_at_profiles` rebuilds the polydata copying only point arrays → both centerline files in `scratch/uniform_probe/` have cell arrays `[]` and point arrays `EdgeArray, EdgePCoordArray, MaximumInscribedSphereRadius, TCoords`. `extract_groupid_tracts` then silently runs `extract_unique_tracts`. On the template scaffold that means (measured): 2 tracts (95.6 / 94.9 mm, both the full inlet → one-outlet path), 0 junctions, the parent's template vertices assigned to tract 0 or 1 by nearest-point coin flip, 48 tokens per copy so the *daughters* get ~6 tokens each and the parent two redundant token sets, `token_attend` masking each vertex to one copy, `_cross_tract_smooth` down-weighting most parent edges, and an arbitrary inlet/pose choice.

The two defects inside `extract_groupid_tracts` on real (near-coincident, blanked) data and the validated three-line rule are as in the previous review: keep the **longest copy per GroupId**, attach each blanked run to the daughter that follows it, snap endpoints at 1 mm — on SNF00000100: 3 tracts 88.8 / 13.3 / 12.9 mm, 1 junction, 100 % coverage, ≤ 1 % overlap (`prototype_branch_tracts.py`). The generator side needs `clip_centerline_at_profiles` to carry the cell arrays (per kept cell) or convert them to point data first. The silent fallback should be an error for `cleandata` inputs.

### 2.4 `variable_remeshing.py`: stretch detection depends on file winding

```1633:1661:datatransform/template_creation/vessel_pipeline.py
    for i in range(n_pts):
        # ...
        hit = locator.IntersectWithLine(p0, p_end, tol, t, x, pcoords, sub_id, cell_id)
        if hit:
            d = float(np.sqrt((x[0] - p[0]) ** 2 + (x[1] - p[1]) ** 2 + (x[2] - p[2]) ** 2))
            if 0.10 < d <= (3.5 * r_local):
                cid = int(cell_id.get())
                if gt_cell_normals is not None and 0 <= cid < len(gt_cell_normals):
                    if float(np.dot(n, gt_cell_normals[cid])) > 0.2:
                        distances[i] = d
```

`n = -template_normals` (correct for the marching-cubes `open_base_surface`: 0.0 % outward as computed) and `gt_cell_normals` from `vtkPolyDataNormals` with `ConsistencyOn, AutoOrientNormalsOff` — i.e. the GT's *stored winding* decides the sign. Measured (`probe_stretch_orientation.py`, `scan_gt_winding.py`):

| | non-zero stretch | max |
|---|---|---|
| pipeline call as-is on `open_base_surface` vs SNF00000100 | **1 / 17 303** | 0.56 mm |
| same, GT winding flipped | 5 862 | 6.46 mm |
| original files with inward winding (as-is normals disagree with auto-oriented) | `.vtp` SNF/UPF/USFD/ANSYS: 11 of 11 sampled; C0xxx `.vtp`: 0 of 4; `.stl` p*: 1 of 15 | |

So for roughly the AneuX half of the 682 originals the adaptive step is a no-op and `template_mesh` = uniform template (SNF00000100: 7 909 vertices at 0.41 mm median edge; the uniform remesh of the same tube: 7 920). Fix: test `abs(dot) > 0.2`, or `AutoOrientNormalsOn()` for the GT (gives 98.5 % outward here), or orient both by the radial direction from the nearest centerline point (robust for tubes, no closed-surface assumption). Also note the saved template's normals come out **outward** (99.9 %) after remeshing, opposite to `open_base_surface` — nothing downstream may assume a fixed convention (§9.4). Also worth a quality gate: `n_clipped == n_profiles` is a warning only; C0002 lost a branch.

### 2.5 `template_centerline` versus `original_centerline`

Same case, both from `process_centerline_dataset`: 1 922 vs 1 919 points, 6 cells each; Hausdorff 0.12 mm, p95 0.06 mm; MISR difference p50 +0.011 mm, max 0.062 mm. The template is the polyball of the original centerline, so its centerline is that centerline again. Consequences: (i) the folder buys nothing for training; (ii) `_build_data` reads `original_centerline` and then ignores it; (iii) at inference the natural input is the Stage-1 centerline, and routing it through `build_parent_tube` → template → `centerline_creation.py` → tracts adds a VMTK pass and a second, slightly different curve. Recommendation: parametrise, tokenise and pose from `original_centerline` in training and from the Stage-1 centerline at inference; keep `template_centerline` only as an optional consistency check (or drop the folder).

### 2.6 Findings from the legacy cache that still apply

- **Coincident tracts poison everything downstream** — now on the template path (§2.3). 
- **Short centerlines** (94/200 legacy cases covering only the aneurysm neighbourhood) are fixed by construction by re-extracting inlet → all outlets, provided `measure_open_profiles` finds every opening; `hascap.csv` remains the exclusion list.
- **Constant 2 mm radius**: the scaffold geometry no longer depends on it (template), but `TUBE_RADIUS_MM` / `R_MARGIN_MM` still parametrise the loss weights and the head (§9.3).
- **Fixed 1000 fine rings / varying physical resolution**: resolved by the template — 0.41 mm edges, physical.
- **Voronoi centerline bends into large sacs**; the polyball then bulges at the neck (C0002 render). Cache-time diagnostic (deviation from a spline through the healthy segments; MISR-cap hit rate) still recommended; the hemoMesh-removal option is deferred by decision.
- **No loss weight has been validated at convergence** (only checkpoint ever written: epoch 1).

### 2.7 The I/O layer

`cleaned_io.py` is sound and simpler than before: four folders, completeness gating, rawdata rejection, lazy VMTK import, `ensure_derived` for centerlines only, `_init_kwargs` round-trip for the spawn pool. `_build_data` requires templates by default and raises with an actionable message. Minor: `list_cleandata_samples` creates folders as a side effect; the fallback tube path is reachable when `require_templates=False`, which should never be the case on `cleandata`.

---

## 3. Encoder (PointNeXt)

Unchanged: 18.5 M parameters (12.6 M in stage 3 on 64 points; 1.2 M in the head), 64 ms forward. (1) Feed GT normals (cached, unused) and, now that a template exists, the signed distance of each GT point to the template — the encoder's job is precisely the residual. (2) The latent head is a single-head, single-layer positional-query cross-attention from 96 slots to 64 12-mm tokens with no FFN/LayerNorm and no access to stage-2/3 features; angular detail can only arrive through content. Replace with a local-pooling transformer head (§4.4). (3) Halve stage-3 width, move capacity to the head. (4) `logvar.clamp(-8, 2)` kills gradients outside the range.

---

## 4. Latent space and VAE

### 4.1 As built

96 tokens × 128 = 12 288 dims; `LAMBDA_KL = 5e-4` on a KL summed over 128 dims and averaged over tokens (at init on the real sample: KL 7.13 → 0.0036 in the objective vs recon 1.315): an autoencoder with an unsmooth, unscaled latent. `allocate_token_counts` spreads 96 slots over the tracts in proportion to arc length plus junction tokens — spacing therefore *varies with vessel length* (2.0 mm on this 95 mm case, 1 mm on a 50 mm one).

### 4.2 What Stage 1 needs from the latent

Stage 1 emits, per centerline sample at 1 mm, `[x, y, z, r]` and, on every second sample, a texture token `z_1 … z_D`, along the whole tree. The latent is a tree-structured sequence of *local* tokens with fixed per-dimension scale, smoothness (Stage 1 lands near training codes), decoder robustness to Stage 1 error, and ideally a known null code for healthy wall so Stage 1 only has to model where the sac is.

### 4.3 VAE vs AE + noise — trade-off and recommendation

| | β-VAE (meaningful β) | AE + fixed-σ noise + standardisation | **Light per-token KL-VAE with free bits (recommended)** |
|---|---|---|---|
| Reconstruction accuracy | Costs accuracy; steep trade-off on ~200 samples | Maximal | Near-AE: free-bits floor means KL only bites on unused dims |
| Latent scale | Unit by construction | Must be standardised explicitly | Near-unit; final standardisation pass is cheap insurance |
| Healthy segments | Collapse to the prior — *desired* (template explains them) | No pressure toward a null code | Same desirable collapse; per-token KL along the tree becomes a diagnostic of where the template fails |
| Robustness to Stage 1 error | Built in | σ controls it directly | Built in at the learned posterior width; add small extra noise if it narrows |
| Prior sampling without Stage 1 | Yes | No | Healthy tokens yes |
| Failure mode | Global collapse → blurry sacs | σ mis-set → brittle or blurry | Two knobs, wide safe range |

Per-token KL to N(0, I), free bits ≈ 0.1–0.25 nats/dim, β such that total KL is 2–5 % of recon at convergence (start ~1e-2 on a per-dim-averaged KL), 20-epoch warm-up. Log active units (healthy vs sac tokens), the per-token KL profile along the tree, and a noise-robustness curve (recon when decoding μ + σ·ε for σ ∈ {0, 0.25, 0.5, 1}) — the interface contract with Stage 1. Fall back to AE + fixed σ ≈ 0.2–0.3 only if β proves untunable. Fine-tune the decoder on Stage 1 samples once Stage 1 exists.

### 4.4 Token geometry under the 1 mm / 2 mm decision

| `ds_tok` | tokens / sample (80–130 mm tree) | tokens on a sac (neck 3–6, height 3–15 mm) | token width |
|---|---|---|---|
| 1 mm | 80–130 | 5–15 | 8–16 |
| **2 mm** | **40–70** | **3–8** | **16–32** |
| 3–4 mm | 25–40 | 1–4 | 32–64 — sac becomes a blob token |

Concretely, given the decision:
- `TOKEN_SPACING_MM = 2.0` and `CL_SAMPLE_MM = 1.0` as shared constants of both stages; per branch `n_tok = floor(L / 2) + 1` tokens at `u_k = k·2 mm / L`-equivalent arc positions (first token at the branch start; the daughter's first token *is* the junction — no separate junction tokens, the tree gives adjacency). `LATENT_LEN` becomes the padding maximum (e.g. 128) with a validity mask, not the count.
- Tokens on the **original / Stage-1 centerline** (§2.5). The Stage-1 sequence is then: 1 mm geometry samples, every second one carrying a 32-dim token.
- Encoder head: assign each GT point to its nearest centerline sample (Voronoi along the tree), attention-pool PointNeXt features per token, then 2–4 transformer layers over the token tree with arc-length and branch-depth positional encodings.
- Decoder cross-attention restricted to the ~5 nearest tokens along the vertex's branch (plus the neighbouring branch's tokens within ~4 mm of an ostium), instead of softmax over all of a tract's slots.

---

## 5. Decoder

### 5.1 Pseudo-coordinates — still θ-blind on the template

`e_u = 0.5 + 0.5·clamp(Δu / step)` with `step = u_step` (mean |Δu| over same-tract edges — a sensible per-vertex physical step on the template). `e_th = 0.5 + 0.5·Δθ/π`: on the template a 0.41 mm edge on a 1.9 mm radius spans Δθ ≈ 0.22 rad → `e_th ≈ 0.5 ± 0.035`; ring neighbours are indistinguishable to a degree-2 B-spline on 5 knots. Normalise `Δθ` by the local circumferential step (`edge_len / r_local`, both now available) so neighbours land on the kernel corners. `e_kind` (0 / 0.5) still occupies a full 5-knot axis: kernel `(5, 5, 2)`, degree 1 on the last axis, or a separate small conv for cross-branch edges.

### 5.2 Compute profile on the real template scaffold (3080 Ti, SNF00000100)

| | tube path (legacy, 64 000 fine nodes) | **template path (7 909 fine, 1 980 mid, 638 coarse)** |
|---|---|---|
| fwd + bwd, one sample | 4.28 s (with AdamW step) | **0.79 s** |
| peak VRAM | 2.7 GiB | **0.74 GiB** |
| cache build | — | 2.0 s (no ray-casting) |

The 5×5×5 SplineConv (24.6 M of 43.4 M parameters) is still the bulk of the time, but the template's physical resolution removed the 8× node excess. On the A100s this is comfortably a batch of 8–16 per GPU; the point remains not speed but affording 2–5·10⁴ steps, deeper decoders and k-fold sweeps. Kernel `(5,5,2)` and 64-channel fine convs are still worth taking.

### 5.3 Aggregation and normalisation

`aggr="add"`, `root_weight=False`, no normalisation in the decoder convs. Degree on the template is 5–8 (measured mean 6.0 fine, 6.0 mid, 5.9 coarse) — more regular than the junction-coupled tube, but still use `aggr="mean"` or degree normalisation, pre-norm, `root_weight=True`.

### 5.4 Cross-attention from nodes to latent

Content-keyed positional interpolation of the latent along u; single head, no FFN; θ enters only through `out`. Two heads + FFN cost nothing; with fixed 2 mm tokens restrict attention to the nearest tokens (§4.4). The tract mask is currently applied to *coincident* tracts (§2.3) — after the tract fix the mask means what it should.

### 5.5 Hierarchy and upsampling — done, with two refinements

`knn_upsample_tables` (k = 3, inverse distance, `__inc__` correct under batching — verified with a 2-graph batch in the tests and here) replaces bilinear upsampling; measured neighbour distances 0.9 mm (coarse→mid) / 0.53 mm (mid→fine) p50, **0 % / 0.04 % of targets have a neighbour on the opposite wall** — the lumen (≥ 1.7 mm) is wider than the search. Two refinements: restrict candidates to the same branch (after §2.3) and to `n_i·n_j > 0` so thin daughters near an ostium cannot mix with the parent; and use `pyvista.decimate` output only if it keeps the openings and manifoldness (it did here: 3 loops, 1 component at every level) — a cache-time assertion, since `decimate_pro` and the FPS fallback have different guarantees. The decoder still sees no centerline geometry (curvature, torsion, `r_local`, ostium distance) as node features — `r_local` is now in the cache; add it.

### 5.6 Displacement head

Δr = softplus(·) − `R_MARGIN_MM` (constant), Δs = `SHEAR_MAX_MM`·tanh. On a 0.87 mm daughter branch the floor lets a vertex cross the centerline; on the trunk 3 mm of shear cannot reach an overhanging dome. Both bounds should be `r_local`-relative (floor ≈ −0.8·r_local, shear cap ∝ max(3 mm, k·r_local)), and the coarse level should be free 3-D (638 vertices — dome placement is its job).

### 5.7 Output topology

Inherited from the template: one component, `n_profiles` boundary loops, manifold — the CFD prerequisite is now structural. What keeps it fold-free is §10.

---

## 6. Losses on the template path (measured at init on SNF00000100: recon 1.315, kl 7.13, disp 0, lap 0.190, norm 0.009, rad 0.016)

- **Radial Huber vs `r*`** (λ = 1): `nearest_normal_offset_r_star` takes the *nearest GT vertex* and projects it on the template normal. Under the sac the nearest GT vertex is the neck rim or the sac wall beside the vertex, so the target is 0.03 / 0.24 / 0.83 mm (p50/p90/max) where the outward ray reaches the GT at 4.1 / 11.8 / 24.9 mm; the ratio (offset / ray) has median 0.005 and does not exceed 0.26. With `valid = True` on all 7 909 vertices this term pins the sac region to the template while the Chamfer pulls it out. Replace by an outward ray-cast along the template normal (the pipeline's `compute_raycast_stretch_distances` with the orientation fix is exactly this and can be exported per template vertex as `StretchDistance`), mark misses/grazes as invalid, and keep a proper `ambiguous` flag for double hits. Healthy wall: offset p50 0.06 mm, p95 0.32 mm — fine as is.
- **Dirichlet / Laplacian weights**: `dth = du = 0`, `ring_med = r*` → `smoothness_edge_weights ≡ 1` (measured). Replace the grid statistics with edge-based ones: `|r*_i − r*_j| / edge_len` and the 1-ring median of `r*`, which give the same crease-awareness on any mesh; or, better, drive the weight from the GT distance itself (large `StretchDistance` gradient = neck).
- **Chamfer weights**: `pred_weights` uses `|R + n·Δ|` with `R = 2.0` as a stand-in for the distance from the centerline; the GT side uses the true distance. Use `r_local + n·Δ` on the predicted side. Predicted side samples vertices only — a stretched dome is under-sampled; sample on predicted faces.
- **Laplacian on absolute positions** penalises the template's own curvature; the displacement Laplacian would not. Low priority.
- **Missing**: triangle stretch / edge-length-ratio regulariser and a normal-flip penalty — the real failure modes of a fixed-topology deformer on overhanging sacs (§10).
- **KL**: §4.

---

## 7. Training framework

| item | current | issue / suggestion |
|---|---|---|
| batch | bs 1 × accum 8 → ~21 steps/epoch, 4.3 k steps in 200 epochs | plan 2–5·10⁴ steps; real bs 4–8 now fits easily (0.74 GiB/sample) |
| optimizer / schedule | AdamW 2e-4, wd 1e-4 on everything, cosine, no warm-up | exclude norms/biases/gates from wd; 200–500-step LR warm-up |
| EMA | 0.999 per step | horizon ≈ 47 epochs: early `best.pt` selection uses near-initial weights. 0.99–0.995 with warm-up `min(d, (1+n)/(10+n))` |
| validation | every 5 epochs, EMA, ~15 % random split | also log non-EMA val; k-fold (§8) |
| augmentation | none | L/R mirror (before scaffold build — flips Bishop handedness consistently), per-epoch resampling of `x_true` from the full GT (cache all vertices, not 16 384 — 3 099 of the 16 384 are duplicates on this sample), θ-phase and ±5° pose jitter |
| resume | none | add (model/EMA/opt/sched/epoch/RNG) |
| logging | `print`, history at the end | per-epoch CSV/TensorBoard |
| device | `gpu_index = 1 if n_gpu > 1 else 0` | `CUDA_VISIBLE_DEVICES` / argument |
| multi-GPU | none | 4 × A100 as 4 independent configs/folds |
| precision | FP32 tensors, TF32 matmuls | correct |
| split | random 85/15, no test set | 5-fold CV + held-out test fold |

---

## 8. Spending compute for accuracy, not speed

| lever | expected effect | cost |
|---|---|---|
| full GT point set + face-sampled predicted points, resampled every epoch | removes the 16 k sampling floor and dome vertex-density bias | memory only |
| augmentation set of §7 | largest single generalisation gain on ~200 samples | free |
| 5–10× more optimisation steps | nowhere near convergence at 4.3 k steps | time, now cheap |
| decoder depth (6–8 residual convs/level, pre-norm) and multi-head cross-attention + FFN instead of the 5³ kernel | capacity where it is used | modest |
| local-pooling transformer latent head (§4.4) | better latent, less overfitting | small |
| normals + template signed distance into the encoder | cheap accuracy | free |
| densified templates under the sac (§2.4 fix) | fine resolution where the residual lives | regenerate templates |
| curriculum: radial-only warm-start on the parent, then unlock sac terms | stable convergence | free |
| 5-fold CV | error bars on every design decision | 5× per config — what the A100s are for |

---

## 9. Template scaffold: what is done, what is wrong, what remains

### 9.1 Status of the migration items

| component | previous review | now (`d0fdc1b`) |
|---|---|---|
| fine scaffold = template | to do | **done**: posed `template_mesh`, exact identity at init |
| hierarchy | coarse templates needed | **done differently**: `decimate` 25 % / 8 % of the template; openings and manifoldness preserved on the probe (assert it) |
| inter-level upsampling | barycentric / kNN tables | **done**: kNN k = 3 inverse distance, batched correctly, no cross-wall mixing |
| grid bookkeeping (`branch_nl`, `n_radial`) | to neutralise | **done**: `n_radial = 1`, one "branch" per level |
| (u, θ, branch) per vertex | export or project | **done by projection** to the nearest dense-CL sample — onto *coincident* tracts until §2.3 is fixed |
| `r*` | `StretchDistance` along template normal | **wrong**: nearest-vertex normal offset (§6) |
| smoothness weights | edge-based | **disabled**: dth/du zero (§6) |
| `R_local` per vertex | MISR interpolation | **cached** (`r_local`, `r_local_mid`, `r_local_coarse`) but **unused** by model and losses |
| tokens at 2 mm on the original centerline | to do | fixed 96 slots on the template centerline (§4.4, §2.5) |
| pseudo-coords in physical units | to do | u yes (`u_step`), θ no (§5.1) |
| displacement bounds `r_local`-relative, free 3-D at coarse | to do | constants (§5.6) |
| `postprocess.tensor_to_vtp` | template faces | not checked here |

### 9.2 Remaining design (ordered)

1. **Data first**: GT generator (§2.2); cell arrays through the clip + `extract_groupid_tracts` rule (§2.3); orientation-agnostic stretch test and regenerated templates (§2.4); parametrise from `original_centerline` (§2.5). Generate the three products of one case in one process so they share one `build_parent_tube` run; export `R_template` and `StretchDistance` on the final template vertices.
2. **Normals**: orient every level's normals outward by the radial direction from the nearest centerline sample (one dot product; the sign of `r*`, the head floor and the Chamfer weights all depend on it). Today it is correct on 99.8 % of SNF00000100's vertices only because of how that file happens to be wound.
3. **`r*` and weights** as in §6; **`r_local`** in `composed_radius`, Chamfer weights, head bounds; θ pseudo-coordinate in physical units.
4. **Tokens** per §4.4 with arc-length-driven counts and a mask.
5. **Cache the full GT** (vertices + normals) and a mirrored copy; cache-time gates: openings = `n_profiles` at every level, one component, no non-manifold edges, GT→template p99 outside the sac ≤ 0.5·`r_local`, every GT vertex within a few mm of the centerline.
6. **Inference path** = Stage-1 centerline (1 mm, MISR) → `generate_base_surface` + uncap → remesh → Stage 2 decode. Note that *variable* remeshing needs the GT for its stretch field; at inference only a uniform remesh (or a stretch predicted from the tokens) is available. Train with both densities, or train on uniform templates and let the final remesh (§10) handle density — decide before regenerating.

---

## 10. Watertight single manifold for CFD

Topology is now inherited from the template (§5.7). Mandatory to keep it: (1) fold prevention — hinge penalty on `n_pred · n_template < 0` per triangle plus a stretch regulariser (per-triangle singular values or edge-length ratios); (2) displacement bounds that can reach the dome (§5.6); (3) verification in `postprocess.py`: self-intersection count, boundary loops = `n_profiles`, one component, min triangle angle — as validation metrics; (4) a final isotropic remesh of the decoded surface before CFD. Optional: keep boundary-loop vertices in their profile plane during training.

---

## 11. Evaluation protocol

Held-out folds, mm units, non-Huber: symmetric Chamfer mean and p95, Hausdorff, normal angle error; **sac-specific** versions using a sac mask (template vertices with outward ray > 1 mm — 304 of 7 909 on the probe case — or GT points > `r_local` + 1 mm from the centerline); neck-plane error and sac volume error; the mesh-validity counts of §10; the noise-robustness curve of §4.3. The current validation score stays for model selection only.

---

## 12. Tests

55/55 pass (`test_architecture.py`, ~4.5 min). `test_template_scaffold_starts_from_mesh` is a good addition (posed template = fine level, kNN tables, `__inc__` under a 2-graph batch, finite output). What the suite cannot see:

- coaxial synthetic cylinders (template r = 2.0, GT r = 2.4, no sac): the nearest-vertex `r*` is exact there and its failure under a sac is invisible; add a cylinder with a lateral bulge and assert `r*` ≈ ray distance under it;
- both synthetic meshes are wound outward; nothing asserts the scaffold normals point away from the centerline;
- every centerline test uses point-data arrays with identical copies and a 0.4 mm blank; real files have cell arrays (dropped), near-coincident copies and ~1.6 mm blanks — a fixture from `scratch/uniform_probe/original_centerline/SNF00000100.vtp` (or `scratch/cl_probe/SNF00000100_branched_unclipped.vtp`) asserting 3 tracts / 1 junction would have caught §2.3 twice;
- `smoothness_edge_weights` returning all ones; token spacing ≈ 2 mm; `e_th` of a ring neighbour ≈ 0/1; decimation keeping `n_profiles` loops.

Earlier scratch probes still relevant: `analyze_stage2_pipeline.py`, `scan_tract_quality.py`, `probe_variable_templates.py`, `time_stage2_step.py`, `profile_stage2_step.py`, `probe_branch_arrays.py`, `probe_groupid_tracts.py`, `prototype_branch_tracts.py`.

---

## 13. Prioritised change list

**P0 — data generation (nothing should be trained before these; the first three change files in `cleandata/`)**
1. GT generator that remeshes the *original* vessel (§2.2) — `process_gt_remesh_dataset`; `uniform_remeshing.py` as it stands writes the parent tube.
2. Orientation-agnostic (or radially oriented) stretch test in `compute_raycast_stretch_distances` (§2.4); regenerate `template_mesh`; export `R_template` / `StretchDistance` on the final vertices.
3. Carry `GroupIds`/`Blanking`/`CenterlineIds`/`TractIds` through `clip_centerline_at_profiles`; replace the body of `extract_groupid_tracts` with the validated rule; error instead of fallback on `cleandata` (§2.3).
4. One process per case for centerline + GT remesh + template; `n_clipped == n_profiles` as a hard gate; `hascap.csv` exclusions.

**P0 — training-side consumption of the template (small, local edits)**
5. Parametrise/pose/tokenise from `original_centerline`; drop or demote `template_centerline` (§2.5).
6. Outward normal orientation by radial direction at every level (§9.2).
7. `r*` from an outward ray-cast (or the exported `StretchDistance`) with validity/ambiguity; edge-based `dth`/`du`/`ring_med` (§6).
8. `r_local` in `composed_radius`, Chamfer weights and head bounds (§5.6, §6).

**P1 — model**
9. Tokens at 2 mm with arc-length-driven counts and a mask; local-pooling transformer latent head; light per-token KL with free bits; noise-robustness curve at validation (§4).
10. θ pseudo-coordinate in physical units; kernel `(5,5,2)` or separate cross-branch conv; `aggr="mean"`, pre-norm, `root_weight=True`; 6–8 convs/level (§5).
11. Free 3-D displacement at coarse; fold and stretch penalties (§5.6, §10); face-sampled Chamfer on the full GT; robust Dirichlet with GT-driven weights.
12. Encoder: normals + template signed distance in; stage-3 width down; node features `r_local`, curvature, torsion, ostium distance.

**P2 — training framework and evaluation**
13. Resume; per-epoch logging; EMA 0.99–0.995 with warm-up; LR warm-up; wd exclusions; device via env; augmentation set; curriculum; 5-fold harness on the A100s.
14. Metrics of §11 and validity checks of §10 in `postprocess.py`; final isotropic remesh step.
15. Tests of §12.

---

## 14. Open questions

1. **GT smoothing**: should `uniformly_remeshed` be remeshed from the raw original or from the Taubin-smoothed surface that the centerline and template are built from? Smoothed keeps the three products mutually consistent and removes segmentation staircase (CFD wants that anyway); raw keeps the GT literal.
2. **Template density at inference**: variable remeshing needs the GT's stretch field, which Stage 1 cannot provide. Train on uniform templates only (simplest, matches inference exactly), or on variable templates and accept a train/inference density mismatch, or predict a stretch field from the tokens and remesh with it?
3. **`template_centerline`**: drop the folder, or keep it as a per-case consistency check against `original_centerline`?
4. **GT remesh target edge**: "very fine" — 0.25 mm gives ~4× the vertices of the 0.5 mm probe (≈ 30–40 k for SNF00000100); anything finer mostly costs cache size and encoder FPS time. Confirm the number so `N_TRUE` and the encoder input can be sized to it.
