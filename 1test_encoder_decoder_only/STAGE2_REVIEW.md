# Stage 2 (latent → mesh texture) — architecture and training-framework review

Scope: `1test_encoder_decoder_only/train_pipeline/*` (config, dataset, raycast, geometry, ops, model, losses, train), `aneuxai.py`, `postprocess.py`, the cached v7 dataset in `tube_cache/`, the checkpoints in `output/`, and the future data source `datatransform/template_creation/vessel_pipeline.py` (+ `output_variable_remeshed/`). No code in the pipeline was changed; all probing scripts live in `scratch/` (`analyze_stage2_pipeline.py`, `probe_centerline_tracts.py`, `scan_tract_quality.py`, `summarize_tract_quality.py`, `probe_variable_templates.py`, `time_stage2_step.py`, `profile_stage2_step.py`).

Everything below that is a number was measured on the current cache / code on the 3080 Ti box; everything that is a judgement is marked as such.

---

## 0. Executive summary

Ranked by expected impact on whether Stage 2 (and later Stage 1) works:

1. **The scaffold/centerline data feeding the model is the dominant problem, not the network.** `extract_unique_tracts` with `snap=1e-4 mm` does not merge the overlapping VMTK centerline paths, so almost every sample is covered by 2–4 coincident tubes (median duplicated-point fraction 0.75, 89/200 samples have several tracts but zero junctions). 94/200 samples (all UPF/USFD/ANSYS sources) have centerlines that cover only the aneurysm neighbourhood, leaving 20–60 % of the vessel surface >5 mm from any scaffold. Ray-cast `r*` is valid on only ~40 % of fine nodes as a direct consequence. None of the 200 samples passes a "clean scaffold" check. The current model is therefore being asked to fit noise: the only checkpoint on disk (epoch 1) has an epoch-mean recon of 15.5 mm² while a freshly initialised model on a clean long-centerline sample scores 0.6 mm². **Fix the data first; do not tune architecture against this cache.**
2. **The move to the `variable_remeshing.py` templates is not a "slight" code change — it removes the regular (u, θ) grid that the decoder's hierarchy, upsampling, pseudo-coordinates, `r*` statistics and radial clamp all assume.** The 95 produced templates are irregular triangulations with 2.3 k–34 k vertices (median 12.5 k), 2–6 open ends and only a `Normals` array. Section 10 lays out two migration designs; the one I recommend keeps the centerline-intrinsic (tract, u, θ) machinery by *projecting template vertices onto the branched centerline*, and replaces the bilinear grid upsampling with barycentric/kNN interpolation between three isotropic remeshes of the same template.
3. **The decoder is compute-dominated by 12 SplineConvs with a 5×5×5 kernel on 128 channels at 64 k nodes**: 97 % of GPU time (`pyg::spline_weighting` forward+backward ≈ 4.1 s of the 4.3 s per-sample step on the 3080 Ti), 24.8 M of 43.4 M parameters. Two design details make most of that wasted: (a) the third pseudo-coordinate (`e_kind`) only ever takes the values 0 and 0.5, so a 5-knot dimension encodes one bit; (b) `Δθ` is normalised by π rather than by the ring step, so ring neighbours map to 0.5 ± 0.016 and the θ axis of the kernel is effectively blind on the regular grid. Fixing (b) is a correctness fix; fixing (a) plus narrower fine-level channels gives a 5–10× speed-up for free.
4. **The VAE is effectively an autoencoder with a 12 288-dim latent (96 × 128) for 172 training shapes.** `LAMBDA_KL = 5e-4` on a per-token KL sum makes the KL term ≈ 0.5 % of the loss; posterior variance will go to the `logvar` clamp floor. This is fine for pure reconstruction but bad for Stage 1 (a generative prior over Z needs a smooth, compact, well-scaled latent). Recommendation (§15): keep a VAE but make it a *light, per-token* one — per-dim free bits, β sized so KL is a few % of recon — with tokens placed at fixed arc-length spacing along the whole centerline so Stage 1 emits them together with the centerline; monitor noise-robustness explicitly and fall back to AE + fixed-σ noise only if β proves untunable on this dataset.
5. **No augmentation, no resume, no per-epoch logging, EMA horizon ≈ 47 epochs, hardcoded `cuda:1`.** Each is small; together they make a 40-hour HPC run fragile and its early validation numbers meaningless. Left/right mirroring and per-epoch resampling of `x_true` are cheap, anatomically valid augmentations that effectively double the dataset.

What is good and should be kept: the centerline-intrinsic design (displacements in the Bishop frame, (u, θ) Fourier conditioning, tree-valued latent), the decoupled radial/shear head with identity init and radial clamp, the multi-scale Chamfer with point-to-plane Huber, the cache/warm-up infrastructure, the sizeable smoke-test suite, and the decision to stay in FP32/TF32.

---

## 1. What the pipeline does today (as built, not as drawn)

```
vessel.vtp + centerline.vtp
   └─ dataset.py: dedup polyline → extract_unique_tracts → orient → canonical pose
      → allocate rings (40/250/1000 total) → Bishop frames → 3 tube scaffolds (R = 2 mm, 6/12/64 radial)
      → junction coupling edges (k=2 NN within 4 mm of ostia)
      → x_true: 16 384 GT points (75 % FPS + 25 % far-from-centerline FPS) + GT normals
      → raycast.py: r* per fine/mid node along Bishop normal (Voronoi + 4 mm ambiguity guard)
      → 96 latent token slots (tract-proportional + junction tokens), attend masks
      → cached as one 16 MB .pt per sample (v7)

model.py
   Encoder  PointNeXt: stem(xyz) → SA 16384→1024→256→64→64 (r = 1.5/3/6/12 mm) with InvRes blocks
            → CenterlineLatentHead: 96 positional queries γ(u)+tract-emb cross-attend 64 tokens → μ, logσ² (128)
   Latent   reparam → LatentTractSelfAttention (window ±2 along tract, ALiBi, gated)
   Decoder  per level: LatentCrossAttention (node γ(u),γ(θ) queries → z tokens, tract-masked)
            + coarse-only positional self-attention
            + 4 × ResidualSplineConv(128, kernel 5³, degree 2, aggr add, no norm)
            + DecoupledDisplacementHead → Δr (softplus − R), Δs (3·tanh) in Bishop frame
            coarse → bilinear (u,θ) upsample → mid → upsample → fine (64 000 nodes)

losses.py  recon = Σ_levels w_l · weighted Chamfer (pt-to-plane Huber δ=1 mm + 0.2 L2, weights 1+d_cl/R ≤ 4)
           + λ_kl·KL (annealed 20 ep) + 0.15·Dirichlet(Δr,Δs) + 0.05·Laplacian + 0.02·normal consistency
           + 1.0·radial Huber(r_pred vs r*) on valid nodes

train.py   AdamW 2e-4, wd 1e-4, cosine 200 ep, bs 1 × accum 8, clip 1.0, EMA 0.999, val every 5 ep on EMA
```

Relationship to the whiteboard/draft: the draft's Stage 2 is "PointNeXt encoder → x0, SplineConv decoder conditioned by FiLM/AdaIN from the latent, trained with `z = N(x0, I)` noise, inference `z = x0`". The implementation replaces FiLM with per-level cross-attention from nodes to latent tokens (a reasonable, more expressive choice), and replaces the fixed-σ noise with a learned-σ VAE (see §5 — this changes what Stage 1 will see).

---

## 2. Dataset and scaffold quality (measured)

### 2.1 Tract extraction produces overlapping duplicates and fragments

`scan_tract_quality.py` over all 200 ICA samples:

| metric | median | notes |
|---|---|---|
| kept tracts per sample | 3–4 | many samples have 8–16 (capped by `MAX_TRACTS`) |
| duplicated-point fraction (kept tract points within 0.3 mm of another kept tract) | **0.75** | i.e. most of the scaffold is a tube lying on another tube |
| samples with ≥2 tracts and **0 junctions** | **89 / 200** | tracts are copies of each other, not a tree |
| samples with fragmented parent (≥3 tracts, low coverage gain) | 64 / 200 | |
| samples passing "clean scaffold" (dup < 0.1, junctions consistent) | **0 / 200** | |

Root cause: `extract_unique_tracts` merges points only when they coincide to `snap = 1e-4 mm`. VMTK `vmtkcenterlines` emits one polyline per (source, target) pair; the shared parent segment is resampled independently on each polyline, so consecutive samples differ by ~0.01 mm — never equal, never merged. `probe_centerline_tracts.py` on SNF00000100 shows three "tracts" whose points are 0.005–0.02 mm apart along the whole parent ICA.

Downstream consequences (all observed in the v7 cache):

- **`r*` valid on only ~0.39–0.43 of fine nodes** (`analyze_stage2_pipeline.py`). The Voronoi guard `voronoi_ok_hit` rejects a hit whose nearest dense-centerline sample belongs to a different tract; with 3 coincident tracts, 2/3 of rays fail. The radial loss therefore supervises less than half the parent vessel, and the Dirichlet weights (`smoothness_edge_weights`, w = 1 for `valid=False, ambiguous=False`) keep full stiffness exactly where supervision is missing.
- **Cross-tract coupling edges explode**: `couple_ostium_edges` adds k = 2 NN edges between every pair of tracts within 4 mm of an "ostium"; with coincident tracts everything is within 4 mm. USFD_0009 has 172 k cross-tract edges out of 362 k. These edges enter every SplineConv (`aggr="add"` → node degree at junctions 50–100 instead of 6) and every smoothness term.
- **Latent tokens are wasted**: `tok_per_tract` like `[9, 2, 2, 2, …]` — up to 30 of 96 tokens sit on 2-ring degenerate tracts.
- **Duplicated predicted surfaces**: three coincident tubes all pulled to the same wall by the radial loss and Chamfer → the output is self-overlapping by construction. Chamfer cannot penalise this.
- **Orientation/pose instability**: `_choose_inlet` walks from the highest-degree node; with zero junctions every node is a leaf, the inlet tract choice becomes arbitrary, and `_canonical_pose` (built from the first 5 inlet points) and the u-direction of every tract inherit that arbitrariness. Stage 1 will need a canonical orientation; the current one is not reproducible across similar anatomies.

### 2.2 Half the dataset has aneurysm-local centerlines

`summarize_tract_quality.py`: 94/200 samples (UPF, USFD, ANSYS_UNIGE) have raw centerline arc length far below the SNF/p* group, and 20–60 % of GT vessel points lie > 5 mm from any kept tract (`v_far5`). For these, `x_true` still samples the *entire* vessel, so the Chamfer true→pred term drags the tube ends (max shear 3 mm, `R_MARGIN` radial) toward surface the scaffold can never reach. Those samples are effectively label noise with weight up to 4× (the `1 + d_cl/R` weighting rewards exactly the far points).

### 2.3 Constant 2 mm tube radius

ICA radius ≈ 1.5–2.5 mm, ophthalmic/PComA/AChA ≈ 0.4–1.0 mm. On daughter branches Δr must be ≈ −1 to −1.5 mm everywhere, approaching the `softplus − R` floor; `r*` medians of 1.5–1.9 mm in the cache confirm the wall is mostly *inside* the tube. The head's zero-init identity is therefore a poor starting point for anything but the ICA trunk. The MISR-driven templates from `vessel_pipeline.py` solve this outright (another argument for the migration).

### 2.4 Fixed 1000 fine rings regardless of vessel length

`allocate_ring_counts` distributes 40/250/1000 rings proportionally to arc length. Total arc varies ~10 mm (short UPF cases) to > 300 mm (SNF with overlaps counted) → ring spacing 0.01–0.3 mm. Because the spline pseudo-coordinate is Δu normalised *per tract* by the ring step, the physical receptive field of the decoder changes by an order of magnitude across samples. A fixed physical spacing (variable node count) is preferable and is exactly what an isotropic remesh gives you.

### 2.5 Ray-cast `r*` semantics

`select_r_star_from_hits` marks a node ambiguous if any accepted hit is > 4 mm from the nearest one (double wall through the sac) and invalid if no accepted hit within 20 mm or `normal·ray < 0.2`. On the sac the intended behaviour (unlock Dirichlet via `SMOOTH_W_AMBIGUOUS = 0.05`) only triggers on double hits; rays that exit through the neck and miss, or graze at low `normal·ray`, stay `valid=False, ambiguous=False` and keep w = 1. So the smoothness prior is stiffest on part of the neck region where the wall must change fastest. Suggest: treat "no hit / low-dot" nodes whose Chamfer residual is large as soft (or simply use w = `SMOOTH_W_AMBIGUOUS` for all invalid nodes), or derive the weight from the GT itself (distance of the nearest GT point from the centerline) rather than from ray-cast success.

---

## 3. Encoder (PointNeXt) — assessment

Parameters: 18.5 M (16.8 M in the InvRes blocks, 12.6 M of that in stage 3 with 512×4 expansion; 1.2 M latent head; stem 192).

What is fine: mm-scale radii (1.5/3/6/12 mm) match vessel scales; canonical pose removes the need for rotation invariance; packed FPS/ball query through pytorch3d/pyg-lib is fast (64 ms forward for 16 k points).

Issues and suggestions:

1. **Input features are raw xyz only.** GT normals are already cached (`x_true_normal`) but not fed. Concatenating normals (and optionally `d_cl` and the local tube radius) to the stem is standard in PointNeXt and is the cheapest way to make the encoder sensitive to wall orientation and to sac vs parent.
2. **The latent head is a single-head, single-layer, positional-query cross-attention** from 96 queries (γ(u)+tract embedding, *no content*) to the 64 stage-4 tokens. Each latent token is thus a fixed soft-positional average of 64 global (12 mm) features projected by one linear layer; there is no FFN, no LayerNorm, no multi-head, and no way for a token to look at stage-2/3 tokens (3–6 mm) that hold sac-scale detail. Suggest a small transformer head (2 blocks: self-attention over the 64–256 encoder tokens, then cross-attention from the 96 queries, 4 heads, FFN, pre-LN) with keys taken from stage 2 *and* stage 4 (multi-scale keys). This is where "which side of the ICA the sac sits on" must be encoded; angular information reaches the token only through content, so the head needs capacity.
3. **Capacity is misallocated**: 12.6 M params in stage 3 (64 points!) versus 1.2 M in the head that actually forms the latent. Halving stage-3 width (256 hidden) and moving capacity into the head is cost-neutral.
4. **`logvar.clamp(-8, 2)`** kills gradients outside the range; with the tiny β the clamp floor will be reached. See §5.
5. **Stage-4 SA with 12 mm radius over 64 → 64 points** is essentially a global pooling; fine, but note the encoder never sees the scaffold or centerline beyond the nearest-CL `u`/tract used for the keys, so the latent is a shape code of the GT only. That is the right thing for Stage 2, and it is what Stage 1 must later predict.

---

## 4. Latent space and VAE — assessment (and Stage 1 consequences)

- **Size**: 96 tokens × 128 = 12 288 latent dims for 172 training samples. A Stage 1 prior (diffusion/flow over Z conditioned on a clinical vector + centerline) has to model a distribution in this space from ~200 examples. Realistic sizes for this dataset are a few hundred to ~2 k dims (e.g. 32–48 tokens × 32–48 dims), with tokens tied to a *fixed physical spacing* along the centerline so Stage 1 can generate them alongside the centerline.
- **β**: `LAMBDA_KL = 5e-4` multiplies a KL that is summed over 128 dims and averaged over tokens (≈ 15 nats/token at init → 0.007 in the loss versus recon 0.6). Even after warm-up the KL is < 1 % of the objective. Expect an autoencoder with σ → e^{-4}: excellent reconstructions, an unsmooth latent, and a Stage 1 that either overfits the 172 codes or produces off-manifold codes the decoder has never seen. Options: (a) β with free bits per dim (e.g. 0.05–0.1 nats/dim floor), monitoring active dims; (b) drop the VAE, train a deterministic AE and add *fixed* Gaussian noise `z + σ·ε` (σ ≈ 0.1–0.3 of the latent std) during decoder training, as the whiteboard specifies (`z ~ N(x0, I)` — with unit noise you would need to normalise the latent first); (c) latent-diffusion style: AE + light KL (β ≈ 1e-3 per dim with normalisation) then a diffusion prior. (b)/(c) are the modern default for mesh/point latents and are simpler to make robust than tuning β on 172 samples.
- **Token semantics**: tokens are `linspace(0,1)` along each tract plus junction tokens with `u = 1.0`. Junction tokens therefore share their positional key with every incident tract's last token. Give junction tokens their own learned positional embedding (or a distinct `kind` embedding) instead of `γ(1.0)`.
- **`LatentTractSelfAttention`** (window ±2, gate ≤ 0.5, applied identically in `forward` and `decode`) is fine. A wider window or 2 layers would let tokens see the whole tract; with 96 tokens the cost is nil.
- **Padding/`MAX_TRACTS = 16`** with degenerate 2-ring tracts wastes tokens today; after fixing §2.1 the typical ICA sample will have 2–5 tracts, and 96 tokens is then more than needed.

---

## 5. Decoder — assessment

Parameters: 24.8 M, of which 24.6 M are the 12 SplineConvs (2.05 M each: 5³ = 125 basis × 128 × 128). Cross-attention 3 × 59 k, coarse self-attention 82 k, heads 387 each.

### 5.1 SplineConv pseudo-coordinates — one correctness bug, one waste

```56:63:1test_encoder_decoder_only/train_pipeline/geometry.py
    e_u = 0.5 + 0.5 * (du / step).clamp(-1.0, 1.0)
    # ... 
    dth = wrap_pi(theta[dst] - theta[src])
    e_th = 0.5 + 0.5 * (dth / math.pi).clamp(-1.0, 1.0)
    # ...
    return torch.stack([e_u, e_th, e_kind], dim=-1)
```

- `e_u` is normalised by the ring step, so longitudinal neighbours land on 0 and 1 (kernel corners) — correct.
- `e_th` is normalised by π. On a 64-ring the circumferential neighbour has `dth = 2π/64 = 0.098` → `e_th = 0.516`; on the 6-ring coarse level 0.667. For the fine and mid levels the θ axis of the degree-2 B-spline kernel therefore sees 0.484 / 0.5 / 0.516 for (−θ, same, +θ) neighbours: the three basis weights are nearly identical, so the fine conv cannot tell "left ring neighbour" from "right ring neighbour" from "same column". The fine decoder is effectively an isotropic 1-D filter along u with a symmetric θ blur. Normalising `dth` by the local ring step (2π/n_radial), like `du`, puts neighbours on the kernel corners and restores a real 2-D (u, θ) kernel. Cross-tract edges can keep the π normalisation or get their own conv (below).
- `e_kind` is 0.5 for same-tract edges and 0 for cross-tract edges — one bit encoded on a 5-knot spline axis. That axis multiplies the parameter count and the `spline_weighting` FLOPs by 5. Replace with either (i) kernel `(5, 5, 2)` and degree 1 on the last axis (50 basis, −60 %), or (ii) a separate small conv for cross-tract edges and a 2-D kernel for grid edges (25 basis, −80 %).

### 5.2 Compute profile (3080 Ti, one sample: 64 000 fine nodes, 383 k directed edges)

| | time |
|---|---|
| full train step (fwd + bwd + AdamW) | **4.28 s / sample**, 2.7 GiB peak |
| `pyg::spline_weighting` fwd / bwd-x / bwd-w | 0.97 s / 0.95 s / 2.17 s — **97 % of GPU time** |
| encoder forward (eval) | 64 ms |
| decoder forward (eval) | ~1.0 s |
| losses forward (eval) | ~1.0 s (dominated by the sync after Chamfer/knn; knn itself 23 ms) |

So 172 samples ≈ 12 min/epoch, 200 epochs ≈ 41 h on the 3080 Ti; the spline kernel is scalar FP32 CUDA (no tensor cores), so an A100 buys perhaps 1.5–2×, not 5×. Levers, multiplicative: kernel 125 → 50 or 25 (2.5–5×), fine channels 128 → 64 (4×), fine nodes 64 k → ~12–16 k as the variable templates naturally provide (4–5×). Any two of these make a full run an overnight job on the 3080 Ti and a ~2–3 h job on an A100.

### 5.3 Aggregation and normalisation

```646:657:1test_encoder_decoder_only/train_pipeline/model.py
        kernel_size: int = SPLINE_KERNEL_SIZE,
        # ...
            kernel_size=kernel_size,
            degree=degree,
            aggr="add",
            root_weight=False,
```

- `aggr="add"` with `root_weight=False` and no normalisation layer anywhere in the decoder convs. Node degree is 6 on the grid but 50–100 at junctions with the current coupling edges; the pre-activation scale at junctions is 10× the rest. Even after fixing §2.1 the degree is non-uniform at ostia and at level boundaries. Use `aggr="mean"` (or degree-normalised sum) and add a pre-norm (LayerNorm/GraphNorm) inside `ResidualSplineConv`. This is also what will make the irregular template mesh (degree 5–8, varying) behave.
- No `root_weight`: a node's own feature only survives through the residual. Fine given the residual, but `root_weight=True` is cheap and standard.

### 5.4 Cross-attention from nodes to latent

- Queries are purely positional (γ(u), γ(θ)); keys are `w_k([z_k, γ(u_k)])`; values `w_v(z_k)`; output `out([a, γu, γθ])`. This is a content-keyed positional interpolation of the latent along u — sensible. Single head, no FFN, and the θ dependence enters only through `out`. Two heads (one attending "coarsely", one "locally") and a small FFN would help at negligible cost; more important is that angular structure lives entirely inside the 128-dim token, so keep token width ≥ 64 even when shrinking the latent.
- Masking with `-1e4` before softmax is fine in FP32.
- Fine level: 64 k queries × 96 keys is cheap (≈ 59 k params). OK.

### 5.5 Hierarchy and upsampling

`bilinear_cylindrical_upsample` is exact for the current grid (θ-wrapped, endpoints aligned) and is the single component that *cannot* survive an irregular template (see §10). The gating of the upsampled hidden state (`σ(α)`), the identity-initialised heads and the composition `dx_f = up(dx_m) + local` are good.

Missing conditioning: the decoder never sees the centerline geometry (curvature/torsion, local radius, distance to the nearest ostium) or the local tube radius as features — it sees only (u, θ) and dx. Adding `[R_local, κ, τ, d_ostium]` to the node features (cheap, available from the scaffold) gives the network a direct handle on "small daughter branch vs ICA trunk" and on bifurcation geometry.

### 5.6 Displacement head

Δr = softplus(·) − R (floor −R, so r ≥ 0), Δs = 3·tanh (shear ≤ 3 mm). With a constant 2 mm tube the floor is what daughter branches live near (§2.3). With MISR templates the floor should be a fraction of the local radius (e.g. −0.8·R_local) and the shear cap should scale with the sac size actually present (the largest aneurysms need 10–20 mm of *tangential* motion for a fixed-topology tube to wrap a dome; the variable remeshing reduces but does not remove this).

### 5.7 Output topology

Tubes overlap at bifurcations ("not a watertight Boolean"), and with duplicated tracts they overlap everywhere. For downstream CFD/statistics you need a manifold surface; the template pipeline (polyball → marching cubes → remesh) already produces one. This is the strongest architectural reason for the migration: the decoder output topology becomes the template's manifold topology.

---

## 6. Losses — assessment

- **Chamfer** (both directions, `1 + d_cl/R` weights capped at 4, point-to-plane Huber δ = 1 mm + 0.2 L2, normals from the GT side in both directions) is well designed. Two gaps:
  1. Pred side uses *vertices*. Where the tube must stretch 5–10× in area to cover a dome, dome vertices are sparse and the dome is under-supervised; sampling points uniformly on predicted faces (pytorch3d `sample_points_from_meshes` on the fine mesh) fixes the density bias and is compatible with any triangulation.
  2. Weights reward far-from-centerline GT points, which is right for sacs but catastrophic for the short-centerline samples (§2.2) where "far" means "uncovered vessel". Once centerlines cover the vessel this is fine; until then those samples must be excluded or the GT must be clipped to the scaffold's reach.
- **Radial Huber vs `r*`** (λ = 1) is the strongest and cleanest signal on the parent vessel; its coverage is limited by §2.1 (40 % valid). With MISR templates the template vertex *is* the ray origin, and `r*` becomes "signed distance along the template normal", valid almost everywhere except under the sac — much better.
- **Dirichlet on (Δr, Δs)**: λ = 0.15 on squared differences per edge. With 0.3 mm rings and a 10 mm sac over a 3 mm neck, every neck edge costs (1 mm)² unless unlocked by `ambiguous`. §2.5 explains why part of the neck stays locked. Consider a robust (Huber/L1) Dirichlet or a curvature-based (second-difference) term so that sharp-but-smooth necks are not penalised as if they were noise.
- **Uniform Laplacian on absolute positions** (λ = 0.05) penalises the tube's own curvature including the sac; the pytorch3d cotangent or the *displacement* Laplacian would not. Small weight, low priority.
- **Normal consistency** 0.02: fine.
- **KL**: see §4.
- **Missing terms worth having**: face-area / edge-length ratio regulariser (limits triangle stretch, the actual failure mode on domes), and a self-intersection penalty at bifurcations once tubes are replaced by a single manifold (then unnecessary).
- Validation score = recon + rad (no smoothness) — good choice for model selection.

---

## 7. Training framework — assessment

| item | current | issue / suggestion |
|---|---|---|
| batch | bs 1 × accum 8 → 21–22 optimizer steps/epoch, 4 300 steps total | fine for a start; after the speed-ups in §5.2, bs 2–4 real batches are feasible and reduce Python per-graph loops |
| optimizer | AdamW 2e-4, wd 1e-4 (no exclusion of norms/biases/gates) | OK; exclude LayerNorm/bias/gates from wd; a 200–500-step linear LR warm-up before cosine is standard with attention + zero-init gates |
| schedule | cosine over 200 epochs, no warm-up, no restarts | fine |
| clip | global 1.0 | fine |
| EMA | 0.999 per optimizer step | horizon 1 000 steps ≈ **47 epochs**; the first ~8 validations (every 5 epochs) evaluate near-initial EMA weights, and `best.pt` is chosen on them. Use 0.99–0.995, or the standard warm-up `decay = min(d, (1+n)/(10+n))` |
| validation | every 5 epochs, EMA weights, 28 samples | fine; also log the non-EMA val loss so you can see the gap |
| augmentation | none | add: left/right mirror (LICA↔RICA, doubles the data; mirror *before* scaffold generation so frames stay consistent), per-epoch resampling of `x_true` from the full GT (store the full GT points, not 16 384, in the cache), small random shifts of the ring/θ origin phase, small canonical-pose jitter (±5°) |
| resume | none — `last.pt` is written but never loaded | required on the HPC (wall-time limits); resume model/EMA/opt/sched/epoch/RNG |
| logging | `print` + `history` saved only at the end | write CSV/TensorBoard per epoch, save `history.json` every epoch; currently a killed run leaves only `last.pt` (the existing one is epoch 1) |
| device | `gpu_index = 1 if n_gpu > 1 else 0` hardcoded in `aneuxai.py` and `postprocess.py` | on a 4-GPU node this always takes GPU 1; use `CUDA_VISIBLE_DEVICES` / argument |
| multi-GPU | none | 4 × A100 are best used as 4 independent runs (β, latent size, kernel variants). DDP over 172 samples buys little |
| precision | FP32 tensors, TF32 matmuls | correct choice; the spline kernel and knn are FP32 CUDA anyway. Don't add AMP |
| gradient checkpointing | `"off"` | correct: peak is 2.7 GiB, checkpointing would only add recompute |
| dataloader | 3 workers, `torch.load` of 16 MB per item | fine |
| test split | none (train/val only) | keep a 10–15 % test set untouched for the final paper numbers, or do k-fold once the run is < 3 h |
| determinism | seeds set; FPS start random | fine |

Throughput reality check (3080 Ti, current model, current data): 41 h per 200-epoch run, single GPU, 2.7 GiB. On the HPC one A100 ≈ 20–25 h. After §5.2 changes: ≈ 3–6 h on the 3080 Ti.

---

## 8. Checkpoint / run evidence

`output/last.pt` is from **epoch 1**, with epoch-average train metrics `recon 15.5`, `rad 0.96`, `disp 0.45`, `kl 147.6`. For comparison, a *freshly initialised* model on the long-centerline sample SNF00000100 already scores `recon 0.60`, `rad 0.09`: the epoch mean is 25× worse than a clean sample at init, which is the short-centerline / uncovered-vessel group (§2.2) dominating the loss. There is no evidence a full run has ever completed; every loss weight in `config.py` is therefore untested at convergence. Treat the weights as priors, not tuned values, and plan a short sweep (β, λ_disp, λ_rad, λ_lap) once the data is fixed.

---

## 9. Smoke tests (`test_architecture.py`)

Broad and valuable (shapes, KL, pseudo-coords range, upsampling, Bishop orthogonality, FPS counts, Chamfer/mesh losses vs pytorch3d, spline backend, junction coupling, batching, decode-without-vessel, precision flags). Missing tests that would have caught the issues above: (i) tract extraction on a real VMTK centerline with shared parent segments (asserts no coincident tracts); (ii) `e_th` of a circumferential neighbour is far from 0.5; (iii) `r*` valid fraction on a synthetic straight tube ≈ 1.0; (iv) scaffold coverage of `x_true` (fraction of GT within 2 R of a scaffold node) above a threshold; (v) EMA horizon versus steps/epoch.

---

## 10. Fit to the future `variable_remeshing.py` dataset

### 10.1 What the templates are (95 produced so far, `output_variable_remeshed/`)

| property | value |
|---|---|
| samples produced | 95 (SNF + p* only; none of the UPF/USFD/ANSYS short-centerline cases) |
| vertices | 2.3 k – 34 k, median 12.5 k (GT vessel median 22.8 k) |
| edge length | median 0.36 mm; min 0.002–0.1 mm (a few near-degenerate edges per mesh), max ≈ 0.8 mm |
| template → GT distance | median 0.15 mm, p95 0.3–4.5 mm, max 1–6 mm (two outliers p133/p166 at 5 mm) |
| GT → template distance | median 0.2 mm, max median **7.5 mm** (the sac), 36–50 mm in SNF00000063/SNF00000215 (template missed a whole branch — 2.3 k / 5.8 k vertices) |
| open boundaries | 2–6 |
| point data | only `Normals` — no centerline id, no (u, θ), no MISR, no GroupIds |

Implications: the template is a good parent-vessel proxy with MISR-correct radii and a single manifold; the aneurysm is the only large residual (as intended). But it is an irregular triangulation with per-sample topology, so:

### 10.2 Which parts of the current pipeline break

| component | assumes | on the template |
|---|---|---|
| `allocate_ring_counts`, `_generate_level`, `branch_nl`/`n_radial` bookkeeping | regular rings × radial grid per tract | no rings; vertex count varies 15× |
| `bilinear_cylindrical_upsample` / `upsample_branch_concat` | source and target grids per branch | no grids |
| `intrinsic_spline_pseudo_coords` | Δu in ring steps, Δθ | needs a (tract, u, θ) per vertex — not present |
| `r_star_grid_stats` (`dth`, `du`, ring medians) and `smoothness_edge_weights` | ring/column neighbours | no rings |
| `radial_bias_for_zero_init`, `clamp_residual_radial`, `DecoupledDisplacementHead` | constant tube R | MISR radius varies per vertex |
| `_build_latent_tokens` (tract-proportional, junction tokens) | tracts from `extract_unique_tracts` | needs the pipeline's `vmtkBranchExtractor` GroupIds instead |
| coarse positional self-attention masks, `attend` tables | per-tract node ranges | per-vertex tract id from projection |
| `postprocess.tensor_to_vtp` | grid faces | template faces (actually simpler) |
| Chamfer, radial Huber, normal consistency, Laplacian | any mesh | unchanged |
| Encoder | GT point cloud | unchanged |

Roughly: encoder and losses survive; `dataset.py`, `raycast.py` (partly), `geometry.py` (upsampling, pseudo-coords, clamps) and the decoder's level plumbing need rework. This is a redesign of the scaffold layer, not a "slight change".

### 10.3 Recommended migration design (keeps the centerline-intrinsic idea)

1. **Extend `vessel_pipeline.py` to export, per template vertex**: nearest branch (`GroupId` from `vmtkBranchExtractor`), normalised arc position `u` on that branch, angle `θ` in that branch's Bishop frame, MISR radius `R_local`, and the branch tree (parent/child, ostium positions). All of this exists inside the pipeline at template-build time; writing it as point-data arrays costs nothing and removes `extract_unique_tracts`, `_orient_tracts` and most of `dataset.py`'s scaffold code. Also export the smoothed branched centerline used to build the template.
2. **Build the hierarchy as three isotropic remeshes of the same template** (edge ≈ 1.5 / 0.8 / 0.36 mm — the pipeline already has the remesher) or by quadric decimation. Inter-level upsampling then becomes **barycentric interpolation onto the nearest coarse face** (or 3-NN inverse-distance) precomputed once per sample and cached as index/weight tensors. This replaces `bilinear_cylindrical_upsample` with a general sparse-matrix multiply that is identical for every mesh.
3. **Keep SplineConv with pseudo-coords (Δu/ℓ, Δθ/ℓ_θ, kind)** where ℓ is the level's target edge length in u (arc mm) and ℓ_θ the equivalent angular step at `R_local` — i.e. normalise by *physical* edge length so kernels are scale-consistent across samples (this is also the fix for §5.1). Edges = template triangle edges (+ optional kNN across branches near ostia, k small).
4. **Displacement in the template's own frame**: Δr along the template vertex normal (or the radial direction from the branch centerline — nearly identical on a MISR tube), Δs in the tangent plane; floor `−0.8·R_local`, shear cap ∝ max(3 mm, k·R_local). Ray-cast `r*` from the template vertex along its normal (with the existing Voronoi/ambiguity logic, now trivially valid on the parent vessel).
5. **Latent tokens at fixed physical spacing** (e.g. one token per 3–4 mm of branch + one per ostium, capped ≈ 32–48) so Stage 1 can emit them together with the centerline.
6. **Cache** the full GT point set (not 16 k) and resample `x_true` per epoch; store the mirror-augmented copy as a second sample.
7. **Quality gates at cache time**, so the model never sees samples like SNF00000063/SNF00000215: GT→template max ≤ some multiple of the largest sac height, coverage of the GT by the template ≥ 95 % within 2·R_local outside the aneurysm, no template edge < 0.05 mm, ≥ 2 openings, all GT points within X mm of the exported centerline.

The alternative — dropping (u, θ) entirely and using a generic mesh GNN (e.g. SplineConv on 3-D relative coordinates or a MeshCNN/DiffusionNet block) with FiLM/AdaIN from a global latent, as the draft suggests — is simpler to implement but loses the along-centerline latent structure that makes Stage 1 tractable. I would keep the intrinsic design.

### 10.4 Template coverage of the dataset

Only 95/200 templates exist and they are exactly the long-centerline SNF/p* group. The 94 short-centerline cases will not produce usable templates until they get full-vessel centerlines (re-extract with sources/targets at the actual vessel openings). Decide early whether the Stage 2 dataset is "the 95 (+ mirror = 190)" or whether centerline re-extraction for the rest is worth it; it changes how much regularisation you need.

---

## 11. Stage 1 integration notes

- Stage 1 must produce (a) a branched centerline with radii and (b) the latent tokens. The template pipeline turns (a) into a mesh deterministically (polyball → marching cubes → remesh) — so at inference the *same* pipeline runs on a generated centerline, and Stage 2 only adds the aneurysm/texture. This is a clean split, and it means the decoder's job at training time should be "residual on top of a MISR template", i.e. mostly the sac. Make sure Stage 2 is trained on templates built from *smoothed* centerlines/radii of the kind Stage 1 can plausibly output, not from ground-truth-exact ones, otherwise Stage 2 will over-trust the template.
- Because Stage 1 will output *imperfect* latents, the Stage 2 decoder should be trained with latent noise (learned posterior width or fixed σ — see §15) and the latent should be normalised (zero-mean, unit-var per dim, or a learned LayerNorm on z) so a diffusion/flow prior has a well-scaled target.
- Token count/positions must be a deterministic function of the centerline (fixed spacing + ostium tokens), not of arbitrary VMTK path enumeration.
- Canonical pose must be computable from a generated centerline alone (it is: inlet position + tangent + a third axis from the first bifurcation direction), and must be *stable* — the current highest-degree-node heuristic is not.

---

## 12. Prioritised change list (Part I version — superseded by §18 after the answers in §13)

**P0 — before any further training**
1. Fix tract extraction: use `vmtkBranchExtractor` GroupIds/`CenterlineIds` (or merge polylines with tolerance ≈ 0.25·MISR), assert no coincident tracts; re-derive junctions from the branch tree. Recache.
2. Exclude (or re-extract centerlines for) the 94 short-centerline samples; add coverage/quality gates at cache time.
3. `e_th` normalisation by ring step (`geometry.intrinsic_spline_pseudo_coords`).
4. EMA decay 0.99–0.995 with warm-up; add resume; per-epoch history/CSV; device selection via env/arg.

**P1 — architecture (do together with the template migration, §10.3)**
5. Kernel `(5,5,2)` degree-(2,2,1) or separate cross-tract conv; fine level 64 channels; `aggr="mean"`, pre-LayerNorm, `root_weight=True`.
6. Encoder: normals (+ `d_cl`) as input features; 2-block transformer latent head with multi-scale keys; shrink stage-3 width.
7. Latent: 32–48 tokens × 32–64 dims at fixed spacing; either β-VAE with free bits or AE + fixed-σ noise + latent normalisation. Decide with Stage 1 in mind.
8. Decoder node features: `R_local`, curvature, torsion, distance to nearest ostium.
9. Losses: face-sampled pred points for Chamfer; robust Dirichlet; unlock smoothness on all `valid=False` nodes or drive weights from GT-to-centerline distance; MISR-relative radial floor/shear cap.
10. Augmentation: L/R mirror (before scaffold build), per-epoch `x_true` resampling, θ-phase and pose jitter.

**P2 — nice to have**
11. Small LR warm-up; wd exclusion for norms/biases/gates; log non-EMA val too; held-out test set.
12. Tests for tract uniqueness, `e_th`, `r*` validity on a synthetic tube, scaffold coverage.
13. Use the 4 A100s for parallel sweeps (β, latent size, λ_disp, kernel) rather than DDP.

---

## 13. Questions asked after Part I, and the answers (9 Sep)

| question | answer |
|---|---|
| Dataset scope | Everything will be regenerated from the original files: `centerline_creation.py` for centerlines, `variable_remeshing.py` for the templates, uniform remeshing for all cases; small training-pipeline changes to consume them. |
| VAE vs AE + noise | Open — asked for the trade-off and a recommendation (→ §15). |
| Token placement for Stage 1 | Tokens will be generated along the *entire* centerline. |
| Watertight / single manifold | Mandatory (CFD downstream), open outlets allowed. |
| HPC wall-time | Effectively unlimited; compute should not be wasted, but accuracy of the final model matters more than training speed. VRAM headroom is available for anything worth it. |

---

# Part II — refined analysis given the answers

## 14. Data pipeline: what the regenerated dataset will actually contain, and what must still change

All three scripts call the same `build_parent_tube` (Taubin → open-profile detection → flow extensions → capped Voronoi centerline with MISR → 0.1 mm spline resampling → Laplacian smoothing with MISR re-copied → `vmtkBranchExtractor` → polyball/marching-cubes tube → pipe-section uncap). Consequences for the training pipeline:

1. **The centerline problem of §2.1 disappears only if the training code consumes the branch arrays.** `extract_branches` runs `vmtkBranchExtractor`, and `clip_centerline_at_profiles` copies every point-data array (`GroupIds`, `CenterlineIds`, `TractIds`, `Blanking`, `MaximumInscribedSphereRadius`) onto the clipped output. The output is still one polyline per (centerline, tract), so parent segments still appear once per source→target path; but every duplicate carries the same `GroupId`. Replace `extract_unique_tracts` with "one representative polyline per `GroupId`" (or run `vmtkCenterlineMerge`, which does exactly this) and derive the tree from group adjacency (`Blanking == 1` tracts are the bifurcation regions). This is the single most important data change and it is small.
2. **The 94 short-centerline samples will be fixed by construction** — the new centerlines run inlet→every outlet on the *original* vessel — provided those vessels have detectable open profiles. `1or0.py` / `hascap.csv` (the keep/discard labelling of `VESSELS_ORIGINAL`) is exactly the curation this needs; keep the discard list as a cache-time exclusion.
3. **Build the three outputs from one `build_parent_tube` call.** Today `centerline_creation.py`, `uniform_remeshing.py` and `variable_remeshing.py` each rebuild the tube independently. VMTK's Voronoi centerline is deterministic for identical input, but the flow-extension retry path in `extract_centerlines_for_tube` (extended vs anatomical seeds) and the smoothing can diverge across runs if any parameter differs. One process writing `{id}_centerline.vtp`, `{id}_uniform.vtp`, `{id}_variable.vtp` guarantees the centerline the training code projects onto is the one the template was built from. Also write, per template vertex, the arrays the decoder needs: `GroupId`, `u` (normalised arc position within the group), `theta` (angle in the group's Bishop frame), `R_template` (already computed by `compute_template_local_radii`), and `StretchDistance` (already computed by `compute_raycast_stretch_distances` — this *is* the fine-level `r*`, for free).
4. **The uniform template is the natural coarse level.** Uniform and variable remeshes sample the *same* open base surface, so barycentric projection between them is exact up to remesh error. Hierarchy: uniform at ~1.2–1.5 mm (coarse, ~1–2 k vertices), uniform at 0.5 mm (mid, the current `DEFAULT_TARGET_EDGE_LENGTH`), variable (fine, 0.5 mm base, down to 0.01 mm under the sac). Export one extra coarse uniform remesh per case; the interpolation index/weight tensors are computed once at cache time.
5. **Known geometric caveat to check on the regenerated data: Voronoi centerlines are attracted into the sac.** VMTK paths minimise ∫1/R, so for large sidewall aneurysms the path bends toward the sac centre and the MISR balloons; `clamp_misr_for_parent_tube` caps the radius but not the path. Template→GT max distances of 1–6 mm in §10.1 are consistent with that. Two consequences: the "healthy" template bulges under the sac (less for Stage 2 to explain — fine), and the centerline itself leaks sac position/size (Stage 1 must then generate that bend, which is consistent but makes the "centerline = healthy parent" story less clean). A diagnostic worth adding at cache time: per sample, max deviation between the centerline and a straight-through spline fitted on the healthy segments either side of the sac, and the MISR cap hit rate. If the deviation is large on many cases, the cleaner alternative is to build templates from aneurysm-removed vessels — the `hemoMesh --removal` Voronoi reconstruction already exists in the repo and has a point-picking tool, at the cost of one manual pick per case.
6. **GT for the loss**: use `VESSELS_ORIGINAL` (all vertices, with normals) rather than a 16 384-point subsample; cache the full point set and resample per epoch (§7). The original meshes have 15–50 k vertices — trivially within VRAM.
7. **Template quality gates at cache time** (replacing the `assert_template_*` checks that only guard the remesher): GT→template p99 outside the sac ≤ 0.5·R_local; template→GT max ≤ 1.5·max(R_local) away from the sac; every anatomical opening present (`n_clipped == n_profiles`, currently only a warning); no edge < 0.05 mm; one connected component; exactly `n_profiles` boundary loops. SNF00000063 / SNF00000215 (36–50 mm GT→template) must fail these.

## 15. VAE vs AE + noise — trade-off and recommendation

Framing: Stage 1 will generate, along the whole centerline, a sequence of (x, y, z, r, z_i) per centerline sample. The Stage 2 latent is therefore a *tree-structured sequence of local tokens*, not a global code, and Stage 1 is a conditional sequence generator (diffusion/flow-matching or autoregressive transformer) over that tree. What Stage 2's latent regulariser must deliver for that consumer:

- a fixed scale per dimension (Stage 1 losses and noise schedules assume it),
- smoothness: nearby codes decode to nearby shapes (Stage 1 will land *near*, not *on*, training codes),
- decoder robustness to the kind of error Stage 1 will make,
- ideally: healthy segments map to a known "null" code so Stage 1 only has to model where the aneurysm is.

| | β-VAE (meaningful β) | AE + fixed-σ noise + standardisation | Light per-token KL-VAE with free bits (recommended) |
|---|---|---|---|
| Reconstruction accuracy | Costs accuracy: β trades recon for KL directly; on 172 samples the trade-off is steep and hard to tune. | Maximal; nothing competes with recon. | Near-AE accuracy: the free-bits floor means KL only bites on dims the model does not need. |
| Latent scale | Automatically unit-scale. | Must be standardised explicitly (per-dim mean/std from the training set, or a LayerNorm on z); σ is then in those units. | Automatically near unit-scale; a final standardisation pass is still cheap insurance. |
| Healthy segments | Posterior collapses to the prior on tokens that carry no information — here that is *desired*: the template already explains the healthy wall, so those tokens should be ≈ N(0, I). | No pressure toward a null code; healthy tokens get arbitrary codes Stage 1 must learn to reproduce. | Same desirable collapse on healthy tokens; sac tokens stay informative. Per-token KL along the centerline becomes a direct diagnostic of "where does the template fail to explain the wall". |
| Robustness to Stage 1 error | Built in (posterior sampling). | Built in, and the level is under direct control (σ). | Built in at the learned posterior width; add a small extra noise term if the posterior gets too narrow. |
| Prior sampling without Stage 1 | Possible — good sanity check. | Not possible. | Possible for healthy tokens; sac tokens are off-prior by design, so full-shape prior samples are not meaningful. |
| Diagnostics | Active-unit count tells you the effective latent size. | None built in. | Active units + per-token KL profile. |
| Failure modes | Posterior collapse of *everything* (blurry sacs) if β too high. | Latent uses capacity wastefully; σ mis-set gives either a brittle decoder or a blurry one. | β/free-bits still two knobs, but with much wider safe range. |
| Match to whiteboard | — | Matches `z ~ N(x0, I)` (with standardisation; unit noise on an unstandardised latent is arbitrary). | Equivalent to the whiteboard with a learned rather than fixed noise level. |

**Recommendation**: the third column. Concretely: per-token KL to N(0, I) with a free-bits floor of ~0.1–0.25 nats per dim, β set so that the total KL is roughly 2–5 % of the reconstruction term at convergence (start at β ≈ 1e-2 on a per-dim-averaged KL and adjust), 20-epoch warm-up kept. Monitor three things: active units per token type (healthy vs sac), the per-token KL profile along the centerline, and a *noise-robustness curve* — reconstruction error when decoding μ + σ·ε for σ ∈ {0, 0.25, 0.5, 1} in standardised units. That curve is the interface contract with Stage 1: Stage 1 must land within the σ where Stage 2 still reconstructs acceptably. If β turns out untunable on 200 samples (recon degrades before healthy tokens collapse), fall back to column 2 with σ ≈ 0.2–0.3 standardised. In either case, once Stage 1 exists, fine-tune the Stage 2 decoder on Stage 1 samples (teacher-forced centerline, generated tokens) — the real error distribution beats any noise model.

Latent geometry that follows from "tokens along the whole centerline":

- One token per centerline sample at a fixed spacing `ds_tok` shared by both stages (2 mm is a good starting value: ~40–70 tokens for an ICA segment with branches; 1 mm doubles that). Ostium tokens are then unnecessary — the bifurcation is where two branches' token sequences meet, and the branch tree provides the adjacency.
- Token width 16–32 dims. Total latent 1–2 k dims, but *local*: each token has to describe the wall within roughly ±`ds_tok` and a few radii. The aneurysm is 5–15 tokens.
- Encoder head that produces exactly this: assign each GT point to its nearest centerline sample (Voronoi along the tree), pool PointNeXt features per sample (attention pooling), then 2–4 transformer layers over the token tree with arc-length + branch-depth positional encodings. This replaces `CenterlineLatentHead` and makes the latent explicitly local, which is the right inductive bias for 200 samples: every sample contributes 40–70 local training examples of "wall residual given local code".
- Decoder cross-attention is then a short-range operation (a template vertex attends to the ~5 nearest tokens along its branch, plus the neighbouring branch's tokens near an ostium) — cheaper and more stable than the current all-tokens-of-the-tract softmax.
- Stage 1 sequence element: `[x, y, z, r, z_1 … z_D]` per centerline sample, exactly the whiteboard's `x, y, z, r, c` row.

## 16. Watertight, single-manifold output for CFD

Deforming the template mesh makes the output inherit the template's topology (one component, `n_profiles` open loops), so watertightness is preserved *iff* the deformation does not fold. Required additions:

1. **Outlet planes**: boundary-loop vertices must stay in the anatomical profile plane. Constrain their displacement to the plane (project Δ onto the plane; allow radius change) or freeze them; the GT was clipped on the same planes, so this costs nothing in accuracy.
2. **Fold prevention**: penalise triangles whose deformed normal flips relative to the template normal (hinge on `n_pred · n_template < 0`) and a stretch regulariser (per-triangle singular-value or edge-length-ratio penalty, ARAP-style). Both are cheap and they are the actual failure modes of a fixed-topology deformer on overhanging (bottleneck) sacs.
3. **Displacement parametrisation**: the current 3 mm shear cap and the radial floor were sized for a 2 mm tube. On bottleneck aneurysms, dome vertices need to travel several radii *tangentially* from their template location; a hard 3 mm cap makes those unreachable. Suggest: free 3-D displacement at the coarse level (few vertices, dome placement), local-frame residuals with MISR-relative bounds at mid/fine.
4. **Verification in `postprocess.py`**: self-intersection count (`vtkIntersectionPolyDataFilter` self-intersection mode or pymeshlab), boundary-loop count, connected components, min triangle angle / aspect ratio. Make these part of the validation metrics, not an afterthought.
5. **Final remesh**: an isotropic remesh (the pipeline's own remesher with `PreserveBoundaryEdges`) of the decoded surface before CFD is normal practice and should be assumed; it decouples the training mesh (dense under the sac) from the CFD mesh.
6. Because the variable template is dense under the sac and the deformation stretches those triangles, the *deformed* fine mesh is roughly uniform — that was the point of the stretch-driven `k = 1 + d/R` sizing, and it is the right design for CFD-quality output.

## 17. Spending compute for accuracy, not speed

Given unlimited wall-time and spare VRAM, the priorities change: speed-ups matter only insofar as they buy more optimisation steps and more experiments. Where extra capacity/compute actually converts into accuracy here:

| lever | expected effect | cost |
|---|---|---|
| Denser supervision: full original GT point set (15–50 k) + face-sampled predicted points, resampled each epoch | Direct: removes the 16 k sampling floor in Chamfer and the vertex-density bias on the dome | Memory only; knn is 23 ms |
| Augmentation (L/R mirror, θ-phase, ±5° pose jitter) | Largest single generalisation gain on 200 samples | Free |
| Many more optimisation steps (currently 4.3 k). Mesh deformers typically need 2–5·10⁴ | Training is far from converged at 200 epochs of 21 steps | Time; affordable once the spline kernel is fixed |
| Decoder depth (6–8 residual convs/level with pre-norm) and multi-head cross-attention with FFN, instead of the 5³ kernel | Capacity where it is used | Modest after the kernel fix |
| Transformer latent head with local pooling (§15) | Better latent, less overfitting | Small |
| Encoder input normals | Cheap accuracy | Free |
| An extra finest level at the template's native resolution if the variable mesh is coarsened for mid | Sac detail | VRAM (available) |
| Curriculum: radial-only warm-start on the parent vessel, then unlock sac terms | Faster, more stable convergence | Free |
| k-fold CV (5 folds) instead of a single 172/28 split | Error bars on every design decision; with unlimited wall-time this is what makes sweeps trustworthy | 5× compute per config |
| Batch size > 1 | Stabilises gradients; does not improve accuracy by itself | Use to fill VRAM, not as a goal |
| DDP over 4 A100 | Not worth it at this data size | — |

Evaluation protocol to make "accuracy" measurable (report on the held-out fold, mm units, non-Huber): symmetric Chamfer mean and p95, Hausdorff, normal angle error, and *sac-specific* versions using a sac mask (GT points with `StretchDistance` > 1 mm or the hemoMesh picked-point neighbourhood), plus neck-plane error, sac volume error, and the mesh-validity counts from §16.4. The current validation score (Huber recon + radial) is fine for model selection but not for reporting.

## 18. Revised prioritised change list (supersedes §12)

**P0 — data (blocking everything else)**
1. One-process regeneration of centerline + uniform + variable templates from `VESSELS_ORIGINAL` (gated by `hascap.csv`), exporting per-vertex `GroupId`, `u`, `theta`, `R_template`, `StretchDistance` and the branch tree; cache-time quality gates (§14.7); sac-attraction diagnostic (§14.5).
2. Training-side scaffold from the branch arrays (one polyline per `GroupId`), fixed-spacing tokens along the tree, barycentric inter-level interpolation, full GT point set cached.

**P1 — model (with the migration)**
3. Pseudo-coords normalised by physical edge length in u and θ; kernel `(5,5,2)`/separate cross-branch conv; `aggr="mean"`, pre-norm, `root_weight=True`; 6–8 convs/level at 64–128 channels.
4. Local-pooling transformer latent head; tokens 16–32 dims at 2 mm; light per-token KL with free bits; noise-robustness curve logged every validation.
5. Displacement: free 3-D at coarse, MISR-relative local-frame residuals at mid/fine; outlet-plane constraint; fold and stretch penalties.
6. Losses: face-sampled Chamfer on full GT, robust Dirichlet, smoothness weights from `StretchDistance` rather than ray-cast success, radial Huber against `StretchDistance` on the template.
7. Encoder: normals in; stage-3 width down, head capacity up.

**P2 — training framework**
8. Resume, per-epoch logging, EMA 0.99–0.995 with warm-up, LR warm-up, wd exclusions, device via env; augmentation set of §17; curriculum; 5-fold CV harness; the metrics of §17 and validity checks of §16.4 in `postprocess.py`.
9. Tests: one-polyline-per-GroupId, `e_th` of a ring neighbour ≈ 0/1, `r*` ≈ `StretchDistance` on a synthetic tube, inter-level interpolation reproduces a linear field exactly, outlet vertices stay in plane, mirror augmentation round-trips the Bishop frame handedness.

## 19. Remaining open questions

1. "Uniform remeshing for all of them": is the uniform template meant as the coarse hierarchy level (my assumption in §14.4), or as a separate training variant? If the latter, the coarse level still needs an extra ~1.5 mm remesh export.
2. Are you willing to spend one manual pick per case for `hemoMesh --removal` if the sac-attraction diagnostic (§14.5) shows the Voronoi centerline bends into the sac on many cases? It changes what "template" and "centerline" mean for both stages.
3. Token spacing for Stage 1: 1 mm or 2 mm along the centerline (fixes latent length and the Stage 1 sequence length)?
4. Does the CFD downstream need the outlets exactly planar and circular, or only planar? (Decides whether boundary vertices are frozen or only constrained to the plane.)
