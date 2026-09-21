"""Hyperparameters for the hierarchical PointNeXt–SplineConv deformation VAE.

Values follow the architectural specification. Physical units are millimetres.

This package trains Stage 2: a geometry VAE that maps a vessel surface to a
tree-valued latent Z on the centerline and decodes (Z, centerline tube) to a
deformed mesh. A future Stage 1 will map a condition vector to a centerline
and to Z on that centerline, then call Stage 2's `decode()`.

Stage-2 precision (see configure_stage2_precision):
  CPU preprocessing in dataset.py is NumPy float64 (splines, Bishop frames,
  arc-length, COM). All model parameters, GPU tensors, and Data fields are
  torch.float32. Ampere matmuls use TF32 execution with FP32 storage.
  Autocast / bfloat16 / float16 are not used in Stage 2.
"""

import math
import os

import torch


def configure_cuda_allocator():
    """Linux CUDA caching allocator can grow in-place; Windows cannot."""
    key = "PYTORCH_CUDA_ALLOC_CONF"
    if os.name == "nt":
        conf = os.environ.get(key, "")
        parts = [
            p.strip()
            for p in conf.split(",")
            if p.strip() and "expandable_segments" not in p
        ]
        if parts:
            os.environ[key] = ",".join(parts)
        else:
            os.environ.pop(key, None)
        return
    os.environ.setdefault(key, "expandable_segments:True")


def configure_stage2_precision():
    """FP32 tensor storage, TF32 Ampere matmuls, no autocast or reduced dtypes."""
    configure_cuda_allocator()
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.sparse.check_sparse_tensor_invariants.disable()


configure_stage2_precision()

# --- Scaffold / data ---
# Fallback only when r_local is unavailable. The real radius is the cached
# per-vertex r_local (§7.3, Appendix B item 7); do not use this as a stand-in
# for Chamfer weights, composed_radius, or the displacement-head floor.
TUBE_RADIUS_MM = 2.0
N_TRUE = 16384
N_TRUE_FAR_FRAC = 0.25
FAR_CL_MARGIN_MM = 1.0
# Cache contents change with this bump: full GT, latent_valid, r*, 2 mm tokens.
CACHE_VERSION = 10

# (n_length, n_radial) per hierarchy level
LEVEL_COARSE = (40, 6)
LEVEL_MID = (250, 12)
LEVEL_FINE = (1000, 64)
HIERARCHY_LEVELS = (LEVEL_COARSE, LEVEL_MID, LEVEL_FINE)

# Template hierarchy is built at cache time by decimating template_mesh.
# No coarse_remeshed folder: mid/coarse are derived from the same template.
TEMPLATE_MID_KEEP = 0.25
TEMPLATE_COARSE_KEEP = 0.08
TEMPLATE_MIN_MID = 256
TEMPLATE_MIN_COARSE = 64
TEMPLATE_UPSAMPLE_K = 3

JUNCTION_COUPLE_RADIUS_MM = 4.0
JUNCTION_COUPLE_K = 2

MIN_RINGS_PER_BRANCH = 2
MAX_TRACTS = 16
DENSE_CL_SPACING_MM = 0.2
# Snap tract endpoints when rebuilding the GroupId tree (§2.4.3).
GROUPID_ENDPOINT_SNAP_MM = 1.0

# --- Intrinsic Fourier encodings ---
K_U = 8
K_THETA = 6
GAMMA_U_DIM = 2 * K_U  # 16
GAMMA_THETA_DIM = 2 * K_THETA  # 12

# --- Latent trajectory (tree-valued; 1 mm / 2 mm contract, §5.4) ---
# Shared with Stage 1. Distinct from DENSE_CL_SPACING_MM (cache-time samples).
TOKEN_SPACING_MM = 2.0
CL_SAMPLE_MM = 1.0
# LATENT_LEN is a padding maximum, not the token count. Per branch
# n_tok = floor(L / TOKEN_SPACING_MM) + 1; unused slots are masked by
# latent_valid (head, mixer, decoder cross-attention, and KL). No separate
# junction tokens — the daughter's first token is the junction. p95 tree
# ~216 mm / 2 mm ≈ 108 tokens plus margin → 128.
LATENT_LEN = 128
LATENT_DIM = 16  # measured §5.3.3a; Option C §5.3.6 item 1
# Soft σ bound (§5.3.6 item 2):
#   logvar = LOGVAR_MIN + (LOGVAR_MAX - LOGVAR_MIN) * sigmoid(raw)
# Gradient never dies, unlike a hard clamp (§4.2 item 4).
SIGMA_MIN = 0.1
SIGMA_MAX = math.e  # log σ²_max = 2
LOGVAR_MIN = 2.0 * math.log(SIGMA_MIN)  # ≈ -4.605
LOGVAR_MAX = 2.0 * math.log(SIGMA_MAX)  # 2.0
# LOGVAR_CLAMP = (-8.0, 2.0)  # deprecated hard clamp (σ down to e^{-4} ≈ 0.018).
# Live alias so existing imports still resolve until items 17/19 migrate.
LOGVAR_CLAMP = (LOGVAR_MIN, LOGVAR_MAX)
# Decoder cross-attention locality (§5.4): K nearest tokens on the vertex's
# branch, plus neighbouring-branch tokens within OSTIUM_NEIGHBOR_MM of an ostium.
TOKEN_ATTEND_K = 5
OSTIUM_NEIGHBOR_MM = 4.0
# Active-unit test for post-training standardisation (§5.3.6 item 7, §5.3.7).
ACTIVE_UNIT_KL_THRESH = 0.01
Z_ATTN_HEADS = 4
Z_ATTN_ALPHA_INIT = 0.1
Z_ATTN_RADIUS = 2
Z_ATTN_ALIBI = 8.0
Z_ATTN_GATE_MAX = 0.5
COARSE_ATTN_RINGS = 4
COARSE_ATTN_OSTIUM_U = 0.15
COARSE_ATTN_GATE_MAX = 0.5

# --- PointNeXt encoder ---
STEM_DIM = 32
RADIUS_NEIGHBOR_CAP = 256
SA_STAGES = (
    # n_out, radius_mm, neighbor_cap, hidden_dim, n_invres
    (1024, 1.5, RADIUS_NEIGHBOR_CAP, 64, 2),
    (256, 3.0, RADIUS_NEIGHBOR_CAP, 128, 2),
    (64, 6.0, RADIUS_NEIGHBOR_CAP, 256, 2),
    (64, 12.0, RADIUS_NEIGHBOR_CAP, 512, 2),
)
INVRES_EXPANSION = 4
INVRES_ALPHA_INIT = 0.1

# --- SplineConv decoder ---
DECODER_HIDDEN_DIM = 128
ATTN_DIM = 128
# (u, θ, kind). Kind is one bit: 2 knots (§6.2). Mean aggregation so
# updates do not scale with degree (§6.4). PyG SplineConv takes one integer
# `degree`, not a per-axis tuple, so the conv uses 2 (the u/θ value). Kernel
# size (5,5,2) is what cuts parameters vs a 5³ kernel; degree is interpolation
# order, not weight count.
SPLINE_KERNEL_SIZE = (5, 5, 2)
SPLINE_DEGREE = 2
SPLINE_AGGR = "mean"
SPLINE_ROOT_WEIGHT = True  # §6.4; vertex self-weight, not residual-only
N_CONV_PER_LEVEL = 6  # 6–8 residual convs per level (§6, §9); pick 6
# Kept until item 28 switches the decoder to N_CONV_PER_LEVEL.
N_SPLINE_COARSE = 4
N_SPLINE_MID = 4
N_SPLINE_FINE = 4
COARSE_ATTN_HEADS = 4
TRACT_EMB_DIM = 32
SKIP_GATE_INIT = 0.1
COARSE_ATTN_ALPHA_INIT = SKIP_GATE_INIT
# r_local-relative displacement bounds (§6.7, item 25). Floor ≈ −0.8 · r_local
# (approach but do not cross the centerline). Shear cap ∝ max(3 mm, k · r_local).
RADIAL_FLOOR_FRAC = 0.8
SHEAR_MAX_BASE_MM = 3.0
SHEAR_MAX_MM = SHEAR_MAX_BASE_MM  # fallback alias; prefer SHEAR_MAX_BASE_MM
# Softplus margin fallback when r_local is unavailable. Prefer
# −RADIAL_FLOOR_FRAC * r_local. Mid/fine residuals stay ≥ that floor.
R_MARGIN_MM = TUBE_RADIUS_MM

# --- Optimisation ---
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
KL_WARMUP_EPOCHS = 20
EMA_DECAY = 0.999
# "off" = no checkpointing (faster backward, more VRAM).
# "fine" = only the 64k-node SplineConvs (OOM fallback).
# "all" = encoder InvRes + every SplineConv + coarse attn (least VRAM).
GRADIENT_CHECKPOINTING = "off"


def normalize_gradient_checkpointing(value=None):
    """Map a user flag to 'off' | 'fine' | 'all'."""
    if value is None:
        value = GRADIENT_CHECKPOINTING
    if isinstance(value, bool):
        return "all" if value else "off"
    key = str(value).strip().lower()
    if key in ("off", "false", "0", "no", "none"):
        return "off"
    if key in ("fine", "fine_only", "fine-only"):
        return "fine"
    if key in ("all", "true", "1", "yes", "on"):
        return "all"
    raise ValueError(
        f"gradient_checkpointing must be False/'off', True/'all', or 'fine'; got {value!r}"
    )

LAMBDA_RECON = 1.0
# Fixed λ kept for DEFAULT_LOSS_WEIGHTS until item 19 wires GECO.
LAMBDA_KL = 5e-4
# Rate-controlled KL, GECO-style (Rezende & Viola 2018; §5.3.6 item 3).
# β ← clip(β · exp(η · (KL̄_raw − R*)), β_min, β_max), once per optimiser step.
# Constraint is on the mean over valid tokens, so healthy tokens may spend ≈ 0
# and sac tokens more. KL_WARMUP_EPOCHS ramps the β_max ceiling from
# GECO_BETA_INIT to GECO_BETA_MAX (never below GECO_BETA_MIN). R* is a starting
# midpoint of the 8–16 nats/token band at D = 16; set finally by the §5.3.7 sweep.
RATE_TARGET_NATS = 12.0
GECO_BETA_INIT = 1.0  # standard-VAE weight; the dual then adapts
GECO_BETA_MIN = 1e-4  # floor so the rate term never vanishes
GECO_BETA_MAX = 10.0  # cap so reconstruction is not starved if the dual overshoots
GECO_ETA = 1e-3  # log-space step per nat of (KL̄_raw − R*) per optimiser step
TOKEN_KL_FLOOR_NATS = 0.5  # per-token warm-up insurance only, §5.3.6 item 3

LAMBDA_DISP = 0.15
LAMBDA_LAP = 0.05
LAMBDA_NORM = 0.02
LAMBDA_RAD = 1.0
LAMBDA_CD_MID = 0.5
LAMBDA_CD_COARSE = 0.05
LAMBDA_RAD_MID = 0.5

CHAMFER_WEIGHT_CAP = 4.0
RADIAL_HUBER_DELTA_MM = 1.0
PLANE_HUBER_DELTA_MM = 1.0
PLANE_L2_MIX = 0.2

SMOOTH_BETA_THETA = 1.0
SMOOTH_BETA_U = 1.0
SMOOTH_BETA_R = 0.25
SMOOTH_DELTA_R_MM = 1.0
SMOOTH_W_AMBIGUOUS = 0.05
CROSS_TRACT_SMOOTH_W = 0.05

# --- Cached radial GT (r*) ray-casting ---
R_STAR_T_MAX_MM = 20.0
R_STAR_INWARD_MM = 1.0
R_STAR_T_EPS_MM = 0.05
R_STAR_NORMAL_DOT = 0.2
R_STAR_RING_SLACK = 2.0
R_STAR_ARC_SLACK_MM = 2.0
R_STAR_AMBIGUOUS_MM = 4.0
R_STAR_HIT_TOL = 1e-4

DEFAULT_LOSS_WEIGHTS = {
    "recon": LAMBDA_RECON,
    "kl": LAMBDA_KL,
    "disp": LAMBDA_DISP,
    "lap": LAMBDA_LAP,
    "norm": LAMBDA_NORM,
    "rad": LAMBDA_RAD,
}

# DataLoader node sets that do not match the fine-graph node count.
FOLLOW_BATCH = [
    "x_true",
    "x_true_cl_dist",
    "x_true_normal",
    "pos_coarse",
    "pos_mid",
    "cl_dense",
    "cl_tract_id",
    "branch_nl_coarse",
    "branch_nl_mid",
    "branch_nl_fine",
    "r_star_mid",
    "r_star_valid_mid",
    "r_star_ambiguous_mid",
    "latent_u",
    "latent_tract_id",
    "latent_is_junction",
    "latent_pos",
    "latent_valid",
    "token_attend",
]
