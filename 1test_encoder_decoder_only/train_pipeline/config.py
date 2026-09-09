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

import torch


def configure_stage2_precision():
    """FP32 tensor storage, TF32 Ampere matmuls, no autocast or reduced dtypes."""
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.sparse.check_sparse_tensor_invariants.disable()


configure_stage2_precision()

# --- Scaffold / data ---
TUBE_RADIUS_MM = 2.0
N_TRUE = 16384
N_TRUE_FAR_FRAC = 0.25
FAR_CL_MARGIN_MM = 1.0
CACHE_VERSION = 8

# (n_length, n_radial) per hierarchy level
LEVEL_COARSE = (40, 6)
LEVEL_MID = (250, 12)
LEVEL_FINE = (1000, 64)
HIERARCHY_LEVELS = (LEVEL_COARSE, LEVEL_MID, LEVEL_FINE)

JUNCTION_COUPLE_RADIUS_MM = 4.0
JUNCTION_COUPLE_K = 2

MIN_RINGS_PER_BRANCH = 2
MAX_TRACTS = 16
MIN_TOKENS_PER_TRACT = 2
DENSE_CL_SPACING_MM = 0.2

# --- Intrinsic Fourier encodings ---
K_U = 8
K_THETA = 6
GAMMA_U_DIM = 2 * K_U  # 16
GAMMA_THETA_DIM = 2 * K_THETA  # 12

# --- Latent trajectory (tree-valued; LATENT_LEN includes junction tokens) ---
LATENT_LEN = 96
LATENT_DIM = 128
LOGVAR_CLAMP = (-8.0, 2.0)
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
SPLINE_KERNEL_SIZE = 5
SPLINE_DEGREE = 2
N_SPLINE_COARSE = 4
N_SPLINE_MID = 4
N_SPLINE_FINE = 4
COARSE_ATTN_HEADS = 4
TRACT_EMB_DIM = 32
SKIP_GATE_INIT = 0.1
COARSE_ATTN_ALPHA_INIT = SKIP_GATE_INIT
SHEAR_MAX_MM = 3.0
# Softplus margin on each head: Δr ≥ -R_MARGIN. Mid/fine residuals are further
# clamped so the composed radial offset n·Δx_total stays ≥ -R_MARGIN.
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
LAMBDA_KL = 5e-4
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
    "token_attend",
]
