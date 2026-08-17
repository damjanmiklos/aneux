"""Hyperparameters for the hierarchical PointNeXt–SplineConv deformation VAE.

Values follow the architectural specification. Physical units are millimetres.

Stage-1 precision (see configure_stage1_precision):
  CPU preprocessing in dataset.py is NumPy float64 (splines, Bishop frames,
  arc-length, COM). All model parameters, GPU tensors, and Data fields are
  torch.float32. Ampere matmuls use TF32 execution with FP32 storage.
  Autocast / bfloat16 / float16 are not used in Stage 1.
"""

import torch


def configure_stage1_precision():
    """FP32 tensor storage, TF32 Ampere matmuls, no autocast or reduced dtypes."""
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.sparse.check_sparse_tensor_invariants.disable()


configure_stage1_precision()

# --- Scaffold / data ---
TUBE_RADIUS_MM = 2.0
N_TRUE = 4096
CACHE_VERSION = 3

# (n_length, n_radial) per hierarchy level
LEVEL_COARSE = (40, 6)
LEVEL_MID = (250, 12)
LEVEL_FINE = (1000, 50)
HIERARCHY_LEVELS = (LEVEL_COARSE, LEVEL_MID, LEVEL_FINE)

MIN_RINGS_PER_BRANCH = 2

# --- Intrinsic Fourier encodings ---
K_U = 8
K_THETA = 6
GAMMA_U_DIM = 2 * K_U  # 16
GAMMA_THETA_DIM = 2 * K_THETA  # 12

# --- Latent trajectory ---
LATENT_LEN = 64
LATENT_DIM = 64
LOGVAR_CLAMP = (-30.0, 20.0)

# --- PointNeXt encoder ---
STEM_DIM = 32
SA_STAGES = (
    # n_out, radius_mm, k_neighbors, hidden_dim, n_invres
    (1024, 1.5, 32, 64, 2),
    (256, 3.0, 32, 128, 2),
    (64, 6.0, 32, 256, 2),
    (16, 12.0, 16, 512, 2),
)
INVRES_EXPANSION = 4
INVRES_ALPHA_INIT = 0.1

# --- SplineConv decoder ---
DECODER_HIDDEN_DIM = 64
ATTN_DIM = 128
SPLINE_KERNEL_SIZE = 5
SPLINE_DEGREE = 2
EDGE_MAX_MM = 2.0
N_SPLINE_COARSE = 4
N_SPLINE_MID = 2
N_SPLINE_FINE = 2
SHEAR_MAX_MM = 3.0
# Softplus margin: Δr > -R_MARGIN so the lumen cannot invert through the centerline.
R_MARGIN_MM = TUBE_RADIUS_MM

# --- Optimisation ---
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
KL_WARMUP_EPOCHS = 20

LAMBDA_RECON = 1.0
LAMBDA_KL = 5e-4
LAMBDA_DISP = 0.15
LAMBDA_LAP = 0.05
LAMBDA_NORM = 0.02
# Auxiliary Chamfer on coarser decoder stages (multi-scale reconstruction).
LAMBDA_CD_MID = 0.5
LAMBDA_CD_COARSE = 0.25

DEFAULT_LOSS_WEIGHTS = {
    "recon": LAMBDA_RECON,
    "kl": LAMBDA_KL,
    "disp": LAMBDA_DISP,
    "lap": LAMBDA_LAP,
    "norm": LAMBDA_NORM,
}

# DataLoader node sets that do not match the fine-graph node count.
FOLLOW_BATCH = [
    "x_true",
    "pos_coarse",
    "pos_mid",
    "cl_dense",
    "branch_nl_coarse",
    "branch_nl_mid",
    "branch_nl_fine",
]
