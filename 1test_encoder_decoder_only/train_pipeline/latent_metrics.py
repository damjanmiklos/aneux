"""§5.3.7 latent-channel instrumentation for Stage 2 (Appendix B item 21).

`train.py` is expected to call these helpers at validation / epoch end. Nothing
here opens TensorBoard; every public summary is a CSV-friendly dict of Python
scalars (and lists of such dicts for per-token / per-s curves).

Definitions follow STAGE2_REVIEW.md §5.3.6–§5.3.7 and the §8 logging row:

- raw per-token KL is unclamped and *before* β: KL(N(μ, σ²) ‖ N(0, I)) in nats,
  summed over latent dimensions, then masked by `latent_valid`;
- bits per case is Σ_tokens KL / ln 2 (valid tokens only);
- the rate gap is KL̄_raw − R*, with KL̄_raw the mean over valid tokens;
- active units are dimensions whose mean raw per-dimension KL exceeds 0.01 nats,
  split healthy / sac;
- sac tokens are those whose wall neighbourhood intersects the AneuX dome
  (`aneurysms/original/{id}_dome`) when a mask or dome is available;
- the noise-robustness curve decodes μ + s·ε in *standardised* units for
  s ∈ {0, 0.25, 0.5, 1} (Stage-1 contract, also §12).

The 50-case rate–distortion sweep of §5.3.7 is *not* implemented here: it waits
on the tract / token / r* fixes. See `rate_distortion_sweep`.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

try:
    from config import ACTIVE_UNIT_KL_THRESH, RATE_TARGET_NATS, TOKEN_SPACING_MM
except ImportError:  # scratch tests that only add this file to path
    ACTIVE_UNIT_KL_THRESH = 0.01
    RATE_TARGET_NATS = 12.0
    TOKEN_SPACING_MM = 2.0

LN2 = math.log(2.0)
NOISE_SCALES = (0.0, 0.25, 0.5, 1.0)
INTERP_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
# §5.3.3a contact distance used when labelling sac tokens against the dome.
DOME_CONTACT_MM = 0.3
# GT remesh target (§2.2); null-code "within remesh noise" uses this ceiling.
REMESH_NOISE_MM = 0.15
# "a few percent" in the sampled- vs μ-path acceptance line of §5.3.7.
SAMPLED_MU_REL_TOL = 0.05
STD_EPS = 1e-8

ANEURYSMS_ORIGINAL_SUBDIR = os.path.join(
    "models-v1.0", "models", "aneurysms", "original"
)
DOME_SUFFIXES = ("_dome.vtp", "_dome.stl", "_dome.vtk", ".vtp")

# Appendix B item 26 has not landed on postprocess.py; names we will try first.
_POSTPROCESS_VALIDITY_NAMES = (
    "mesh_validity_metrics",
    "validity_counts",
    "evaluate_mesh_validity",
    "mesh_validity",
)

_VALIDITY_PLACEHOLDER_COMMENT = (
    "postprocess §11 validity counts are Appendix B item 26 and are not "
    "importable yet; returning placeholders."
)

__all__ = [
    "ACTIVE_UNIT_KL_THRESH",
    "DOME_CONTACT_MM",
    "INTERP_SCALES",
    "LN2",
    "NOISE_SCALES",
    "RATE_TARGET_NATS",
    "LatentMetricAccumulator",
    "as_csv_row",
    "beta_and_rate_gap",
    "bits_per_case",
    "compare_sampled_vs_mu_path",
    "count_active_units",
    "epoch_latent_log_row",
    "interp_z_case_validity",
    "kl_profile_along_tree",
    "noise_robustness_curve",
    "null_code_template_error",
    "rate_distortion_sweep",
    "raw_kl_per_dim",
    "raw_kl_per_token",
    "resolve_dome_path",
    "sac_tokens_from_wall_neighbourhood",
    "summarize_raw_kl",
    "surface_distance_metrics",
    "train_val_gap_row",
]


# ---------------------------------------------------------------------------
# Tensor / CSV helpers
# ---------------------------------------------------------------------------

def _token_valid(valid: Tensor | None, token_kl: Tensor) -> Tensor:
    """Boolean mask aligned with a per-token tensor (`[B, L]` or `[L]`)."""
    if valid is None:
        return torch.ones(token_kl.shape, dtype=torch.bool, device=token_kl.device)
    t = valid if valid.dtype == torch.bool else valid != 0
    if t.shape == token_kl.shape:
        return t
    if t.numel() == token_kl.numel():
        return t.reshape(token_kl.shape)
    if t.dim() == 1 and token_kl.dim() == 2 and t.numel() == token_kl.size(-1):
        return t.unsqueeze(0).expand_as(token_kl)
    raise ValueError(
        f"valid shape {tuple(valid.shape)} does not match tokens {tuple(token_kl.shape)}"
    )


def _ensure_bld(x: Tensor) -> Tensor:
    """Promote μ / logvar / per-dim KL to [B, L, D]."""
    if x.dim() == 1:
        return x.view(1, 1, -1)
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() != 3:
        raise ValueError(f"expected [D], [L, D] or [B, L, D]; got {tuple(x.shape)}")
    return x


def _python_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        return bool(value)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        if value.numel() == 1:
            item = value.detach().cpu().reshape(-1)[0]
            if value.dtype == torch.bool:
                return bool(item.item())
            if value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                return int(item.item())
            return float(item.item())
        return [_python_scalar(v) for v in value.detach().cpu().reshape(-1)]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return float(value)
        return float(value)
    if isinstance(value, (int,)):
        return int(value)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    return value


def as_csv_row(record: Mapping[str, Any] | None) -> dict[str, Any]:
    """Flatten nested dicts; JSON-encode lists of dicts; scalarise tensors."""
    if not record:
        return {}
    out: dict[str, Any] = {}
    for key, value in record.items():
        if value is None:
            out[str(key)] = None
        elif isinstance(value, dict):
            nested = as_csv_row(value)
            for nk, nv in nested.items():
                out[f"{key}_{nk}"] = nv
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
            out[str(key)] = json.dumps(value)
        elif isinstance(value, (list, tuple)):
            out[str(key)] = json.dumps([_python_scalar(v) for v in value])
        else:
            out[str(key)] = _python_scalar(value)
    return out


def _finite_mean(values: Tensor) -> float:
    if values.numel() == 0:
        return float("nan")
    ok = torch.isfinite(values)
    if not bool(ok.any()):
        return float("nan")
    return float(values[ok].mean().item())


# ---------------------------------------------------------------------------
# Raw KL (unclamped, before β) — §5.3.6 items 3–4, §5.3.7
# ---------------------------------------------------------------------------

def raw_kl_per_dim(mu: Tensor, logvar: Tensor) -> Tensor:
    """Unclamped diagonal-Gaussian KL vs N(0, I), nats per dimension.

    KL = ½ (μ² + σ² − 1 − log σ²) with σ² = exp(logvar). No log-variance
    clamp and no β: this is the quantity the rate controller and the
    active-unit test read. Shape matches `mu` (`[B, L, D]` after promotion).
    """
    mu_b = _ensure_bld(mu).double()
    lv = _ensure_bld(logvar).double()
    if mu_b.shape != lv.shape:
        raise ValueError(f"mu {tuple(mu.shape)} vs logvar {tuple(logvar.shape)}")
    return -0.5 * (1.0 + lv - mu_b.pow(2) - lv.exp())


def raw_kl_per_token(mu: Tensor, logvar: Tensor) -> Tensor:
    """Unclamped KL summed over D, nats per token. Shape `[B, L]`."""
    return raw_kl_per_dim(mu, logvar).sum(dim=-1)


def bits_per_case(kl_per_token: Tensor, valid: Tensor | None = None) -> dict[str, Any]:
    """Σ_valid tokens KL / ln 2, then mean over the batch.

    `kl_per_token` is nats per token `[B, L]` or `[L]`. Padding / invalid
    slots contribute 0. Returns CSV scalars plus the per-case list.
    """
    kt = kl_per_token.double()
    if kt.dim() == 1:
        kt = kt.unsqueeze(0)
    if kt.dim() != 2:
        raise ValueError(f"kl_per_token expected [B, L] or [L]; got {tuple(kl_per_token.shape)}")
    mask = _token_valid(valid, kt)
    per_case = (kt.masked_fill(~mask, 0.0).sum(dim=-1)) / LN2
    n_valid = mask.sum(dim=-1)
    return {
        "bits_per_case": _finite_mean(per_case),
        "bits_per_case_list": [float(v) for v in per_case.detach().cpu()],
        "n_valid_tokens_mean": _finite_mean(n_valid.double()),
        "n_cases": int(per_case.numel()),
    }


def summarize_raw_kl(
    mu: Tensor,
    logvar: Tensor,
    valid: Tensor | None = None,
) -> dict[str, Any]:
    """Mean raw per-token KL (nats) and bits per case over valid tokens."""
    kl_tok = raw_kl_per_token(mu, logvar)
    mask = _token_valid(valid, kl_tok)
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        kl_mean = float("nan")
        kl_sum = 0.0
    else:
        kl_mean = float(kl_tok[mask].mean().item())
        kl_sum = float(kl_tok.masked_fill(~mask, 0.0).sum().item())
    bits = bits_per_case(kl_tok, valid=mask)
    return {
        "kl_mean_raw": kl_mean,
        "kl_sum_nats": kl_sum,
        "n_valid_tokens": n_valid,
        **bits,
    }


def beta_and_rate_gap(
    kl_mean_raw: float,
    beta: float,
    rate_target: float | None = None,
) -> dict[str, Any]:
    """β (GECO dual) and gap KL̄_raw − R* (§5.3.6 item 3, §5.3.7, §8)."""
    r_star = float(RATE_TARGET_NATS if rate_target is None else rate_target)
    kl_bar = float(kl_mean_raw)
    return {
        "beta": float(beta),
        "rate_target_nats": r_star,
        "kl_mean_raw": kl_bar,
        "rate_gap": kl_bar - r_star,
    }


# ---------------------------------------------------------------------------
# Active units, healthy / sac — §5.3.7
# ---------------------------------------------------------------------------

def _mean_kl_per_dim(kl_dim: Tensor, token_mask: Tensor) -> Tensor:
    """Mean over tokens selected by `token_mask` `[B, L]` → `[D]`."""
    d = kl_dim.size(-1)
    weights = token_mask.double().unsqueeze(-1)
    n = float(token_mask.sum().item())
    if n <= 0:
        return kl_dim.new_full((d,), float("nan"))
    return (kl_dim.double() * weights).sum(dim=(0, 1)) / n


def count_active_units(
    kl_per_dim: Tensor | None = None,
    *,
    mu: Tensor | None = None,
    logvar: Tensor | None = None,
    valid: Tensor | None = None,
    sac_mask: Tensor | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Count dimensions with mean raw KL > threshold, split healthy / sac.

    Sac tokens: caller passes a boolean mask (True = sac), typically from
    `sac_tokens_from_wall_neighbourhood` (wall neighbourhood ∩ AneuX dome).
    If `sac_mask` is omitted the sac split is skipped (`n_active_sac` is
    null) and healthy is computed on every valid token.
    """
    if kl_per_dim is None:
        if mu is None or logvar is None:
            raise ValueError("count_active_units needs kl_per_dim or mu and logvar")
        kl_per_dim = raw_kl_per_dim(mu, logvar)
    else:
        kl_per_dim = _ensure_bld(kl_per_dim)
    thresh = float(ACTIVE_UNIT_KL_THRESH if threshold is None else threshold)
    token_kl = kl_per_dim.sum(dim=-1)
    valid_m = _token_valid(valid, token_kl)
    healthy_m = valid_m
    sac_m = None
    sac_available = sac_mask is not None
    if sac_available:
        sac_m = _token_valid(sac_mask, token_kl) & valid_m
        healthy_m = valid_m & ~sac_m

    def _count(mask: Tensor) -> tuple[int | None, list[int], list[float]]:
        if mask is None:
            return None, [], []
        mean_d = _mean_kl_per_dim(kl_per_dim, mask)
        if not torch.isfinite(mean_d).any() and int(mask.sum().item()) == 0:
            return 0, [], [float("nan")] * int(mean_d.numel())
        active = torch.isfinite(mean_d) & (mean_d > thresh)
        idxs = [int(i) for i in torch.nonzero(active, as_tuple=False).reshape(-1)]
        means = [float(v) if math.isfinite(float(v)) else None for v in mean_d.detach().cpu()]
        return int(active.sum().item()), idxs, means

    n_all, idx_all, mean_all = _count(valid_m)
    n_h, idx_h, _ = _count(healthy_m)
    n_s, idx_s, _ = _count(sac_m) if sac_available else (None, [], [])
    return {
        "active_unit_threshold_nats": thresh,
        "n_active_all": n_all,
        "n_active_healthy": n_h,
        "n_active_sac": n_s,
        "active_dims_all": idx_all,
        "active_dims_healthy": idx_h,
        "active_dims_sac": idx_s,
        "mean_kl_per_dim": mean_all,
        "n_tokens_healthy": int(healthy_m.sum().item()),
        "n_tokens_sac": int(sac_m.sum().item()) if sac_m is not None else None,
        "sac_mask_available": bool(sac_available),
    }


def resolve_dome_path(
    dataset_id: str,
    aneurysms_original_dir: str | None = None,
    rawdata_root: str | None = None,
) -> str | None:
    """`aneurysms/original/{id}_dome` when that archive is on disk; else None.

    Training never writes here. Missing files are not an error — the sac
    split is optional.
    """
    candidates: list[str] = []
    if aneurysms_original_dir:
        candidates.append(aneurysms_original_dir)
    if rawdata_root:
        candidates.append(os.path.join(rawdata_root, ANEURYSMS_ORIGINAL_SUBDIR))
    try:
        from aneux_paths import RAWDATA

        candidates.append(os.path.join(RAWDATA, ANEURYSMS_ORIGINAL_SUBDIR))
    except ImportError:
        pass
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates.append(os.path.join(repo, "rawdata", ANEURYSMS_ORIGINAL_SUBDIR))
    stem = str(dataset_id)
    seen: set[str] = set()
    for folder in candidates:
        if not folder or folder in seen:
            continue
        seen.add(folder)
        for suffix in DOME_SUFFIXES:
            path = os.path.join(folder, stem + suffix)
            if os.path.isfile(path):
                return path
    return None


def _load_xyz(path: str) -> Tensor | None:
    try:
        import pyvista as pv
    except ImportError:
        return None
    mesh = pv.read(path)
    pts = torch.as_tensor(mesh.points, dtype=torch.float64)
    return pts if pts.numel() else None


def sac_tokens_from_wall_neighbourhood(
    token_pos: Tensor,
    wall_xyz: Tensor | None,
    dome_xyz: Tensor | str | None = None,
    *,
    valid: Tensor | None = None,
    contact_mm: float = DOME_CONTACT_MM,
    dataset_id: str | None = None,
    aneurysms_original_dir: str | None = None,
) -> dict[str, Any]:
    """Boolean sac mask: token wall neighbourhood intersects the dome.

    Each wall vertex is assigned to its nearest token (`token_pos`, typically
    `data.latent_pos`). A valid token is sac if any assigned vertex lies
    within `contact_mm` of a dome sample. This is the §5.3.7 *intersection*
    rule (stricter 5 % occupancy in §5.3.3a is a measurement convention, not
    the logging definition).

    `dome_xyz` may be points `[M, 3]`, a filesystem path, or omitted (then
    `resolve_dome_path(dataset_id)` is tried). If the dome is unavailable
    the mask is all-False and `sac_mask_available` is False.
    """
    pos = token_pos.double()
    if pos.dim() == 3:
        if pos.size(0) != 1:
            raise ValueError("batched token_pos not supported; pass one case [L, 3]")
        pos = pos[0]
    n_tok = pos.size(0)
    valid_flat = valid.reshape(-1)[:n_tok] if valid is not None else None
    valid_m = _token_valid(valid_flat, pos.new_zeros(n_tok))

    dome_pts = None
    dome_path = None
    if isinstance(dome_xyz, str):
        dome_path = dome_xyz
        dome_pts = _load_xyz(dome_xyz)
    elif torch.is_tensor(dome_xyz):
        dome_pts = dome_xyz.double().reshape(-1, 3)
    elif dataset_id is not None:
        dome_path = resolve_dome_path(dataset_id, aneurysms_original_dir)
        if dome_path:
            dome_pts = _load_xyz(dome_path)

    reason = None
    mask = torch.zeros(n_tok, dtype=torch.bool, device=pos.device)
    if wall_xyz is None or wall_xyz.numel() == 0:
        reason = "no wall neighbourhood points"
    elif dome_pts is None or dome_pts.numel() == 0:
        reason = "AneuX dome not available (optional)"
    else:
        wall = wall_xyz.double().reshape(-1, 3)
        tok_d = torch.cdist(wall, pos)
        nearest = tok_d.argmin(dim=1)
        dome_d = torch.cdist(wall, dome_pts).min(dim=1).values
        near_dome = dome_d <= float(contact_mm)
        if bool(near_dome.any()):
            hit = nearest[near_dome]
            mask.scatter_(0, hit, True)
        mask = mask & valid_m

    available = reason is None
    return {
        "sac_mask": mask,
        "sac_mask_available": available,
        "n_sac_tokens": int((mask & valid_m).sum().item()) if available else None,
        "dome_path": dome_path,
        "reason": reason,
        "contact_mm": float(contact_mm),
        "token_spacing_mm": float(TOKEN_SPACING_MM),
    }


# ---------------------------------------------------------------------------
# Per-token KL profile along the tree — §5.3.7, §8, §12
# ---------------------------------------------------------------------------

def kl_profile_along_tree(
    mu: Tensor,
    logvar: Tensor,
    *,
    valid: Tensor | None = None,
    sac_mask: Tensor | None = None,
    case_id: str | None = None,
    latent_u: Tensor | None = None,
    latent_tract_id: Tensor | None = None,
    token_index: Tensor | None = None,
) -> dict[str, Any]:
    """Per-token raw KL for one case (or batch-1) plus a peak-at-sac flag.

    A flat profile (similar KL on healthy and sac tokens) means the latent
    is not local; no rate setting fixes that (§5.3.7).
    """
    kl_tok = raw_kl_per_token(mu, logvar)
    if kl_tok.size(0) != 1:
        # Profiles are stored per validation case; average only as a fallback.
        kl_tok = kl_tok[:1]
    kl = kl_tok[0]
    mask = _token_valid(valid[0] if valid is not None and valid.dim() > 1 else valid, kl)
    sac = None
    if sac_mask is not None:
        sac = _token_valid(
            sac_mask[0] if sac_mask.dim() > 1 else sac_mask, kl
        ) & mask

    def _opt_list(t: Tensor | None) -> list[Any] | None:
        if t is None:
            return None
        v = t[0] if t.dim() > 1 and t.size(0) == 1 else t
        v = v.detach().cpu().reshape(-1)[: kl.numel()]
        if v.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            return [int(x) for x in v]
        if v.dtype == torch.bool:
            return [bool(x) for x in v]
        return [float(x) for x in v]

    u_list = _opt_list(latent_u)
    tract_list = _opt_list(latent_tract_id)
    idx_list = _opt_list(token_index)
    rows: list[dict[str, Any]] = []
    for i in range(int(kl.numel())):
        if idx_list is not None:
            ti = idx_list[i]
        else:
            ti = i
        row = {
            "case_id": case_id,
            "token_index": int(ti),
            "kl_nats": float(kl[i].item()),
            "valid": bool(mask[i].item()),
        }
        if sac is not None:
            row["is_sac"] = bool(sac[i].item())
        if u_list is not None:
            row["latent_u"] = u_list[i]
        if tract_list is not None:
            row["latent_tract_id"] = tract_list[i]
        rows.append(row)

    valid_rows = [r for r in rows if r["valid"]]
    peak = None
    peak_at_sac = None
    if valid_rows:
        peak = max(valid_rows, key=lambda r: r["kl_nats"])
        if sac is not None:
            peak_at_sac = bool(peak.get("is_sac", False))
    healthy_mean = None
    sac_mean = None
    if sac is not None and bool(mask.any()):
        if bool((mask & ~sac).any()):
            healthy_mean = float(kl[mask & ~sac].mean().item())
        if bool(sac.any()):
            sac_mean = float(kl[sac].mean().item())
    return {
        "case_id": case_id,
        "tokens": rows,
        "peak_token_index": None if peak is None else peak["token_index"],
        "peak_kl_nats": None if peak is None else peak["kl_nats"],
        "peak_at_sac": peak_at_sac,
        "kl_mean_healthy": healthy_mean,
        "kl_mean_sac": sac_mean,
        "n_valid_tokens": int(mask.sum().item()),
    }


# ---------------------------------------------------------------------------
# Surface metrics used by the noise-robustness curve (§12, millimetre, non-Huber)
# ---------------------------------------------------------------------------

def _min_dists(src: Tensor, dst: Tensor) -> Tensor:
    if src.numel() == 0 or dst.numel() == 0:
        return src.new_zeros(src.size(0))
    src_f = src.float().reshape(-1, 3)
    dst_f = dst.float().reshape(-1, 3)
    try:
        from pytorch3d.ops import knn_points

        dist2 = knn_points(
            src_f.unsqueeze(0), dst_f.unsqueeze(0), K=1, return_nn=False
        ).dists.reshape(-1)
        return dist2.clamp_min(0.0).sqrt()
    except Exception:
        chunk = 4096
        parts = []
        for i in range(0, src_f.size(0), chunk):
            d = torch.cdist(src_f[i : i + chunk], dst_f)
            parts.append(d.min(dim=1).values)
        return torch.cat(parts, dim=0)


def _quantile(x: Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.float(), q).item())


def surface_distance_metrics(
    pred: Tensor,
    true: Tensor,
    *,
    pred_normal: Tensor | None = None,
    true_normal: Tensor | None = None,
    sac_pred_mask: Tensor | None = None,
    sac_true_mask: Tensor | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    """Symmetric Chamfer mean / p95, Hausdorff, optional normal-angle error.

    Millimetre units, no Huber (§12). Optional sac masks report the same
    numbers on the sac subset only (keys suffixed `_sac`).
    """
    p = pred.float().reshape(-1, 3)
    t = true.float().reshape(-1, 3)
    d_pt = _min_dists(p, t)
    d_tp = _min_dists(t, p)
    row = {
        f"{prefix}chamfer_mean": 0.5 * (float(d_pt.mean()) + float(d_tp.mean())),
        f"{prefix}chamfer_p95": 0.5 * (_quantile(d_pt, 0.95) + _quantile(d_tp, 0.95)),
        f"{prefix}hausdorff": float(max(float(d_pt.max()), float(d_tp.max()))),
    }
    if (
        pred_normal is not None
        and true_normal is not None
        and pred_normal.numel()
        and true_normal.numel()
    ):
        # Angle at each pred point vs the GT normal of its nearest true point.
        try:
            from pytorch3d.ops import knn_points

            idx = knn_points(
                p.unsqueeze(0), t.unsqueeze(0), K=1, return_nn=False
            ).idx.reshape(-1)
        except Exception:
            idx = torch.cdist(p, t).argmin(dim=1)
        n_p = pred_normal.float().reshape(-1, 3)
        n_t = true_normal.float().reshape(-1, 3)[idx]
        n_p = n_p / n_p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        n_t = n_t / n_t.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cos = (n_p * n_t).sum(dim=-1).clamp(-1.0, 1.0)
        row[f"{prefix}normal_angle_mean_deg"] = float(
            torch.acos(cos).mean().mul(180.0 / math.pi).item()
        )
    if sac_pred_mask is not None or sac_true_mask is not None:
        p_s = p[sac_pred_mask.reshape(-1).bool()] if sac_pred_mask is not None else p
        t_s = t[sac_true_mask.reshape(-1).bool()] if sac_true_mask is not None else t
        if p_s.numel() and t_s.numel():
            row.update(
                surface_distance_metrics(p_s, t_s, prefix=f"{prefix}" + "sac_")
            )
    return row


def _as_verts(decoded: Any) -> Tensor:
    if torch.is_tensor(decoded):
        return decoded
    if isinstance(decoded, dict):
        for key in ("x_pred", "verts", "vertices"):
            if key in decoded and torch.is_tensor(decoded[key]):
                return decoded[key]
    for key in ("x_pred", "verts", "vertices"):
        if hasattr(decoded, key) and torch.is_tensor(getattr(decoded, key)):
            return getattr(decoded, key)
    raise TypeError(
        f"decode_fn must return a [N, 3] tensor or an object with x_pred; got {type(decoded)!r}"
    )


def _standardise_mu(
    mu: Tensor,
    latent_mean: Tensor | None,
    latent_std: Tensor | None,
    active_mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (mu_std, mean, std, active) broadcast to `[B, L, D]`."""
    mu_b = _ensure_bld(mu).float()
    d = mu_b.size(-1)
    if latent_mean is None:
        mean = mu_b.new_zeros(d)
    else:
        mean = latent_mean.float().reshape(-1)[:d]
    if latent_std is None:
        std = mu_b.new_ones(d)
    else:
        std = latent_std.float().reshape(-1)[:d].clamp_min(STD_EPS)
    if active_mask is None:
        active = torch.ones(d, dtype=torch.bool, device=mu_b.device)
    else:
        active = active_mask.reshape(-1)[:d]
        if active.dtype != torch.bool:
            active = active != 0
    mean_b = mean.view(1, 1, d)
    std_b = std.view(1, 1, d)
    act_b = active.view(1, 1, d)
    mu_std = torch.where(act_b, (mu_b - mean_b) / std_b, torch.zeros_like(mu_b))
    return mu_std, mean, std, active


def _destandardise(z_std: Tensor, mean: Tensor, std: Tensor, active: Tensor, mu_raw: Tensor) -> Tensor:
    d = z_std.size(-1)
    mean_b = mean.view(1, 1, d)
    std_b = std.view(1, 1, d)
    act_b = active.view(1, 1, d)
    raw = z_std * std_b + mean_b
    return torch.where(act_b, raw, mu_raw)


# ---------------------------------------------------------------------------
# Noise-robustness curve — §5.3.7, §8, §12
# ---------------------------------------------------------------------------

def noise_robustness_curve(
    mu: Tensor,
    decode_fn: Callable[[Tensor], Any],
    *,
    latent_mean: Tensor | float | None = None,
    latent_std: Tensor | float | None = None,
    active_mask: Tensor | None = None,
    scales: Sequence[float] = NOISE_SCALES,
    eps: Tensor | None = None,
    generator: torch.Generator | None = None,
    x_true: Tensor | None = None,
    pred_normal_fn: Callable[[Tensor], Tensor | None] | None = None,
    true_normal: Tensor | None = None,
    sac_pred_mask: Tensor | None = None,
    sac_true_mask: Tensor | None = None,
    metric_fn: Callable[[Tensor, Tensor | None], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Reconstruction vs noise scale in standardised latent units.

    Stage 1's expected error must lie on the flat part of this curve. For
    each s ∈ `scales` this destandardises `μ_std + s·ε` (inactive dimensions
    stay at raw μ — they are dropped before standardising, §5.3.6 item 7)
    and calls `decode_fn`. Chamfer / Hausdorff / normals follow §12 when
    `x_true` is given; `metric_fn` can replace or extend that.

    `decode_fn` is injected so this file does not import `model.py`. A dummy
    callable is enough to unit-test the wiring.
    """
    mu_b = _ensure_bld(mu).float()
    mean_t = None if latent_mean is None else torch.as_tensor(latent_mean, device=mu_b.device, dtype=mu_b.dtype)
    std_t = None if latent_std is None else torch.as_tensor(latent_std, device=mu_b.device, dtype=mu_b.dtype)
    mu_std, mean, std, active = _standardise_mu(mu_b, mean_t, std_t, active_mask)
    if eps is None:
        noise = torch.randn(
            mu_std.shape, dtype=mu_std.dtype, device=mu_std.device, generator=generator
        )
    else:
        noise = _ensure_bld(eps).to(device=mu_std.device, dtype=mu_std.dtype)
        if noise.shape != mu_std.shape:
            noise = noise.expand_as(mu_std)
    act_b = active.view(1, 1, -1)
    noise = torch.where(act_b, noise, torch.zeros_like(noise))

    rows: list[dict[str, Any]] = []
    for s in scales:
        s_f = float(s)
        z_std = mu_std + s_f * noise
        z_raw = _destandardise(z_std, mean, std, active, mu_b)
        decoded = decode_fn(z_raw)
        pred = _as_verts(decoded)
        row: dict[str, Any] = {
            "s": s_f,
            "z_std_rms": float(z_std.pow(2).mean().sqrt().item()),
        }
        extra: Mapping[str, Any] = {}
        if metric_fn is not None:
            extra = dict(metric_fn(pred, x_true))
        elif x_true is not None:
            n_pred = pred_normal_fn(pred) if pred_normal_fn is not None else None
            extra = surface_distance_metrics(
                pred,
                x_true,
                pred_normal=n_pred,
                true_normal=true_normal,
                sac_pred_mask=sac_pred_mask,
                sac_true_mask=sac_true_mask,
            )
        row.update(as_csv_row(extra) if extra else {})
        rows.append(as_csv_row(row))
    return rows


# ---------------------------------------------------------------------------
# Acceptance helpers — §5.3.7 last paragraph, §12 latent sanity
# ---------------------------------------------------------------------------

def compare_sampled_vs_mu_path(
    recon_mu: float | Tensor,
    recon_sampled: float | Tensor,
    *,
    rel_tol: float = SAMPLED_MU_REL_TOL,
) -> dict[str, Any]:
    """Sampled-path recon should sit within a few percent of the μ path."""
    mu = float(_python_scalar(recon_mu))
    samp = float(_python_scalar(recon_sampled))
    denom = max(abs(mu), 1e-12)
    rel = abs(samp - mu) / denom
    return {
        "recon_mu": mu,
        "recon_sampled": samp,
        "rel_gap": float(rel),
        "rel_tol": float(rel_tol),
        "within_few_percent": bool(rel <= rel_tol),
    }


def train_val_gap_row(
    train_mu: float | None = None,
    train_sampled: float | None = None,
    val_mu: float | None = None,
    val_sampled: float | None = None,
) -> dict[str, Any]:
    """Train/validation gap on both the μ path and the sampled path."""
    row: dict[str, Any] = {
        "train_recon_mu": _python_scalar(train_mu),
        "train_recon_sampled": _python_scalar(train_sampled),
        "val_recon_mu": _python_scalar(val_mu),
        "val_recon_sampled": _python_scalar(val_sampled),
    }
    if train_mu is not None and val_mu is not None:
        row["gap_mu"] = float(val_mu) - float(train_mu)
    else:
        row["gap_mu"] = None
    if train_sampled is not None and val_sampled is not None:
        row["gap_sampled"] = float(val_sampled) - float(train_sampled)
    else:
        row["gap_sampled"] = None
    return row


def null_code_template_error(
    x_decoded: Tensor,
    x_template: Tensor,
    *,
    remesh_noise_mm: float = REMESH_NOISE_MM,
) -> dict[str, Any]:
    """Null code (prior mean 0 in raw space) should decode to the template."""
    diff = (x_decoded.float().reshape(-1, 3) - x_template.float().reshape(-1, 3)).norm(dim=-1)
    rms = float(diff.pow(2).mean().sqrt().item()) if diff.numel() else float("nan")
    p95 = _quantile(diff, 0.95)
    mx = float(diff.max().item()) if diff.numel() else float("nan")
    return {
        "null_rms_mm": rms,
        "null_p95_mm": p95,
        "null_max_mm": mx,
        "remesh_noise_mm": float(remesh_noise_mm),
        "within_remesh_noise": bool(rms <= float(remesh_noise_mm)),
    }


def _validity_placeholder(t: float | None = None) -> dict[str, Any]:
    row = {
        "n_components": None,
        "n_boundary_loops": None,
        "n_self_intersections": None,
        "min_angle_deg": None,
        "n_nonmanifold_edges": None,
        "validity_source": "placeholder",
        "comment": _VALIDITY_PLACEHOLDER_COMMENT,
    }
    if t is not None:
        row["t"] = float(t)
    return row


def _try_postprocess_validity(verts: Tensor, faces: Tensor | None) -> dict[str, Any] | None:
    """Item 26 metrics if `postprocess` exports them; never required.

    `postprocess.py` currently imports the training stack at module level and
    does not yet expose §11 counts, so a failed or empty import is expected.
    """
    try:
        import postprocess
    except Exception:
        return None
    fn = None
    for name in _POSTPROCESS_VALIDITY_NAMES:
        cand = getattr(postprocess, name, None)
        if callable(cand):
            fn = cand
            break
    if fn is None:
        return None
    try:
        raw = fn(verts, faces)
    except TypeError:
        try:
            raw = fn(verts)
        except Exception:
            return None
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    out = {k: _python_scalar(v) for k, v in raw.items()}
    out["validity_source"] = getattr(fn, "__name__", "postprocess")
    out.setdefault("comment", None)
    return out


def interp_z_case_validity(
    z_case: Tensor,
    decode_fn: Callable[[Tensor], Any],
    *,
    faces: Tensor | None = None,
    scales: Sequence[float] = INTERP_SCALES,
    validity_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Decode t · z_case, t ∈ {0, 0.25, 0.5, 0.75, 1}, and collect §11 counts.

    Acceptance: every mesh has one component, three boundary loops, and no
    non-manifold edges. Counts come from `validity_fn` or a postprocess
    helper when importable; otherwise placeholders (item 26).
    """
    z = z_case.float()
    rows: list[dict[str, Any]] = []
    for t in scales:
        t_f = float(t)
        pred = _as_verts(decode_fn(t_f * z))
        if validity_fn is not None:
            try:
                metrics = dict(validity_fn(pred, faces))
            except TypeError:
                metrics = dict(validity_fn(pred))
            metrics = {k: _python_scalar(v) for k, v in metrics.items()}
            metrics.setdefault("validity_source", "validity_fn")
            metrics["t"] = t_f
        else:
            imported = _try_postprocess_validity(pred, faces)
            if imported is None:
                metrics = _validity_placeholder(t_f)
            else:
                imported["t"] = t_f
                metrics = imported
        rows.append(as_csv_row(metrics))
    return rows


# ---------------------------------------------------------------------------
# Epoch CSV row — §8
# ---------------------------------------------------------------------------

def epoch_latent_log_row(
    *,
    mu: Tensor | None = None,
    logvar: Tensor | None = None,
    valid: Tensor | None = None,
    sac_mask: Tensor | None = None,
    beta: float | None = None,
    rate_target: float | None = None,
    kl_summary: Mapping[str, Any] | None = None,
    active: Mapping[str, Any] | None = None,
    noise_rows: Sequence[Mapping[str, Any]] | None = None,
    profile: Mapping[str, Any] | None = None,
    sampled_vs_mu: Mapping[str, Any] | None = None,
    null_code: Mapping[str, Any] | None = None,
    epoch: int | None = None,
) -> dict[str, Any]:
    """One per-epoch record: raw KL, bits, β, gap, active units, optional curves."""
    summary = dict(kl_summary) if kl_summary is not None else {}
    if not summary and mu is not None and logvar is not None:
        summary = summarize_raw_kl(mu, logvar, valid=valid)
    active_row = dict(active) if active is not None else {}
    if not active_row and mu is not None and logvar is not None:
        active_row = count_active_units(mu=mu, logvar=logvar, valid=valid, sac_mask=sac_mask)
    gap = {}
    if beta is not None:
        kl_bar = summary.get("kl_mean_raw", float("nan"))
        gap = beta_and_rate_gap(kl_bar, beta, rate_target=rate_target)

    row: dict[str, Any] = {}
    if epoch is not None:
        row["epoch"] = int(epoch)
    row.update(
        {
            "kl_mean_raw": summary.get("kl_mean_raw"),
            "bits_per_case": summary.get("bits_per_case"),
            "n_valid_tokens_mean": summary.get("n_valid_tokens_mean", summary.get("n_valid_tokens")),
            "beta": gap.get("beta", beta),
            "rate_target_nats": gap.get("rate_target_nats"),
            "rate_gap": gap.get("rate_gap"),
            "n_active_healthy": active_row.get("n_active_healthy"),
            "n_active_sac": active_row.get("n_active_sac"),
            "n_active_all": active_row.get("n_active_all"),
            "sac_mask_available": active_row.get("sac_mask_available"),
        }
    )
    if sampled_vs_mu:
        row.update(as_csv_row({"sampled_vs_mu": dict(sampled_vs_mu)}))
    if null_code:
        row.update(as_csv_row({"null": dict(null_code)}))
    if profile:
        row["kl_profile_peak_token"] = profile.get("peak_token_index")
        row["kl_profile_peak_kl"] = profile.get("peak_kl_nats")
        row["kl_profile_peak_at_sac"] = profile.get("peak_at_sac")
        row["kl_profile_mean_healthy"] = profile.get("kl_mean_healthy")
        row["kl_profile_mean_sac"] = profile.get("kl_mean_sac")
    if noise_rows:
        for nr in noise_rows:
            s = nr.get("s")
            if s is None:
                continue
            tag = str(s).replace(".", "p")
            for key, value in nr.items():
                if key == "s":
                    continue
                row[f"noise_s{tag}_{key}"] = value
    return as_csv_row(row)


class LatentMetricAccumulator:
    """Running sums over a loader so `train.py` can log one row per epoch."""

    def __init__(self, threshold: float | None = None):
        self.threshold = float(ACTIVE_UNIT_KL_THRESH if threshold is None else threshold)
        self.reset()

    def reset(self) -> None:
        self._kl_sum = 0.0
        self._n_valid = 0
        self._bits_sum = 0.0
        self._n_cases = 0
        self._d = None
        self._kl_dim_healthy = None
        self._kl_dim_sac = None
        self._n_tok_healthy = 0
        self._n_tok_sac = 0
        self._saw_sac_mask = False

    @torch.no_grad()
    def update(
        self,
        mu: Tensor,
        logvar: Tensor,
        valid: Tensor | None = None,
        sac_mask: Tensor | None = None,
    ) -> None:
        kl_dim = raw_kl_per_dim(mu, logvar)
        kl_tok = kl_dim.sum(dim=-1)
        mask = _token_valid(valid, kl_tok)
        n_valid = int(mask.sum().item())
        if n_valid:
            self._kl_sum += float(kl_tok[mask].sum().item())
            self._n_valid += n_valid
        bits = bits_per_case(kl_tok, valid=mask)
        self._bits_sum += float(bits["bits_per_case"]) * int(bits["n_cases"])
        self._n_cases += int(bits["n_cases"])
        d = int(kl_dim.size(-1))
        if self._d is None:
            self._d = d
            self._kl_dim_healthy = torch.zeros(d, dtype=torch.float64)
            self._kl_dim_sac = torch.zeros(d, dtype=torch.float64)
        # Split; without a mask every valid token is healthy.
        if sac_mask is None:
            healthy = mask
            sac = mask.new_zeros(mask.shape, dtype=torch.bool)
        else:
            self._saw_sac_mask = True
            sac = _token_valid(sac_mask, kl_tok) & mask
            healthy = mask & ~sac
        n_h = int(healthy.sum().item())
        n_s = int(sac.sum().item())
        if n_h:
            self._kl_dim_healthy = self._kl_dim_healthy + _mean_kl_per_dim(kl_dim, healthy).cpu() * n_h
            self._n_tok_healthy += n_h
        if n_s:
            self._kl_dim_sac = self._kl_dim_sac + _mean_kl_per_dim(kl_dim, sac).cpu() * n_s
            self._n_tok_sac += n_s

    def compute(self, beta: float | None = None, rate_target: float | None = None) -> dict[str, Any]:
        kl_mean = self._kl_sum / self._n_valid if self._n_valid else float("nan")
        bits_mean = self._bits_sum / self._n_cases if self._n_cases else float("nan")
        summary = {
            "kl_mean_raw": kl_mean,
            "bits_per_case": bits_mean,
            "n_valid_tokens": self._n_valid,
            "n_valid_tokens_mean": (self._n_valid / self._n_cases) if self._n_cases else 0.0,
            "n_cases": self._n_cases,
        }

        def _active(sum_dim: Tensor | None, n_tok: int) -> tuple[int | None, list[int]]:
            if sum_dim is None or self._d is None:
                return None, []
            if n_tok <= 0:
                return 0, []
            mean_d = sum_dim / float(n_tok)
            active = torch.isfinite(mean_d) & (mean_d > self.threshold)
            idxs = [int(i) for i in torch.nonzero(active, as_tuple=False).reshape(-1)]
            return int(active.sum().item()), idxs

        n_h, idx_h = _active(self._kl_dim_healthy, self._n_tok_healthy)
        if self._saw_sac_mask:
            n_s, idx_s = _active(self._kl_dim_sac, self._n_tok_sac)
        else:
            n_s, idx_s = None, []
        n_all, idx_all = _active(
            (self._kl_dim_healthy if self._kl_dim_healthy is not None else 0)
            + (self._kl_dim_sac if self._kl_dim_sac is not None else 0),
            self._n_tok_healthy + self._n_tok_sac,
        )
        active = {
            "n_active_healthy": n_h,
            "n_active_sac": n_s,
            "n_active_all": n_all,
            "active_dims_healthy": idx_h,
            "active_dims_sac": idx_s,
            "active_dims_all": idx_all,
            "sac_mask_available": self._saw_sac_mask,
            "active_unit_threshold_nats": self.threshold,
        }
        return epoch_latent_log_row(
            beta=beta,
            rate_target=rate_target,
            kl_summary=summary,
            active=active,
        )


@torch.no_grad()
def compute_latent_epoch_metrics(
    *,
    model: Any,
    dataloader: Any,
    device: Any,
    beta: float | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """The per-epoch latent row `train.py` logs: raw KL, bits/case, active units.

    Runs the encoder and the tract mixer only (the posterior the KL term sees),
    never the decoder, so it costs a small part of a validation pass.  R* is
    read from `config` at call time so a runtime override is reported.
    """
    try:
        import config as _config

        rate_target = float(getattr(_config, "RATE_TARGET_NATS", RATE_TARGET_NATS))
    except ImportError:
        rate_target = float(RATE_TARGET_NATS)
    was_training = bool(model.training)
    model.eval()
    acc = LatentMetricAccumulator()
    try:
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= int(max_batches):
                break
            batch = batch.to(device)
            mu_raw, logvar_raw = model.encode(batch)
            mu, logvar = model.mix_posterior(mu_raw, logvar_raw, batch)
            acc.update(mu.float(), logvar.float(), valid=getattr(batch, "latent_valid", None))
    finally:
        model.train(was_training)
    return acc.compute(beta=beta, rate_target=rate_target)


def rate_distortion_sweep(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """50-case D × R* grid of §5.3.7 — not run from this module.

    The sweep is an experiment after the tract, token layout, and r* fixes
    (STAGE2_REVIEW.md §5.3.7, last paragraph). Item 21 only ships the
    functions that *log* rate, active units, and the noise-robustness curve.
    """
    return {
        "skipped": True,
        "reason": (
            "Rate-distortion sweep (D in {4,8,16,32} x R* in {2,8,16,32} "
            "nats/token on 50 cases) is deferred until tract/token/r* fixes "
            "land; this module implements the logging functions only."
        ),
    }
