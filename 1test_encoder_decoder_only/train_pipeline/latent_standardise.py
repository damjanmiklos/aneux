"""Post-training latent standardisation (STAGE2_REVIEW §5.3.6 item 7).

Run the encoder on the training split, drop dimensions the rate controller
left inactive (mean raw KL < 0.01 nats, §5.3.7), then store per-dimension
mean and std of μ over valid tokens. Unused dimensions have μ ≈ 0 and a
near-zero spread: dividing by that std would amplify numerical noise into
the Stage-1 target.

Stage 1 trains on standardised codes of width ``surviving_count``. Stage 2
de-standardises (and scatters dropped dimensions back to 0, the raw-space
prior mean) before decoding.

The affine statistics are always computed from μ. Posterior samples
``μ + σ·ε`` are the preferred Stage-1 target and are a flag on
``standardise`` (free augmentation; stats themselves stay μ-based).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from config import ACTIVE_UNIT_KL_THRESH

__all__ = [
    "compute_latent_standardisation",
    "apply_standardisation_to_checkpoint",
    "standardise",
    "destandardise",
    "raw_kl_per_dim",
    "STATS_KEY",
    "LATENT_MEAN_KEY",
    "LATENT_STD_KEY",
    "LATENT_KEPT_DIMS_KEY",
    "LATENT_SURVIVING_COUNT_KEY",
    "STAGE1_SAMPLE_POSTERIOR",
]

# Nested blob and the four fields Appendix B item 20 requires in the checkpoint.
STATS_KEY = "latent_standardisation"
LATENT_MEAN_KEY = "latent_mean"
LATENT_STD_KEY = "latent_std"
LATENT_KEPT_DIMS_KEY = "latent_kept_dims"
LATENT_SURVIVING_COUNT_KEY = "latent_surviving_count"

# Documented Stage-1 default: train on standardised posterior samples.
STAGE1_SAMPLE_POSTERIOR = True

_STD_EPS = 1e-6
_DEFAULT_SAC_KEY = "latent_is_sac"


def raw_kl_per_dim(mu: Tensor, logvar: Tensor) -> Tensor:
    """Unclamped per-dimension KL of N(μ, σ²) against N(0, I), in nats.

    ``logvar`` is log(σ²). Returns the same shape as ``mu``.
    """
    return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())


def compute_latent_standardisation(
    encoder_or_model,
    dataloader,
    device,
    latent_valid_key: str = "latent_valid",
    kl_thresh: float = 0.01,
    sac_mask_key: str = _DEFAULT_SAC_KEY,
    eps: float = _STD_EPS,
) -> dict:
    """Encoder pass over the training split → drop inactive dims → μ mean/std.

    Parameters
    ----------
    encoder_or_model
        ``GraphVAE`` (uses ``encode``), an encoder module, or any callable
        that returns ``(mu, logvar)``, a ``VAEOutput``, or a dict with those
        keys. ``mu`` / ``logvar`` are ``[B, L, D]`` (or any shape with D last).
    dataloader
        Training-split iterator. Each batch is moved to ``device`` when it
        implements ``.to(device)``.
    device
        Device for the encoder forward.
    latent_valid_key
        Batch attribute / dict key for the token validity mask. Missing mask
        → every token is valid (padding-by-duplication era).
    kl_thresh
        Active-unit threshold in nats (§5.3.7). Dimensions with mean raw KL
        **below or equal** to this are dropped. Default 0.01; config
        ``ACTIVE_UNIT_KL_THRESH`` is the same number.
    sac_mask_key
        Optional per-token sac mask (wall neighbourhood ∩ dome mesh). When
        the key is absent — dome meshes not in the batch — only global
        stats are computed. Healthy = valid & ~sac when the mask is present.
    eps
        Floor on kept-dimension std so a surviving but near-constant dim
        cannot explode.

    Returns a dict consumed by ``standardise`` / ``destandardise`` /
    ``apply_standardisation_to_checkpoint``. Transform stats (``mean``,
    ``std``, ``kept_dims``, ``surviving_count``) are **global** over valid
    tokens. ``healthy`` / ``sac`` are report-only (None if no sac mask).
    """
    if kl_thresh is None:
        kl_thresh = float(ACTIVE_UNIT_KL_THRESH)
    kl_thresh = float(kl_thresh)
    eps = float(eps)
    device = torch.device(device) if device is not None else torch.device("cpu")

    module = encoder_or_model if isinstance(encoder_or_model, nn.Module) else None
    was_training = bool(module.training) if module is not None else False
    if module is not None:
        module.eval()

    acc_all = None
    acc_h = None
    acc_s = None
    saw_sac_mask = False
    n_batches = 0
    latent_dim = None

    try:
        with torch.no_grad():
            for batch in dataloader:
                n_batches += 1
                batch = _move_batch(batch, device)
                mu, logvar = _run_encoder(encoder_or_model, batch)
                mu = _as_token_matrix(mu)
                if logvar is None:
                    logvar = torch.zeros_like(mu)
                else:
                    logvar = _as_token_matrix(logvar)
                if mu.shape != logvar.shape:
                    raise ValueError(
                        f"mu shape {tuple(mu.shape)} != logvar shape {tuple(logvar.shape)}"
                    )
                d = int(mu.size(-1))
                if latent_dim is None:
                    latent_dim = d
                    acc_all = _new_acc(d)
                    acc_h = _new_acc(d)
                    acc_s = _new_acc(d)
                elif d != latent_dim:
                    raise ValueError(
                        f"latent width changed mid-pass: {latent_dim} -> {d}"
                    )

                n_tok = int(mu.size(0))
                valid = _token_mask(batch, latent_valid_key, n_tok, device=mu.device)
                sac = _token_mask(
                    batch, sac_mask_key, n_tok, device=mu.device, missing_ok=True
                )
                kl = raw_kl_per_dim(mu, logvar)

                mu_cpu = mu.detach().to("cpu")
                kl_cpu = kl.detach().to("cpu")
                valid_cpu = valid.detach().to("cpu")
                _acc_update(acc_all, mu_cpu[valid_cpu], kl_cpu[valid_cpu])

                if sac is not None:
                    saw_sac_mask = True
                    sac_cpu = sac.detach().to("cpu") & valid_cpu
                    healthy_cpu = (~sac.detach().to("cpu")) & valid_cpu
                    _acc_update(acc_s, mu_cpu[sac_cpu], kl_cpu[sac_cpu])
                    _acc_update(acc_h, mu_cpu[healthy_cpu], kl_cpu[healthy_cpu])
    finally:
        if module is not None and was_training:
            module.train()

    if acc_all is None or acc_all["n"] == 0:
        raise ValueError(
            "latent standardisation saw no valid tokens "
            f"(batches={n_batches}, latent_valid_key={latent_valid_key!r})"
        )

    global_stats = _finalize_acc(acc_all, eps)
    mean_kl = global_stats["mean_raw_kl"]
    # §5.3.7: active units are dimensions whose mean raw KL *exceeds* 0.01 nats.
    active = mean_kl > kl_thresh
    kept_dims = torch.nonzero(active, as_tuple=False).reshape(-1).to(dtype=torch.long)
    inactive_dims = torch.nonzero(~active, as_tuple=False).reshape(-1).to(dtype=torch.long)
    if kept_dims.numel() == 0:
        raise ValueError(
            f"every latent dimension is inactive (mean raw KL <= {kl_thresh} nats); "
            "refusing to standardise an empty code"
        )

    mean = global_stats["mean"][kept_dims].to(dtype=torch.float32)
    std = global_stats["std"][kept_dims].to(dtype=torch.float32)

    healthy_report = (
        _report_group(acc_h, eps, kl_thresh) if saw_sac_mask else None
    )
    sac_report = _report_group(acc_s, eps, kl_thresh) if saw_sac_mask else None

    return {
        "mean": mean,
        "std": std,
        "kept_dims": kept_dims,
        "surviving_count": int(kept_dims.numel()),
        "inactive_dims": inactive_dims,
        "mean_raw_kl": mean_kl.to(dtype=torch.float32),
        "active_mask": active,
        "kl_thresh": kl_thresh,
        "n_valid_tokens": int(acc_all["n"]),
        "n_dropped": int(inactive_dims.numel()),
        "latent_dim_in": int(latent_dim),
        "eps": eps,
        "prefer_posterior_samples": STAGE1_SAMPLE_POSTERIOR,
        "sac_mask_available": bool(saw_sac_mask),
        "healthy": healthy_report,
        "sac": sac_report,
    }


def apply_standardisation_to_checkpoint(ckpt: dict, stats: dict) -> dict:
    """Copy ``ckpt`` and store standardisation fields beside the weights.

    A raw ``state_dict`` (every value a tensor, no ``model`` / ``epoch`` key)
    is wrapped as ``{"model": <state_dict>, ...stats}`` so
    ``load_state_dict`` is not given unexpected keys. A training checkpoint
    that already has ``model`` / ``optimizer`` / ``epoch`` keeps that layout
    and gains the same stats keys.
    """
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint must be a dict, got {type(ckpt)!r}")
    payload = _checkpoint_payload(stats)
    nested = _cpu_stats(stats)
    if _is_raw_state_dict(ckpt):
        out = {"model": ckpt}
        out.update(payload)
        out[STATS_KEY] = nested
        return out
    out = dict(ckpt)
    out.update(payload)
    out[STATS_KEY] = nested
    return out


def standardise(
    mu: Tensor,
    stats: dict,
    logvar: Optional[Tensor] = None,
    sample_posterior: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Raw encoder codes → Stage-1 targets of width ``surviving_count``.

    Drops inactive dimensions then ``(x - mean) / std``. ``mu`` may be any
    shape with last dim ``latent_dim_in`` (or already the kept width).

    Set ``sample_posterior=True`` to use ``μ + σ·ε`` in raw space before the
    affine map (preferred Stage-1 target; requires ``logvar``). The stored
    mean/std remain those of μ.
    """
    x = _to_tensor(mu)
    kept, mean, std, d_in, k = _unpack_transform(stats, x.device, x.dtype)
    x = _select_kept(x, kept, d_in, k)
    if sample_posterior:
        if logvar is None:
            raise ValueError("sample_posterior=True requires logvar")
        lv = _select_kept(_to_tensor(logvar, like=x), kept, d_in, k)
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        x = x + torch.exp(0.5 * lv) * noise
    return (x - mean) / std


def destandardise(z: Tensor, stats: dict, restore_inactive: bool = True) -> Tensor:
    """Stage-1 codes → raw decoder codes.

    Undoes ``(x - mean) / std`` on kept dimensions. By default scatters back
    to width ``latent_dim_in`` with inactive dimensions filled by 0 (the
    null code / prior mean in raw space) so a decoder trained at full D
    still runs. ``restore_inactive=False`` returns only the kept raw dims.
    """
    x = _to_tensor(z)
    kept, mean, std, d_in, k = _unpack_transform(stats, x.device, x.dtype)
    if int(x.size(-1)) != k:
        raise ValueError(
            f"destandardise expected last dim {k} (surviving_count), got {int(x.size(-1))}"
        )
    raw_kept = x * std + mean
    if not restore_inactive:
        return raw_kept
    full = raw_kept.new_zeros(raw_kept.shape[:-1] + (d_in,))
    full[..., kept] = raw_kept
    return full


def _run_encoder(encoder_or_model, batch):
    encode = getattr(encoder_or_model, "encode", None)
    if callable(encode):
        out = encode(batch)
    elif isinstance(encoder_or_model, nn.Module):
        out = encoder_or_model(batch)
    elif callable(encoder_or_model):
        out = encoder_or_model(batch)
    else:
        raise TypeError(
            "encoder_or_model must be a module with encode/forward or a callable, "
            f"got {type(encoder_or_model)!r}"
        )
    return _extract_mu_logvar(out)


def _extract_mu_logvar(out) -> tuple[Tensor, Optional[Tensor]]:
    if isinstance(out, (tuple, list)):
        if len(out) < 1:
            raise ValueError("encoder returned an empty tuple")
        mu = out[0]
        logvar = out[1] if len(out) > 1 else None
        return mu, logvar
    if isinstance(out, dict):
        if "mu" not in out:
            raise KeyError("encoder dict output is missing 'mu'")
        return out["mu"], out.get("logvar")
    mu = getattr(out, "mu", None)
    if mu is None:
        raise TypeError(
            "encoder output must be (mu, logvar), a dict, or an object with .mu; "
            f"got {type(out)!r}"
        )
    return mu, getattr(out, "logvar", None)


def _move_batch(batch, device):
    to = getattr(batch, "to", None)
    if callable(to):
        try:
            return to(device)
        except TypeError:
            return batch
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


def _as_token_matrix(t: Tensor) -> Tensor:
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    if t.ndim == 0:
        raise ValueError("latent tensor is scalar")
    if t.ndim == 1:
        return t.reshape(-1, 1)
    return t.reshape(-1, t.size(-1))


def _lookup(batch, key: str):
    if batch is None or not key:
        return None
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


def _token_mask(
    batch,
    key: str,
    n_tokens: int,
    device,
    missing_ok: bool = False,
) -> Optional[Tensor]:
    raw = _lookup(batch, key)
    if raw is None:
        if missing_ok:
            return None
        return torch.ones(n_tokens, dtype=torch.bool, device=device)
    if not torch.is_tensor(raw):
        raw = torch.as_tensor(raw)
    mask = raw.reshape(-1) != 0
    if mask.numel() == 1 and n_tokens != 1:
        mask = mask.expand(n_tokens)
    if mask.numel() != n_tokens:
        raise ValueError(
            f"{key!r} has {mask.numel()} elements, expected {n_tokens} tokens"
        )
    return mask.to(device=device, dtype=torch.bool)


def _new_acc(d: int) -> dict[str, Any]:
    return {
        "n": 0,
        "sum": torch.zeros(d, dtype=torch.float64),
        "sumsq": torch.zeros(d, dtype=torch.float64),
        "sumkl": torch.zeros(d, dtype=torch.float64),
    }


def _acc_update(acc: dict, mu: Tensor, kl: Tensor) -> None:
    if mu.numel() == 0:
        return
    m = mu.detach().to(dtype=torch.float64, device="cpu")
    k = kl.detach().to(dtype=torch.float64, device="cpu")
    acc["n"] += int(m.size(0))
    acc["sum"] += m.sum(dim=0)
    acc["sumsq"] += m.pow(2).sum(dim=0)
    acc["sumkl"] += k.sum(dim=0)


def _finalize_acc(acc: dict, eps: float) -> Optional[dict]:
    n = int(acc["n"])
    if n == 0:
        return None
    mean = acc["sum"] / n
    var = (acc["sumsq"] / n) - mean.pow(2)
    std = var.clamp_min(0.0).sqrt().clamp_min(eps)
    mean_kl = acc["sumkl"] / n
    return {
        "n_tokens": n,
        "mean": mean.to(dtype=torch.float32),
        "std": std.to(dtype=torch.float32),
        "mean_raw_kl": mean_kl.to(dtype=torch.float32),
    }


def _report_group(acc: dict, eps: float, kl_thresh: float) -> Optional[dict]:
    fin = _finalize_acc(acc, eps)
    if fin is None:
        return None
    active = fin["mean_raw_kl"] > kl_thresh
    kept = torch.nonzero(active, as_tuple=False).reshape(-1).to(dtype=torch.long)
    fin["n_active"] = int(kept.numel())
    fin["kept_dims"] = kept
    return fin


def _to_tensor(x, like: Optional[Tensor] = None) -> Tensor:
    if torch.is_tensor(x):
        t = x
    else:
        t = torch.as_tensor(x)
    if like is not None:
        t = t.to(device=like.device, dtype=like.dtype)
    return t


def _unpack_transform(stats: dict, device, dtype):
    kept = _to_tensor(stats["kept_dims"]).to(device=device, dtype=torch.long)
    mean = _to_tensor(stats["mean"]).to(device=device, dtype=dtype)
    std = _to_tensor(stats["std"]).to(device=device, dtype=dtype)
    d_in = int(stats.get("latent_dim_in", int(kept.max().item()) + 1 if kept.numel() else 0))
    k = int(stats["surviving_count"])
    if mean.numel() != k or std.numel() != k or kept.numel() != k:
        raise ValueError(
            "stats mean/std/kept_dims length must equal surviving_count "
            f"({k}); got {int(mean.numel())}/{int(std.numel())}/{int(kept.numel())}"
        )
    std = std.clamp_min(float(stats.get("eps", _STD_EPS)))
    return kept, mean, std, d_in, k


def _select_kept(x: Tensor, kept: Tensor, d_in: int, k: int) -> Tensor:
    last = int(x.size(-1))
    if last == d_in:
        return x[..., kept]
    if last == k:
        return x
    raise ValueError(
        f"expected last dim {d_in} (raw) or {k} (kept), got {last}"
    )


def _is_raw_state_dict(ckpt: dict) -> bool:
    if any(k in ckpt for k in ("model", "epoch", "optimizer", STATS_KEY, LATENT_MEAN_KEY)):
        return False
    if not ckpt:
        return False
    return all(torch.is_tensor(v) for v in ckpt.values())


def _cpu_tensor(x):
    if torch.is_tensor(x):
        return x.detach().to("cpu").clone()
    return x


def _cpu_stats(stats: dict) -> dict:
    out = {}
    for key, value in stats.items():
        if isinstance(value, dict) or value is None:
            out[key] = None if value is None else _cpu_stats(value)
        else:
            out[key] = _cpu_tensor(value)
    return out


def _checkpoint_payload(stats: dict) -> dict:
    return {
        LATENT_MEAN_KEY: _cpu_tensor(stats["mean"]),
        LATENT_STD_KEY: _cpu_tensor(stats["std"]),
        LATENT_KEPT_DIMS_KEY: _cpu_tensor(stats["kept_dims"]),
        LATENT_SURVIVING_COUNT_KEY: int(stats["surviving_count"]),
    }
