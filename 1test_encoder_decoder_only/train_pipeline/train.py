import contextlib
import csv
import inspect
import math
import os
import random
import signal
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dist_utils import (
    all_reduce_sum_pair,
    barrier,
    distributed_active,
    is_main_process,
    reduce_mean_dict,
    unwrap_model,
    world_size,
    wrap_ddp,
)
from run_report import (
    TopKCheckpoints,
    capture_slurm_job_stats,
    dump_json,
    hardware_snapshot,
    plot_training_history,
    selection_score,
    write_history_tables,
    write_run_readme,
    write_run_summary,
)
from resource_monitor import ResourceMonitor

from config import (
    DEFAULT_LOSS_WEIGHTS,
    FOLLOW_BATCH,
    GRAD_CLIP,
    KL_WARMUP_EPOCHS,
    LAMBDA_KL,
    LEARNING_RATE,
    N_TRUE,
    N_TRUE_FAR_FRAC,
    TUBE_RADIUS_MM,
    WEIGHT_DECAY,
    configure_stage2_precision,
)
from dataset import _CPU_TENSOR_KEYS
import losses as _losses_mod
from losses import compute_losses

# §8: 0.99–0.995 with warm-up. Do not use config.EMA_DECAY (still 0.999).
DEFAULT_EMA_DECAY = 0.993
LR_WARMUP_STEPS = 300
POSE_JITTER_DEG = 5.0

try:
    from latent_standardise import standardise_and_prune_latent as _standardise_and_prune_latent
except ImportError:
    _standardise_and_prune_latent = None
try:
    from latent_standardise import compute_latent_standardisation as _compute_latent_standardisation
except ImportError:
    _compute_latent_standardisation = None
try:
    from latent_standardise import apply_standardisation_to_checkpoint as _apply_standardisation_to_checkpoint
except ImportError:
    _apply_standardisation_to_checkpoint = None
try:
    from latent_metrics import compute_latent_epoch_metrics as _compute_latent_epoch_metrics
except ImportError:
    _compute_latent_epoch_metrics = None


def _device_type(device):
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":")[0]


def _gib(nbytes):
    return nbytes / float(1024 ** 3)


def _cuda_index(device):
    dev = torch.device(device)
    if dev.index is not None:
        return dev.index
    return torch.cuda.current_device()


def resolve_train_device(device="cuda"):
    """Use the caller's `device` argument. `auto` / None → first visible CUDA.

    `aneuxai.py` owns the default GPU pick. This helper does not hardcode
    `gpu_index = 1`; if CUDA_VISIBLE_DEVICES is set, `cuda:0` is that device.
    """
    if isinstance(device, torch.device):
        return device
    if device is None or str(device).strip().lower() in ("", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def _ema_decay_default(explicit=None):
    """Prefer a value aneuxai exported; otherwise §8's 0.993."""
    if explicit is not None:
        return float(explicit)
    try:
        import aneuxai
        val = getattr(aneuxai, "EMA_DECAY", None)
        if val is not None:
            return float(val)
    except Exception:
        pass
    return float(DEFAULT_EMA_DECAY)


def _print_vram(tag, device, model=None):
    """Snapshot of live CUDA memory. Peak is since the last reset_peak_memory_stats."""
    if _device_type(device) != "cuda":
        return
    idx = _cuda_index(device)
    torch.cuda.synchronize(idx)
    free_b, total_b = torch.cuda.mem_get_info(idx)
    lines = [
        f"VRAM {tag}:",
        f"  GPU: {torch.cuda.get_device_name(idx)} (cuda:{idx})  {_gib(total_b):.2f} GiB total",
    ]
    if model is not None:
        param_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lines.append(
            f"  Parameters (fp32): {_gib(param_bytes):.3f} GiB  ({n_params:,} params)"
        )
    lines.extend(
        [
            f"  Current allocated: {_gib(torch.cuda.memory_allocated(idx)):.3f} GiB",
            f"  Current reserved:  {_gib(torch.cuda.memory_reserved(idx)):.3f} GiB",
            f"  Peak allocated:    {_gib(torch.cuda.max_memory_allocated(idx)):.3f} GiB",
            f"  Peak reserved:     {_gib(torch.cuda.max_memory_reserved(idx)):.3f} GiB",
            f"  Free (driver):     {_gib(free_b):.2f} / {_gib(total_b):.2f} GiB",
        ]
    )
    tqdm.write("\n".join(lines))


def _keep_meta_on_cpu(batch):
    for key in _CPU_TENSOR_KEYS:
        val = getattr(batch, key, None)
        if torch.is_tensor(val) and val.device.type != "cpu":
            setattr(batch, key, val.cpu())
    return batch


def _face_from_batch(batch):
    face = getattr(batch, "face", None)
    if face is not None and face.numel() > 0:
        return face
    faces = getattr(batch, "faces", None)
    if faces is not None and faces.numel() > 0:
        return faces
    return None


def kl_anneal_weight(epoch, max_weight=LAMBDA_KL, warmup_epochs=KL_WARMUP_EPOCHS):
    """Linear KL anneal: 0 at epoch 1, `max_weight` from epoch `warmup_epochs` onward.

    Used only when GECO is unavailable. With GECO, β is the KL weight and the
    ceiling β_max ramps from GECO_BETA_INIT to GECO_BETA_MAX (never 0).
    """
    if warmup_epochs <= 1:
        return float(max_weight)
    t = min(1.0, max(0.0, (epoch - 1) / float(warmup_epochs - 1)))
    return float(max_weight) * t


def init_geco_beta():
    """Initial dual variable. Prefer aneuxai, else config.GECO_BETA_INIT, else 1.0."""
    try:
        import aneuxai

        val = getattr(aneuxai, "GECO_BETA_INIT", None)
        if val is not None:
            return float(val)
    except Exception:
        pass
    import config as cfg

    return float(getattr(cfg, "GECO_BETA_INIT", 1.0))


def geco_is_available():
    """True when losses.py exposes the dual update *and* compute_losses takes kl_beta."""
    return (
        getattr(_losses_mod, "update_geco_beta", None) is not None
        and _fn_has_param(compute_losses, "kl_beta")
    )


def step_geco_beta(beta, kl_mean_raw, epoch=None, warmup_epochs=None):
    """One optimiser-step GECO update. No-op if ``update_geco_beta`` is missing."""
    fn = getattr(_losses_mod, "update_geco_beta", None)
    if fn is None or beta is None or kl_mean_raw is None:
        # losses.update_geco_beta missing — cannot adapt β (item 19 stays inert).
        return beta
    kwargs = {}
    if _fn_has_param(fn, "epoch") and epoch is not None:
        kwargs["epoch"] = epoch
    if _fn_has_param(fn, "warmup_epochs") and warmup_epochs is not None:
        kwargs["warmup_epochs"] = warmup_epochs
    return fn(beta, kl_mean_raw, **kwargs)


def _geco_beta_max(epoch, warmup_epochs=None):
    fn = getattr(_losses_mod, "geco_beta_max_for_epoch", None)
    if fn is None:
        # losses.geco_beta_max_for_epoch missing — skip logging the ramp ceiling.
        return None
    kwargs = {}
    if _fn_has_param(fn, "warmup_epochs") and warmup_epochs is not None:
        kwargs["warmup_epochs"] = warmup_epochs
    return float(fn(epoch, **kwargs))


def _tensor_scalar(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return value.reshape(-1).float().mean()
    return torch.tensor(float(value))


def weighted_total(terms, weights):
    total = (
        weights["recon"] * terms["recon"]
        + weights["kl"] * terms["kl"]
        + weights["disp"] * terms["disp"]
        + weights["lap"] * terms["lap"]
        + weights["norm"] * terms["norm"]
    )
    if "rad" in terms:
        total = total + float(weights.get("rad", 0.0)) * terms["rad"]
    for extra in ("fold", "conf", "stretch"):
        if extra in terms and extra in weights:
            total = total + float(weights[extra]) * terms[extra]
    return total


def _weighted_total(terms, weights):
    return weighted_total(terms, weights)


def ema_warmup_decay(decay, n_updates):
    """§8 EMA warm-up: min(d, (1 + n) / (10 + n)), n = updates already applied.

    Applied in the 0.99–0.995 regime the review specified. Lower decay already
    has a short horizon and keeps constant d (keeps existing ModelEMA tests).
    """
    decay = float(decay)
    n = max(0, int(n_updates))
    if decay < 0.99:
        return decay
    return min(decay, (1.0 + n) / (10.0 + n))


def adamw_param_groups(model, weight_decay):
    """AdamW groups: no weight decay on LayerNorms, biases, and gates (§8)."""
    skip = set()
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            for param in module.parameters(recurse=False):
                skip.add(id(param))
        for pname, param in module.named_parameters(recurse=False):
            key = pname.lower()
            if key in ("alpha", "alpha_raw", "alpha_c_raw", "alpha_m_raw") or "gate" in key:
                skip.add(id(param))
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        key = name.lower()
        if (
            param.ndim <= 1
            or key.endswith("bias")
            or id(param) in skip
            or "gate" in key
            or key.endswith(".alpha")
            or "alpha_raw" in key
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        groups.append({"params": list(model.parameters()), "weight_decay": float(weight_decay)})
    return groups


def build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, eta_min):
    """Linear LR warm-up for `warmup_steps` optimiser steps, then cosine."""
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    if warmup_steps <= 0:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
    warmup_steps = min(warmup_steps, max(1, total_steps - 1))
    remain = max(1, total_steps - warmup_steps)
    warmup = LinearLR(
        optimizer,
        start_factor=max(1e-6, 1.0 / float(warmup_steps)),
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(optimizer, T_max=remain, eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cuda_state)
        except Exception:
            pass


def _optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


class ModelEMA:
    """Exponential moving average of model weights for validation and export.

    Shadows live on CPU so training does not keep a second 43M-parameter copy
    in VRAM (Windows WDDM otherwise pages that into system RAM).
    Decay uses §8 warm-up `min(d, (1 + n) / (10 + n))`.
    """

    def __init__(self, model, decay=DEFAULT_EMA_DECAY):
        self.decay = float(decay)
        self.shadow = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
        self._backup = None
        self.n_updates = 0

    @torch.no_grad()
    def update(self, model):
        decay = ema_warmup_decay(self.decay, self.n_updates)
        msd = model.state_dict()
        for key, shadow in self.shadow.items():
            value = msd[key].detach()
            if value.device != shadow.device:
                value = value.to(device=shadow.device)
            if shadow.dtype.is_floating_point:
                shadow.mul_(decay).add_(value, alpha=1.0 - decay)
            else:
                shadow.copy_(value)
        self.n_updates += 1

    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def store(self, model):
        self._backup = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def restore(self, model):
        if self._backup is not None:
            model.load_state_dict(self._backup, strict=True)
            self._backup = None

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow, "n_updates": int(self.n_updates)}

    def load_state_dict(self, state):
        if not state:
            return
        if "decay" in state:
            self.decay = float(state["decay"])
        if "shadow" in state and state["shadow"] is not None:
            self.shadow = state["shadow"]
        self.n_updates = int(state.get("n_updates", 0))


def save_training_checkpoint(
    path,
    *,
    model,
    ema,
    optimizer,
    scheduler,
    epoch,
    metrics,
    global_step=0,
    geco_beta=None,
):
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": unwrap_model(model).state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
        "rng": capture_rng_state(),
        "geco_beta": None if geco_beta is None else float(geco_beta),
    }
    torch.save(payload, path)
    return payload


def load_training_checkpoint(
    path,
    *,
    model,
    ema=None,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
):
    """Restore model / EMA / optimiser / scheduler / epoch / RNG from last.pt."""
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"{path} is not a training checkpoint (expected last.pt dict)")
    model.load_state_dict(ckpt["model"], strict=True)
    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        device = next(model.parameters()).device
        _optimizer_to_device(optimizer, device)
    if scheduler is not None and ckpt.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as exc:
            tqdm.write(f"Warning: could not restore scheduler from {path}: {exc}")
    restore_rng_state(ckpt.get("rng"))
    return ckpt


def resolve_resume_path(resume, ckpt_dir):
    """How resume is invoked: resume=True/'auto'/path; False disables it."""
    if resume is False or resume is None:
        return None
    if isinstance(resume, str) and resume.strip().lower() not in ("", "auto", "true", "1", "yes"):
        return resume
    if ckpt_dir:
        for candidate in (
            os.path.join(ckpt_dir, "checkpoints", "last.pt"),
            os.path.join(ckpt_dir, "last.pt"),
        ):
            if os.path.isfile(candidate):
                return candidate
    return None


def losses_from_output(out, batch, kl_beta=None):
    batch_mid = getattr(batch, "pos_mid_batch", None)
    batch_coarse = getattr(batch, "pos_coarse_batch", None)
    cl_batch = getattr(batch, "cl_dense_batch", None)
    extra = {}
    if _fn_has_param(compute_losses, "kl_beta"):
        extra["kl_beta"] = kl_beta
    # else: compute_losses has no kl_beta — GECO cannot scale the KL tensor.
    if _fn_has_param(compute_losses, "latent_valid"):
        extra["latent_valid"] = getattr(batch, "latent_valid", None)
    terms = compute_losses(
        out.x_pred,
        batch.x_true,
        out.mu,
        out.logvar,
        batch.x,
        batch.edge_index,
        batch.x_true_batch,
        batch.num_graphs,
        face=_face_from_batch(batch),
        batch_tube=batch.batch,
        delta_x=out.delta_x,
        delta_r=getattr(out, "delta_r", None),
        delta_s=getattr(out, "delta_s", None),
        x_pred_mid=out.x_pred_mid,
        batch_mid=batch_mid,
        x_pred_coarse=out.x_pred_coarse,
        batch_coarse=batch_coarse,
        x_true_cl_dist=getattr(batch, "x_true_cl_dist", None),
        cl_dense=getattr(batch, "cl_dense", None),
        cl_dense_batch=cl_batch,
        r_star=getattr(batch, "r_star", None),
        r_star_valid=getattr(batch, "r_star_valid", None),
        r_star_ambiguous=getattr(batch, "r_star_ambiguous", None),
        r_dth=getattr(batch, "r_dth", None),
        r_du=getattr(batch, "r_du", None),
        r_ring_med=getattr(batch, "r_ring_med", None),
        r_star_mid=getattr(batch, "r_star_mid", None),
        r_star_valid_mid=getattr(batch, "r_star_valid_mid", None),
        normal=getattr(batch, "normal", None),
        normal_mid=getattr(batch, "normal_mid", None),
        pos_mid=getattr(batch, "pos_mid", None),
        x_true_normal=getattr(batch, "x_true_normal", None),
        has_true_normal=getattr(batch, "has_true_normal", None),
        tract_id=getattr(batch, "tract_id", None),
        pos_coarse=getattr(batch, "pos_coarse", None),
        normal_coarse=getattr(batch, "normal_coarse", None),
        # Without these the radial term read r = 2 mm + n·Δx against
        # r* = r_local + stretch (a 0.5 mm branch was driven 1.5 mm inward,
        # through its own axis) and the Chamfer target was the 16k subsample.
        r_local=getattr(batch, "r_local", None),
        r_local_mid=getattr(batch, "r_local_mid", None),
        r_local_coarse=getattr(batch, "r_local_coarse", None),
        gt_points=getattr(batch, "gt_points", None),
        gt_normals=getattr(batch, "gt_normals", None),
        gt_batch=getattr(batch, "gt_points_batch", None),
        gt_cl_dist=getattr(batch, "gt_cl_dist", None),
        x_template=batch.x,
        face_mid=getattr(batch, "face_mid", None),
        face_coarse=getattr(batch, "face_coarse", None),
        edge_index_mid=getattr(batch, "edge_index_mid", None),
        edge_index_coarse=getattr(batch, "edge_index_coarse", None),
        **extra,
    )
    if isinstance(terms, tuple) and len(terms) == 2 and isinstance(terms[0], dict):
        terms, info = terms
    else:
        info = _kl_info_from_output(terms, out, batch, kl_beta)
    terms = dict(terms)
    for key in ("kl_mean_raw", "rate_gap", "beta"):
        if key in info and key not in terms:
            terms[key] = info[key]
    return terms


def _kl_info_from_output(terms, out, batch, kl_beta):
    """Read kl_mean_raw from the losses info dict (compute_losses currently drops it)."""
    if isinstance(terms, dict):
        nested = terms.get("kl_info")
        if isinstance(nested, dict):
            return nested
        if "kl_mean_raw" in terms:
            return terms
    vae_kl = getattr(_losses_mod, "vae_kl_loss", None)
    if vae_kl is None or getattr(out, "mu", None) is None:
        # losses.vae_kl_loss missing — cannot recover kl_mean_raw for the dual.
        return {}
    kwargs = {}
    if _fn_has_param(vae_kl, "latent_valid"):
        kwargs["latent_valid"] = getattr(batch, "latent_valid", None)
    if _fn_has_param(vae_kl, "beta"):
        kwargs["beta"] = kl_beta
    result = vae_kl(out.mu, out.logvar, **kwargs)
    if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], dict):
        return result[1]
    return {}


def _fn_has_param(fn, name):
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _forward_model(model, batch, sample=None):
    """Run GraphVAE.forward. Pass `sample` when model.py exposes it (item 9).

    DDP's wrapper ``forward`` is ``(*inputs, **kwargs)`` and has no
    ``reparameterize``. One GPU never wraps, so this only shows up on the
    4-GPU job, on the validation-σ pass. Inspect and patch the inner module;
    still call the wrapper so NCCL brackets the forward.
    """
    core = unwrap_model(model)
    fwd = core.forward
    if _fn_has_param(fwd, "sample"):
        if sample is None:
            return model(batch)
        return model(batch, sample=bool(sample))

    want = None if sample is None else bool(sample)
    if want is None:
        return model(batch)
    if want and model.training:
        return model(batch)
    if (not want) and (not model.training):
        return model(batch)

    orig = core.reparameterize
    if want and _fn_has_param(orig, "sample"):
        def _forced(mu, logvar, *args, **kwargs):
            kwargs["sample"] = True
            return orig(mu, logvar, *args, **kwargs)

        core.reparameterize = _forced
    elif want:
        def _sampled(mu, logvar, *args, **kwargs):
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std

        core.reparameterize = _sampled
    else:
        def _mu(mu, logvar, *args, **kwargs):
            return mu

        core.reparameterize = _mu
    try:
        return model(batch)
    finally:
        core.reparameterize = orig


def _accum_window_len(step, n_batches, accum_steps):
    window_start = (step // accum_steps) * accum_steps
    return min(accum_steps, n_batches - window_start)


def _zero_meters():
    return {
        "loss": 0.0,
        "recon": 0.0,
        "kl": 0.0,
        "disp": 0.0,
        "lap": 0.0,
        "norm": 0.0,
        "rad": 0.0,
        "kl_mean_raw": 0.0,
        "rate_gap": 0.0,
        "fold": 0.0,
        "conf": 0.0,
        "fold_tpl": 0.0,
        "stretch": 0.0,
    }


def _add_meter(acc, key, value, scale):
    if torch.is_tensor(value):
        num = float((value.detach() * scale).reshape(-1).mean().item())
    else:
        num = float(value) * float(scale)
    acc[key] = float(acc.get(key) or 0.0) + num


def _flush_meters(acc, total_samples):
    if total_samples == 0:
        return _zero_meters()
    out = {}
    for key, value in acc.items():
        if value is None:
            out[key] = 0.0
        elif torch.is_tensor(value):
            out[key] = float(value.item()) / total_samples
        else:
            out[key] = float(value) / total_samples
    return out


def _zero_tensor_meters():
    return {k: None for k in _zero_meters()}


def _worker_init(_worker_id):
    torch.set_num_threads(1)


def _wrap_pi(theta):
    return torch.remainder(theta + math.pi, 2.0 * math.pi) - math.pi


def _batch_tensor(batch, name):
    val = getattr(batch, name, None)
    return val if torch.is_tensor(val) and val.numel() > 0 else None


def _num_graphs(batch):
    n = getattr(batch, "num_graphs", None)
    if n is not None:
        return int(n)
    b = getattr(batch, "batch", None)
    if torch.is_tensor(b) and b.numel() > 0:
        return int(b.max().item()) + 1
    return 1


def _index_tensor(batch, name, n, device):
    val = getattr(batch, name, None)
    if torch.is_tensor(val) and val.numel() == n:
        return val.to(device=device)
    return None


def _cpu_generator(generator):
    if generator is None:
        return None
    try:
        if generator.device.type != "cpu":
            return None
    except Exception:
        return generator
    return generator


def _subset_indices(points, n_keep, generator=None):
    n_pts = int(points.size(0))
    n_keep = int(max(0, min(n_keep, n_pts)))
    if n_keep <= 0:
        return points.new_zeros((0,), dtype=torch.long)
    if n_keep == n_pts:
        return torch.arange(n_pts, device=points.device)
    # ops.fps_indices is deterministic (centroid start). Seeded diversity comes
    # from a random permutation; FPS is applied only on a proper random pool.
    g = _cpu_generator(generator)
    perm = torch.randperm(n_pts, device="cpu", generator=g).to(device=points.device)
    pool_n = min(n_pts, max(n_keep * 4, n_keep))
    if pool_n >= n_pts:
        return perm[:n_keep]
    pool = perm[:pool_n]
    try:
        from ops import fps_indices

        loc = fps_indices(points[pool], n_keep)
        return pool[loc]
    except Exception:
        return perm[:n_keep]


def _hybrid_subset_indices(points, n_keep, cl_xyz=None, generator=None):
    n_keep = int(n_keep)
    n_far = int(round(n_keep * float(N_TRUE_FAR_FRAC)))
    n_uni = max(1, n_keep - n_far)
    uni = _subset_indices(points, n_uni, generator=generator)
    if n_far <= 0 or cl_xyz is None or cl_xyz.numel() == 0:
        extra = _subset_indices(points, n_far, generator=generator) if n_far else uni[:0]
        return torch.cat([uni, extra], dim=0)[:n_keep]
    cl = cl_xyz.to(device=points.device, dtype=points.dtype).reshape(-1, 3)
    dmin = torch.cdist(points.float(), cl.float()).min(dim=1).values
    far_mask = dmin > (float(TUBE_RADIUS_MM) + 1.0)
    far_idx = far_mask.nonzero(as_tuple=False).view(-1)
    if far_idx.numel() == 0:
        extra = _subset_indices(points, n_far, generator=generator)
    else:
        extra_local = _subset_indices(points[far_idx], min(n_far, int(far_idx.numel())), generator=generator)
        extra = far_idx[extra_local]
        if extra.numel() < n_far:
            pad = _subset_indices(points, n_far - int(extra.numel()), generator=generator)
            extra = torch.cat([extra, pad], dim=0)
    return torch.cat([uni, extra], dim=0)[:n_keep]


def _pack_resampled(batch, new_pts, new_nrm, new_dist, new_bidx):
    batch.x_true = new_pts
    if new_nrm is not None:
        batch.x_true_normal = new_nrm
    if new_dist is not None:
        batch.x_true_cl_dist = new_dist
    if new_bidx is not None:
        batch.x_true_batch = new_bidx
    return batch


def _local_sample_x_true(batch, n_true=None, generator=None):
    """Fallback when dataset.sample_x_true is not importable yet (item 27)."""
    gt = _batch_tensor(batch, "gt_points")
    if gt is None:
        return batch
    n_graphs = _num_graphs(batch)
    gt_batch = _index_tensor(batch, "gt_points_batch", gt.size(0), gt.device)
    nrm = _batch_tensor(batch, "gt_normals")
    if nrm is None:
        nrm = _batch_tensor(batch, "gt_points_normal")
    dist = _batch_tensor(batch, "gt_cl_dist")
    cl = _batch_tensor(batch, "cl_dense")
    cl_xyz = cl[:, :3] if cl is not None and cl.size(-1) >= 3 else None
    x_true = _batch_tensor(batch, "x_true")
    x_batch = None
    if x_true is not None:
        x_batch = _index_tensor(batch, "x_true_batch", x_true.size(0), x_true.device)

    if n_true is None:
        if x_true is not None and n_graphs <= 1:
            n_true = int(x_true.size(0))
        elif x_true is not None and x_batch is not None:
            n_true = None
        else:
            n_true = int(N_TRUE)

    pieces, nrm_p, dist_p, b_p = [], [], [], []
    for g in range(n_graphs):
        if gt_batch is None:
            if n_graphs > 1:
                break
            sl = torch.arange(gt.size(0), device=gt.device)
        else:
            sl = (gt_batch == g).nonzero(as_tuple=False).view(-1)
        if sl.numel() == 0:
            continue
        if n_true is None and x_batch is not None:
            n_g = int((x_batch == g).sum().item())
        else:
            n_g = int(n_true)
        n_g = max(1, n_g)
        cl_g = None
        if cl_xyz is not None:
            cl_b = _index_tensor(batch, "cl_dense_batch", cl_xyz.size(0), cl_xyz.device)
            cl_g = cl_xyz if cl_b is None else cl_xyz[cl_b == g]
        idx = _hybrid_subset_indices(gt[sl], n_g, cl_xyz=cl_g, generator=generator)
        take = sl[idx]
        pieces.append(gt[take])
        if nrm is not None and nrm.size(0) == gt.size(0):
            nrm_p.append(nrm[take])
        if dist is not None and dist.size(0) == gt.size(0):
            dist_p.append(dist[take])
        b_p.append(torch.full((take.numel(),), g, device=gt.device, dtype=torch.long))
    if not pieces:
        return batch
    new_pts = torch.cat(pieces, dim=0)
    new_nrm = torch.cat(nrm_p, dim=0) if nrm_p else None
    new_dist = torch.cat(dist_p, dim=0) if dist_p else None
    new_bidx = torch.cat(b_p, dim=0) if b_p else None
    return _pack_resampled(batch, new_pts, new_nrm, new_dist, new_bidx)


def resample_x_true(batch, n_true=None, generator=None):
    """Draw a fresh x_true from cached full GT. Does not write caches.

    Prefers `dataset.sample_x_true` when the dataset sibling has landed;
    otherwise FPS/random subset of `gt_points`.
    """
    if _batch_tensor(batch, "gt_points") is None:
        return batch
    try:
        from dataset import sample_x_true as _ds_sample_x_true
    except ImportError:
        _ds_sample_x_true = None

    if _ds_sample_x_true is not None:
        kwargs = {}
        if _fn_has_param(_ds_sample_x_true, "n_true") and n_true is not None:
            kwargs["n_true"] = n_true
        if _fn_has_param(_ds_sample_x_true, "generator") and generator is not None:
            kwargs["generator"] = generator
        out = None
        try:
            out = _ds_sample_x_true(batch, **kwargs)
        except TypeError:
            try:
                n = int(n_true if n_true is not None else N_TRUE)
                out = _ds_sample_x_true(batch.gt_points, n)
            except TypeError:
                out = None
        if out is batch or getattr(out, "x_true", None) is not None:
            return out
        if torch.is_tensor(out):
            batch.x_true = out
            return batch
        if isinstance(out, (tuple, list)) and out:
            batch.x_true = out[0]
            if len(out) > 1 and torch.is_tensor(out[1]):
                batch.x_true_normal = out[1]
            return batch
    return _local_sample_x_true(batch, n_true=n_true, generator=generator)


# L/R mirror, x -> -x in the posed frame.  Reflected on the fly, per graph,
# so the scaffold, the frames and the GT turn together (cache v12 no longer
# stores mirrored copies of the GT).
#   positions and polar vectors (normals, tangents)  ->  M v
#   binormals (b = ±n x t, an axial vector)           -> -M b, frame handedness kept
#   theta, torsion (handedness-dependent scalars)     ->  negated
#   triangles                                         ->  winding flipped, normals stay outward
# Radii, distances, u, |dr/dtheta| and the upsample tables are unchanged;
# pose_R (bookkeeping only) takes M on its rows.
_MIRROR_POS_KEYS = (
    "x", "pos_mid", "pos_coarse", "x_true", "gt_points", "latent_pos", "token_pos",
    "boundary_plane_origin", "boundary_plane_origin_mid", "boundary_plane_origin_coarse",
)
_MIRROR_VEC_KEYS = (
    "normal", "normal_mid", "normal_coarse",
    "tangent", "tangent_mid", "tangent_coarse",
    "x_true_normal", "gt_normals", "gt_points_normal",
    "boundary_plane_normal", "boundary_plane_normal_mid", "boundary_plane_normal_coarse",
)
_MIRROR_AXIAL_KEYS = ("binormal", "binormal_mid", "binormal_coarse")
_MIRROR_ODD_SCALARS = ("theta", "theta_mid", "theta_coarse", "torsion", "torsion_mid", "torsion_coarse")
# face tensor -> the point set its indices address
_MIRROR_FACES = (("face", "x"), ("face_mid", "pos_mid"), ("face_coarse", "pos_coarse"), ("gt_faces", "gt_points"))


def _mirror_rows_graph(batch, key, n, n_graphs):
    """Graph index of each row of ``batch.<key>``, or None if it cannot be told."""
    if n_graphs == 1:
        return torch.zeros(n, dtype=torch.long)
    own = getattr(batch, f"{key}_batch", None)
    if torch.is_tensor(own) and own.numel() == n:
        return own
    for suffix, ref in (("_mid", "pos_mid_batch"), ("_coarse", "pos_coarse_batch")):
        if key.endswith(suffix):
            b = getattr(batch, ref, None)
            return b if torch.is_tensor(b) and b.numel() == n else None
    for ref in ("batch", "x_true_batch", "gt_points_batch"):
        b = getattr(batch, ref, None)
        if torch.is_tensor(b) and b.numel() == n:
            return b
    if key == "token_pos" and n % n_graphs == 0:
        return torch.arange(n) // (n // n_graphs)
    return None


def apply_mirror(batch, p=0.5, generator=None, flip=None):
    """Reflect each graph of the batch with probability ``p`` (see the table above).

    ``flip`` ([n_graphs] bool) overrides the draw.  Returns (batch, flip).
    """
    n_graphs = _num_graphs(batch)
    if flip is None:
        flip = torch.rand(n_graphs, generator=_cpu_generator(generator)) < float(p)
    flip = torch.as_tensor(flip, dtype=torch.bool).reshape(-1).cpu()
    if flip.numel() != n_graphs:
        raise ValueError(f"apply_mirror: flip has {flip.numel()} entries for {n_graphs} graphs")
    if not bool(flip.any()):
        return batch, flip

    def rows(key, n, device):
        g = _mirror_rows_graph(batch, key, n, n_graphs)
        if g is None:
            raise ValueError(f"apply_mirror: cannot tell which graph each row of {key!r} belongs to")
        return flip.to(device)[g.to(device)]

    def reflect(key, sign_other=1.0, sign_x=-1.0):
        v = _batch_tensor(batch, key)
        if v is None or v.size(-1) < 3:
            return
        m = rows(key, v.size(0), v.device).unsqueeze(-1)
        s = v.new_tensor([sign_x, sign_other, sign_other])
        out = v.clone()
        out[..., :3] = torch.where(m, v[..., :3] * s, v[..., :3])
        setattr(batch, key, out)

    for key in _MIRROR_POS_KEYS + _MIRROR_VEC_KEYS:
        reflect(key)
    for key in _MIRROR_AXIAL_KEYS:
        reflect(key, sign_other=-1.0, sign_x=1.0)
    cl = _batch_tensor(batch, "cl_dense")
    if cl is not None:
        reflect("cl_dense")
    for key in _MIRROR_ODD_SCALARS:
        v = _batch_tensor(batch, key)
        if v is None:
            continue
        m = rows(key, v.reshape(-1).numel(), v.device).reshape(v.shape)
        neg = -v
        if key.startswith("theta"):
            # theta lives in [-pi, pi); only -pi leaves it when negated.  Fixing that one
            # value (instead of _wrap_pi) keeps every other angle bit-exact.
            neg = torch.where(neg >= math.pi, neg - 2.0 * math.pi, neg)
        setattr(batch, key, torch.where(m, neg, v))
    for fkey, pkey in _MIRROR_FACES:
        f = _batch_tensor(batch, fkey)
        pts = _batch_tensor(batch, pkey)
        if f is None or pts is None:
            continue
        g = _mirror_rows_graph(batch, pkey, pts.size(0), n_graphs)
        if g is None:
            raise ValueError(f"apply_mirror: cannot assign {fkey!r} to graphs")
        m = flip.to(f.device)[g.to(f.device)[f[0]]]
        out = f.clone()
        out[1] = torch.where(m, f[2], f[1])
        out[2] = torch.where(m, f[1], f[2])
        setattr(batch, fkey, out)
    # bookkeeping only (posed = pose_R @ world): fold M in so pose_R still maps to the posed frame
    pose_R = getattr(batch, "pose_R", None)
    if torch.is_tensor(pose_R) and pose_R.numel() == 9 * n_graphs:
        R = pose_R.reshape(n_graphs, 3, 3).clone()
        R[flip.to(R.device), 0, :] *= -1.0
        batch.pose_R = R.reshape(pose_R.shape)
    return batch, flip


def apply_theta_phase(batch, generator=None):
    """Add a random θ-phase in (−π, π] on the already-built scaffold."""
    phase = (torch.rand((), generator=generator) * 2.0 - 1.0) * math.pi
    phase_v = float(phase.item())
    for key in ("theta", "theta_mid", "theta_coarse"):
        th = _batch_tensor(batch, key)
        if th is not None:
            setattr(batch, key, _wrap_pi(th + phase_v))
    return batch, phase_v


def _axis_angle_rotation(max_deg, device, dtype, generator=None):
    axis = torch.randn(3, generator=generator, dtype=torch.float32)
    axis = axis / axis.norm().clamp_min(1e-8)
    ang = (torch.rand((), generator=generator) * 2.0 - 1.0) * math.radians(float(max_deg))
    x, y, z = axis.tolist()
    c = float(torch.cos(ang))
    s = float(torch.sin(ang))
    C = 1.0 - c
    R = torch.tensor(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        device=device,
        dtype=dtype,
    )
    return R


# Every xyz position and every direction in the batch must turn together:
# a rim plane left behind drags the rim off its ostium by the jitter angle.
_POS_KEYS = (
    "x",
    "pos_mid",
    "pos_coarse",
    "x_true",
    "latent_pos",
    "token_pos",
    "gt_points",
    "boundary_plane_origin",
    "boundary_plane_origin_mid",
    "boundary_plane_origin_coarse",
)
_VEC_KEYS = (
    "normal",
    "normal_mid",
    "normal_coarse",
    "tangent",
    "tangent_mid",
    "tangent_coarse",
    "binormal",
    "binormal_mid",
    "binormal_coarse",
    "x_true_normal",
    "gt_normals",
    "gt_points_normal",
    "boundary_plane_normal",
    "boundary_plane_normal_mid",
    "boundary_plane_normal_coarse",
)


def apply_pose_jitter(batch, max_deg=POSE_JITTER_DEG, generator=None):
    """±5° rotation of the already-built scaffold (positions and frames)."""
    ref = _batch_tensor(batch, "x")
    if ref is None:
        ref = _batch_tensor(batch, "x_true")
    if ref is None:
        return batch, None
    R = _axis_angle_rotation(max_deg, device=ref.device, dtype=ref.dtype, generator=generator)
    rt = R.T
    for key in _POS_KEYS:
        pts = _batch_tensor(batch, key)
        if pts is None or pts.size(-1) < 3:
            continue
        out = pts.clone()
        out[..., :3] = pts[..., :3].to(dtype=R.dtype) @ rt
        setattr(batch, key, out)
    cl = _batch_tensor(batch, "cl_dense")
    if cl is not None and cl.size(-1) >= 3:
        out = cl.clone()
        out[..., :3] = cl[..., :3].to(dtype=R.dtype) @ rt
        batch.cl_dense = out
    for key in _VEC_KEYS:
        vec = _batch_tensor(batch, key)
        if vec is None or vec.size(-1) < 3:
            continue
        out = vec.clone()
        out[..., :3] = vec[..., :3].to(dtype=R.dtype) @ rt
        setattr(batch, key, out)
    pose_R = getattr(batch, "pose_R", None)
    if torch.is_tensor(pose_R) and pose_R.numel() >= 9:
        R_cpu = R.detach().to(device=pose_R.device, dtype=pose_R.dtype)
        if pose_R.dim() == 3 and pose_R.size(-2) == 3 and pose_R.size(-1) == 3:
            batch.pose_R = R_cpu.unsqueeze(0) @ pose_R
        elif pose_R.dim() == 2 and tuple(pose_R.shape[-2:]) == (3, 3):
            if pose_R.shape == (3, 3):
                batch.pose_R = R_cpu @ pose_R
            elif pose_R.size(0) % 3 == 0:
                bsz = pose_R.size(0) // 3
                batch.pose_R = R_cpu.unsqueeze(0) @ pose_R.view(bsz, 3, 3)
        elif pose_R.dim() == 2 and pose_R.size(-1) == 3 and pose_R.size(0) % 3 == 0:
            bsz = pose_R.size(0) // 3
            batch.pose_R = R_cpu.unsqueeze(0) @ pose_R.view(bsz, 3, 3)
    return batch, R


def apply_train_augmentations(batch, generator=None):
    """§8 train-time augs: L/R mirror (per graph) and ±5° pose jitter.

    No θ-phase.  It rotated only the decoder's θ labels while the encoder saw
    the unrotated points and the scaffold kept its geometry, so the angular
    position the decoder queries the latent with no longer matched where the
    aneurysm is -- the latent could not say *where* around the vessel anything
    sits, and the decoder fell back to finding the sac from template density
    (z = 0 decoded the same mesh as μ on the failed run).
    """
    apply_mirror(batch, p=0.5, generator=generator)
    apply_pose_jitter(batch, max_deg=POSE_JITTER_DEG, generator=generator)
    resample_x_true(batch, generator=generator)
    return batch


def train_epoch(
    model,
    dataloader,
    optimizer,
    weights,
    device,
    accum_steps=1,
    grad_clip=GRAD_CLIP,
    vram_probe=False,
    ema=None,
    scheduler=None,
    global_step=0,
    augment=True,
    geco_beta=None,
    epoch=1,
    kl_warmup_epochs=KL_WARMUP_EPOCHS,
):
    model.train()
    totals = _zero_tensor_meters()
    total_samples = 0
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(dataloader)
    use_cuda = _device_type(device) == "cuda"
    logged_first_batch = False
    logged_first_step = False
    window_kl_sum = 0.0
    window_kl_n = 0
    use_geco = geco_beta is not None and geco_is_available()
    if vram_probe and use_cuda:
        torch.cuda.reset_peak_memory_stats(_cuda_index(device))

    show_bar = is_main_process()
    for step, batch in enumerate(tqdm(dataloader, desc="Training", disable=not show_bar)):
        batch = _keep_meta_on_cpu(batch.to(device))
        if augment:
            apply_train_augmentations(batch)
        batch_size = int(batch.num_graphs)
        total_samples += batch_size
        window_len = _accum_window_len(step, n_batches, accum_steps)

        out = _forward_model(model, batch, sample=True)
        terms = losses_from_output(out, batch, kl_beta=geco_beta if use_geco else None)
        loss = _weighted_total(terms, weights) / window_len

        is_update = (step + 1) % accum_steps == 0 or (step + 1) == n_batches
        sync_ctx = (
            model.no_sync()
            if (hasattr(model, "no_sync") and not is_update)
            else contextlib.nullcontext()
        )
        with sync_ctx:
            loss.backward()
        if vram_probe and use_cuda and not logged_first_batch:
            _print_vram("after first batch (forward+backward, no optimizer.step yet)", device, unwrap_model(model))
            logged_first_batch = True

        kl_raw = _tensor_scalar(terms.get("kl_mean_raw"))
        if kl_raw is not None:
            window_kl_sum += float(kl_raw.item()) * batch_size
            window_kl_n += batch_size

        if is_update:
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(unwrap_model(model))
            if scheduler is not None:
                scheduler.step()
            if use_geco:
                window_kl_sum, window_kl_n = all_reduce_sum_pair(
                    window_kl_sum, window_kl_n, device=device if use_cuda else None
                )
            if use_geco and window_kl_n > 0:
                geco_beta = step_geco_beta(
                    geco_beta,
                    window_kl_sum / float(window_kl_n),
                    epoch=epoch,
                    warmup_epochs=kl_warmup_epochs,
                )
            window_kl_sum = 0.0
            window_kl_n = 0
            global_step += 1
            if vram_probe and use_cuda and not logged_first_step:
                _print_vram("after first optimizer.step (AdamW moments allocated)", device, unwrap_model(model))
                logged_first_step = True

        if vram_probe and os.environ.get("ANEUX_VRAM_PROBE_ONLY", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            if not is_update:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(unwrap_model(model))
                if use_cuda and not logged_first_step:
                    _print_vram(
                        "after first optimizer.step (AdamW moments allocated)",
                        device,
                        unwrap_model(model),
                    )
                    logged_first_step = True
            tqdm.write("ANEUX_VRAM_PROBE_ONLY: stopping after the first batch")
            break

        scale = float(window_len * batch_size)
        _add_meter(totals, "loss", loss, scale)
        for key in ("recon", "kl", "disp", "lap", "norm", "rad", "kl_mean_raw", "rate_gap", "fold", "conf", "fold_tpl", "stretch"):
            if key in terms:
                val = terms[key]
                if not torch.is_tensor(val):
                    val = _tensor_scalar(val)
                if val is not None:
                    _add_meter(totals, key, val, float(batch_size))
        del out, terms, loss, batch

    if vram_probe and use_cuda:
        _print_vram("end of epoch 1 (peak over all train batches)", device, unwrap_model(model))

    metrics = _flush_meters(totals, total_samples)
    if geco_beta is not None:
        metrics["geco_beta"] = float(geco_beta)
        metrics["beta"] = float(geco_beta)
    return metrics, global_step, geco_beta


def evaluate_epoch(model, dataloader, weights, device, sample=False, kl_beta=None):
    """Validation recon. `sample=False` is the μ path (σ=0, used for best.pt)."""
    model.eval()
    totals = _zero_tensor_meters()
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc="Validation σ" if sample else "Validation μ",
            disable=not is_main_process(),
        ):
            batch = _keep_meta_on_cpu(batch.to(device))
            batch_size = int(batch.num_graphs)
            total_samples += batch_size
            out = _forward_model(model, batch, sample=bool(sample))
            terms = losses_from_output(out, batch, kl_beta=kl_beta)
            loss = _weighted_total(terms, weights)
            _add_meter(totals, "loss", loss, float(batch_size))
            for key in ("recon", "kl", "disp", "lap", "norm", "rad", "kl_mean_raw", "rate_gap", "fold", "conf", "fold_tpl", "stretch"):
                if key in terms:
                    val = terms[key]
                    if not torch.is_tensor(val):
                        val = _tensor_scalar(val)
                    if val is not None:
                        _add_meter(totals, key, val, float(batch_size))
            del out, terms, loss, batch

    return _flush_meters(totals, total_samples)


def _format_metrics(metrics, weights, tag, epoch, epochs):
    rad = metrics.get("rad", 0.0)
    w_rad = rad * float(weights.get("rad", 0.0))
    return (
        f"Epoch {epoch:03d}/{epochs:03d} [{tag}] | "
        f"Total: {metrics['loss']:.4f} | "
        f"Recon: {metrics['recon']:.4f} | "
        f"Rad: {rad:.4f} | "
        f"KL: {metrics['kl']:.4f} | "
        f"Disp: {metrics['disp']:.4f} | "
        f"Lap: {metrics['lap']:.4f} | "
        f"Norm: {metrics['norm']:.4f} | "
        f"Fold: {metrics.get('fold', 0.0):.4f} | "
        f"Conf: {metrics.get('conf', 0.0):.4f} | "
        f"FoldTpl: {metrics.get('fold_tpl', 0.0):.4f} | "
        f"w_recon: {metrics['recon'] * weights['recon']:.4f} | "
        f"w_rad: {w_rad:.4f} | "
        f"w_kl: {metrics['kl'] * weights['kl']:.4f} | "
        f"w_disp: {metrics['disp'] * weights['disp']:.4f} | "
        f"w_lap: {metrics['lap'] * weights['lap']:.4f} | "
        f"w_norm: {metrics['norm'] * weights['norm']:.4f}"
    )


def _scalarize(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return None
    if isinstance(value, (float, int, np.floating, np.integer, bool)):
        return float(value)
    return value if isinstance(value, str) else None


def _flatten_metrics(data, prefix=""):
    out = {}
    if data is None:
        return out
    if not isinstance(data, dict):
        val = _scalarize(data)
        if val is not None:
            out[prefix or "value"] = val
        return out
    for key, value in data.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}_{key}"
        if isinstance(value, dict):
            out.update(_flatten_metrics(value, name))
        else:
            val = _scalarize(value)
            if val is not None:
                out[name] = val
    return out


class CsvEpochLogger:
    """Print + CSV under the experiment output directory (TensorBoard optional)."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._rows = []
        if csv_path:
            parent = os.path.dirname(csv_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

    def log(self, row, epoch=None, tag=""):
        flat = dict(row)
        if epoch is not None:
            flat.setdefault("epoch", epoch)
        line = " ".join(
            f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in list(flat.items())[:24]
        )
        tqdm.write(f"[metrics{(' ' + tag) if tag else ''}] {line}")
        self._rows.append(flat)
        if not self.csv_path:
            return
        fields = []
        for rec in self._rows:
            for key in rec:
                if key not in fields:
                    fields.append(key)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for rec in self._rows:
                writer.writerow({k: rec.get(k, "") for k in fields})


def _try_tb_writer(log_dir):
    if not log_dir:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=log_dir)
    except Exception:
        return None


def _call_maybe(fn, **kwargs):
    if fn is None:
        return None
    try:
        params = inspect.signature(fn).parameters
        var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if var_kw:
            return fn(**kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in params}
        return fn(**filtered)
    except Exception as exc:
        tqdm.write(f"Warning: {fn.__name__} failed: {exc}")
        return None


def run_post_training_standardisation(model, dataloader, device, ckpt_path=None, ckpt=None):
    """Item 20 call site: prune inactive dims, store μ mean/std on the checkpoint."""
    fn = _standardise_and_prune_latent
    compute_fn = _compute_latent_standardisation
    apply_fn = _apply_standardisation_to_checkpoint
    if fn is None and compute_fn is None:
        raise RuntimeError(
            "latent_standardise.py is not available yet. Wait for the item-20 "
            "module (standardise_and_prune_latent / compute_latent_standardisation) "
            "before running post-training standardisation."
        )
    stats = None
    if fn is not None:
        stats = _call_maybe(fn, model=model, dataloader=dataloader, device=device)
        if stats is None:
            try:
                stats = fn(model, dataloader, device)
            except TypeError:
                stats = None
    if stats is None and compute_fn is not None:
        stats = _call_maybe(
            compute_fn,
            encoder_or_model=model,
            model=model,
            dataloader=dataloader,
            device=device,
        )
    if stats is None:
        raise RuntimeError(
            "standardise_and_prune_latent / compute_latent_standardisation returned "
            "no statistics. Check latent_standardise.py."
        )
    if ckpt_path:
        if ckpt is None:
            if os.path.isfile(ckpt_path):
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                except TypeError:
                    ckpt = torch.load(ckpt_path, map_location="cpu")
            else:
                ckpt = {}
        if not isinstance(ckpt, dict):
            ckpt = {"model": ckpt}
        if apply_fn is not None:
            applied = _call_maybe(apply_fn, ckpt=ckpt, stats=stats)
            if isinstance(applied, dict):
                ckpt = applied
        ckpt["latent_standardisation"] = stats
        torch.save(ckpt, ckpt_path)
        tqdm.write(f"Stored latent standardisation stats on {ckpt_path}")
    return stats


def _checkpoint_root(ckpt_dir):
    if not ckpt_dir:
        return None
    nested = os.path.join(ckpt_dir, "checkpoints")
    if os.path.isdir(nested) or os.path.isdir(os.path.join(ckpt_dir, "plots")):
        os.makedirs(nested, exist_ok=True)
        return nested
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def _loader_mp_context(explicit):
    if explicit is not None:
        return explicit
    if os.name == "nt":
        return "spawn"
    if torch.cuda.is_available():
        return "spawn"
    return None


def train_model(
    model,
    train_dataset,
    val_dataset,
    epochs=100,
    batch_size=4,
    lr=LEARNING_RATE,
    weights=None,
    device="cuda",
    accum_steps=1,
    val_every=5,
    num_workers=0,
    ckpt_dir=None,
    grad_clip=GRAD_CLIP,
    weight_decay=WEIGHT_DECAY,
    kl_max=LAMBDA_KL,
    kl_warmup_epochs=KL_WARMUP_EPOCHS,
    ema_decay=None,
    resume="auto",
    lr_warmup_steps=LR_WARMUP_STEPS,
    augment=True,
    pin_memory=False,
    prefetch_factor=2,
    persistent_workers=None,
    multiprocessing_context=None,
    topk_train=3,
    topk_val=3,
    write_plots=True,
    run_meta=None,
    find_unused_parameters=False,
):
    """Train loop.

    Resume: pass ``resume=True`` / ``"auto"`` (default) to load ``last.pt``
    when it exists, ``resume=False`` to start fresh, or ``resume=path/to/last.pt``.
    Restores model, EMA, optimiser, scheduler, epoch, and RNG.

    When torchrun has initialized a process group, each rank trains a shard
    (DistributedSampler) and gradients sync with NCCL DDP.
    """
    if weights is None:
        weights = dict(DEFAULT_LOSS_WEIGHTS)

    configure_stage2_precision()
    device = resolve_train_device(device)
    ema_decay = _ema_decay_default(ema_decay)
    main = is_main_process()
    n_gpu = world_size() if distributed_active() else 1
    global_batch = int(batch_size) * int(max(1, accum_steps)) * int(n_gpu)

    def _log(*args, **kwargs):
        if main:
            print(*args, **kwargs)

    _log(
        f"Stage-2 precision: matmul={torch.get_float32_matmul_precision()} "
        f"tf32_matmul={torch.backends.cuda.matmul.allow_tf32} "
        f"tf32_cudnn={torch.backends.cudnn.allow_tf32}"
    )
    env_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    _log(
        f"Train device: {device}  CUDA_VISIBLE_DEVICES={env_cvd!r}  "
        f"world_size={n_gpu}  ema_decay={ema_decay}  "
        f"lr_warmup_steps={int(lr_warmup_steps)}"
    )

    use_cuda = _device_type(device) == "cuda"
    mp_ctx = _loader_mp_context(multiprocessing_context) if num_workers else None
    persist = persistent_workers if persistent_workers is not None else (num_workers > 0)
    loader_kwargs = dict(
        follow_batch=FOLLOW_BATCH,
        pin_memory=bool(pin_memory) and use_cuda,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(max(1, prefetch_factor))
        loader_kwargs["persistent_workers"] = bool(persist)
        loader_kwargs["worker_init_fn"] = _worker_init
        if mp_ctx:
            loader_kwargs["multiprocessing_context"] = mp_ctx

    train_sampler = None
    val_sampler = None
    if distributed_active() and n_gpu > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        follow_batch=FOLLOW_BATCH,
        pin_memory=bool(pin_memory) and use_cuda,
    )
    latent_loader = val_loader
    if main and val_sampler is not None:
        latent_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            follow_batch=FOLLOW_BATCH,
        )

    model = model.to(device)
    if use_cuda:
        torch.cuda.synchronize()
        from ops import assert_optimized_cuda_kernels, fps_indices

        fps_indices(torch.randn(64, 3, device=device), 8)
        assert_optimized_cuda_kernels(device)
        torch.cuda.synchronize()
    optimizer = AdamW(adamw_param_groups(model, weight_decay), lr=lr)
    n_batches = max(1, len(train_loader))
    steps_per_epoch = max(1, math.ceil(n_batches / float(max(1, accum_steps))))
    total_opt_steps = max(1, int(epochs) * steps_per_epoch)
    eta_min = max(lr * 1e-2, 1e-7)
    scheduler = build_warmup_cosine_scheduler(
        optimizer, lr_warmup_steps, total_opt_steps, eta_min
    )
    ema = ModelEMA(model, decay=ema_decay) if ema_decay and ema_decay > 0.0 else None

    ckpt_root = _checkpoint_root(ckpt_dir)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(os.path.join(ckpt_dir, "data"), exist_ok=True)
        if main and write_plots:
            write_run_readme(ckpt_dir)
            dump_json(os.path.join(ckpt_dir, "data", "hardware.json"), hardware_snapshot())
            if run_meta is not None:
                dump_json(os.path.join(ckpt_dir, "data", "run_config.json"), run_meta)

    start_epoch = 1
    global_step = 0
    use_geco = geco_is_available()
    geco_beta = init_geco_beta() if use_geco else None
    if use_geco:
        _log(
            f"GECO dual: β_init={geco_beta:.6g}  "
            f"(KL weight=1; {kl_warmup_epochs}-epoch ramp of β_max)"
        )
    else:
        _log("GECO unavailable; using annealed LAMBDA_KL")
    resume_path = resolve_resume_path(resume, ckpt_dir)
    if resume_path and os.environ.get("ANEUX_VRAM_PROBE_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        _log("ANEUX_VRAM_PROBE_ONLY: ignoring resume checkpoint")
        resume_path = None
    if resume_path:
        _log(f"Resuming from {resume_path}")
        ckpt = load_training_checkpoint(
            resume_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        if use_geco and ckpt.get("geco_beta") is not None:
            geco_beta = float(ckpt["geco_beta"])
        _log(
            f"  restored epoch {ckpt.get('epoch')}  next={start_epoch}  "
            f"step={global_step}"
            + (f"  geco_beta={geco_beta:.6g}" if geco_beta is not None else "")
        )

    model = wrap_ddp(model, device, find_unused_parameters=find_unused_parameters)
    raw_model = unwrap_model(model)

    _log(
        f"Per-GPU batch {batch_size} x accum {accum_steps} x {n_gpu} GPU(s) "
        f"= global {global_batch}"
    )
    _log(f"Train dataset size: {len(train_dataset)}")
    _log(f"Val dataset size: {len(val_dataset)}")
    _log(
        f"Schedule: {steps_per_epoch} opt steps/epoch, {total_opt_steps} total, "
        f"warmup {int(lr_warmup_steps)} then cosine"
    )

    csv_path = None
    if ckpt_dir and main:
        csv_path = os.path.join(ckpt_dir, "data", "epoch_metrics.csv")
    logger = CsvEpochLogger(csv_path)
    tb = _try_tb_writer(os.path.join(ckpt_dir, "tb") if ckpt_dir and main else None)

    monitor = None
    probe_only = os.environ.get("ANEUX_VRAM_PROBE_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if main and ckpt_dir and not probe_only:
        interval = float(os.environ.get("ANEUX_MONITOR_SEC", "15"))
        monitor = ResourceMonitor(
            os.path.join(ckpt_dir, "data"),
            interval=interval,
            disk_path=ckpt_dir,
        )
        monitor.start()

    top_train = (
        TopKCheckpoints(ckpt_root, "best_train", k=int(topk_train))
        if ckpt_root and main and topk_train
        else None
    )
    top_val = (
        TopKCheckpoints(ckpt_root, "best_val", k=int(topk_val))
        if ckpt_root and main and topk_val
        else None
    )

    history = []
    stop = {"flag": False}

    def _on_signal(signum, _frame):
        stop["flag"] = True
        tqdm.write(f"signal {signum}: finishing this epoch then saving last.pt")

    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        epoch_weights = dict(weights)
        if use_geco:
            epoch_weights["kl"] = 1.0
        else:
            epoch_weights["kl"] = kl_anneal_weight(
                epoch, max_weight=kl_max, warmup_epochs=kl_warmup_epochs
            )
        t0 = time.perf_counter()
        metrics, global_step, geco_beta = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch_weights,
            device,
            accum_steps,
            grad_clip=grad_clip,
            vram_probe=(use_cuda and epoch == start_epoch and main),
            ema=ema,
            scheduler=scheduler,
            global_step=global_step,
            augment=augment,
            geco_beta=geco_beta,
            epoch=epoch,
            kl_warmup_epochs=kl_warmup_epochs,
        )
        epoch_s = time.perf_counter() - t0
        metrics = reduce_mean_dict(metrics, device=device if use_cuda else None)
        n_seen = max(1, len(train_dataset))
        metrics["epoch"] = int(epoch)
        metrics["epoch_seconds"] = float(epoch_s)
        metrics["samples_per_sec"] = float(n_seen) / max(epoch_s, 1e-6)
        if use_cuda:
            idx = _cuda_index(device)
            metrics["peak_vram_gib"] = float(torch.cuda.max_memory_allocated(idx) / 1024**3)
        last_lr = scheduler.get_last_lr()[0] if scheduler is not None else lr
        ema_d = ema_warmup_decay(ema.decay, max(0, ema.n_updates - 1)) if ema is not None else 0.0
        extra = f" | kl_lambda: {epoch_weights['kl']:.6f} | lr: {last_lr:.2e} | ema_d: {ema_d:.4f}"
        if geco_beta is not None:
            extra += f" | β: {float(geco_beta):.6g}"
            if metrics.get("rate_gap") is not None:
                extra += f" | rate_gap: {metrics['rate_gap']:.4f}"
        extra += f" | {metrics['samples_per_sec']:.2f} samples/s"
        _log(_format_metrics(metrics, epoch_weights, "TRAIN", epoch, epochs) + extra)
        if os.environ.get("ANEUX_VRAM_PROBE_ONLY", "").strip().lower() in ("1", "true", "yes"):
            history.append(metrics)
            _log("ANEUX_VRAM_PROBE_ONLY: skipping val, plots, and remaining epochs")
            break
        metrics["lr"] = float(last_lr)
        metrics["kl_lambda"] = float(epoch_weights["kl"])
        metrics["global_step"] = int(global_step)
        metrics["ema_decay_used"] = float(ema_d)
        if geco_beta is not None:
            metrics["geco_beta"] = float(geco_beta)
            metrics["beta"] = float(geco_beta)
            bmax = _geco_beta_max(epoch, warmup_epochs=kl_warmup_epochs)
            if bmax is not None:
                metrics["geco_beta_max"] = bmax

        if main:
            latent_row = _call_maybe(
                _compute_latent_epoch_metrics,
                model=raw_model,
                dataloader=latent_loader if len(val_dataset) else train_loader,
                device=device,
                epoch=epoch,
                beta=geco_beta if geco_beta is not None else epoch_weights["kl"],
                r_star=None,
                weights=epoch_weights,
                kl_lambda=epoch_weights["kl"],
                kl_mean_raw=metrics.get("kl_mean_raw"),
                rate_gap=metrics.get("rate_gap"),
            )
            if isinstance(latent_row, dict):
                metrics.update(_flatten_metrics(latent_row, prefix="latent"))

        do_val = epoch % val_every == 0 or epoch == epochs or stop["flag"]
        if do_val:
            if ema is not None:
                live_mu = evaluate_epoch(
                    model, val_loader, epoch_weights, device, sample=False, kl_beta=geco_beta
                )
                live_mu = reduce_mean_dict(live_mu, device=device if use_cuda else None)
                _log(_format_metrics(live_mu, epoch_weights, "VAL live μ", epoch, epochs))
                for key, value in live_mu.items():
                    metrics[f"val_live_{key}"] = value
                ema.store(raw_model)
                ema.copy_to(raw_model)
            val_metrics = evaluate_epoch(
                model, val_loader, epoch_weights, device, sample=False, kl_beta=geco_beta
            )
            val_sampled = evaluate_epoch(
                model, val_loader, epoch_weights, device, sample=True, kl_beta=geco_beta
            )
            val_metrics = reduce_mean_dict(val_metrics, device=device if use_cuda else None)
            val_sampled = reduce_mean_dict(val_sampled, device=device if use_cuda else None)
            if ema is not None:
                ema.restore(raw_model)
            _log(_format_metrics(val_metrics, epoch_weights, "VAL μ  ", epoch, epochs))
            _log(_format_metrics(val_sampled, epoch_weights, "VAL σ  ", epoch, epochs))
            gap = float(val_sampled["recon"] - val_metrics["recon"])
            _log(f"  recon(σ) − recon(μ) = {gap:.4f}")
            for key, value in val_metrics.items():
                metrics[f"val_{key}"] = value
            for key, value in val_sampled.items():
                metrics[f"val_sample_{key}"] = value
            metrics["val_recon_sample_gap"] = gap

            if top_val is not None:
                payload = {
                    "epoch": int(epoch),
                    "model": (ema.shadow if ema is not None else raw_model.state_dict()),
                    "metrics": {k: metrics[k] for k in metrics if k.startswith("val_")},
                    "score": selection_score(val_metrics, epoch_weights),
                }
                if top_val.consider(payload["score"], payload, epoch, extra={"split": "val"}):
                    _log(
                        f"  saved val top-{topk_val} (score {payload['score']:.4f}) [EMA μ path]"
                    )

        if top_train is not None:
            payload = {
                "epoch": int(epoch),
                "model": raw_model.state_dict(),
                "metrics": {k: metrics[k] for k in ("loss", "recon", "rad", "kl") if k in metrics},
                "score": selection_score(metrics, epoch_weights),
            }
            if top_train.consider(payload["score"], payload, epoch, extra={"split": "train"}):
                _log(f"  saved train top-{topk_train} (score {payload['score']:.4f})")

        if main:
            logger.log(metrics, epoch=epoch, tag="epoch")
            if tb is not None:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        tb.add_scalar(key, float(value), epoch)
            if ckpt_root:
                save_training_checkpoint(
                    os.path.join(ckpt_root, "last.pt"),
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=metrics,
                    global_step=global_step,
                    geco_beta=geco_beta,
                )
            if ckpt_dir:
                write_history_tables(history + [metrics], ckpt_dir)
                if write_plots:
                    plot_training_history(history + [metrics], ckpt_dir)
                if monitor is not None:
                    monitor.write_csv()
                    if write_plots:
                        monitor.plot()

        history.append(metrics)
        barrier()
        if stop["flag"]:
            _log("stop flag set; leaving the training loop")
            break

    if ckpt_dir and main and os.environ.get("ANEUX_VRAM_PROBE_ONLY", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        best_path = os.path.join(ckpt_root or ckpt_dir, "best_val_1.pt")
        legacy_best = os.path.join(ckpt_dir, "best.pt")
        load_path = best_path if os.path.isfile(best_path) else legacy_best
        if os.path.isfile(load_path):
            try:
                best_sd = torch.load(load_path, map_location=device, weights_only=False)
            except TypeError:
                best_sd = torch.load(load_path, map_location=device)
            if isinstance(best_sd, dict) and "model" in best_sd:
                raw_model.load_state_dict(best_sd["model"], strict=True)
            elif isinstance(best_sd, dict) and all(torch.is_tensor(v) for v in best_sd.values()):
                raw_model.load_state_dict(best_sd)
            print(f"Restored best validation weights from {load_path}")
        try:
            last_path = os.path.join(ckpt_root or ckpt_dir, "last.pt")
            stats = run_post_training_standardisation(
                raw_model,
                train_loader,
                device,
                ckpt_path=last_path,
            )
            if os.path.isfile(best_path) and stats is not None:
                try:
                    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
                except TypeError:
                    best_ckpt = torch.load(best_path, map_location="cpu")
                if not isinstance(best_ckpt, dict):
                    best_ckpt = {"model": best_ckpt}
                applied = _call_maybe(_apply_standardisation_to_checkpoint, ckpt=best_ckpt, stats=stats)
                if isinstance(applied, dict):
                    best_ckpt = applied
                best_ckpt["latent_standardisation"] = stats
                torch.save(best_ckpt, best_path)
                print(f"Stored latent standardisation stats on {best_path}")
        except RuntimeError as exc:
            print(f"Post-training latent standardisation skipped: {exc}")
        slurm = capture_slurm_job_stats()
        if slurm:
            dump_json(os.path.join(ckpt_dir, "data", "slurm_jobstats.json"), slurm)
        write_history_tables(history, ckpt_dir)
        if write_plots:
            plot_training_history(history, ckpt_dir)
        write_run_summary(ckpt_dir, history)

    if monitor is not None:
        monitor.stop()
    if tb is not None:
        tb.close()

    return raw_model, history

