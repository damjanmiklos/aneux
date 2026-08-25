import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from config import (
    DEFAULT_LOSS_WEIGHTS,
    FOLLOW_BATCH,
    GRAD_CLIP,
    KL_WARMUP_EPOCHS,
    LAMBDA_KL,
    LEARNING_RATE,
    WEIGHT_DECAY,
    configure_stage2_precision,
)
from losses import compute_losses


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


def _face_from_batch(batch):
    face = getattr(batch, "face", None)
    if face is not None and face.numel() > 0:
        return face
    faces = getattr(batch, "faces", None)
    if faces is not None and faces.numel() > 0:
        return faces
    return None


def kl_anneal_weight(epoch, max_weight=LAMBDA_KL, warmup_epochs=KL_WARMUP_EPOCHS):
    """Linear KL anneal: 0 at epoch 1, `max_weight` from epoch `warmup_epochs` onward."""
    if warmup_epochs <= 1:
        return float(max_weight)
    t = min(1.0, max(0.0, (epoch - 1) / float(warmup_epochs - 1)))
    return float(max_weight) * t


def weighted_total(terms, weights):
    return (
        weights["recon"] * terms["recon"]
        + weights["kl"] * terms["kl"]
        + weights["disp"] * terms["disp"]
        + weights["lap"] * terms["lap"]
        + weights["norm"] * terms["norm"]
    )


def _weighted_total(terms, weights):
    return weighted_total(terms, weights)


def losses_from_output(out, batch):
    batch_mid = getattr(batch, "pos_mid_batch", None)
    batch_coarse = getattr(batch, "pos_coarse_batch", None)
    cl_batch = getattr(batch, "cl_dense_batch", None)
    return compute_losses(
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
    )


def _accum_window_len(step, n_batches, accum_steps):
    window_start = (step // accum_steps) * accum_steps
    return min(accum_steps, n_batches - window_start)


def _zero_meters():
    return {"loss": 0.0, "recon": 0.0, "kl": 0.0, "disp": 0.0, "lap": 0.0, "norm": 0.0}


def train_epoch(
    model, dataloader, optimizer, weights, device, accum_steps=1, grad_clip=GRAD_CLIP, vram_probe=False
):
    model.train()
    totals = _zero_meters()
    total_samples = 0
    optimizer.zero_grad()
    n_batches = len(dataloader)
    use_cuda = _device_type(device) == "cuda"
    logged_first_batch = False
    logged_first_step = False
    if vram_probe and use_cuda:
        torch.cuda.reset_peak_memory_stats(_cuda_index(device))

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        batch = batch.to(device)
        batch_size = batch.num_graphs
        total_samples += batch_size
        window_len = _accum_window_len(step, n_batches, accum_steps)

        out = model(batch)
        terms = losses_from_output(out, batch)
        loss = _weighted_total(terms, weights) / window_len

        loss.backward()
        if vram_probe and use_cuda and not logged_first_batch:
            _print_vram("after first batch (forward+backward, no optimizer.step yet)", device, model)
            logged_first_batch = True

        if (step + 1) % accum_steps == 0 or (step + 1) == n_batches:
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            if vram_probe and use_cuda and not logged_first_step:
                _print_vram("after first optimizer.step (AdamW moments allocated)", device, model)
                logged_first_step = True

        totals["loss"] += loss.item() * window_len * batch_size
        for key in ("recon", "kl", "disp", "lap", "norm"):
            totals[key] += terms[key].item() * batch_size

    if vram_probe and use_cuda:
        _print_vram("end of epoch 1 (peak over all train batches)", device, model)

    if total_samples == 0:
        return _zero_meters()
    return {k: v / total_samples for k, v in totals.items()}


def evaluate_epoch(model, dataloader, weights, device):
    model.eval()
    totals = _zero_meters()
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch = batch.to(device)
            batch_size = batch.num_graphs
            total_samples += batch_size
            out = model(batch)
            terms = losses_from_output(out, batch)
            loss = _weighted_total(terms, weights)
            totals["loss"] += loss.item() * batch_size
            for key in ("recon", "kl", "disp", "lap", "norm"):
                totals[key] += terms[key].item() * batch_size

    if total_samples == 0:
        return _zero_meters()
    return {k: v / total_samples for k, v in totals.items()}


def _format_metrics(metrics, weights, tag, epoch, epochs):
    return (
        f"Epoch {epoch:03d}/{epochs:03d} [{tag}] | "
        f"Total: {metrics['loss']:.4f} | "
        f"Recon: {metrics['recon']:.4f} | "
        f"KL: {metrics['kl']:.4f} | "
        f"Disp: {metrics['disp']:.4f} | "
        f"Lap: {metrics['lap']:.4f} | "
        f"Norm: {metrics['norm']:.4f} | "
        f"w_recon: {metrics['recon'] * weights['recon']:.4f} | "
        f"w_kl: {metrics['kl'] * weights['kl']:.4f} | "
        f"w_disp: {metrics['disp'] * weights['disp']:.4f} | "
        f"w_lap: {metrics['lap'] * weights['lap']:.4f} | "
        f"w_norm: {metrics['norm'] * weights['norm']:.4f}"
    )


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
):
    if weights is None:
        weights = dict(DEFAULT_LOSS_WEIGHTS)

    configure_stage2_precision()
    print(
        f"Stage-2 precision: matmul={torch.get_float32_matmul_precision()} "
        f"tf32_matmul={torch.backends.cuda.matmul.allow_tf32} "
        f"tf32_cudnn={torch.backends.cudnn.allow_tf32}"
    )

    use_cuda = _device_type(device) == "cuda"
    loader_kwargs = dict(
        follow_batch=FOLLOW_BATCH,
        pin_memory=use_cuda,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        **loader_kwargs,
    )

    model = model.to(device)
    if use_cuda:
        torch.cuda.synchronize()
        from ops import fps_indices

        fps_indices(torch.randn(64, 3, device=device), 8)
        torch.cuda.synchronize()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=max(lr * 1e-2, 1e-7))

    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Effective batch size: {batch_size} x {accum_steps} = {batch_size * accum_steps}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    history = []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        epoch_weights = dict(weights)
        epoch_weights["kl"] = kl_anneal_weight(epoch, max_weight=kl_max, warmup_epochs=kl_warmup_epochs)
        metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch_weights,
            device,
            accum_steps,
            grad_clip=grad_clip,
            vram_probe=(use_cuda and epoch == 1),
        )
        scheduler.step()
        print(_format_metrics(metrics, epoch_weights, "TRAIN", epoch, epochs)
              + f" | kl_lambda: {epoch_weights['kl']:.6f} | lr: {scheduler.get_last_lr()[0]:.2e}")

        if epoch % val_every == 0 or epoch == epochs:
            val_metrics = evaluate_epoch(model, val_loader, epoch_weights, device)
            print(_format_metrics(val_metrics, epoch_weights, "VAL  ", epoch, epochs))
            for key, value in val_metrics.items():
                metrics[f"val_{key}"] = value

            if ckpt_dir and val_metrics["recon"] < best_val:
                best_val = val_metrics["recon"]
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))
                print(f"  saved best checkpoint (val_recon {best_val:.4f})")

        if ckpt_dir:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "metrics": metrics,
                },
                os.path.join(ckpt_dir, "last.pt"),
            )

        history.append(metrics)

    if ckpt_dir:
        best_path = os.path.join(ckpt_dir, "best.pt")
        if os.path.isfile(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
            print(f"Restored best validation weights from {best_path}")

    return model, history
