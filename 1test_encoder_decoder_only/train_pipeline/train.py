import os
from contextlib import nullcontext

import torch
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from losses import compute_losses


def _device_type(device):
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":")[0]


def _autocast(device):
    if _device_type(device) == "cuda" and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _face_from_batch(batch):
    face = getattr(batch, "face", None)
    if face is not None and face.numel() > 0:
        return face
    faces = getattr(batch, "faces", None)
    if faces is not None and faces.numel() > 0:
        return faces
    return None


def losses_from_batch(x_pred, mu, logvar, batch):
    return compute_losses(
        x_pred,
        batch.x_true,
        mu,
        logvar,
        batch.x,
        batch.edge_index,
        batch.x_true_batch,
        batch.num_graphs,
        face=_face_from_batch(batch),
        batch_tube=batch.batch,
    )


def _accum_window_len(step, n_batches, accum_steps):
    window_start = (step // accum_steps) * accum_steps
    return min(accum_steps, n_batches - window_start)


def train_epoch(model, dataloader, optimizer, weights, device, accum_steps=1, grad_clip=1.0):
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_geom = 0.0
    total_samples = 0

    optimizer.zero_grad()
    n_batches = len(dataloader)

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        batch = batch.to(device)
        batch_size = batch.num_graphs
        total_samples += batch_size
        window_len = _accum_window_len(step, n_batches, accum_steps)

        with _autocast(device):
            x_pred, mu, logvar = model(batch)
            loss_recon, loss_kl, loss_geom = losses_from_batch(x_pred, mu, logvar, batch)
            loss = (
                weights["recon"] * loss_recon
                + weights["kl"] * loss_kl
                + weights["geom"] * loss_geom
            ) / window_len

        loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == n_batches:
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * window_len * batch_size
        total_recon += loss_recon.item() * batch_size
        total_kl += loss_kl.item() * batch_size
        total_geom += loss_geom.item() * batch_size

    if total_samples == 0:
        return {"loss": 0.0, "recon": 0.0, "kl": 0.0, "geom": 0.0}

    return {
        "loss": total_loss / total_samples,
        "recon": total_recon / total_samples,
        "kl": total_kl / total_samples,
        "geom": total_geom / total_samples,
    }


def evaluate_epoch(model, dataloader, weights, device):
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_geom = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch = batch.to(device)
            batch_size = batch.num_graphs
            total_samples += batch_size

            with _autocast(device):
                x_pred, mu, logvar = model(batch)
                loss_recon, loss_kl, loss_geom = losses_from_batch(x_pred, mu, logvar, batch)
                loss = (
                    weights["recon"] * loss_recon
                    + weights["kl"] * loss_kl
                    + weights["geom"] * loss_geom
                )

            total_loss += loss.item() * batch_size
            total_recon += loss_recon.item() * batch_size
            total_kl += loss_kl.item() * batch_size
            total_geom += loss_geom.item() * batch_size

    if total_samples == 0:
        return {"loss": 0.0, "recon": 0.0, "kl": 0.0, "geom": 0.0}

    return {
        "loss": total_loss / total_samples,
        "recon": total_recon / total_samples,
        "kl": total_kl / total_samples,
        "geom": total_geom / total_samples,
    }


def train_model(
    model,
    train_dataset,
    val_dataset,
    epochs=100,
    batch_size=4,
    lr=1e-4,
    weights=None,
    device="cuda",
    accum_steps=1,
    val_every=5,
    num_workers=0,
    ckpt_dir=None,
    grad_clip=1.0,
):
    if weights is None:
        weights = {"recon": 1.0, "kl": 0.001, "geom": 0.1}

    use_cuda = _device_type(device) == "cuda"
    loader_kwargs = dict(
        follow_batch=["x_true"],
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
    optimizer = AdamW(model.parameters(), lr=lr)

    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Effective batch size: {batch_size} x {accum_steps} = {batch_size * accum_steps}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    history = []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        metrics = train_epoch(
            model, train_loader, optimizer, weights, device, accum_steps, grad_clip=grad_clip
        )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} [TRAIN] | "
            f"Total: {metrics['loss']:.4f} | "
            f"Recon: {metrics['recon']:.4f} | "
            f"KL: {metrics['kl']:.4f} | "
            f"Geom: {metrics['geom']:.4f} | "
            f"weighted recon: {metrics['recon'] * weights['recon']:.4f} | "
            f"weighted kl: {metrics['kl'] * weights['kl']:.4f} | "
            f"weighted geom: {metrics['geom'] * weights['geom']:.4f}"
        )

        if epoch % val_every == 0 or epoch == epochs:
            val_metrics = evaluate_epoch(model, val_loader, weights, device)
            print(
                f"Epoch {epoch:03d}/{epochs:03d} [VAL]   | "
                f"Total: {val_metrics['loss']:.4f} | "
                f"Recon: {val_metrics['recon']:.4f} | "
                f"KL: {val_metrics['kl']:.4f} | "
                f"Geom: {val_metrics['geom']:.4f} | "
                f"weighted recon: {val_metrics['recon'] * weights['recon']:.4f} | "
                f"weighted kl: {val_metrics['kl'] * weights['kl']:.4f} | "
                f"weighted geom: {val_metrics['geom'] * weights['geom']:.4f}"
            )
            metrics["val_loss"] = val_metrics["loss"]
            metrics["val_recon"] = val_metrics["recon"]
            metrics["val_kl"] = val_metrics["kl"]
            metrics["val_geom"] = val_metrics["geom"]

            if ckpt_dir and val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))
                print(f"  saved best checkpoint (val {best_val:.4f})")

        if ckpt_dir:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
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
