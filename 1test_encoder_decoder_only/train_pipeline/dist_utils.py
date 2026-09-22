"""Single-node (and multi-node) DDP helpers for Stage-2 training.

Komondor launches this with torchrun + NCCL (docs.hpc.dkf.hu/AI/pytorch.html).
When WORLD_SIZE is unset the helpers are no-ops so aneuxai.py stays single-GPU.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def env_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def env_rank():
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))


def env_local_rank():
    return int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))


def distributed_active():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    if not distributed_active():
        return True
    return dist.get_rank() == 0


def world_size():
    if distributed_active():
        return dist.get_world_size()
    return 1


def rank():
    if distributed_active():
        return dist.get_rank()
    return 0


def init_distributed(backend=None):
    """Initialize the default process group when torchrun/Slurm set WORLD_SIZE>1."""
    if distributed_active():
        return True
    if env_world_size() <= 1:
        return False
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    local = env_local_rank()
    idx = None
    if torch.cuda.is_available():
        nvis = torch.cuda.device_count()
        idx = 0 if nvis <= 1 else max(0, min(local, nvis - 1))
        torch.cuda.set_device(idx)
    init_kwargs = {}
    if idx is not None and backend == "nccl":
        # PyTorch 2.11 warns on barrier() unless the group knows its device.
        init_kwargs["device_id"] = torch.device("cuda", idx)
    try:
        dist.init_process_group(backend=backend, init_method="env://", **init_kwargs)
    except TypeError:
        dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        nvis = torch.cuda.device_count()
        idx = 0 if nvis <= 1 else max(0, min(env_local_rank(), nvis - 1))
        torch.cuda.set_device(idx)
    return True


def barrier():
    if distributed_active():
        dist.barrier()


def broadcast_object(obj, src=0):
    """Broadcast a pickleable object from ``src``. No-op when not distributed."""
    if not distributed_active():
        return obj
    packed = [obj]
    dist.broadcast_object_list(packed, src=int(src))
    return packed[0]


def destroy_distributed():
    """Tear down the process group without a barrier.

    A barrier here is harmful: if one rank already failed, the others are
    waiting on an earlier collective and this rendezvous unblocks them into
    a dead NCCL group (SIGABRT in the watchdog).
    """
    if not distributed_active():
        return
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def wrap_ddp(model, device, find_unused_parameters=False):
    if not distributed_active() or world_size() < 2:
        return model
    if device.type == "cuda":
        model = model.to(device)
        idx = torch.cuda.current_device()
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[idx],
            output_device=idx,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )
    return torch.nn.parallel.DistributedDataParallel(
        model,
        find_unused_parameters=find_unused_parameters,
    )


def reduce_mean_dict(metrics, device=None):
    """Average scalar metrics across ranks. Non-finite values become 0 before the mean."""
    if not distributed_active() or world_size() < 2:
        return metrics
    keys = sorted(metrics)
    if not keys:
        return metrics
    vec = []
    for key in keys:
        val = metrics[key]
        try:
            num = float(val)
        except (TypeError, ValueError):
            vec.append(0.0)
            continue
        if num != num or num in (float("inf"), float("-inf")):
            num = 0.0
        vec.append(num)
    tensor = torch.tensor(vec, device=device or "cpu", dtype=torch.float64)
    if tensor.device.type == "cpu" and dist.get_backend() == "nccl":
        tensor = tensor.cuda()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / float(world_size())
    out = dict(metrics)
    cpu = tensor.detach().cpu().tolist()
    for key, num in zip(keys, cpu):
        try:
            float(metrics[key])
        except (TypeError, ValueError):
            continue
        out[key] = float(num)
    return out


def all_reduce_sum_pair(a, b, device=None):
    """Sum two scalars across ranks. No-op when not distributed."""
    if not distributed_active() or world_size() < 2:
        return float(a), float(b)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor([float(a), float(b)], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    vals = tensor.detach().cpu().tolist()
    return float(vals[0]), float(vals[1])


def print0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)
