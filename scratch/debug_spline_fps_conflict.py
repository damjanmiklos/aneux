"""Check whether SplineConv CUDA init breaks pytorch3d FPS."""
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from pytorch3d.ops import sample_farthest_points
from torch_geometric.nn import fps as pyg_fps
from torch_geometric.nn.conv import SplineConv

x = torch.randn(256, 3, device="cuda")
batch = torch.zeros(256, dtype=torch.long, device="cuda")


def p3d():
    _, loc = sample_farthest_points(x.unsqueeze(0), K=32, random_start_point=False)
    torch.cuda.synchronize()
    return int(loc.max())


print("p3d before spline", p3d(), flush=True)
conv = SplineConv(8, 8, dim=3, kernel_size=5).cuda()
torch.cuda.synchronize()
print("SplineConv on cuda", type(conv), flush=True)
try:
    print("p3d after spline", p3d(), flush=True)
except Exception as exc:
    print("p3d after spline FAILED", type(exc).__name__, exc, flush=True)

try:
    idx = pyg_fps(x, batch, ratio=0.125)
    torch.cuda.synchronize()
    print("pyg fps after spline", idx.shape, int(idx.max()), flush=True)
except Exception as exc:
    print("pyg fps FAILED", type(exc).__name__, exc, flush=True)

# dummy spline forward
h = torch.randn(256, 8, device="cuda")
ei = torch.randint(0, 256, (2, 64), device="cuda")
pseudo = torch.rand(64, 3, device="cuda")
try:
    y = conv(h, ei, pseudo)
    torch.cuda.synchronize()
    print("spline forward", y.shape, flush=True)
except Exception as exc:
    print("spline forward FAILED", type(exc).__name__, exc, flush=True)

try:
    print("p3d after spline forward", p3d(), flush=True)
except Exception as exc:
    print("p3d after spline forward FAILED", type(exc).__name__, exc, flush=True)
