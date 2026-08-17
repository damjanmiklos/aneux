"""Does constructing the full GraphVAE on CUDA break pytorch3d FPS?"""
import os
import sys

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "1test_encoder_decoder_only", "train_pipeline"))

import torch
from pytorch3d.ops import sample_farthest_points
import ops
from model import GraphVAE


def probe(tag):
    ops._PYTORCH3D_CUDA_FPS = None
    try:
        x = torch.zeros(1, 4, 3, device="cuda")
        sample_farthest_points(x, K=2, random_start_point=False)
        torch.cuda.synchronize()
        print(tag, "p3d OK", flush=True)
    except Exception as exc:
        print(tag, "p3d FAIL", type(exc).__name__, exc, flush=True)


print("step1", flush=True)
probe("before model")
print("building GraphVAE", flush=True)
model = GraphVAE()
print("created CPU", flush=True)
probe("after CPU ctor")
print("moving to cuda", flush=True)
model = model.cuda()
torch.cuda.synchronize()
print("on cuda", flush=True)
probe("after .cuda()")
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
probe("after AdamW")
print("stem", flush=True)
x = torch.randn(4096, 3, device="cuda")
h = model.encoder.stem(x)
torch.cuda.synchronize()
print("stem ok", h.shape, flush=True)
probe("after stem")
