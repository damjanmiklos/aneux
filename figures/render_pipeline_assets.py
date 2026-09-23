"""Render the mesh / point-cloud panels used by make_pipeline_figure.py.

Everything comes from one cached training graph (the exact tensors the model
sees, canonical pose, mm), so the figure shows real data rather than cartoons:

    gt_surface      S_GT, the uniformly remeshed ground truth
    x_true          the 16 384-point encoder input, coloured by SDF to the template
    fps_1024/256/64 nested farthest-point centres of the four SA stages
    centerline      the centerline tree, one colour per tract
    tokens          latent token positions (2 mm spacing) on the tree
    template        template mesh T (fine level)
    level_coarse/mid/fine  the three decoder scaffolds, coloured by tract

Run in the aneurysmgnn env:
    python figures/render_pipeline_assets.py [--case p462_EwAADxURDAABCwMWEQAcCxAB]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyvista as pv
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PIPELINE = os.path.join(REPO, "1test_encoder_decoder_only", "train_pipeline")
CACHE = os.path.join(REPO, "1test_encoder_decoder_only", "tube_cache")
OUT = os.path.join(HERE, "pipeline_assets")
sys.path.insert(0, PIPELINE)
import dataset  # noqa: E402,F401  (registers AneurysmData for torch.load)

TRACT_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
                "#76B7B2", "#EDC948", "#9C755F", "#FF9DA7", "#BAB0AC"]
GT_GREY = "#d9d6cf"
TEMPLATE_BLUE = "#b7cde3"
FPS_BLUE = "#2f5f8f"
WIN = 1400


def _sdf_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("sdf", ["#2166ac", "#a9a9a9", "#b2182b"])


SDF_CMAP = _sdf_cmap()


def poly(points, faces):
    f = np.asarray(faces).T
    return pv.PolyData(np.asarray(points), np.hstack([np.full((len(f), 1), 3), f]).ravel())


def fps(points, k, seed=0):
    n = len(points)
    if k >= n:
        return np.arange(n)
    idx = np.empty(k, dtype=np.int64)
    idx[0] = seed
    d = np.linalg.norm(points - points[seed], axis=1)
    for i in range(1, k):
        idx[i] = int(d.argmax())
        d = np.minimum(d, np.linalg.norm(points - points[idx[i]], axis=1))
    return idx


def tract_rgb(tract):
    t = np.asarray(tract).clip(min=0) % len(TRACT_COLORS)
    lut = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in TRACT_COLORS], dtype=np.uint8)
    return lut[t]


class Renderer:
    """Same orthographic camera for every asset so the panels line up."""

    def __init__(self, gt: pv.PolyData, view=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
        self.center = np.array(gt.center)
        self.view = np.asarray(view, dtype=float)
        self.up = up
        b = np.array(gt.bounds).reshape(3, 2)
        self.scale = 0.60 * max(b[1, 1] - b[1, 0], b[2, 1] - b[2, 0])

    def plotter(self):
        pl = pv.Plotter(off_screen=True, window_size=(WIN, WIN))
        pl.enable_parallel_projection()
        pl.set_background("white")
        return pl

    def shoot(self, pl, name):
        pl.camera.position = tuple(self.center + 200.0 * self.view)
        pl.camera.focal_point = tuple(self.center)
        pl.camera.up = self.up
        pl.camera.parallel_scale = self.scale
        path = os.path.join(OUT, f"{name}.png")
        pl.screenshot(path, transparent_background=True)
        pl.close()
        return path


def crop_all(paths, ref_path, pad=12):
    """Crop every image with the GT's alpha bounding box so scales stay equal."""
    a = np.asarray(Image.open(ref_path))[..., 3]
    ys, xs = np.nonzero(a > 8)
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
           min(xs.max() + pad, a.shape[1]), min(ys.max() + pad, a.shape[0]))
    for p in paths:
        Image.open(p).crop(box).save(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="p462_EwAADxURDAABCwMWEQAcCxAB")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pv.OFF_SCREEN = True

    path = os.path.join(CACHE, f"{args.case}_v10_L40R6_L250R12_L1000R64_n16384_rad2.0_Z128.pt")
    d = torch.load(path, map_location="cpu", weights_only=False)
    gt = poly(d.gt_points.numpy(), d.gt_faces.numpy())
    r = Renderer(gt)
    ghost = dict(color=GT_GREY, opacity=0.13, smooth_shading=True)
    shots = []

    pl = r.plotter()
    pl.add_mesh(gt, color=GT_GREY, smooth_shading=True, specular=0.25)
    ref = r.shoot(pl, "gt_surface")
    shots.append(ref)

    xt = d.x_true.numpy()
    sdf = d.x_true_template_sdf.numpy().reshape(-1)
    lim = float(np.percentile(np.abs(sdf), 97))
    pl = r.plotter()
    pl.add_points(pv.PolyData(xt), scalars=np.clip(sdf, -lim, lim), cmap=SDF_CMAP,
                  clim=(-lim, lim), point_size=4, render_points_as_spheres=True,
                  show_scalar_bar=False)
    shots.append(r.shoot(pl, "x_true"))

    centres = xt
    for k, size in ((1024, 9), (256, 15), (64, 24)):
        centres = centres[fps(centres, k)]
        pl = r.plotter()
        pl.add_mesh(gt, **ghost)
        pl.add_points(pv.PolyData(centres), color=FPS_BLUE, point_size=size,
                      render_points_as_spheres=True)
        shots.append(r.shoot(pl, f"fps_{k}"))

    cl = d.cl_dense.numpy()[:, :3]
    cl_t = d.cl_tract_id.numpy()
    pl = r.plotter()
    pl.add_mesh(gt, **ghost)
    for t in np.unique(cl_t):
        pts = cl[cl_t == t]
        if len(pts) < 2:
            continue
        line = pv.lines_from_points(pts).tube(radius=0.32)
        pl.add_mesh(line, color=TRACT_COLORS[int(t) % len(TRACT_COLORS)], smooth_shading=True)
    shots.append(r.shoot(pl, "centerline"))

    valid = d.latent_valid.numpy().astype(bool)
    tok = d.latent_pos.numpy()[valid]
    tok_t = d.latent_tract_id.numpy()[valid]
    pl = r.plotter()
    pl.add_mesh(gt, color=GT_GREY, opacity=0.08, smooth_shading=True)
    for t in np.unique(cl_t):
        pts = cl[cl_t == t]
        if len(pts) >= 2:
            pl.add_mesh(pv.lines_from_points(pts).tube(radius=0.12), color="#8a8a8a")
    for t in np.unique(tok_t):
        pl.add_mesh(pv.PolyData(tok[tok_t == t]).glyph(geom=pv.Sphere(radius=0.62), scale=False,
                                                     orient=False),
                    color=TRACT_COLORS[int(t) % len(TRACT_COLORS)], smooth_shading=True)
    shots.append(r.shoot(pl, "tokens"))

    fine = poly(d.x.numpy(), d.face.numpy())
    pl = r.plotter()
    pl.add_mesh(fine, color=TEMPLATE_BLUE, smooth_shading=True, specular=0.25)
    shots.append(r.shoot(pl, "template"))

    levels = (
        ("coarse", d.pos_coarse, d.face_coarse, d.tract_id_coarse, 1.4),
        ("mid", d.pos_mid, d.face_mid, d.tract_id_mid, 0.9),
        ("fine", d.x, d.face, d.tract_id, 0.35),
    )
    for name, pos, face, tract, lw in levels:
        m = poly(pos.numpy(), face.numpy())
        m.point_data["rgb"] = tract_rgb(tract.numpy())
        pl = r.plotter()
        pl.add_mesh(m, scalars="rgb", rgb=True, show_edges=True, edge_color="#3a3a3a",
                    line_width=lw, smooth_shading=False, ambient=0.25)
        shots.append(r.shoot(pl, f"level_{name}"))

    crop_all(shots, ref)

    info = {
        "case": args.case,
        "n_gt": int(d.gt_points.shape[0]),
        "n_true": int(d.x_true.shape[0]),
        "n_fine": int(d.x.shape[0]),
        "n_mid": int(d.pos_mid.shape[0]),
        "n_coarse": int(d.pos_coarse.shape[0]),
        "n_tokens": int(valid.sum()),
        "latent_len": int(valid.shape[0]),
        "n_tracts": int(d.n_tracts),
        "tok_tract": [int(t) for t in tok_t],
        "sdf_lim": lim,
    }
    import json
    with open(os.path.join(OUT, "case_info.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print(info)


if __name__ == "__main__":
    main()
