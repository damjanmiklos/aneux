"""Paper-style overview of the full two-stage model (fullscreen, 16:9).

Stage 1 follows the design draft (conditioning, latent diffusion with
classifier-free guidance, Transformer decoder to a serialized tree). Stage 2
is the implemented geometry VAE of make_pipeline_figure.py, condensed. The
draft's Stage 2 (a tube conditioned by FiLM on a global z) is replaced by
what the code does now; the interface between the stages follows the Stage-1
contract in STAGE2_REVIEW §5.2 / §10.2: per 1 mm sample [x, y, z, r, s],
a texture token every 2 mm, codes standardised by latent_standardise.py.

Layout plan (canvas 160 x 90 units, 1 unit = 1/8 in, figure 20 x 11.25 in):

  Title row      title, subtitle with the line-style key, legend at right.
  Stage 1 band   (y 48-84.5), vertical tab on the left.
    inference lane: clinical vector -> embedding / frequency encoding /
      missing token -> concat -> MLP -> c (and learned null condition) ->
      diffusion box [x_t node, two U-Net passes (c and null), guidance,
      denoising step, loop x1000] -> x0 -> Transformer decoder ->
      generated sequence table -> decoded tree render.
    training lane (dashed, step 3): S_GT -> PointNeXt (global) -> x0 ->
      diffusion loss; z ~ N(x0, I) -> Transformer; sequence targets / loss.
  Hand-off wire  along the gap between the bands (non-differentiable).
  Stage 2 band   (y 1-46.3), vertical tab on the left.
    lane I: parse sequence -> de-standardise -> Z grid, Z bus.
    lane D: template construction -> T -> coarse / mid / fine decoder
      levels (kNN upsampling between) -> post-processing -> surface render.
    lane T (dashed, steps 1 and 2): S_GT -> PointNeXt blocks -> latent head
      -> mu, log sigma^2 -> tract mixer -> reparameterise -> z joins the Z bus;
      standardise -> Stage-1 targets; Stage-2 losses.

    python figures/make_two_stage_figure.py
"""
from __future__ import annotations

import os

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Rectangle

import make_pipeline_figure as base
from make_pipeline_figure import GREY, INK, KIND, TRACT_COLORS, Z_COL, Canvas, fmt, load_info

# Arial lacks circled digits, subscript digits and angle brackets; fall back per glyph
base.plt.rcParams["font.family"] = ["Arial", "Segoe UI Symbol", "DejaVu Sans"]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "two_stage_pipeline")

KIND.update({
    "diff": ("#ece3f5", "#6a4c93"),
    "cond": ("#efefef", "#6b6b6b"),
})
BAND = {
    1: ("#f8f5fb", "#d9cce8", "#6a4c93"),
    2: ("#f5faf4", "#c9e0c6", "#3d8b4b"),
}
TRAIN = (0, (4.5, 2.5))
LOSS_EC = KIND["loss"][1]


def band(c, key, y0, y1, title, sub):
    fc, ec, tab = BAND[key]
    c.rbox(1, y0, 158, y1 - y0, fc, ec, lw=1.0, r=1.2, z=0)
    c.rbox(1, y0, 3.6, y1 - y0, tab, tab, lw=1.0, r=1.2, z=1)
    c.text(2.35, (y0 + y1) / 2, title, fs=12.5, weight="bold", color="white", rotation=90)
    c.text(3.75, (y0 + y1) / 2, sub, fs=8.6, color="white", rotation=90)


def badge(c, x, y, n, r=0.72, col="#555555"):
    """Numbered training-order badge (Arial has no circled digits)."""
    c.ax.add_patch(Circle((x, y), r, fc=col, ec="none", zorder=9))
    c.text(x, y - 0.03, str(n), fs=7.6, weight="bold", color="white", z=10)


def lane_label(c, x, y, n, s):
    badge(c, x + 0.72, y, n)
    c.text(x + 1.9, y, s, fs=9.2, weight="bold", color="#555555", ha="left")


def unet(c, x, y, w, h, ec=KIND["diff"][1], fc=KIND["diff"][0]):
    """Small U-Net glyph: encoder bars stepping down, decoder bars stepping up."""
    depth = [0, 1, 2, 3, 2, 1, 0]
    n = len(depth)
    bw = w / (n * 1.45)
    step = (w - bw) / (n - 1)
    for i, dpt in enumerate(depth):
        bh = h * (0.92 - 0.2 * dpt)
        by = y + h - bh - dpt * h * 0.05
        bx = x + i * step
        c.ax.add_patch(Rectangle((bx, by), bw, bh, fc=fc, ec=ec, lw=0.9, zorder=4))
        if i < n // 2:
            j = n - 1 - i
            yy = by + bh * 0.82
            c.ax.plot([bx + bw, x + j * step], [yy, yy], color=ec, lw=0.6, ls=(0, (2, 1.5)),
                      zorder=3)


def vec_glyph(c, x, y, w, h, n, col, ec):
    for i in range(n):
        c.ax.add_patch(Rectangle((x, y + i * h / n), w, h / n, fc=col, ec=ec, lw=0.6, zorder=4))


def z_grid(c, info, gx, gy, gw, gh):
    tok = np.asarray(info["tok_tract"])
    order = np.argsort(tok, kind="stable")
    n = len(order)
    cw = gw / n
    for j, t in enumerate(tok[order]):
        base_c = np.array(to_rgb(TRACT_COLORS[t % len(TRACT_COLORS)]))
        c.ax.add_patch(Rectangle((gx + j * cw, gy), cw, gh, fc=1 - 0.45 * (1 - base_c),
                                 ec="none", zorder=3))
        c.ax.add_patch(Rectangle((gx + j * cw, gy + gh), cw, 0.5,
                                 fc=TRACT_COLORS[t % len(TRACT_COLORS)], ec="none", zorder=3))
    for i in range(17):
        c.ax.plot([gx, gx + gw], [gy + i * gh / 16] * 2, color="white", lw=0.35, zorder=4)
    c.ax.add_patch(Rectangle((gx, gy), gw, gh + 0.5, fc="none", ec=Z_COL, lw=1.3, zorder=5))


# ---------------------------------------------------------------------------
def draw_header(c):
    c.text(1.5, 88.3, "Two-stage generative model: clinical conditions → vessel surface",
           fs=18.5, weight="bold", ha="left")
    c.text(1.5, 85.9,
           "Stage 1 generates the centerline tree with texture tokens; Stage 2 decodes the "
           "tokens onto a template mesh.   Solid: inference · dashed: training only · "
           "numbered badges: training order",
           fs=10, color=GREY, ha="left")
    entries = [
        ("cond", "condition"), ("diff", "diffusion"), ("attn", "attention / transformer"),
        ("enc", "encoder"), ("lat", "latent / stochastic"), ("dec", "decoder"),
        ("op", "fixed op (non-diff.)"), ("loss", "loss"),
    ]
    x0, y0 = 106.5, 88.7
    for i, (k, lab) in enumerate(entries):
        col, row = i % 4, i // 4
        xx, yy = x0 + col * 13.4, y0 - row * 2.2
        fc, ec = KIND[k]
        c.rbox(xx, yy - 0.5, 2.2, 1.0, fc, ec, lw=0.9, ls=(0, (3, 1.6)) if k == "op" else "-",
               r=0.2, z=3)
        c.text(xx + 2.8, yy, lab, fs=8.2, ha="left")


def draw_stage1(c, info):
    band(c, 1, 47.7, 84.4, "STAGE 1 · generation", "design draft")

    # clinical vector
    c.text(10.7, 82.9, "clinical vector", fs=9.8, weight="bold")
    c.rbox(6.0, 65.0, 9.4, 16.8, KIND["cond"][0], KIND["cond"][1], lw=1.1, r=0.5, z=3)
    rows = ["sex = female", "age = 60 y", "m$_1$ = 1.3", "m$_2$ = 9.8", "m$_3$ = NULL"]
    ry = [79.4, 76.4, 73.4, 70.4, 67.4]
    for s, y in zip(rows, ry):
        c.text(6.7, y, s, fs=8.8, ha="left")

    # encoders
    encs = [
        (76.6, "Embedding", "categorical", [0]),
        (71.0, "Frequency enc.", "numerical, N freq.", [1, 2, 3]),
        (65.4, "Missing token", "NULL (learned)", [4]),
    ]
    for yb, title, sub, src in encs:
        c.block(18.0, yb, 12.4, 4.6, "cond", title, [sub], tfs=9.0, fs=7.9)
        for k in src:
            c.arrow([(15.5, ry[k]), (18.0, yb + 2.3)], lw=0.9, ms=7)
        c.arrow([(30.4, yb + 2.3), (32.4, 73.3)], lw=0.9, ms=7)
    c.ax.add_patch(Circle((33.2, 73.3), 0.8, fc="white", ec=INK, lw=1.0, zorder=5))
    c.text(33.2, 73.3, "+", fs=10)
    c.text(33.2, 75.1, "concat", fs=7.6, color=GREY)
    c.arrow([(34.0, 73.3), (35.2, 73.3)], ms=7)
    c.block(35.2, 67.6, 3.0, 11.4, "cond", "MLP", rot=90, tfs=9.2)
    c.arrow([(38.2, 73.3), (39.7, 73.3)], ms=7)
    vec_glyph(c, 39.7, 70.3, 1.5, 6.0, 6, "#dcd3ea", KIND["diff"][1])
    c.text(40.45, 77.3, r"$c\in\mathbb{R}^{256}$", fs=9.2)
    vec_glyph(c, 39.7, 64.4, 1.5, 3.3, 3, "#ffffff", KIND["diff"][1])
    c.text(40.45, 68.6, r"$\varnothing$ learned", fs=8.2, color=GREY)

    # diffusion
    dx0, dx1, dy0, dy1 = 44.3, 101.0, 63.5, 83.2
    c.rbox(dx0, dy0, dx1 - dx0, dy1 - dy0, "#f3eef9", KIND["diff"][1], lw=1.2, r=0.9, z=2)
    c.text((dx0 + dx1) / 2, 82.0, "Latent diffusion with classifier-free guidance",
           fs=10.5, weight="bold")
    # x_t node
    c.ax.add_patch(Circle((48.6, 73.2), 1.5, fc=KIND["lat"][0], ec=KIND["lat"][1], lw=1.2,
                          zorder=5))
    c.text(48.6, 73.2, r"$x_t$", fs=10.5)
    c.text(48.6, 77.7, r"$x_T\sim\mathcal{N}(0,I)$", fs=8.6)
    c.arrow([(48.6, 76.9), (48.6, 74.7)], ms=7)
    # U-Nets
    unet(c, 54.2, 75.2, 12.2, 5.4)
    unet(c, 54.2, 67.4, 12.2, 5.4)
    c.text(60.3, 74.3, r"U-Net $\epsilon_\theta(x_t,t,c)$", fs=8.6)
    c.text(60.3, 66.5, r"same U-Net, $\epsilon_\theta(x_t,t,\varnothing)$", fs=8.6)
    c.arrow([(50.1, 73.2), (52.3, 73.2), (52.3, 77.4), (54.2, 77.4)], ms=7)
    c.arrow([(52.3, 73.2), (52.3, 70.8), (54.2, 70.8)], ms=7)
    c.arrow([(41.2, 74.8), (43.0, 74.8), (43.0, 79.9), (54.2, 79.9)], color=KIND["diff"][1],
            lw=1.1, ms=7)
    c.arrow([(41.2, 66.0), (43.0, 66.0), (43.0, 68.6), (54.2, 68.6)], color=KIND["diff"][1],
            lw=1.1, ms=7)
    # guidance and step
    c.arrow([(66.4, 77.9), (68.0, 77.9), (68.0, 75.0), (69.4, 75.0)], ms=7)
    c.arrow([(66.4, 70.1), (68.0, 70.1), (68.0, 72.6), (69.4, 72.6)], ms=7)
    c.block(69.4, 69.3, 15.8, 9.0, "diff", "Guidance",
            [r"$\hat\epsilon=\epsilon_\varnothing+w\,(\epsilon_c-\epsilon_\varnothing)$",
             "w: guidance scale"], tfs=9.6, fs=8.6)
    c.arrow([(85.2, 73.8), (86.8, 73.8)], ms=7)
    c.block(86.8, 69.3, 12.6, 9.0, "diff", "Denoising step",
            [r"$x_{t-1}=\mathrm{step}(x_t,\hat\epsilon)$", "t = T … 1"], tfs=9.6, fs=8.6)
    c.arrow([(93.1, 69.3), (93.1, 64.7), (48.6, 64.7), (48.6, 71.7)], color=KIND["diff"][1],
            lw=1.1, ms=8)
    c.text(71.0, 64.7, "repeat T = 1000 steps", fs=8.2, color=KIND["diff"][1],
           bg="#f3eef9")
    # x0 -> Transformer
    c.arrow([(99.4, 73.8), (104.0, 73.8)])
    c.text(102.3, 75.4, r"$z=x_0$", fs=9.4)

    c.block(104.0, 64.6, 12.4, 17.0, "attn", "Transformer",
            ["decoder", "", "z → serialized", "tree, one row", "per step", "", "+ MLP heads"],
            tfs=10, fs=8.5)
    c.arrow([(116.4, 73.1), (118.6, 73.1)])

    # generated sequence
    tx, tw = 118.6, 21.0
    c.text(tx + tw / 2, 82.9, "generated sequence", fs=9.8, weight="bold")
    rows = [
        ("hdr", "x   y   z   r   s", r"$\tau\in\mathbb{R}^{16}$"),
        ("sp", r"$\langle$branch start$\rangle$", ""),
        ("row", "x₁ y₁ z₁ r₁ s₁", "τ₁"),
        ("row", "x₂ y₂ z₂ r₂ s₂", "–"),
        ("row", "x₃ y₃ z₃ r₃ s₃", "τ₂"),
        ("dots", "⋮", "⋮"),
        ("sp", r"$\langle$branch end$\rangle$", ""),
        ("sp", r"$\langle$branch start$\rangle$", ""),
        ("dots", "⋮", "⋮"),
    ]
    rh = 1.8
    y = 81.5
    split = tx + 14.2
    for kind, left, right in rows:
        y0 = y - rh
        if kind == "hdr":
            c.ax.add_patch(Rectangle((tx, y0), tw, rh, fc="#e6e6e6", ec="#9a9a9a", lw=0.6,
                                     zorder=3))
        elif kind == "sp":
            c.ax.add_patch(Rectangle((tx, y0), tw, rh, fc="#f4f0e2", ec="#9a9a9a", lw=0.6,
                                     zorder=3))
        else:
            c.ax.add_patch(Rectangle((tx, y0), split - tx, rh, fc="white", ec="#9a9a9a",
                                     lw=0.6, zorder=3))
            c.ax.add_patch(Rectangle((split, y0), tx + tw - split, rh,
                                     fc=KIND["lat"][0] if right.startswith("τ") else "white",
                                     ec="#9a9a9a", lw=0.6, zorder=3))
        if kind == "sp":
            c.text(tx + tw / 2, y0 + rh / 2, left, fs=8.3, color="#7a5d10")
        else:
            c.text((tx + split) / 2, y0 + rh / 2, left, fs=8.3)
            c.text((split + tx + tw) / 2, y0 + rh / 2, right, fs=8.3)
        y = y0
    c.text(tx + tw / 2, y - 0.9, "1 mm samples · τ on every 2nd (2 mm)", fs=7.9, color=GREY)
    c.text(tx + tw / 2, y - 2.1, "r = MISR · s = template stretch", fs=7.9, color=GREY)

    c.arrow([(139.6, 73.1), (141.2, 73.1)], ms=8)
    c.image("stage1_tree", 141.2, 64.8, 16.6, 16.0)
    c.text(149.5, 82.9, "decoded tree", fs=9.8, weight="bold")

    # training lane (step 3)
    lane_label(c, 6.0, 62.0, 3, "Stage-1 training (Stage 2 frozen)")
    c.image("gt_surface", 5.6, 49.0, 10.8, 10.0)
    c.text(11.0, 48.7, r"$S_{GT}$", fs=8.6, color=GREY)
    c.arrow([(16.4, 54.2), (18.2, 54.2)], ls=TRAIN, color=GREY, ms=8)
    c.block(18.2, 50.3, 12.6, 8.0, "enc", "PointNeXt", ["encoder", "(global pooling)"],
            tfs=9.6, fs=8.3)
    c.arrow([(30.8, 54.2), (33.1, 54.2)], ls=TRAIN, color=GREY, ms=8)
    c.ax.add_patch(Circle((34.4, 54.2), 1.3, fc=KIND["lat"][0], ec=KIND["lat"][1], lw=1.1,
                          zorder=5))
    c.text(34.4, 54.2, r"$x_0$", fs=9.6)
    c.arrow([(35.7, 54.2), (44.3, 54.2)], ls=TRAIN, color=GREY, ms=8)
    c.block(44.3, 49.6, 56.7, 9.1, "loss", r"$\mathcal{L}_{diff}=\|\epsilon-\epsilon_\theta(x_t,t,c)\|^2$",
            [r"$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$,   $\epsilon\sim\mathcal{N}(0,I)$",
             r"c randomly replaced by $\varnothing$, which trains the unconditional branch"],
            tfs=9.8, fs=8.6)
    c.arrow([(72.6, 58.7), (72.6, 63.5)], ls=TRAIN, color=LOSS_EC, lw=1.1, ms=8)
    c.arrow([(34.4, 55.5), (34.4, 60.6), (110.2, 60.6), (110.2, 64.6)], ls=TRAIN, color=GREY,
            ms=8)
    c.text(104.6, 60.6, r"$z\sim\mathcal{N}(x_0,I)$", fs=8.8, color=INK, bg=BAND[1][0])
    c.block(112.6, 48.8, 33.2, 12.0, "loss", r"$\mathcal{L}_{seq}$  sequence reconstruction",
            ["targets per 1 mm sample:",
             "x, y, z, r: original centerline + MISR",
             "s: StretchDistance of the template",
             "τ: standardised Stage-2 posterior samples (step 2)",
             r"$\langle$branch$\rangle$ tokens: tree order"], tfs=9.6, fs=8.1)


def draw_handoff(c):
    y = 47.2
    c.arrow([(149.5, 64.8), (149.5, y), (14.0, y), (14.0, 44.2)], lw=1.6)
    c.text(82.0, y, "inference hand-off (non-differentiable): generated sequence → Stage 2",
           fs=8.8, weight="bold", bg="white", z=8)


def draw_stage2(c, info):
    band(c, 2, 1, 46.4, "STAGE 2 · geometry VAE", "implemented")

    # lane I: parse, de-standardise, Z
    c.block(6.0, 36.4, 16.0, 7.8, "op", "Parse sequence",
            [r"$\mathcal{C}$: xyz, MISR r, branches", "s: stretch · τ: tokens"], tfs=9.6, fs=8.1)
    c.arrow([(22.0, 40.3), (24.6, 40.3)], ms=8)
    c.block(24.6, 36.4, 16.4, 7.8, "op", "De-standardise",
            [r"$\tau\,\sigma_d+\mu_d$ per dimension", "inactive dims → 0"], tfs=9.6, fs=8.1)
    c.arrow([(41.0, 40.3), (44.6, 40.3)], color=Z_COL, lw=1.4, ms=9)
    z_grid(c, info, 44.6, 37.3, 30.0, 5.9)
    c.text(76.4, 42.1, r"$Z\in\mathbb{R}^{L\times16}$", fs=10.5, ha="left")
    c.text(76.4, 40.1, "one 16-d code per 2 mm token", fs=8.3, color=GREY, ha="left")
    c.text(76.4, 38.6, f"(p462: {len(info['tok_tract'])} tokens, L = 128 slots)", fs=8.3,
           color=GREY, ha="left")

    # lane D: template, levels, post-processing
    c.arrow([(14.0, 36.4), (14.0, 34.1)], ms=8)
    c.block(6.0, 18.6, 16.0, 15.5, "op", "Template (VMTK)",
            [r"base surface from $\mathcal{C}$ + MISR", "uncap outlets",
             "remesh, edge length", "set by s", "→ T (variable density)",
             "+ scaffold (u, θ, tract)"], tfs=9.6, fs=8.1)
    c.image("template", 22.4, 19.5, 13.0, 11.0)
    c.text(28.9, 31.3, "T", fs=10.5, weight="bold")
    levels = [
        ("coarse", "Coarse", info["n_coarse"], 38.2,
         ["cross-attn → Z", "pos. self-attn", "6 × SplineConv", "free Δx", "rim-plane proj."]),
        ("mid", "Mid", info["n_mid"], 64.8,
         ["cross-attn → Z", "skip  h↑, Δx↑", "6 × SplineConv", "Δr n + Δs (t, b)",
          "rim-plane proj."]),
        ("fine", "Fine", info["n_fine"], 91.4,
         ["cross-attn → Z", "skip  h↑, Δx↑", "6 × SplineConv", "Δr n + Δs (t, b)",
          "rim-plane proj."]),
    ]
    lw_, ly0, lh = 23.6, 18.6, 15.0
    c.arrow([(35.4, 25.0), (38.2, 25.0)], ms=8)
    for key, name, n, x0, lines in levels:
        c.rbox(x0, ly0, lw_, lh, KIND["dec"][0], KIND["dec"][1], lw=1.2, r=0.7, z=3)
        c.text(x0 + 0.9, ly0 + lh - 1.2, f"{name} · {fmt(n)} v.", fs=9.4, weight="bold",
               ha="left")
        c.image(f"level_{key}", x0 + 0.5, ly0 + 0.7, 10.0, 10.8, anchor="bottom", z=4)
        for j, s in enumerate(lines):
            col = "#7a5a0a" if s.startswith(("cross", "pos.")) else "#1f5c2a" if "Spline" in s else "#333333"
            c.text(x0 + 17.3, ly0 + lh - 3.8 - 2.05 * j, s, fs=8.3, color=col)
    for xa, xb in ((61.8, 64.8), (88.4, 91.4)):
        c.arrow([(xa, 25.0), (xb, 25.0)], color=KIND["dec"][1], ms=8)
        c.text((xa + xb) / 2, 26.4, "kNN↑", fs=7.6, color=KIND["dec"][1])
    # Z bus
    zb = 35.4
    c.arrow([(59.6, 37.3), (59.6, zb)], color=Z_COL, lw=1.6, head=False, z=7)
    c.arrow([(50.0, zb), (116.6, zb)], color=Z_COL, lw=1.6, head=False, z=7)
    for x0 in (38.2, 64.8, 91.4):
        c.arrow([(x0 + 11.8, zb), (x0 + 11.8, ly0 + lh)], color=Z_COL, lw=1.6, z=7, ms=9)
    # post-processing and output
    c.arrow([(115.0, 25.0), (118.2, 25.0)], ms=8)
    c.block(118.2, 20.0, 9.2, 10.0, "op", "Post-process",
            ["fold checks", "isotropic", "remesh"], tfs=9.0, fs=8.1)
    c.arrow([(127.4, 25.0), (129.2, 25.0)], ms=8)
    c.image("gt_surface", 129.0, 21.6, 28.8, 22.6)
    c.text(143.4, 20.7, r"$\hat{S}$: watertight, open outlets → CFD", fs=9.2, weight="bold")
    c.text(143.4, 19.2, "(shown: GT surface of case p462)", fs=8.1, color=GREY)

    # lane T: stage-2 training
    lane_label(c, 6.0, 16.3, 1, r"Stage-2 training (VAE):  $\mathcal{C}$ = original centerline,  "
               r"T = cleandata template,  target $S_{GT}$")
    c.image("gt_surface", 5.6, 2.3, 10.6, 10.0)
    c.arrow([(16.2, 7.6), (17.8, 7.6)], ls=TRAIN, color=GREY, ms=7)
    # mini PointNeXt blocks
    x = 17.8
    for n, ch in ((16384, 32), (1024, 64), (256, 128), (64, 128), (64, 256)):
        h = 0.48 * 0.95 * n ** 0.3
        w = 0.19 * ch ** 0.5
        c.cuboid(x, 7.6 - h / 2, w, h)
        x += w + base.CUBE_DX * 0.6 + 1.3
    c.text(26.6, 2.6, "PointNeXt, 16384 → 64 pts", fs=7.9, color=GREY)
    c.arrow([(x - 0.4, 7.6), (38.2, 7.6)], ls=TRAIN, color=GREY, ms=7)
    c.block(38.2, 3.8, 11.2, 7.6, "enc", "Latent head", ["pool → tokens", "transformer × 3"],
            tfs=9.2, fs=8.0)
    c.arrow([(49.4, 9.4), (51.0, 9.4)], ls=TRAIN, color=GREY, ms=7)
    c.arrow([(49.4, 5.8), (51.0, 5.8)], ls=TRAIN, color=GREY, ms=7)
    c.block(51.0, 7.9, 6.0, 3.4, "lat", r"$\mu$", tfs=10)
    c.block(51.0, 3.9, 6.0, 3.4, "lat", r"$\log\sigma^2$", tfs=9.4)
    c.arrow([(57.0, 9.4), (58.8, 9.4)], ls=TRAIN, color=GREY, ms=7)
    c.arrow([(57.0, 5.8), (58.8, 5.8)], ls=TRAIN, color=GREY, ms=7)
    c.block(58.8, 3.8, 10.6, 7.6, "attn", "Tract mixer", ["on μ"], tfs=9.2, fs=8.0)
    c.arrow([(69.4, 7.6), (71.2, 7.6)], ls=TRAIN, color=GREY, ms=7)
    c.block(71.2, 3.8, 11.2, 7.6, "lat", "Reparam.", [r"$z=\tilde\mu+\sigma\odot\varepsilon$"],
            tfs=9.2, fs=8.6)
    c.arrow([(82.4, 7.6), (84.6, 7.6)], ls=TRAIN, color=GREY, ms=7)
    c.block(84.6, 2.6, 17.4, 10.0, "op", "Standardise",
            ["drop inactive dims", r"$(\mu+\sigma\varepsilon-\mu_d)\,/\,\sigma_d$",
             "→ τ targets of Stage 1"], tfs=9.2, fs=8.1)
    # training z joins the bus
    c.arrow([(76.8, 11.4), (76.8, 14.3), (116.6, 14.3), (116.6, zb)], color=Z_COL, lw=1.4,
            ls=TRAIN, head=False, bridge=True, z=6)
    c.text(108.0, 14.3, "z (training)", fs=8.2, color=Z_COL, bg=BAND[2][0], z=8)
    # losses
    c.arrow([(112.0, ly0), (112.0, 16.9), (135.0, 16.9), (135.0, 14.2)], ls=TRAIN,
            color=LOSS_EC, lw=1.1, ms=8)
    c.block(119.6, 1.9, 38.2, 12.3, "loss", "Stage-2 losses",
            [r"$\mathcal{L}_{CD}+\mathcal{L}_{rad}+\lambda_{KL}\,\beta\,\mathcal{L}_{KL}"
             r"+0.15\,\mathcal{L}_{disp}+0.05\,\mathcal{L}_{lap}+0.02\,\mathcal{L}_{n}$",
             "Chamfer at 3 levels · radial Huber vs r* · KL rate by GECO", ""],
            tfs=9.6, fs=8.4)
    badge(c, 85.9, 11.5, 2)
    badge(c, 124.6, 3.6, 4, col=LOSS_EC)
    c.text(125.8, 3.6, "planned: fine-tune the Stage-2 decoder on Stage-1 samples", fs=8.4,
           color=LOSS_EC, ha="left", weight="bold")


def main():
    info = load_info()
    c = Canvas()
    draw_header(c)
    draw_stage1(c, info)
    draw_handoff(c)
    draw_stage2(c, info)
    c.fig.savefig(OUT + ".png", dpi=200)
    c.fig.savefig(OUT + ".pdf")
    c.fig.savefig(OUT + ".svg")
    print("wrote", OUT + ".{png,pdf,svg}")


if __name__ == "__main__":
    main()
