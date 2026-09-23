"""Paper-style overview of the full two-stage model (fullscreen, 16:9).

Follows the design draft. Stage 1 turns clinical conditions into a latent z
(latent diffusion with classifier-free guidance) and z into the serialized
centerline tree (Transformer decoder): per 1 mm row x, y, z, radius r and
mesh density s. Stage 2 is the template SplineConv decoder of the current code.

The stages meet three times:
  * geometry, non-differentiable: the rows build the template T (VMTK tube,
    uncap, remesh at edge length s). Nothing is back-propagated through T.
  * global, differentiable: z -> MLP -> (gamma, beta) modulates every
    SplineConv layer (FiLM or AdaIN).
  * local, differentiable: the Transformer's last hidden state h_i of every
    row is laid on its centerline sample and interpolated onto the vertices
    between neighbouring samples, h(v) = (1 - lam) h_i + lam h_{i+1}.
So the surface loss trains the Transformer and the encoder through z and h,
while the template itself stays a fixed input.

Layout (canvas 160 x 90 units, 1 unit = 1/8 in). A U-shaped read: Stage 1
left to right on top, the three connections drop straight down, Stage 2
reads right to left from the template to the output surface.

    python figures/make_two_stage_figure.py
"""
from __future__ import annotations

import os

from matplotlib.patches import Circle, Rectangle

import make_pipeline_figure as base
from make_pipeline_figure import GREY, INK, KIND, Z_COL, Canvas, fmt, load_info

# Arial lacks subscript digits and angle brackets; fall back per glyph
base.plt.rcParams["font.family"] = ["Arial", "Segoe UI Symbol", "DejaVu Sans"]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "two_stage_pipeline")

H_COL = "#1f8a8a"
KIND.update({
    "diff": ("#ece3f5", "#6a4c93"),
    "cond": ("#efefef", "#6b6b6b"),
    "hid": ("#d8eeec", H_COL),
})
BAND = {
    1: ("#f8f5fb", "#d9cce8", "#6a4c93"),
    2: ("#f5faf4", "#c9e0c6", "#3d8b4b"),
}
TRAIN = (0, (4.5, 2.5))
LOSS_EC = KIND["loss"][1]
TF, BF, SF = 10.6, 9.7, 9.3  # block title, body, smallest


def band(c, key, y0, y1, title, sub):
    fc, ec, tab = BAND[key]
    c.rbox(1, y0, 158, y1 - y0, fc, ec, lw=1.0, r=1.2, z=0)
    c.rbox(1, y0, 3.6, y1 - y0, tab, tab, lw=1.0, r=1.2, z=1)
    c.text(2.25, (y0 + y1) / 2, title, fs=13.5, weight="bold", color="white", rotation=90)
    c.text(3.85, (y0 + y1) / 2, sub, fs=SF, color="white", rotation=90)


def badge(c, x, y, n, r=0.82, col="#555555"):
    """Numbered training-order badge (Arial has no circled digits)."""
    c.ax.add_patch(Circle((x, y), r, fc=col, ec="none", zorder=9))
    c.text(x, y - 0.03, str(n), fs=8.8, weight="bold", color="white", z=10)


def lane_label(c, x, y, n, s):
    badge(c, x + 0.82, y, n)
    c.text(x + 2.1, y, s, fs=10.1, weight="bold", color="#555555", ha="left")


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
            yy = by + bh * 0.82
            c.ax.plot([bx + bw, x + (n - 1 - i) * step], [yy, yy], color=ec, lw=0.6,
                      ls=(0, (2, 1.5)), zorder=3)


def vec_glyph(c, x, y, w, h, n, col, ec):
    for i in range(n):
        c.ax.add_patch(Rectangle((x, y + i * h / n), w, h / n, fc=col, ec=ec, lw=0.6, zorder=4))


def node(c, x, y, s, r=1.3, fs=10.1):
    c.ax.add_patch(Circle((x, y), r, fc=KIND["lat"][0], ec=KIND["lat"][1], lw=1.2, zorder=5))
    c.text(x, y, s, fs=fs)


def texts(c, x, y_top, lines, fs=BF, step=None, color="#333333", ha="center", weight="normal"):
    step = step or base.line_units(fs)
    for j, s in enumerate(lines):
        c.text(x, y_top - j * step, s, fs=fs, color=color, ha=ha, weight=weight)


# ---------------------------------------------------------------------------
def draw_header(c):
    c.text(1.5, 88.3, "Two-stage generative model: clinical conditions → vessel surface",
           fs=19, weight="bold", ha="left")
    c.text(1.5, 85.8,
           "Stage 1 generates the centerline tree; Stage 2 deforms a template built from it, "
           "conditioned on z and on the Transformer's hidden states.",
           fs=10.6, color=GREY, ha="left")
    entries = [
        ("cond", "condition"), ("diff", "diffusion"), ("attn", "attention / transformer"),
        ("enc", "encoder"), ("lat", "latent / stochastic"), ("hid", "hidden states h"),
        ("dec", "decoder"), ("loss", "loss"), ("op", "fixed op (non-diff.)"),
        ("_solid", "inference"), ("_dash", "training only"), ("_badge", "training order"),
    ]
    x0, y0 = 99.5, 89.0
    for i, (k, lab) in enumerate(entries):
        col, row = i % 4, i // 4
        xx, yy = x0 + col * 15.0, y0 - row * 1.95
        if k == "_solid":
            c.arrow([(xx, yy), (xx + 2.4, yy)], lw=1.3, ms=7)
        elif k == "_dash":
            c.arrow([(xx, yy), (xx + 2.4, yy)], lw=1.3, ls=TRAIN, color=GREY, ms=7)
        elif k == "_badge":
            badge(c, xx + 1.1, yy, 1, r=0.7)
        else:
            fc, ec = KIND[k]
            c.rbox(xx, yy - 0.5, 2.4, 1.0, fc, ec, lw=0.9,
                   ls=(0, (3, 1.6)) if k == "op" else "-", r=0.2, z=3)
        c.text(xx + 3.0, yy, lab, fs=SF, ha="left")


def draw_stage1(c):
    band(c, 1, 47.6, 84.4, "STAGE 1 · centerline generation", "design draft")

    # clinical vector and its encoders
    c.text(10.7, 82.9, "clinical vector", fs=TF, weight="bold")
    c.rbox(6.0, 65.0, 9.4, 16.8, KIND["cond"][0], KIND["cond"][1], lw=1.1, r=0.5, z=3)
    rows = ["sex = female", "age = 60 y", "m$_1$ = 1.3", "m$_2$ = 9.8", "m$_3$ = NULL"]
    ry = [79.4, 76.4, 73.4, 70.4, 67.4]
    for s, y in zip(rows, ry):
        c.text(6.7, y, s, fs=BF, ha="left")
    encs = [
        (76.6, "Embedding", "categorical", [0]),
        (71.0, "Frequency enc.", "numerical, N freq.", [1, 2, 3]),
        (65.4, "Missing token", "NULL (learned)", [4]),
    ]
    for yb, title, sub, src in encs:
        c.block(17.6, yb, 11.8, 4.6, "cond", title, [sub], tfs=10.1, fs=SF)
        for k in src:
            c.arrow([(15.4, ry[k]), (17.6, yb + 2.3)], lw=0.9, ms=7)
        c.arrow([(29.4, yb + 2.3), (30.6, 73.3)], lw=0.9, ms=7)
    c.ax.add_patch(Circle((31.4, 73.3), 0.8, fc="white", ec=INK, lw=1.0, zorder=5))
    c.text(31.4, 73.3, "+", fs=10)
    c.arrow([(32.2, 73.3), (33.0, 73.3)], ms=7)
    c.block(33.0, 67.6, 3.0, 11.4, "cond", "concat → MLP", rot=90, tfs=10.1)
    c.arrow([(36.0, 73.3), (37.2, 73.3)], ms=7)
    vec_glyph(c, 37.2, 70.3, 1.5, 6.0, 6, "#dcd3ea", KIND["diff"][1])
    c.text(38.9, 77.4, r"$c\in\mathbb{R}^{256}$", fs=BF)
    vec_glyph(c, 37.2, 64.4, 1.5, 3.3, 3, "#ffffff", KIND["diff"][1])
    c.text(37.95, 63.3, r"$\varnothing$ learned", fs=SF, color=GREY)

    # latent diffusion
    dx0, dx1, dy0, dy1 = 41.5, 95.5, 63.5, 83.2
    c.rbox(dx0, dy0, dx1 - dx0, dy1 - dy0, "#f3eef9", KIND["diff"][1], lw=1.2, r=0.9, z=2)
    c.text((dx0 + dx1) / 2, 82.0, "Latent diffusion with classifier-free guidance",
           fs=TF, weight="bold")
    node(c, 45.6, 73.2, r"$x_t$", r=1.5, fs=11)
    c.text(45.6, 77.7, r"$x_T\sim\mathcal{N}(0,I)$", fs=BF)
    c.arrow([(45.6, 76.9), (45.6, 74.7)], ms=7)
    unet(c, 50.6, 75.2, 11.2, 5.4)
    unet(c, 50.6, 67.4, 11.2, 5.4)
    c.text(56.2, 74.3, r"U-Net $\epsilon_\theta(x_t,t,c)$", fs=BF)
    c.text(56.2, 66.5, r"same U-Net, $\epsilon_\theta(x_t,t,\varnothing)$", fs=BF)
    c.arrow([(47.1, 73.2), (48.9, 73.2), (48.9, 77.4), (50.6, 77.4)], ms=7)
    c.arrow([(48.9, 73.2), (48.9, 70.8), (50.6, 70.8)], ms=7)
    c.arrow([(38.7, 74.8), (40.3, 74.8), (40.3, 79.9), (50.6, 79.9)], color=KIND["diff"][1],
            lw=1.1, ms=7)
    c.arrow([(38.7, 66.0), (40.3, 66.0), (40.3, 68.6), (50.6, 68.6)], color=KIND["diff"][1],
            lw=1.1, ms=7)
    c.arrow([(61.8, 77.9), (63.2, 77.9), (63.2, 75.0), (64.6, 75.0)], ms=7)
    c.arrow([(61.8, 70.1), (63.2, 70.1), (63.2, 72.6), (64.6, 72.6)], ms=7)
    c.block(64.6, 69.3, 14.8, 9.0, "diff", "Guidance",
            [r"$\hat\epsilon=\epsilon_\varnothing+w\,(\epsilon_c-\epsilon_\varnothing)$",
             "w: guidance scale"], tfs=TF, fs=BF)
    c.arrow([(79.4, 73.8), (81.0, 73.8)], ms=7)
    c.block(81.0, 69.3, 13.0, 9.0, "diff", "Denoising step",
            [r"$x_{t-1}=\mathrm{step}(x_t,\hat\epsilon)$", "t = T … 1"], tfs=TF, fs=BF)
    c.arrow([(87.5, 69.3), (87.5, 64.7), (45.6, 64.7), (45.6, 71.7)], color=KIND["diff"][1],
            lw=1.1, ms=8)
    c.text(66.5, 64.7, "repeat T = 1000 steps", fs=SF, color=KIND["diff"][1], bg="#f3eef9")

    # z: x0 at inference, a noisy x0 in training
    zx, zy = 97.5, 60.6
    c.arrow([(94.0, 73.8), (zx, 73.8), (zx, zy + 1.3)], lw=1.3, ms=8)
    c.text(96.5, 75.1, r"$x_0$", fs=BF)
    c.text(99.1, 67.3, r"inference: $z=x_0$", fs=SF, rotation=90)
    node(c, zx, zy, r"$z$")
    c.arrow([(zx + 1.3, zy), (100.8, zy)], ms=7)

    # Transformer -> hidden states -> MLP head -> rows
    c.block(100.8, 52.0, 10.8, 30.0, "attn", "Transformer",
            ["decoder", "", "z → rows of", "the serialized", "tree"], tfs=TF, fs=BF)
    c.arrow([(111.6, 73.3), (113.4, 73.3)], ms=7)
    vec_glyph(c, 113.4, 56.0, 1.8, 23.0, 12, KIND["hid"][0], H_COL)
    c.text(114.3, 80.6, r"$h_i$", fs=11, color=H_COL, weight="bold")
    c.arrow([(115.2, 73.3), (117.0, 73.3)], ms=7)
    c.block(117.0, 67.6, 3.0, 11.4, "attn", "MLP head", rot=90, tfs=10.1)
    c.arrow([(120.0, 73.3), (121.8, 73.3)], ms=7)

    tx, tw = 121.8, 19.2
    c.text(tx + tw / 2, 82.9, "generated sequence", fs=TF, weight="bold")
    seq = [
        ("hdr", "x    y    z    r    s"),
        ("sp", r"$\langle$branch start$\rangle$"),
        ("row", "x₁  y₁  z₁  r₁  s₁"),
        ("row", "x₂  y₂  z₂  r₂  s₂"),
        ("row", "x₃  y₃  z₃  r₃  s₃"),
        ("row", "⋮"),
        ("sp", r"$\langle$branch end$\rangle$"),
        ("sp", r"$\langle$branch start$\rangle$  …"),
    ]
    rh, y = 1.9, 81.5
    for kind, s in seq:
        y0 = y - rh
        fc = {"hdr": "#e6e6e6", "sp": "#f4f0e2", "row": "white"}[kind]
        c.ax.add_patch(Rectangle((tx, y0), tw, rh, fc=fc, ec="#9a9a9a", lw=0.6, zorder=3))
        c.text(tx + tw / 2, y0 + rh / 2, s, fs=BF, color="#7a5d10" if kind == "sp" else INK)
        y = y0
    c.text(tx + tw / 2, y - 1.0, "one row per 1 mm · r: radius (MISR)", fs=SF, color=GREY)
    c.text(tx + tw / 2, y - 2.3, "s: mesh density (target edge length)", fs=SF, color=GREY)
    c.arrow([(tx + tw, 73.3), (142.4, 73.3)], ms=7)
    c.image("stage1_tree", 142.4, 64.6, 15.8, 16.4)
    c.text(150.3, 82.9, "decoded tree", fs=TF, weight="bold")

    # training lane
    lane_label(c, 6.0, 61.6, 1, "Training (end-to-end with Stage 2)")
    c.image("gt_surface", 5.6, 49.0, 10.8, 10.0)
    c.text(11.0, 48.6, r"$S_{GT}$", fs=BF, color=GREY)
    c.arrow([(16.4, 54.2), (18.0, 54.2)], ls=TRAIN, color=GREY, ms=8)
    c.block(18.0, 50.3, 12.4, 8.0, "enc", "PointNeXt", ["encoder", "(global pooling)"],
            tfs=TF, fs=BF)
    c.arrow([(30.4, 54.2), (32.3, 54.2)], ls=TRAIN, color=GREY, ms=8)
    node(c, 33.6, 54.2, r"$x_0$")
    c.arrow([(34.9, 54.2), (41.5, 54.2)], ls=TRAIN, color=GREY, ms=8)
    c.block(41.5, 49.6, 54.0, 9.1, "loss",
            r"$\mathcal{L}_{diff}=\|\epsilon-\epsilon_\theta(x_t,t,c)\|^2$",
            [r"$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$  on the latents "
             r"$x_0$ of the trained encoder",
             r"c randomly replaced by $\varnothing$, which trains the unconditional branch"],
            tfs=TF, fs=BF)
    badge(c, 42.5, 57.7, 2, col=LOSS_EC)
    c.arrow([(68.5, 58.7), (68.5, 63.5)], ls=TRAIN, color=LOSS_EC, lw=1.1, ms=8)
    c.arrow([(33.6, 55.5), (33.6, zy), (zx - 1.3, zy)], ls=TRAIN, color=GREY, ms=8)
    c.text(84.0, zy, r"training: $z\sim\mathcal{N}(x_0,I)$", fs=BF, bg=BAND[1][0])
    c.block(116.4, 48.6, 33.4, 12.4, "loss", r"$\mathcal{L}_{seq}$  sequence reconstruction",
            ["targets per 1 mm row, from the GT centerline:",
             "x, y, z and r (MISR)",
             "s: template edge length (StretchDistance)",
             r"$\langle$branch start / end$\rangle$: tree order"], tfs=TF, fs=BF)
    return zx


def draw_links(c, zx):
    """The three places where Stage 1 feeds Stage 2."""
    top = 44.6
    c.arrow([(zx, 59.3), (zx, top)], color=Z_COL, lw=1.9, ms=10, z=7)
    c.text(zx + 0.8, 46.6, r"$z$  (global)", fs=BF, color=Z_COL, weight="bold", ha="left",
           bg="white", z=8)
    c.arrow([(114.3, 56.0), (114.3, top)], color=H_COL, lw=1.9, ms=10, z=7)
    c.text(115.1, 46.6, r"$h_i$  (one per row)", fs=BF, color=H_COL, weight="bold",
           ha="left", bg="white", z=8)
    c.arrow([(152.0, 64.6), (152.0, top)], color=INK, lw=1.9, ms=10, z=7)
    c.text(151.2, 46.6, "rows x, y, z, r, s", fs=BF, weight="bold", ha="right", bg="white",
           z=8)


def draw_stage2(c, info, zx):
    band(c, 2, 1, 45.8, "STAGE 2 · template decoder", "SplineConv decoder of the current code")

    # --- conditioning and template (top row), fed from above ---------------
    ty0, ty1 = 34.6, 44.6
    # global: FiLM / AdaIN
    c.rbox(71.0, ty0, 28.5, ty1 - ty0, KIND["lat"][0], Z_COL, lw=1.3, r=0.7, z=3)
    c.text(85.25, 43.1, "Global: z → every layer", fs=TF, weight="bold")
    texts(c, 85.25, 41.0, [r"MLP$(z)\to(\gamma_\ell,\beta_\ell)$ per SplineConv layer $\ell$",
                           r"FiLM:  $y\leftarrow\gamma\odot y+\beta$",
                           r"AdaIN:  $y\leftarrow\gamma\odot(y-\mu_y)/\sigma_y+\beta$"],
          step=1.95)
    # local: hidden states interpolated onto the mesh
    c.rbox(101.5, ty0, 32.0, ty1 - ty0, KIND["hid"][0], H_COL, lw=1.3, r=0.7, z=3)
    c.text(112.4, 43.1, "Local: h on the mesh", fs=TF, weight="bold")
    texts(c, 112.4, 41.0, [r"$h_i$ sits on centerline sample $i$",
                           r"vertex $v$ between samples $i,\,i{+}1$:",
                           r"$h(v)=(1-\lambda)\,h_i+\lambda\,h_{i+1}$"], step=1.95)
    c.image("hidden_interp", 123.6, 35.2, 9.6, 8.8, z=4)
    # template generation: the only non-differentiable step
    c.rbox(135.6, ty0, 22.6, ty1 - ty0, "white", KIND["op"][1], lw=1.4, ls=(0, (4, 2.5)),
           r=0.7, z=3)
    c.text(146.9, 43.1, "Template generation", fs=TF, weight="bold")
    texts(c, 146.9, 41.1, ["VMTK tube from x, y, z, r", "uncap · remesh at edge length s",
                           "→ T + scaffold (u, θ, tract)"], step=1.75)
    c.text(146.9, 35.6, "non-differentiable", fs=BF, weight="bold", color=LOSS_EC)
    c.arrow([(135.6, 37.4), (133.5, 37.4)], ms=7)
    c.text(134.55, 38.6, "T", fs=SF, weight="bold")

    # gradient note
    c.rbox(48.0, ty0, 21.0, ty1 - ty0, "white", "#9a9a9a", lw=0.9, ls=(0, (1.5, 1.5)), r=0.7,
           z=3)
    c.text(58.5, 43.1, "Gradient paths", fs=TF, weight="bold")
    texts(c, 58.5, 40.9, ["z and h are differentiable, so", "the surface loss also trains",
                          "the Transformer and encoder;", "T is a fixed input"],
          step=1.6, fs=SF)

    # --- decoder row, right to left ----------------------------------------
    c.arrow([(147.0, ty0), (147.0, 31.8)], ms=8)
    c.image("template", 139.2, 15.2, 19.0, 16.4)
    c.text(140.2, 30.6, "T", fs=11, weight="bold", ha="left")
    c.text(148.7, 13.9, "levels: 8 / 25 / 100 % of T", fs=SF, color=GREY)
    c.text(148.7, 12.5, r"geometry: $r_{local},\,\kappa,\,\tau,\,d_{ost}$", fs=SF, color=GREY)

    ly0, lh, lw_ = 13.5, 18.0, 23.0
    levels = [
        ("coarse", "Coarse", info["n_coarse"], 114.0,
         ["in: h(v) ‖ geometry", "pos. self-attn", "6 × SplineConv", "+ FiLM/AdaIN each",
          "free Δx", "rim-plane proj."]),
        ("mid", "Mid", info["n_mid"], 87.0,
         ["in: h(v) ‖ geometry", "skip  h↑, Δx↑", "6 × SplineConv", "+ FiLM/AdaIN each",
          "Δr n + Δs (t, b)", "rim-plane proj."]),
        ("fine", "Fine", info["n_fine"], 60.0,
         ["in: h(v) ‖ geometry", "skip  h↑, Δx↑", "6 × SplineConv", "+ FiLM/AdaIN each",
          "Δr n + Δs (t, b)", "rim-plane proj."]),
    ]
    yc = 22.0
    c.arrow([(139.2, yc), (137.0, yc)], ms=8)
    for key, name, n, x0, lines in levels:
        c.rbox(x0, ly0, lw_, lh, KIND["dec"][0], KIND["dec"][1], lw=1.2, r=0.7, z=3)
        c.text(x0 + 0.9, ly0 + lh - 1.3, f"{name} · {fmt(n)} v.", fs=TF, weight="bold",
               ha="left")
        c.image(f"level_{key}", x0 + 0.4, ly0 + 0.6, 9.6, 11.2, anchor="bottom", z=4)
        for j, s in enumerate(lines):
            col = (H_COL if s.startswith("in:") else Z_COL if "FiLM" in s
                   else "#1f5c2a" if "Spline" in s else "#7a5a0a" if s.startswith("pos.")
                   else "#333333")
            c.text(x0 + 17.0, ly0 + lh - 4.0 - 1.95 * j, s, fs=BF, color=col)
    for xa, xb in ((114.0, 110.0), (87.0, 83.0)):
        c.arrow([(xa, yc), (xb, yc)], color=KIND["dec"][1], ms=8)
        c.text((xa + xb) / 2, yc + 1.5, "kNN↑", fs=SF, color=KIND["dec"][1])

    # buses: (gamma, beta) to every level, h(v) to every level
    gb, hb = 33.5, 32.6
    c.arrow([(85.25, ty0), (85.25, gb)], color=Z_COL, lw=1.6, head=False, z=7)
    c.arrow([(60.0 + 13.6, gb), (114.0 + 13.6, gb)], color=Z_COL, lw=1.6, head=False, z=7)
    c.arrow([(117.5, ty0), (117.5, hb)], color=H_COL, lw=1.6, head=False, z=7, bridge=True)
    c.arrow([(60.0 + 20.4, hb), (114.0 + 20.4, hb)], color=H_COL, lw=1.6, head=False, z=7)
    for x0 in (60.0, 87.0, 114.0):
        c.arrow([(x0 + 13.6, gb), (x0 + 13.6, ly0 + lh)], color=Z_COL, lw=1.6, z=7, ms=9,
                bridge=True)
        c.arrow([(x0 + 20.4, hb), (x0 + 20.4, ly0 + lh)], color=H_COL, lw=1.6, z=7, ms=9)
    c.text(60.0 + 13.0, gb, r"$(\gamma_\ell,\beta_\ell)$", fs=BF, color=Z_COL, ha="right")
    c.text(114.0 + 21.0, hb - 0.1, r"$h(v)$", fs=BF, color=H_COL, ha="left")

    # post-processing and output
    c.arrow([(60.0, yc), (57.4, yc)], ms=8)
    c.block(47.2, 17.0, 10.2, 10.0, "op", "Post-process", ["fold checks", "isotropic", "remesh"],
            tfs=TF, fs=BF)
    c.arrow([(47.2, yc), (45.4, yc)], ms=8)
    c.image("gt_surface", 5.6, 14.6, 40.4, 30.0)
    c.text(25.8, 13.4, r"$\hat{S}$: watertight, open outlets → CFD", fs=10.1, weight="bold")
    c.text(25.8, 11.9, "(shown: GT surface of case p462)", fs=SF, color=GREY)

    # losses (the same end-to-end training as step 1)
    c.arrow([(71.5, ly0), (71.5, 11.0)], ls=TRAIN, color=LOSS_EC, lw=1.1, ms=8)
    c.rbox(48.4, 1.9, 109.8, 9.1, KIND["loss"][0], LOSS_EC, lw=1.3, r=0.7, z=3)
    badge(c, 49.9, 9.5, 1)
    c.text(51.3, 9.5, r"Surface losses vs $S_{GT}$", fs=TF, weight="bold", ha="left")
    c.text(103.3, 7.0,
           r"$\mathcal{L}_{CD}$ (fine + 0.5 mid + 0.05 coarse) $+\ \mathcal{L}_{rad}"
           r"+0.15\,\mathcal{L}_{disp}+0.05\,\mathcal{L}_{lap}+0.02\,\mathcal{L}_{n}$",
           fs=BF)
    c.text(103.3, 4.9, r"trained together with $\mathcal{L}_{seq}$ · in training T is built "
           r"from the GT centerline (cleandata template)", fs=BF)
    c.text(103.3, 3.0, "radial Huber vs r* · Chamfer: Huber point-to-plane + 0.2 L2",
           fs=SF, color=GREY)


def main():
    info = load_info()
    c = Canvas()
    draw_header(c)
    zx = draw_stage1(c)
    draw_links(c, zx)
    draw_stage2(c, info, zx)
    c.fig.savefig(OUT + ".png", dpi=200)
    c.fig.savefig(OUT + ".pdf")
    c.fig.savefig(OUT + ".svg")
    print("wrote", OUT + ".{png,pdf,svg}")


if __name__ == "__main__":
    main()
