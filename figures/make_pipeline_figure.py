"""Paper-style overview of the Stage-2 training pipeline (fullscreen, 16:9).

Layout plan (canvas 160 x 90 units, 1 unit = 1/8 inch, figure 20 x 11.25 in):

  (a) Encoder, top row, left to right
      S_GT render -> 16 384-point input cloud (coloured by SDF to T)
      -> stem MLP -> four set-abstraction stages (FPS centre renders above)
      -> centerline latent head (pool to tokens, token embedding,
         3-layer transformer) -> mu, log sigma^2
  (b) Latent, middle row, right to left (the flow wraps down from (a))
      mu, log sigma^2 -> tract mixer -> reparameterisation -> Z grid;
      the centerline C and its 2 mm tokens sit on the left of the row,
      and a token-table line feeds the encoder head.
  (c) Decoder, bottom row: three stacked level rows (coarse, mid, fine),
      each left to right: scaffold render -> cross-attention to Z ->
      skip / geometry fuse (and positional self-attention on coarse)
      -> 6 SplineConvs -> displacement head -> rim-plane projection -> X^.
      Z is a bus down the left; kNN upsampling runs in the gaps between rows.
      Template T and the scaffold construction box feed every level.
  (d) Objective, right-hand column: total loss, target S_GT, legend,
      then the KL, regulariser, radial and Chamfer terms with arrows from
      the mixer (KL) and from the fine output X^.

Colour code: blue encoder, yellow attention, orange latent / stochastic,
green decoder (dark green SplineConv), dashed white fixed geometric op,
red loss. Every render comes from one real cached training graph, made by
render_pipeline_assets.py.

    python figures/make_pipeline_figure.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "pipeline_assets")
OUT = os.path.join(HERE, "stage2_pipeline")

W, H = 160.0, 90.0
UNIT_IN = 0.125

plt.rcParams.update({
    "font.family": "Arial",
    "mathtext.fontset": "stixsans",
    "font.size": 10,
})

KIND = {
    "enc": ("#dce8f5", "#3c6e9f"),
    "attn": ("#fdf0c4", "#a8841c"),
    "lat": ("#fde2cc", "#c0692a"),
    "dec": ("#dcefdb", "#3d8b4b"),
    "conv": ("#b8deb5", "#2c7438"),
    "op": ("#ffffff", "#6b6b6b"),
    "loss": ("#f7dcdc", "#b0474b"),
}
PANEL = {
    "a": ("#f5f8fc", "#c3d3e5"),
    "b": ("#fdf8f2", "#e8d3bd"),
    "c": ("#f5faf4", "#c9e0c6"),
    "d": ("#fcf5f5", "#e8cccc"),
}
TRACT_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
                "#76B7B2", "#EDC948", "#9C755F", "#FF9DA7", "#BAB0AC"]
INK = "#2b2b2b"
GREY = "#6a6a6a"
Z_COL = "#c0692a"
CUBE_DX, CUBE_DY = 1.3, 0.9


def line_units(fs):
    return fs * 1.38 / 72.0 / UNIT_IN


class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(W * UNIT_IN, H * UNIT_IN))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

    # -- primitives -------------------------------------------------------
    def rbox(self, x, y, w, h, fc, ec, lw=1.2, ls="-", r=0.7, z=2, alpha=1.0):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                           fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, alpha=alpha)
        self.ax.add_patch(p)
        return p

    def text(self, x, y, s, fs=10, color=INK, ha="center", va="center", weight="normal",
             z=6, bg=None, **kw):
        bbox = None
        if bg is not None:
            bbox = dict(boxstyle="square,pad=0.12", fc=bg, ec="none")
        return self.ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va,
                            fontweight=weight, zorder=z, bbox=bbox, **kw)

    def block(self, x, y, w, h, kind, title, lines=(), tfs=10.5, fs=8.9, rot=0):
        fc, ec = KIND[kind]
        ls = (0, (4, 2.5)) if kind == "op" else "-"
        self.rbox(x, y, w, h, fc, ec, lw=1.3, ls=ls, z=3)
        if rot:
            self.text(x + w / 2, y + h / 2, title, fs=tfs, weight="bold", rotation=rot)
            return
        heights = [line_units(tfs)] + [line_units(fs)] * len(lines)
        total = sum(heights)
        cy = y + h / 2 + total / 2
        for i, (s, hh) in enumerate(zip([title, *lines], heights)):
            cy -= hh / 2
            self.text(x + w / 2, cy, s, fs=tfs if i == 0 else fs,
                      weight="bold" if i == 0 else "normal",
                      color=INK if i == 0 else "#333333")
            cy -= hh / 2

    def cuboid(self, x, y, w, h, fc=None, ec=None, z=3):
        """Pseudo-3-D feature block: front face (w x h) plus top and side faces."""
        fc = fc or KIND["enc"][0]
        ec = ec or KIND["enc"][1]
        base = np.array(to_rgb(fc))
        top = tuple(np.clip(base + 0.06, 0, 1))
        side = tuple(base * 0.86)
        dx, dy = CUBE_DX, CUBE_DY
        faces = [
            ([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], fc),
            ([(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)], top),
            ([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)], side),
        ]
        for verts, col in faces:
            self.ax.add_patch(Polygon(verts, closed=True, fc=col, ec=ec, lw=1.0, zorder=z,
                                      joinstyle="round"))

    def arrow(self, pts, color=INK, lw=1.3, ls="-", head=True, z=4, bridge=False, ms=11):
        verts = [tuple(p) for p in pts]
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
        style = "-|>" if head else "-"
        a = FancyArrowPatch(path=Path(verts, codes), arrowstyle=style, mutation_scale=ms,
                            color=color, lw=lw, ls=ls, zorder=z, shrinkA=0, shrinkB=0,
                            joinstyle="miter", capstyle="butt")
        if bridge:
            a.set_path_effects([pe.Stroke(linewidth=lw + 3.2, foreground="white"), pe.Normal()])
        self.ax.add_patch(a)
        return a

    def image(self, name, x, y, w, h, anchor="center", z=3):
        im = plt.imread(os.path.join(ASSETS, f"{name}.png"))
        ih, iw = im.shape[:2]
        ar = iw / ih
        if w / h > ar:
            ww, hh = h * ar, h
        else:
            ww, hh = w, w / ar
        x0 = x + (w - ww) / 2
        y0 = {"center": y + (h - hh) / 2, "bottom": y, "top": y + h - hh}[anchor]
        self.ax.imshow(im, extent=(x0, x0 + ww, y0, y0 + hh), zorder=z, interpolation="lanczos")
        return x0, y0, ww, hh

    def panel(self, key, x, y, w, h, label):
        fc, ec = PANEL[key]
        self.rbox(x, y, w, h, fc, ec, lw=1.0, r=1.2, z=0)
        self.text(x + 1.2, y + h - 1.25, label, fs=14, weight="bold", ha="left", z=5)


def load_info():
    with open(os.path.join(ASSETS, "case_info.json")) as fh:
        return json.load(fh)


def fmt(n):
    return f"{n:,}".replace(",", " ")


# ---------------------------------------------------------------------------
def draw_encoder(c, info):
    c.panel("a", 1, 55, 126.5, 28.5,
            r"(a) Encoder  $q_\phi(Z \mid S_{GT},\,\mathcal{C})$")
    yc = 70.2

    # S_GT -> x_true
    c.image("gt_surface", 2.5, 64.4, 15, 11.6)
    c.text(10, 63.2, r"$S_{GT}$  ground truth", fs=10.5, weight="bold")
    c.text(10, 61.6, f"{fmt(info['n_gt'])} vertices", fs=8.9, color=GREY)
    c.text(10, 60.2, "(uniformly remeshed)", fs=8.9, color=GREY)

    c.arrow([(17.9, yc), (21.2, yc)])
    c.text(19.55, yc + 2.3, "resample\nper epoch", fs=7.8, color=GREY, linespacing=1.1)
    c.text(19.55, yc - 2.5, "FPS, 25 %\nfar from $\\mathcal{C}$", fs=7.8, color=GREY,
           linespacing=1.1)

    c.image("x_true", 21.5, 64.4, 15.5, 11.6)
    c.text(29.25, 63.2, r"$x_{true}\in\mathbb{R}^{16384\times 7}$", fs=10.5)
    c.text(29.25, 61.6, "xyz · normal · SDF to T", fs=8.9, color=GREY)
    lim = info["sdf_lim"]
    cm = LinearSegmentedColormap.from_list("sdf", ["#2166ac", "#a9a9a9", "#b2182b"])
    grad = np.linspace(0, 1, 256)[None, :]
    c.ax.imshow(grad, extent=(24.5, 34, 59.3, 60.0), cmap=cm, aspect="auto", zorder=4)
    c.ax.add_patch(Rectangle((24.5, 59.3), 9.5, 0.7, fc="none", ec=GREY, lw=0.5, zorder=5))
    c.text(24.5, 58.3, f"−{lim:.1f}", fs=7.5, color=GREY)
    c.text(29.25, 58.3, "0", fs=7.5, color=GREY)
    c.text(34.0, 58.3, f"+{lim:.1f} mm", fs=7.5, color=GREY)

    # feature blocks: height ~ points^0.3, width ~ sqrt(channels), so the
    # 256x drop in points and 8x rise in channels both stay visible
    def block_h(n):
        return 0.95 * n ** 0.3

    def block_w(ch):
        return 0.30 * ch ** 0.5

    yc_f = 66.0
    tensors = [
        (16384, 32, None),
        (1024, 64, "fps_1024"),
        (256, 128, "fps_256"),
        (64, 128, "fps_64"),
        (64, 256, "fps_64"),
    ]
    ops = [
        ("Stem", ["MLP 7 → 32"]),
        ("SA$_1$", ["FPS → 1024", "r = 1.5 mm"]),
        ("SA$_2$", ["FPS → 256", "r = 3 mm"]),
        ("SA$_3$", ["FPS → 64", "r = 6 mm"]),
        ("SA$_4$", ["FPS → 64", "r = 12 mm"]),
    ]
    x_lo, x_hi = 37.3, 92.6
    solid = sum(block_w(ch) + CUBE_DX for _, ch, _ in tensors)
    gap = (x_hi - x_lo - solid) / len(tensors)
    x = x_lo
    for (n, ch, img), (name, lines) in zip(tensors, ops):
        c.arrow([(x, yc_f), (x + gap, yc_f)])
        c.text(x + gap / 2, yc_f + 1.25, name, fs=9.4, weight="bold")
        for j, s_ in enumerate(lines):
            c.text(x + gap / 2, yc_f - 1.25 - 1.3 * j, s_, fs=7.9, color="#333333")
        x += gap
        w_, h_ = block_w(ch), block_h(n)
        c.cuboid(x, yc_f - h_ / 2, w_, h_)
        cx = x + (w_ + CUBE_DX) / 2
        c.text(cx, yc_f - h_ / 2 - 1.25, f"{n} × {ch}", fs=8.3, color=GREY)
        if img:
            c.image(img, cx - 5.0, 71.2, 10.0, 7.4, anchor="bottom")
        x += w_ + CUBE_DX
    prev_right = x
    c.text(68.75, 81.9, "PointNeXt hierarchy  (set centres after FPS, shown on the GT)",
           fs=8.9, color=GREY)
    c.text(51.5, 56.2, "every SA stage is followed by 2 InvResMLP blocks · "
           "block height ~ points$^{0.3}$, width ~ channels$^{0.5}$",
           fs=7.9, color=GREY, ha="left")

    # centerline latent head
    hx, hw = 95.0, 17.5
    c.rbox(hx, 57.3, hw, 23.3, "#eef3f9", KIND["enc"][1], lw=1.1, r=0.9, z=2)
    c.text(hx + hw / 2, 79.1, "Centerline latent head", fs=10.5, weight="bold")
    c.block(hx + 1, 58.3, hw - 2, 5.4, "op", "Pool to tokens",
            ["centre → nearest token", "(same tract, max-pool)"], tfs=9.6, fs=8.3)
    c.block(hx + 1, 64.7, hw - 2, 4.7, "enc", "Token embedding",
            [r"$\oplus$ $\gamma(u)$, tract emb., depth"], tfs=9.6, fs=8.3)
    c.block(hx + 1, 70.4, hw - 2, 6.9, "attn", "Transformer × 3",
            ["token self-attention", "4 heads, 256-d, pad mask"], tfs=9.6, fs=8.3)
    c.arrow([(hx + hw / 2, 63.7), (hx + hw / 2, 64.7)], ms=8)
    c.arrow([(hx + hw / 2, 69.4), (hx + hw / 2, 70.4)], ms=8)
    c.arrow([(prev_right, 66), (93.8, 66), (93.8, 61.0), (hx + 1, 61.0)])

    # mu, log sigma^2
    c.arrow([(hx + hw - 1, 73.85), (113.9, 73.85), (113.9, 75.9), (115.2, 75.9)])
    c.arrow([(113.9, 73.85), (113.9, 64.3), (115.2, 64.3)])
    c.block(115.2, 73.0, 10.6, 5.8, "lat", r"$\mu$", ["128 × 16"], tfs=12)
    c.block(115.2, 61.4, 10.6, 5.8, "lat", r"$\log\sigma^2$",
            [r"$\sigma\in[0.1,\,e]$ (soft)"], tfs=11.5, fs=8.3)


def draw_latent(c, info):
    c.panel("b", 1, 37.6, 126.5, 16.6, "(b) Tree-valued latent")

    # centerline + tokens
    c.image("centerline", 2.5, 38.4, 14, 11.2)
    c.text(22.4, 48.3, r"$\mathcal{C}$  centerline", fs=10.5, weight="bold")
    c.text(22.4, 46.7, f"{info['n_tracts']} tracts (GroupIds)", fs=8.9, color=GREY)
    c.text(22.4, 45.3, "canonical ICA pose", fs=8.9, color=GREY)
    c.arrow([(27.8, 42.4), (37.8, 42.4)])
    c.text(32.8, 43.6, "token every 2 mm", fs=8.3, color=GREY)
    c.text(32.8, 41.2, "(u, tract, xyz)", fs=8.3, color=GREY)
    c.image("tokens", 38.0, 38.3, 15.5, 11.6)
    c.text(55.4, 46.6, "≡", fs=17, color=GREY)

    # token table up to the encoder head
    c.arrow([(46.0, 49.6), (46.0, 54.55), (103.75, 54.55), (103.75, 58.3)],
            color=GREY, lw=1.1, ls=(0, (4, 2)))
    c.text(74.0, 54.55, "token table → encoder pooling (and decoder attention masks)",
           fs=8.3, color=GREY, bg=PANEL["b"][0])

    # Z grid: columns = tokens grouped by tract, rows = 16 latent dims
    tok = np.asarray(info["tok_tract"])
    order = np.argsort(tok, kind="stable")
    n = len(order)
    gx, gy, gw, gh = 57.3, 43.4, 28.6, 7.1
    cw, chh = gw / n, gh / 16
    for j, t in enumerate(tok[order]):
        base = np.array(to_rgb(TRACT_COLORS[t % len(TRACT_COLORS)]))
        col = 1 - 0.45 * (1 - base)
        c.ax.add_patch(Rectangle((gx + j * cw, gy), cw, gh, fc=col, ec="none", zorder=3))
        c.ax.add_patch(Rectangle((gx + j * cw, gy + gh), cw, 0.6,
                                 fc=TRACT_COLORS[t % len(TRACT_COLORS)], ec="none", zorder=3))
    for i in range(17):
        c.ax.plot([gx, gx + gw], [gy + i * chh] * 2, color="white", lw=0.45, zorder=4)
    for j in range(n + 1):
        c.ax.plot([gx + j * cw] * 2, [gy, gy + gh], color="white", lw=0.3, zorder=4)
    c.ax.add_patch(Rectangle((gx, gy), gw, gh + 0.6, fc="none", ec=Z_COL, lw=1.3, zorder=5))
    c.text(gx + gw / 2, 52.4, r"$Z\in\mathbb{R}^{L\times 16}$ : one 16-d code per centerline token",
           fs=10.2)
    c.text(gx, 42.2, f"{n} valid tokens, grouped by tract ({info['latent_len']} slots, rest masked)",
           fs=8.3, color=GREY, ha="left")

    # mixer and reparameterisation
    c.block(105.0, 42.6, 20.6, 8.9, "attn", "Tract mixer",
            [r"self-attention on $\mu$ within a tract", "±2 tokens · ALiBi · gate ≤ 0.5",
             r"($\log\sigma^2$ passes through)"], fs=8.3)
    c.arrow([(125.8, 75.9), (126.75, 75.9), (126.75, 47.05), (125.6, 47.05)])
    c.arrow([(120.5, 61.4), (120.5, 51.5)])
    c.block(89.2, 42.6, 13.0, 8.9, "lat", "Reparameterise",
            [r"$z=\tilde{\mu}+\sigma\odot\varepsilon$", r"$\varepsilon\sim\mathcal{N}(0,I)$",
             r"($z=\tilde{\mu}$ at inference)"], fs=8.9)
    c.arrow([(105.0, 47.05), (102.2, 47.05)])
    c.arrow([(89.2, 47.05), (86.1, 47.05)], color=Z_COL)

    # KL tap
    c.arrow([(115.3, 42.6), (115.3, 39.3), (130.0, 39.3)], color=KIND["loss"][1], lw=1.2)
    c.text(121.5, 40.2, r"$\tilde{\mu},\ \log\sigma^2$", fs=8.9, color=KIND["loss"][1])


def draw_decoder(c, info):
    c.panel("c", 1, 1, 126.5, 35.8,
            r"(c) Decoder  $p_\theta(\hat{S} \mid Z,\,T,\,\mathcal{C})$")

    # template and scaffold construction
    c.image("template", 2.2, 22.6, 14.8, 11.3)
    c.text(9.6, 21.5, "T  template mesh", fs=10.5, weight="bold")
    c.text(9.6, 20.0, f"{fmt(info['n_fine'])} vertices", fs=8.9, color=GREY)
    c.arrow([(9.6, 19.2), (9.6, 17.9)])
    c.block(2.0, 2.0, 16.4, 15.9, "op", "Scaffold construction",
            ["decimate T: 8 % · 25 % · 100 %",
             r"project onto $\mathcal{C}$ → $(u,\theta)$, tract",
             "Bishop frame (t, n, b)",
             r"$r_{local}$, κ, τ, $d_{ost}$",
             "rim (ostium) planes",
             "kNN (k = 3) upsample maps"], tfs=9.8, fs=8.3)
    # centerline into the scaffold box
    c.arrow([(16.6, 38.8), (17.6, 38.8), (17.6, 17.9)], color=GREY, lw=1.1, ls=(0, (4, 2)))
    c.text(18.35, 28.0, r"$\mathcal{C}$", fs=9, color=GREY)

    rows = [
        ("coarse", "Coarse", info["n_coarse"], 29.45, "c"),
        ("mid", "Mid", info["n_mid"], 17.85, "m"),
        ("fine", "Fine", info["n_fine"], 6.25, "f"),
    ]
    rh = 9.5
    # trunk from scaffold box to the three scaffold renders
    c.arrow([(18.4, 10.0), (20.3, 10.0)], head=False)
    c.arrow([(20.3, 6.25), (20.3, 29.45)], head=False)
    for _, _, _, yc, _ in rows:
        c.arrow([(20.3, yc), (22.2, yc)], ms=8)

    bx = 34.4  # Z bus
    for key, name, n, yc, sub in rows:
        y0 = yc - rh / 2
        c.rbox(21.6, y0, 105.0, rh, "#ffffff", PANEL["c"][1], lw=0.9, r=0.8, z=1, alpha=0.85)
        c.text(22.4, y0 + rh - 0.95, f"{name} · {fmt(n)} v.", fs=9.4, weight="bold", ha="left")
        c.image(f"level_{key}", 22.2, y0 + 0.25, 10.4, rh - 2.1, anchor="bottom")
        bh = 6.9
        by = yc - bh / 2 - 0.35
        c.arrow([(32.6, yc - 0.35), (36.2, yc - 0.35)])
        c.block(36.2, by, 12.4, bh, "attn", "Cross-attn → Z",
                [r"Q: $\gamma(u),\gamma(\theta)$", "K, V: 5 nearest tokens"], tfs=9.8, fs=8.3)
        if key == "coarse":
            c.block(50.6, by, 9.6, bh, "dec", "Geometry",
                    [r"$\oplus$ $r_{local}$, κ,", r"τ, $d_{ost}$"], tfs=9.8, fs=8.3)
            c.block(62.2, by, 11.8, bh, "attn", "Pos. self-attn",
                    ["4 heads · ±4 rings", "cross-tract at ostia"], tfs=9.8, fs=8.3)
        else:
            c.block(50.6, by, 9.6, bh, "dec", "Skip",
                    [r"+ $\sigma(\alpha)\,h\!\uparrow$", r"+ Lin$(\Delta x\!\uparrow)$"],
                    tfs=9.8, fs=8.3)
            c.block(62.2, by, 11.8, bh, "dec", "Geometry",
                    [r"$\oplus$ $r_{local}$, κ, τ, $d_{ost}$", "→ Linear"], tfs=9.8, fs=8.3)
        c.block(76.0, by, 13.6, bh, "conv", "6 × Res-SplineConv",
                [r"pseudo-coords $(\Delta u,\Delta\theta,$ kind$)$", "kernel 5×5×2, mean aggr."],
                tfs=9.8, fs=8.0)
        if key == "coarse":
            c.block(91.6, by, 13.8, bh, "dec", "Free head",
                    [r"$\Delta x\in\mathbb{R}^3$ per vertex", "(zero-init)"], tfs=9.8, fs=8.3)
        else:
            c.block(91.6, by, 13.8, bh, "dec", "Decoupled head",
                    [r"$\Delta r\,n+\Delta s\,(t,b)$",
                     r"$\Delta r\geq-0.8r_{loc}$, $|\Delta s|$ capped"], tfs=9.8, fs=7.9)
        c.block(107.4, by, 8.6, bh, "op", "Rim-plane",
                ["projection", "(ostia stay", "in cut plane)"], tfs=9.3, fs=7.9)
        for xa, xb in ((48.6, 50.6), (60.2, 62.2), (74.0, 76.0), (89.6, 91.6), (105.4, 107.4)):
            c.arrow([(xa, yc - 0.35), (xb, yc - 0.35)], ms=8)
        c.arrow([(116.0, yc - 0.35), (117.8, yc - 0.35)], ms=8)
        c.text(121.9, yc + 1.2, rf"$\hat{{X}}_{sub}$", fs=13)
        c.text(121.9, yc - 1.6, f"{fmt(n)} × 3", fs=8.3, color=GREY)

    # kNN upsampling between levels
    for (_, _, _, ya, _), (_, _, _, yb, _) in zip(rows[:-1], rows[1:]):
        gap = (ya - rh / 2 + yb + rh / 2) / 2
        c.arrow([(121.9, ya - 2.4), (121.9, gap), (55.4, gap), (55.4, yb + 3.1)],
                color=KIND["dec"][1], lw=1.3)
        c.text(88.0, gap, r"kNN upsample (k = 3):  $\Delta x\!\uparrow$ and gated $h\!\uparrow$",
               fs=8.3, color=KIND["dec"][1], bg=PANEL["c"][0])

    # Z bus: from the grid, along the gap between (b) and (c), down the left
    ztop = 37.15
    c.arrow([(83.4, 43.4), (83.4, ztop), (bx, ztop), (bx, 12.05)],
            color=Z_COL, lw=1.6, head=False, bridge=True, z=7)
    c.arrow([(42.4, ztop), (42.4, 29.45 + 3.1)], color=Z_COL, lw=1.6, z=7)
    for (_, _, _, ya, _), (_, _, _, yb, _) in zip(rows[:-1], rows[1:]):
        gap = (ya - rh / 2 + yb + rh / 2) / 2
        c.arrow([(bx, gap), (42.4, gap), (42.4, yb + 3.1)], color=Z_COL, lw=1.6, z=7)
    c.text(60.0, ztop, "Z  (keys and values at every level)", fs=8.5, color=Z_COL,
           weight="bold", bg="white", z=8)


def draw_objective(c, info):
    c.panel("d", 128.6, 1, 30.4, 82.5, "(d) Training objective")
    x0 = 130.0
    c.text(x0, 79.0, r"$\mathcal{L}=\mathcal{L}_{CD}+\mathcal{L}_{rad}+\lambda_{KL}\,\beta\,\mathcal{L}_{KL}$",
           fs=11.5, ha="left")
    c.text(x0 + 1.5, 76.8,
           r"$+\,0.15\,\mathcal{L}_{disp}+0.05\,\mathcal{L}_{lap}+0.02\,\mathcal{L}_{n}$",
           fs=11.5, ha="left")
    c.text(x0, 74.6, "AdamW 2·10$^{-4}$, cosine · EMA 0.993", fs=8.3, color=GREY, ha="left")
    c.text(x0, 73.3, "batch 4 × 8 accum. · 200 epochs", fs=8.3, color=GREY, ha="left")

    c.image("gt_surface", 133.0, 60.4, 20.0, 11.8)
    c.text(143.0, 59.4, r"target $S_{GT}$", fs=10, weight="bold")

    # legend
    ly = 57.4
    c.text(x0, ly, "Legend", fs=9.8, weight="bold", ha="left")
    entries = [
        ("enc", "encoder layer (learned)"),
        ("attn", "attention block"),
        ("lat", "latent / stochastic"),
        ("dec", "decoder layer (learned)"),
        ("conv", "spline convolution"),
        ("op", "fixed geometric operation"),
        ("loss", "loss term"),
    ]
    for i, (k, lab) in enumerate(entries):
        yy = ly - 1.6 * (i + 1)
        fc, ec = KIND[k]
        c.rbox(x0, yy - 0.55, 2.6, 1.1, fc, ec, lw=1.0, ls=(0, (3, 1.6)) if k == "op" else "-",
               r=0.25, z=3)
        c.text(x0 + 3.4, yy, lab, fs=8.6, ha="left")
    yy = ly - 1.6 * 8
    c.ax.plot([x0, x0 + 2.6], [yy, yy], color=Z_COL, lw=1.6, zorder=4)
    c.text(x0 + 3.4, yy, "latent Z broadcast", fs=8.6, ha="left")

    lw = 26.3
    c.block(x0, 35.6, lw, 6.6, "loss", r"$\mathcal{L}_{KL}$  per-token KL to $\mathcal{N}(0,I)$",
            ["mean over valid tokens", r"$\beta$ by GECO, target 12 nats / token"], tfs=9.6, fs=8.3)
    c.block(x0, 24.6, lw, 9.3, "loss", "Regularisers on the mesh",
            [r"$\mathcal{L}_{disp}$  Dirichlet energy of $(\Delta r,\Delta s)$",
             r"$\mathcal{L}_{lap}$  Laplacian smoothing",
             r"$\mathcal{L}_{n}$  normal consistency",
             "(fold, stretch: logged, weight 0)"], tfs=9.6, fs=8.3)
    c.block(x0, 14.8, lw, 8.1, "loss", r"$\mathcal{L}_{rad}$  radial Huber",
            [r"composed radius vs ray-cast $r^{*}$", "fine + 0.5 · mid"], tfs=9.6, fs=8.3)
    c.block(x0, 2.2, lw, 11.0, "loss", r"$\mathcal{L}_{CD}$  weighted Chamfer",
            ["Huber point-to-plane + 0.2 · L2",
             "fine + 0.5 · mid + 0.05 · coarse",
             r"weights from distance to $\mathcal{C}$ (cap 4)",
             "16 384 area samples of " + r"$\hat{X}$"], tfs=9.6, fs=8.3)

    # X^ into CD, rad, regularisers
    c.arrow([(126.1, 5.9), (x0, 5.9)])
    c.arrow([(129.2, 5.9), (129.2, 29.25), (x0, 29.25)], ms=8)
    c.arrow([(129.2, 18.85), (x0, 18.85)], ms=8)
    c.text(128.0, 7.3, r"$\hat{X}$", fs=10)
    # S_GT into CD and rad
    xr = x0 + lw + 1.3
    c.arrow([(153.2, 66.3), (xr, 66.3), (xr, 7.7), (x0 + lw, 7.7)], color=GREY, lw=1.1)
    c.arrow([(xr, 18.85), (x0 + lw, 18.85)], color=GREY, lw=1.1, ms=8)


def main():
    info = load_info()
    c = Canvas()
    c.text(2.0, 88.0, "Stage-2 geometry VAE: training pipeline", fs=19, weight="bold", ha="left")
    c.text(2.0, 85.4,
           "Hierarchical PointNeXt encoder → tree-valued centerline latent Z → progressive "
           "SplineConv decoder that deforms the template mesh.   "
           f"Renders: case {info['case'].split('_')[0]} from cleandata, canonical pose, mm.",
           fs=10.5, color=GREY, ha="left")
    draw_encoder(c, info)
    draw_latent(c, info)
    draw_decoder(c, info)
    draw_objective(c, info)
    c.fig.savefig(OUT + ".png", dpi=200)
    c.fig.savefig(OUT + ".pdf")
    c.fig.savefig(OUT + ".svg")
    print("wrote", OUT + ".{png,pdf,svg}")


if __name__ == "__main__":
    main()
