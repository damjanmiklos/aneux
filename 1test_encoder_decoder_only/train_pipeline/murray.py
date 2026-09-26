"""Murray's law prior on junction radii: r_parent^k ~ sum r_child^k.

Why this is not a plug-in formula
---------------------------------
* A junction is not a point. vmtk's GroupId split hands the whole bifurcation
  region ("blanked" run) to the daughters, so a daughter's first ~2 parent
  radii of arc length are the flare, not the daughter. Radii are read in
  windows set back from the split by a multiple of the parent radius, where
  the lumen is a tube again.
* Splits come in quick succession (ACom complexes, early branches): segments
  of 2-6 mm leave no room for a window clear of both flares. Such a segment is
  collapsed into its neighbours -- Murray is additive, so
  r_p^k = r_a^k + r_s^k and r_s^k = r_b^k + r_c^k give
  r_p^k = r_a^k + r_b^k + r_c^k -- instead of being guessed at.
* The aneurysm sac, a flare that reaches further than the set-back, or a
  stenosis all show up as a GT radius that is not flat inside the window.
  Such a window slides further from the split to the first tube-like
  position, or is dropped (judged on the GT, never on the prediction).
  A junction some branch of which cannot be measured is left out entirely:
  an incomplete sum of daughters would be a wrong constraint, not a weak one.
* Murray's law holds only roughly in cerebral arteries. The loss is a hinge
  in log-radius units with a band of max(tolerance, the GT's own deviation):
  it never pulls a reconstruction away from a GT that itself breaks the law,
  and only penalises a prediction that is less Murray-like than both.

The windows are computed on the CPU per case from cached fields
(`attach_murray_fields`, called from dataset._finalize_item), stored as a
per-vertex window id plus a small relation table, and the loss reads the
predicted radius (r_local + n . dx, the same radius the `rad` term uses) and
the GT radius (r_star) through the identical estimator.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

# Daughter starts sit 0-0.75 mm from their parent's end in the stage-4
# centerlines; unrelated tracts are several mm apart.
JUNCTION_SNAP_MM = 1.5
# Windows are set back from a split in units of the junction's parent radius.
# Measured on 1764 daughters / 867 parents (v11 cache, 300 cases): a
# daughter's GT radius is +42 % at the split, +10 % at 1.5 R, within 2 % from
# 2.25 R; a parent only dips (-6 %) inside the last 1 R.
CHILD_SETBACK_R = 2.25
PARENT_SETBACK_R = 1.0
# Window length WINDOW_R x the branch radius, clamped to [WINDOW_MIN_MM, WINDOW_MAX_MM].
WINDOW_R = 2.0
WINDOW_MIN_MM = 1.0
WINDOW_MAX_MM = 4.0
# A window that fails the GT shape test slides away from its split in steps of
# half a window, at most SLIDE_MAX_R parent radii past the set-back, and the
# first tube-like position is used (the split's extent varies per junction).
SLIDE_MAX_R = 2.0
# Keep clear of an inlet/outlet rim by RIM_MARGIN_R x the branch radius.
RIM_MARGIN_R = 0.5
# A window needs this many vertices with a valid, unambiguous GT radius ...
MIN_WINDOW_VERTS = 24
MIN_VALID_FRAC = 0.8
# ... and a GT lumen that is a tube there. Axial: spread of the ring medians
# of AXIAL_BINS slices along the window (flare, taper, a sac's shoulder),
# relative to the median. Bulge: share of vertices whose GT radius is more
# than BULGE_REL off the median (a sac or a side branch on part of the ring).
# A ring IQR would also reject oval vessels, which are fine.
AXIAL_BINS = 4
MAX_GT_AXIAL_REL = 0.15
BULGE_REL = 0.4
MAX_GT_BULGE_FRAC = 0.1
# GT window radius vs the template's own radius there: outside this is a sac
# or a stenosis, not the branch calibre.
GT_TEMPLATE_RATIO = (0.6, 1.6)
# Collapse at most this many unmeasurable generations, into at most this many children.
MAX_COLLAPSE_DEPTH = 2
MAX_CHILDREN = 4
# Trimmed mean: drop this fraction at each end inside a window.
TRIM_FRAC = 0.2

ROLE_PARENT = 0
ROLE_CHILD = 1


def _tract_geometry(cl_dense: np.ndarray, cl_tract_id: np.ndarray):
    n_t = int(cl_tract_id.max()) + 1 if cl_tract_id.size else 0
    lengths = np.zeros(n_t)
    starts = np.zeros((n_t, 3))
    ends = np.zeros((n_t, 3))
    for t in range(n_t):
        p = cl_dense[cl_tract_id == t, :3]
        if p.shape[0] == 0:
            lengths[t] = 0.0
            continue
        lengths[t] = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if p.shape[0] > 1 else 0.0
        starts[t] = p[0]
        ends[t] = p[-1]
    return n_t, lengths, starts, ends


def tract_tree(cl_dense: np.ndarray, cl_tract_id: np.ndarray, snap_mm: float = JUNCTION_SNAP_MM):
    """parent[t] (-1 for a root) from 'a daughter starts where its parent ends'."""
    n_t, lengths, starts, ends = _tract_geometry(cl_dense, cl_tract_id)
    parent = np.full(n_t, -1, dtype=np.int64)
    if n_t < 2:
        return parent, lengths
    d = np.linalg.norm(starts[:, None, :] - ends[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    for t in range(n_t):
        s = int(np.argmin(d[t]))
        if d[t, s] <= snap_mm and lengths[s] > 0:
            parent[t] = s
    # break any cycle (a mis-oriented tract) by cutting its weakest link
    for t in range(n_t):
        seen = []
        a = t
        while a >= 0 and a not in seen:
            seen.append(a)
            a = int(parent[a])
        if a >= 0:
            cyc = seen[seen.index(a):]
            cut = max(cyc, key=lambda i: d[i, parent[i]])
            parent[cut] = -1
    return parent, lengths


def _flatness(r: np.ndarray, s: np.ndarray, iv):
    med = float(np.median(r))
    edges = np.linspace(iv[0], iv[1], AXIAL_BINS + 1)
    ring = [np.median(r[(s >= a) & (s <= b)]) for a, b in zip(edges[:-1], edges[1:])
            if ((s >= a) & (s <= b)).sum() >= 3]
    axial = (max(ring) - min(ring)) / max(med, 1e-9) if len(ring) >= 2 else np.inf
    bulge = float(np.mean(np.abs(r - med) > BULGE_REL * med))
    return med, float(axial), bulge


def _trimmed_mean(vals: np.ndarray, frac: float = TRIM_FRAC) -> float:
    v = np.sort(vals)
    k = int(np.floor(frac * v.size))
    v = v[k: v.size - k] if v.size - 2 * k > 0 else v
    return float(v.mean())


def build_murray_relations(
    cl_dense,
    cl_tract_id,
    tract_id,
    u,
    r_local,
    r_star,
    r_star_valid,
    r_star_ambiguous=None,
    boundary_mask=None,
    return_debug: bool = False,
):
    """Per-vertex window id (-1 = none) and relation rows [relation, window, role].

    All inputs are one case's numpy arrays (fine level). Windows and relations
    depend only on the template, the centerline and the GT, so they are fixed
    per case and survive pose jitter and mirroring.
    """
    cl_dense = np.asarray(cl_dense, dtype=np.float64)
    cl_tract_id = np.asarray(cl_tract_id, dtype=np.int64).reshape(-1)
    tract_id = np.asarray(tract_id, dtype=np.int64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    r_local = np.asarray(r_local, dtype=np.float64).reshape(-1)
    r_star = np.asarray(r_star, dtype=np.float64).reshape(-1)
    ok = np.asarray(r_star_valid, dtype=bool).reshape(-1) & np.isfinite(r_star) & (r_star > 0)
    if r_star_ambiguous is not None:
        ok &= ~np.asarray(r_star_ambiguous, dtype=bool).reshape(-1)
    n = tract_id.shape[0]
    usable = np.ones(n, dtype=bool)
    if boundary_mask is not None:
        usable &= ~np.asarray(boundary_mask, dtype=bool).reshape(-1)

    win = np.full(n, -1, dtype=np.int64)
    empty_rel = np.zeros((0, 3), dtype=np.int64)
    debug = {"windows": [], "relations": [], "dropped": []}
    if cl_tract_id.size == 0 or n == 0:
        return (win, empty_rel, debug) if return_debug else (win, empty_rel)

    parent, lengths = tract_tree(cl_dense, cl_tract_id)
    n_t = parent.shape[0]
    children = [[] for _ in range(n_t)]
    for t in range(n_t):
        if parent[t] >= 0:
            children[int(parent[t])].append(t)
    s = u * lengths[np.clip(tract_id, 0, n_t - 1)]
    radius = np.zeros(n_t)
    for t in range(n_t):
        m = tract_id == t
        radius[t] = float(np.median(r_local[m])) if m.any() else 0.0


    # candidate interval on each tract for each role; a tract without room
    # for two separate windows measures one shared window for both roles
    def interval(t: int, role: int):
        """Candidate windows, nearest the split first, and the window's tag."""
        L = lengths[t]
        r = radius[t]
        if L <= 0 or r <= 0:
            return [], None
        w = float(np.clip(WINDOW_R * r, WINDOW_MIN_MM, WINDOW_MAX_MM))
        lo_free = CHILD_SETBACK_R * radius[int(parent[t])] if parent[t] >= 0 else RIM_MARGIN_R * r
        hi_free = L - (PARENT_SETBACK_R * r if children[t] else RIM_MARGIN_R * r)
        span = hi_free - lo_free
        if span < WINDOW_MIN_MM:
            return [], None
        if span < 2.0 * w:
            mid = 0.5 * (lo_free + hi_free)
            half = 0.5 * min(span, w)
            return [(mid - half, mid + half)], "shared"
        # slide limit in the junction's parent radius
        r_j = radius[int(parent[t])] if (role == ROLE_CHILD and parent[t] >= 0) else r
        reach = SLIDE_MAX_R * r_j
        out = []
        off = 0.0
        while off <= reach + 1e-9:
            if role == ROLE_CHILD:
                iv = (lo_free + off, lo_free + off + w)
                if iv[1] > hi_free:
                    break
            else:
                iv = (hi_free - off - w, hi_free - off)
                if iv[0] < lo_free:
                    break
            out.append(iv)
            off += 0.5 * w
        return out, role

    windows = {}  # (tract, role or "shared") -> window id or None

    def window(t: int, role: int):
        cands, tag = interval(t, role)
        if not cands:
            debug["dropped"].append((t, role, "no room"))
            return None
        key = (t, tag)
        if key in windows:
            return windows[key]
        windows[key] = None
        why = None
        for iv in cands:
            m = (tract_id == t) & (s >= iv[0]) & (s <= iv[1]) & usable
            n_in = int(m.sum())
            good = m & ok
            n_good = int(good.sum())
            if n_good < MIN_WINDOW_VERTS or n_good < MIN_VALID_FRAC * max(n_in, 1):
                why = f"verts {n_good}/{n_in}"
                continue
            med, axial, bulge = _flatness(r_star[good], s[good], iv)
            if axial > MAX_GT_AXIAL_REL or bulge > MAX_GT_BULGE_FRAC:
                why = f"gt shape axial {axial:.2f} bulge {bulge:.2f}"
                continue
            ratio = med / max(float(np.median(r_local[good])), 1e-9)
            if not (GT_TEMPLATE_RATIO[0] <= ratio <= GT_TEMPLATE_RATIO[1]):
                why = f"gt/template {ratio:.2f}"
                continue
            if bool((win[good] >= 0).any()):
                why = "overlap"
                continue
            wid = sum(1 for v in windows.values() if v is not None)
            win[good] = wid
            windows[key] = wid
            debug["windows"].append(dict(tract=t, role=tag, s=iv, n=n_good, slid=cands.index(iv),
                                         r_gt=_trimmed_mean(r_star[good]), axial=axial, bulge=bulge))
            return wid
        debug["dropped"].append((t, role, why))
        return None

    def frontier(t: int, depth: int):
        """Child windows standing for tract t at its parent's junction, or None."""
        w = window(t, ROLE_CHILD)
        if w is not None:
            return [w]
        if depth >= MAX_COLLAPSE_DEPTH or not children[t]:
            return None
        out = []
        for c in children[t]:
            f = frontier(c, depth + 1)
            if f is None:
                return None
            out += f
        return out

    rows = []
    rel = 0
    for p in range(n_t):
        if not children[p]:
            continue
        wp = window(p, ROLE_PARENT)
        if wp is None:
            continue
        kids = []
        for c in children[p]:
            f = frontier(c, 0)
            if f is None:
                kids = None
                break
            kids += f
        if not kids or len(kids) > MAX_CHILDREN:
            debug["dropped"].append((p, -1, f"relation children {None if kids is None else len(kids)}"))
            continue
        rows.append((rel, wp, ROLE_PARENT))
        rows.extend((rel, w, ROLE_CHILD) for w in kids)
        debug["relations"].append(dict(parent=p, wp=wp, kids=kids))
        rel += 1
    rel_arr = np.asarray(rows, dtype=np.int64).reshape(-1, 3) if rows else empty_rel
    # windows no relation uses would only cost time in the loss
    used = np.unique(rel_arr[:, 1]) if rel_arr.size else np.zeros(0, dtype=np.int64)
    remap = np.full(int(win.max()) + 2, -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    win = np.where(win >= 0, remap[np.clip(win, 0, None)], -1)
    if rel_arr.size:
        rel_arr[:, 1] = remap[rel_arr[:, 1]]
    if return_debug:
        return win, rel_arr, debug
    return win, rel_arr


def attach_murray_fields(data):
    """Add `murray_win` [N] and `murray_rel` [K, 3] to a cached graph (CPU).

    Always attaches both (empty when nothing is measurable), so graphs with and
    without junctions collate into one batch.
    """
    n = int(data.x.size(0)) if getattr(data, "x", None) is not None else int(data.num_nodes or 0)
    win = np.full(n, -1, dtype=np.int64)
    rel = np.zeros((0, 3), dtype=np.int64)
    need = ("cl_dense", "cl_tract_id", "tract_id", "u", "r_local", "r_star", "r_star_valid")
    if all(getattr(data, k, None) is not None for k in need):
        amb = getattr(data, "r_star_ambiguous", None)
        bnd = getattr(data, "boundary_mask", None)
        try:
            win, rel = build_murray_relations(
                data.cl_dense.numpy(), data.cl_tract_id.numpy(), data.tract_id.numpy(), data.u.numpy(),
                data.r_local.numpy(), data.r_star.numpy(), data.r_star_valid.numpy(),
                None if amb is None else amb.numpy(), None if bnd is None else bnd.numpy(),
            )
        except Exception:
            # a prior, not a requirement: a case it cannot read just has no relations
            win = np.full(n, -1, dtype=np.int64)
            rel = np.zeros((0, 3), dtype=np.int64)
    data.murray_win = torch.from_numpy(win)
    data.murray_rel = torch.from_numpy(rel)
    return data


def _window_trimmed_mean(values: Tensor, gid: Tensor, n_groups: int, frac: float = TRIM_FRAC) -> Tensor:
    """Per-group trimmed mean, differentiable in `values` (gid >= 0 only)."""
    v = values.float()
    # sort by (group, value): values are radii in mm, far below the 1e4 stride
    key = gid.double() * 1e4 + v.detach().double().clamp(-1e3, 1e3)
    order = torch.argsort(key)
    g_sorted = gid[order]
    v_sorted = v[order]
    count = torch.zeros(n_groups, device=v.device, dtype=torch.long).index_add_(
        0, gid, torch.ones_like(gid))
    start = torch.cumsum(count, 0) - count
    rank = torch.arange(order.numel(), device=v.device) - start[g_sorted]
    k = torch.floor(frac * count.float()).long()
    keep_hi = count - k
    keep = (rank >= k[g_sorted]) & (rank < keep_hi[g_sorted])
    tot = torch.zeros(n_groups, device=v.device, dtype=v.dtype).index_add_(
        0, g_sorted[keep], v_sorted[keep])
    cnt = torch.zeros(n_groups, device=v.device, dtype=v.dtype).index_add_(
        0, g_sorted[keep], torch.ones_like(v_sorted[keep]))
    return tot / cnt.clamp_min(1.0)


def murray_errors(r_pred: Tensor, r_gt: Tensor, win: Tensor, win_batch: Tensor,
                  rel: Tensor, rel_batch: Tensor, exponent: float):
    """Log-radius Murray error e = log(sum r_c^k / r_p^k) / k for pred and GT.

    `win` is the per-vertex window id local to its graph, `rel` the rows
    [relation, window, role] local to theirs; *_batch give the graph index.
    Returns (e_pred, e_gt), one entry per relation in the batch.
    """
    device = r_pred.device
    if rel.numel() == 0 or not bool((win >= 0).any()):
        z = r_pred.new_zeros(0)
        return z, z
    rel = rel.to(device)
    rel_batch = rel_batch.to(device)
    win = win.to(device)
    win_batch = win_batch.to(device)
    n_graphs = int(max(int(win_batch.max()), int(rel_batch.max()))) + 1
    # globalise window ids: graph-major offsets
    w_per = torch.zeros(n_graphs, device=device, dtype=torch.long)
    w_per.scatter_reduce_(0, rel_batch, rel[:, 1] + 1, reduce="amax")
    w_off = torch.cumsum(w_per, 0) - w_per
    r_per = torch.zeros(n_graphs, device=device, dtype=torch.long)
    r_per.scatter_reduce_(0, rel_batch, rel[:, 0] + 1, reduce="amax")
    r_off = torch.cumsum(r_per, 0) - r_per
    n_win = int(w_per.sum())
    n_rel = int(r_per.sum())
    sel = win >= 0
    gid = win[sel] + w_off[win_batch[sel]]
    rp = _window_trimmed_mean(r_pred.reshape(-1)[sel].clamp_min(1e-3), gid, n_win)
    rg = _window_trimmed_mean(r_gt.reshape(-1)[sel].clamp_min(1e-3), gid, n_win)
    wg = rel[:, 1] + w_off[rel_batch]
    rg_id = rel[:, 0] + r_off[rel_batch]
    child = rel[:, 2] == ROLE_CHILD
    k = float(exponent)

    def err(r_win):
        rk = r_win[wg].pow(k)
        s_child = torch.zeros(n_rel, device=device, dtype=rk.dtype).index_add_(
            0, rg_id[child], rk[child])
        s_par = torch.zeros(n_rel, device=device, dtype=rk.dtype).index_add_(
            0, rg_id[~child], rk[~child])
        return (torch.log(s_child.clamp_min(1e-12)) - torch.log(s_par.clamp_min(1e-12))) / k

    return err(rp), err(rg.detach())


def murray_loss(e_pred: Tensor, e_gt: Tensor, tolerance: float, huber_delta: float = 0.1) -> Tensor:
    """Hinge beyond max(tolerance, |e_gt|), Huber above it; mean over relations."""
    if e_pred.numel() == 0:
        return e_pred.new_zeros(())
    band = torch.clamp(e_gt.abs(), min=float(tolerance))
    excess = torch.relu(e_pred.abs() - band)
    d = float(huber_delta)
    hub = torch.where(excess < d, 0.5 * excess.pow(2) / d, excess - 0.5 * d)
    return hub.mean()
