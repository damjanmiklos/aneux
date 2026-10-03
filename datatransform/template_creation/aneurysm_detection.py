"""Aneurysm detection on a vessel wall.

Numpy and SciPy only. The vessel-aneurysm remesher and the training cache both
call this, so the sac label cannot drift between them. ``detect_aneurysm``
raises ``AneurysmDetectionError`` when no sac is found; the remesher turns
that into its own quality error.
"""
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

R_FLOOR_MM = 0.35
LOBE_MED_MM = 1.5
LOBE_TOUCH_MM = 0.4
RIM_EXCLUDE_R = 1.5
SAC_REL_CAP = 2.5
SAC_CAP_SCALE = 2.0
SAC_SWITCH_MARGIN = 1.5


class AneurysmDetectionError(RuntimeError):
    """No compact sac on this wall."""


def _edge_table(faces):
    """Unique edges (sorted pairs, lexicographic), per-half-edge index, use counts.

    Half-edges are faces' (0,1), (1,2), (2,0) blocks, in that order. Edges are
    keyed as one int64 each, which sorts like the pairs and is several times
    faster than a row-wise unique.
    """
    faces = np.asarray(faces, dtype=np.int64)
    e = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    e.sort(axis=1)
    if len(e) == 0:
        return np.zeros((0, 2), np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    n = int(e[:, 1].max()) + 1
    key, inv, cnt = np.unique(e[:, 0] * n + e[:, 1], return_inverse=True, return_counts=True)
    uniq = np.stack((key // n, key % n), axis=1)
    return uniq, inv.reshape(-1), cnt


def _dedup_centerline(cpts, rad, lines, tol=0.14):
    """One node per blob of duplicated tract samples, and the unique edges."""
    cpts = np.asarray(cpts, dtype=np.float64)
    rad = np.asarray(rad, dtype=np.float64).reshape(-1)
    key = np.round(cpts / tol).astype(np.int64)
    _uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = int(inv.max()) + 1
    up = np.zeros((n, 3), dtype=np.float64)
    np.add.at(up, inv, cpts)
    up /= np.bincount(inv, minlength=n)[:, None]
    ur = np.full(n, -1.0, dtype=np.float64)
    np.maximum.at(ur, inv, rad)
    ur = np.where(ur < 0.0, R_FLOOR_MM, ur)
    raw = np.asarray(lines, dtype=np.int64)
    edges = []
    offset = 0
    while offset < len(raw):
        count = int(raw[offset])
        ids = raw[offset + 1: offset + 1 + count]
        offset += count + 1
        if count < 2:
            continue
        m = inv[ids]
        keep = m[:-1] != m[1:]
        if np.any(keep):
            edges.append(np.stack((m[:-1][keep], m[1:][keep]), axis=1))
    e = np.unique(np.sort(np.vstack(edges), axis=1), axis=0) if edges else np.zeros((0, 2), np.int64)
    return up, ur, e


def _graph_filter(values, edges, steps, reduce):
    v = np.asarray(values, dtype=np.float64).copy()
    if len(edges) == 0:
        return v
    a, b = edges[:, 0], edges[:, 1]
    op = np.minimum if reduce == "min" else np.maximum
    for _ in range(int(steps)):
        m = v.copy()
        op.at(m, a, v[b])
        op.at(m, b, v[a])
        v = m
    return v


def parent_radius(up, ur, edges, window_mm=3.5):
    """Morphological opening of the inscribed radius along the centerline.

    The inscribed sphere swells into the sac at the neck; a spike shorter than
    the window is removed, a real change of calibre is kept.
    """
    if len(edges):
        length = np.linalg.norm(up[edges[:, 0]] - up[edges[:, 1]], axis=1)
        step = float(np.median(length[length > 1e-4])) if np.any(length > 1e-4) else 0.25
    else:
        step = 0.25
    k = int(np.clip(round(window_mm / max(step, 0.05)), 2, 48))
    opened = _graph_filter(_graph_filter(ur, edges, k, "min"), edges, k, "max")
    return np.maximum(opened, 0.2)


def _adjacency(n, edges):
    a = np.concatenate((edges[:, 0], edges[:, 1]))
    b = np.concatenate((edges[:, 1], edges[:, 0]))
    return csr_matrix((np.ones(len(a), dtype=np.float32), (a, b)), shape=(n, n))


def _component_with_seed(mask, adj, seed):
    out = np.zeros(mask.shape[0], dtype=bool)
    if not mask[seed]:
        return out
    sel = np.flatnonzero(mask)
    sub = adj[sel][:, sel]
    _n, lab = connected_components(sub, directed=False)
    pos = int(np.searchsorted(sel, seed))
    out[sel[lab == lab[pos]]] = True
    return out


def _dilate(seed, allowed, adj, hops):
    out = seed.copy()
    front = seed.copy()
    for _ in range(int(hops)):
        nb = (adj @ front.astype(np.float32)) > 0
        new = nb & allowed & ~out
        if not new.any():
            break
        out |= new
        front = new
    return out


def ball_excess(pts, up, r_parent, k=48, tree=None, refine_above=0.3):
    """Signed distance to the union of the parent's inscribed balls, and its node.

    min_j |p - c_j| - r_j, not the nearest node's |p - c| - r: where a small
    branch leaves a large vessel the branch's centerline is nearest to the large
    vessel's wall, and that measure reads the wall as a bulge of the large
    radius. The min only lowers the nearest-node value, so it is refined only
    where that value could matter (above ``refine_above``): first over the 12
    nearest nodes, then over ``k`` for the points where a node beyond the 12th
    could still be lower (its distance less the largest radius is below the
    current minimum).
    """
    tree = cKDTree(up) if tree is None else tree
    d, j = tree.query(pts, k=1, workers=1)
    j = np.asarray(j, dtype=np.int64)
    ex = d - r_parent[j]
    sel = np.flatnonzero(ex > refine_above)
    r_max = float(np.max(r_parent)) if len(up) else 0.0
    for kk in (12, k):
        kk = min(kk, len(up))
        if len(sel) == 0 or kk < 2:
            break
        dk, jk = tree.query(pts[sel], k=kk, workers=1)
        v = dk - r_parent[jk]
        a = np.argmin(v, axis=1)
        rows = np.arange(len(sel))
        lower = v[rows, a] < ex[sel]
        ex[sel[lower]] = v[rows, a][lower]
        j[sel[lower]] = jk[rows, a][lower]
        sel = sel[dk[:, -1] - r_max < ex[sel]]
    return ex, j


def _peak_candidates(pts, excess, n_max=4, sep_mm=4.0, floor_mm=0.6):
    """Highest excess first, then the next highest at least ``sep_mm`` from those taken."""
    order = np.argsort(-excess)
    out = [int(order[0])]
    taken = pts[order[0]][None]
    for i in order[1:]:
        if excess[i] < floor_mm or len(out) >= n_max:
            break
        if np.min(np.linalg.norm(taken - pts[i], axis=1)) >= sep_mm:
            out.append(int(i))
            taken = np.vstack((taken, pts[i]))
    return out


def _sac_core(pts, excess, adj, dome, r_dome):
    """The dome's connected excess level set that is compact and high.

    Scanned from high to low; the chosen level is the last before the set
    spills along the parent.
    """
    peak = float(excess[dome])
    r_dome = max(r_dome, 0.45)
    char = float(np.clip(1.15 * peak + 2.0 * r_dome, 4.0, 16.0))
    d_cap = float(np.clip(1.55 * char, 8.0, 22.0))
    levels = np.linspace(max(0.55, min(0.72 * peak, peak - 0.15)), 0.28, 8)
    best, core = -1.0, None
    for level in levels:
        comp = _component_with_seed(excess >= float(level), adj, dome)
        n_comp = int(comp.sum())
        if n_comp < 40:
            continue
        chunk = pts[comp]
        cen = chunk.mean(axis=0)
        diam = 2.0 * float(np.linalg.norm(chunk - cen, axis=1).max())
        med_ex = float(np.median(excess[comp]))
        if diam > d_cap or n_comp > 0.34 * len(pts) or med_ex < 0.45:
            continue
        score = (n_comp * med_ex) / (1.0 + (diam / char) ** 2)
        if score > best:
            best, core = score, comp
    if core is None:
        core = _component_with_seed(excess >= max(0.4, 0.35 * peak), adj, dome)
    return core


def _candidate_features(pts, excess, core, dome, r_dome, vert_area, rim_tree):
    chunk = pts[core] if core.any() else pts[[dome]]
    cen = chunk.mean(axis=0)
    peak = float(excess[dome])
    rim = float(rim_tree.query(pts[dome], k=1)[0]) if rim_tree is not None else np.inf
    return {
        "dome": int(dome), "core": core, "peak": peak, "r_parent": r_dome,
        "rel": peak / max(r_dome, 0.2),
        "diam": 2.0 * float(np.linalg.norm(chunk - cen, axis=1).max()),
        "n": int(core.sum()),
        "area": float(vert_area[core].sum()),
        "med": float(np.median(excess[core])) if core.any() else 0.0,
        "rim_mm": rim,
    }


def _select_sac(pts, cands):
    """Group the candidates that are lobes of one sac, then pick the best group.

    Two candidates are lobes of one sac when their cores touch and both stand
    well off the parent (median excess >= LOBE_MED_MM); a giant sac otherwise
    splits into a small high lobe and the bulk. A group scores its volume off
    the parent (area * median excess) times its height relative to the parent
    radius (saturating at SAC_REL_CAP), divided by a penalty for a patch broad
    against its height (a bend of the parent, not a sac). A candidate whose
    dome lies within RIM_EXCLUDE_R parent radii of an open end is a cut-end
    artefact and scores zero. The group with the highest peak wins unless
    another scores SAC_SWITCH_MARGIN times more.
    """
    n = len(cands)
    root = list(range(n))

    def find(i):
        while root[i] != i:
            i = root[i]
        return i

    trees = [cKDTree(pts[c["core"]]) if c["n"] else None for c in cands]
    for i in range(n):
        for j in range(i + 1, n):
            if trees[i] is None or trees[j] is None:
                continue
            if min(cands[i]["med"], cands[j]["med"]) < LOBE_MED_MM:
                continue
            if trees[i].query(pts[cands[j]["core"]], k=1, distance_upper_bound=LOBE_TOUCH_MM)[0].min() <= LOBE_TOUCH_MM:
                root[find(j)] = find(i)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    best, best_score = None, -1.0
    for members in groups.values():
        live = [i for i in members if cands[i]["rim_mm"] >= RIM_EXCLUDE_R * cands[i]["r_parent"]]
        area = sum(cands[i]["area"] for i in live)
        if area <= 0.0:
            score = 0.0
        else:
            med = sum(cands[i]["area"] * cands[i]["med"] for i in live) / area
            peak = max(cands[i]["peak"] for i in live)
            rel = max(cands[i]["rel"] for i in live)
            cap = area / (2.0 * np.pi * peak * peak)
            score = area * med * min(rel, SAC_REL_CAP) / (1.0 + (cap / SAC_CAP_SCALE) ** 2)
        if 0 in members:
            score *= SAC_SWITCH_MARGIN
        for i in members:
            cands[i]["score"] = score
        if score > best_score:
            best, best_score = members, score
    return best


def detect_aneurysm(pts, faces, up, r_parent):
    """Sac plus a short neck lip, and almost none of the parent wall.

    ``excess`` is how far a wall vertex stands off the parent tube. The dome
    is its maximum. The sac is the dome's connected level set at the level
    whose set is compact and high (the step before it spills along the
    parent); a geodesic lip of about half a parent radius picks up the neck.
    """
    t0 = time.perf_counter()
    excess, nearest = ball_excess(pts, up, r_parent)
    mesh_edges, _inv, use = _edge_table(faces)
    adj = _adjacency(len(pts), mesh_edges)
    elen = np.linalg.norm(pts[mesh_edges[:, 0]] - pts[mesh_edges[:, 1]], axis=1)
    median_edge = float(np.median(elen))
    vert_area = np.zeros(len(pts))
    a = pts[faces[:, 1]] - pts[faces[:, 0]]
    b = pts[faces[:, 2]] - pts[faces[:, 0]]
    np.add.at(vert_area, faces.ravel(), np.repeat(np.linalg.norm(np.cross(a, b), axis=1) / 6.0, 3))
    rim_v = np.unique(mesh_edges[use == 1])
    rim_tree = cKDTree(pts[rim_v]) if len(rim_v) else None
    cands = []
    for dome in _peak_candidates(pts, excess, n_max=8):
        if any(c["core"][dome] for c in cands):
            continue  # another high point of a sac already taken
        if len(cands) >= 4:
            break
        r_dome = float(r_parent[nearest[dome]])
        core = _sac_core(pts, excess, adj, dome, r_dome)
        cands.append(_candidate_features(pts, excess, core, dome, r_dome, vert_area, rim_tree))
    members = _select_sac(pts, cands)
    core = np.logical_or.reduce([cands[i]["core"] for i in members])
    top = max(members, key=lambda i: cands[i]["peak"])
    dome, peak = cands[top]["dome"], cands[top]["peak"]
    if int(core.sum()) < 30:
        raise AneurysmDetectionError(f"no aneurysm found (peak excess {peak:.2f} mm)")
    chunk = pts[core]
    core_cen = chunk.mean(axis=0)
    core_diam = 2.0 * float(np.linalg.norm(chunk - core_cen, axis=1).max())
    r_neck = float(np.median(r_parent[nearest[core]]))
    neck_mm = float(np.clip(0.70 * r_neck, 0.55, 1.8))
    near = np.linalg.norm(pts - core_cen, axis=1) <= 0.5 * core_diam + neck_mm
    eligible = near & (excess > 0.12)
    hops = int(max(1, round(neck_mm / max(median_edge, 1e-3))))
    mask = _dilate(core, eligible, adj, hops)
    info = {
        "peak_mm": peak,
        "frac": float(mask.mean()),
        "diameter_mm": core_diam,
        "neck_frac": float((mask & ~core).sum() / max(int(mask.sum()), 1)),
        "core": core,
        "candidates": [{k: v for k, v in c.items() if k != "core"} for c in cands],
        "chosen": members,
        "nearest": nearest,
        "adj": adj,
        "seconds": time.perf_counter() - t0,
    }
    return mask, excess, info

