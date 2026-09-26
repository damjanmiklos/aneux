"""Final polish of a remeshed vessel surface: slivers, fins, crossings, winding.

The remesher hands back a surface that is topologically clean -- one component,
genus 0, manifold, one loop per ostium -- and still carries defects anyone
looking at it would see. Measured on the 742 installed GT surfaces
(2026-09-26):

- 4058 triangles with an angle under 1 degree in 553 cases, 98% of them caps
  touching a rim: a rim vertex almost on the chord of its two rim neighbours,
  or an interior vertex a few microns off a rim edge. Their normals are noise,
  which is also what 290 of the 412 "folds" (adjacent normals over 160 degrees
  apart) were.
- 453 self-intersecting triangle pairs in 23 cases: fins on the crease where a
  sac folds back against its parent, and zig-zag fins in a rim.
- 19 surfaces wound inside out (16 templates). The input set is split 287/742
  between the two windings and nothing downstream fixed the sign.

Every repair here keeps the topology and is kept only if it does: the rim
count, the manifoldness, the component count and the genus must come out as
they went in, and neither folds nor crossings may increase, or the surface is
returned untouched. On a surface with none of these defects the polish is a
no-op. On the 742 GT surfaces it moves the geometry by at most 0.016 mm outside
the fin patches, and by at most 0.19 mm inside one (SNF00000426_03's pit).
"""
import numpy as np

# A cap is a triangle with one angle this wide; its apex sits on the chord.
CAP_DEG = 170.0
# Triangles with a smaller minimum angle are what the sliver pass works on.
BAD_MIN_DEG = 3.0
# Adjacent faces whose normals are further apart than this are a fold / fin.
FOLD_DEG = 160.0
# Geometry a sliver repair may move, as a fraction of the median edge.
SLIVER_TOL_FRACTION = 0.05
# Displacement budgets of the fin relaxation, in mm. The second one is what
# SNF00000426_03's pit needs (0.19 mm): a crease the input itself carries.
RELAX_BUDGETS_MM = (0.1, 0.25)
RELAX_MAX_ITER = 200


# ---------------------------------------------------------------- geometry ---

def _seg_tri(P0, P1, A, B, C):
    """Vectorised Moller-Trumbore: does segment P0-P1 cross the open triangle."""
    D = P1 - P0
    e1, e2 = B - A, C - A
    h = np.cross(D, e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > 1e-18
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = P0 - A
    u = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = f * np.einsum("ij,ij->i", D, q)
    t = f * np.einsum("ij,ij->i", e2, q)
    eps = 1e-9
    return ok & (u > eps) & (v > eps) & (u + v < 1 - eps) & (t > eps) & (t < 1 - eps)


def self_intersections(P, F):
    """Pairs of triangles (sharing no vertex) whose interiors cross."""
    from scipy.spatial import cKDTree
    F = np.asarray(F, np.int64)
    if not len(F):
        return np.zeros((0, 2), np.int64)
    C = P[F].mean(axis=1)
    R = np.linalg.norm(P[F] - C[:, None, :], axis=2).max(axis=1)
    tree = cKDTree(C)
    hits = []
    rmax = R.max()
    for lo in range(0, len(F), 20000):
        idx = np.arange(lo, min(lo + 20000, len(F)))
        nb = tree.query_ball_point(C[idx], R[idx] + rmax)
        fi = np.repeat(idx, [len(n) for n in nb])
        fj = np.concatenate([np.asarray(n, dtype=np.int64) for n in nb])
        keep = fj > fi
        fi, fj = fi[keep], fj[keep]
        keep = np.linalg.norm(C[fi] - C[fj], axis=1) <= R[fi] + R[fj]
        fi, fj = fi[keep], fj[keep]
        Fi, Fj = F[fi], F[fj]
        share = np.zeros(len(fi), bool)
        for a in range(3):
            for b in range(3):
                share |= Fi[:, a] == Fj[:, b]
        fi, fj = fi[~share], fj[~share]
        if not len(fi):
            continue
        Ti, Tj = P[F[fi]], P[F[fj]]
        hit = np.zeros(len(fi), bool)
        for s, e in ((0, 1), (1, 2), (2, 0)):
            hit |= _seg_tri(Ti[:, s], Ti[:, e], Tj[:, 0], Tj[:, 1], Tj[:, 2])
            hit |= _seg_tri(Tj[:, s], Tj[:, e], Ti[:, 0], Ti[:, 1], Ti[:, 2])
        hits.append(np.column_stack([fi[hit], fj[hit]]))
    return np.vstack(hits) if hits else np.zeros((0, 2), np.int64)


def _normal(p):
    return np.cross(p[1] - p[0], p[2] - p[0])


def _angles(p):
    out = []
    for k in range(3):
        u = p[(k + 1) % 3] - p[k]
        v = p[(k + 2) % 3] - p[k]
        d = np.linalg.norm(u) * np.linalg.norm(v)
        out.append(180.0 if d < 1e-300 else float(np.degrees(np.arccos(np.clip(u @ v / d, -1, 1)))))
    return out


def _min_angle(p):
    # A triangle with a zero-length edge reads 180 at that corner, and 0 is
    # what its minimum should be.
    a = _angles(p)
    return 0.0 if max(a) >= 180.0 else min(a)


def _min_angles(P, F):
    p = P[F]
    out = np.full(len(F), 180.0)
    for k in range(3):
        u = p[:, (k + 1) % 3] - p[:, k]
        w = p[:, (k + 2) % 3] - p[:, k]
        d = np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1)
        a = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", u, w) / np.maximum(d, 1e-300), -1, 1)))
        out = np.minimum(out, np.where(d < 1e-300, 0.0, a))
    return out


def _pt_seg(p, a, b):
    d = b - a
    L2 = d @ d
    t = 0.0 if L2 < 1e-300 else np.clip((p - a) @ d / L2, 0, 1)
    return float(np.linalg.norm(p - (a + t * d)))


def _pt_tri(p, a, b, c):
    """Point-triangle distance (Ericson, Real-Time Collision Detection 5.1.5)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ab @ ap
    d2 = ac @ ap
    if d1 <= 0 and d2 <= 0:
        return float(np.linalg.norm(ap))
    bp = p - b
    d3 = ab @ bp
    d4 = ac @ bp
    if d3 >= 0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return float(np.linalg.norm(p - (a + d1 / (d1 - d3) * ab)))
    cp = p - c
    d5 = ab @ cp
    d6 = ac @ cp
    if d6 >= 0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return float(np.linalg.norm(p - (a + d2 / (d2 - d6) * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(p - (b + w * (c - b))))
    den = 1.0 / (va + vb + vc)
    return float(np.linalg.norm(p - (a + ab * vb * den + ac * vc * den)))


def _seg_seg(p1, q1, p2, q2):
    """Distance between two segments."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = d1 @ d1
    e = d2 @ d2
    f = d2 @ r
    c = d1 @ r
    b = d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-300 else 0.0
    t = (b * s + f) / e if e > 1e-300 else 0.0
    if t < 0:
        t = 0.0
        s = np.clip(-c / a, 0, 1) if a > 1e-300 else 0.0
    elif t > 1:
        t = 1.0
        s = np.clip((b - c) / a, 0, 1) if a > 1e-300 else 0.0
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def _edges(F, nv):
    """Undirected edge keys, their inverse into the unique list, and counts."""
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    und = np.sort(E, axis=1)
    key = und[:, 0] * nv + und[:, 1]
    _u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return E, und, key, inv, cnt


def signed_volume(P, F):
    """Enclosed volume with every rim fan-capped at its centroid.

    Positive when the triangles are wound with their normals pointing out of
    the lumen. The fan caps are what make the number meaningful on an open
    surface; without them it depends on where the origin is.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    P = np.asarray(P, np.float64)
    F = np.asarray(F, np.int64)
    nv = len(P)
    E, _und, _key, inv, cnt = _edges(F, nv)
    be = E[cnt[inv] == 1]
    vol = np.einsum("ij,ij->i", P[F[:, 0]], np.cross(P[F[:, 1]], P[F[:, 2]])).sum() / 6.0
    if len(be):
        _n, lab = connected_components(
            coo_matrix((np.ones(len(be)), (be[:, 0], be[:, 1])), shape=(nv, nv)), directed=False
        )
        for lid in np.unique(lab[be[:, 0]]):
            sel = lab[be[:, 0]] == lid
            ce = P[np.unique(be[sel])].mean(axis=0)
            # the cap runs each rim edge the other way round
            vol += np.einsum("j,ij->", ce, np.cross(P[be[sel, 1]], P[be[sel, 0]])) / 6.0
    return float(vol)


def _fold_verts(P, F, faces, cos_lim):
    """Vertices of interior edges among ``faces`` whose dihedral exceeds the limit."""
    sub = F[faces]
    N = np.cross(P[sub[:, 1]] - P[sub[:, 0]], P[sub[:, 2]] - P[sub[:, 0]])
    Nu = N / np.maximum(np.linalg.norm(N, axis=1), 1e-300)[:, None]
    E = np.vstack([sub[:, [0, 1]], sub[:, [1, 2]], sub[:, [2, 0]]])
    T = np.tile(np.arange(len(sub)), 3)
    und = np.sort(E, axis=1)
    key = und[:, 0] * len(P) + und[:, 1]
    o = np.argsort(key, kind="stable")
    key, und, T = key[o], und[o], T[o]
    same = np.flatnonzero(key[1:] == key[:-1])
    c = np.einsum("ij,ij->i", Nu[T[same]], Nu[T[same + 1]])
    return set(und[same[c < cos_lim]].ravel().tolist())


def surface_census(P, F):
    """What the polish must not change, and what it is there to reduce."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    P = np.asarray(P, np.float64)
    F = np.asarray(F, np.int64)
    nv = len(P)
    E, und, _key, inv, cnt = _edges(F, nv)
    used = np.unique(F)
    rr = np.concatenate([F[:, 0], F[:, 1]])
    cc = np.concatenate([F[:, 1], F[:, 2]])
    _n, lab = connected_components(coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(nv, nv)), directed=False)
    n_comp = len(np.unique(lab[used]))
    be = und[cnt[inv] == 1]
    if len(be):
        _n, lb = connected_components(
            coo_matrix((np.ones(len(be)), (be[:, 0], be[:, 1])), shape=(nv, nv)), directed=False
        )
        loops = len(np.unique(lb[np.unique(be)]))
    else:
        loops = 0
    chi = len(used) - len(cnt) + len(F)
    return dict(
        components=n_comp,
        loops=loops,
        nonmanifold_edges=int((cnt > 2).sum()),
        genus=(2 * n_comp - chi - loops) / 2.0,
        folds=len(_fold_verts(P, F, np.arange(len(F)), np.cos(np.radians(FOLD_DEG)))),
        crossings=int(len(self_intersections(P, F))),
        slivers=int((_min_angles(P, F) < 1.0).sum()),
    )


# ------------------------------------------------------------ sliver pass ---

class _Mesh:
    """Just enough adjacency to edit a triangle soup locally."""

    def __init__(self, P, F):
        self.P = np.asarray(P, np.float64)
        self.F = np.asarray(F, np.int64).copy()
        self.alive = np.ones(len(self.F), bool)
        self.VF = [set() for _ in range(len(self.P))]
        for t, f in enumerate(self.F):
            for v in f:
                self.VF[int(v)].add(t)

    def ef(self, a, b):
        return self.VF[a] & self.VF[b]

    def nbrs(self, v):
        s = set()
        for t in self.VF[v]:
            s.update(int(x) for x in self.F[t])
        s.discard(v)
        return s

    def is_bnd_edge(self, a, b):
        return len(self.ef(a, b)) == 1

    def is_bnd_vert(self, v):
        return any(self.is_bnd_edge(v, n) for n in self.nbrs(v))

    def rim_nbrs(self, v):
        return [n for n in self.nbrs(v) if self.is_bnd_edge(v, n)]

    def kill(self, t):
        self.alive[t] = False
        for v in self.F[t]:
            self.VF[int(v)].discard(t)

    def set_face(self, t, f):
        for v in self.F[t]:
            self.VF[int(v)].discard(t)
        self.F[t] = f
        for v in f:
            self.VF[int(v)].add(t)

    def tri(self, t):
        return self.P[self.F[t]]


def _try_collapse(M, v, n, tol):
    """Remove ``v`` by moving it onto its neighbour ``n``; True if it was done.

    Refused unless it is safe and pays: the link condition must hold (so no
    two sheets are fused), no surviving face may turn over, the worst angle
    around ``v`` must improve, and ``v`` itself must end up within ``tol`` of
    the new surface -- so the geometry it carried is still there. A rim vertex
    may only slide along its own rim, onto a rim neighbour, and only when it
    lies on the chord of its two rim neighbours; that is the one way to take
    a rim vertex out without moving the opening.
    """
    P = M.P
    e = M.ef(v, n)
    if not e:
        return False
    if M.is_bnd_vert(v):
        if len(e) != 1:
            return False
        rn = M.rim_nbrs(v)
        if len(rn) != 2 or n not in rn:
            return False
        other = rn[0] if rn[1] == n else rn[1]
        if _pt_seg(P[v], P[n], P[other]) > tol:
            return False
    elif len(e) != 2:
        return False
    apexes = set()
    for t in e:
        apexes.update(int(x) for x in M.F[t])
    apexes -= {v, n}
    if (M.nbrs(v) & M.nbrs(n)) != apexes:
        return False
    old_worst = min(_min_angle(M.tri(t)) for t in M.VF[v])
    new_faces = []
    for t in M.VF[v]:
        if t in e:
            continue
        f = M.F[t].copy()
        f[f == v] = n
        p_old = M.tri(t)
        nn = _normal(P[f])
        no = _normal(p_old)
        if np.linalg.norm(nn) < 1e-300:
            return False
        # A well-shaped face must keep its facing; a sliver's normal is noise
        # and is judged against the fan instead, below.
        if _min_angle(p_old) > 5.0 and nn @ no <= np.cos(np.radians(45)) * np.linalg.norm(nn) * np.linalg.norm(no):
            return False
        new_faces.append((t, f))
    fan_n = sum(_normal(M.tri(t)) for t in M.VF[v])
    if any(_normal(P[f]) @ fan_n <= 0 for _t, f in new_faces):
        return False
    if min((_min_angle(P[f]) for _t, f in new_faces), default=180.0) <= old_worst:
        return False
    if new_faces and min(_pt_tri(P[v], *P[f]) for _t, f in new_faces) > tol:
        return False
    for t in e:
        M.kill(t)
    for t, f in new_faces:
        M.set_face(t, f)
    return True


def _loop_len(M, a, cap=10):
    """Edges in the rim loop through boundary vertex ``a`` (walk capped at ``cap``)."""
    prev, cur, n = None, a, 0
    while n < cap:
        nx = [x for x in M.rim_nbrs(cur) if x != prev]
        if not nx:
            return cap
        prev, cur = cur, nx[0]
        n += 1
        if cur == a:
            return n
    return cap


def polish_slivers(P, F, max_rounds=12):
    """Remove caps and needles without moving the surface.

    Four moves, in the order that costs least geometry:

    1. A rim ear -- a triangle hanging off the rim by two boundary edges --
       whose apex lies on its chord is dropped; the chord becomes the rim.
       So is one folded back over its only neighbour: that is a flap, not wall.
    2. A cap whose long edge is interior is flipped across it when the two
       diagonals nearly cross, so the flip does not move the surface, and the
       pair's worst angle improves.
    3. Otherwise the cap's apex is collapsed onto an end of its long edge (see
       ``_try_collapse``). Flipping alone cannot clear a band of caps: the
       neighbour across a cap's long edge is usually another cap, and for a
       near-collinear point set every triangulation has one. Removing the
       apex does, because it carries no geometry the chord does not.
    4. A needle collapses its shortest edge, which moves a vertex by at most
       that edge (capped at a quarter of the median edge).
    """
    P = np.asarray(P, np.float64)
    F = np.asarray(F, np.int64)
    med = float(np.median(np.linalg.norm(P[F[:, 0]] - P[F[:, 1]], axis=1)))
    tol = SLIVER_TOL_FRACTION * med
    cos_fold = np.cos(np.radians(FOLD_DEG))
    M = _Mesh(P, F)
    stats = dict(ear=0, flip=0, collapse=0)
    for _ in range(max_rounds):
        changed = False
        ids = np.flatnonzero(M.alive)
        # folded rim ears
        _E, _und, _key, inv, cnt = _edges(M.F[ids], len(M.P))
        n_bnd = (cnt[inv] == 1).reshape(3, -1).sum(axis=0)
        for t in ids[n_bnd == 2]:
            if not M.alive[t]:
                continue
            f = [int(x) for x in M.F[t]]
            for k in range(3):
                apex, a, b = f[k], f[(k + 1) % 3], f[(k + 2) % 3]
                if M.is_bnd_edge(apex, a) and M.is_bnd_edge(apex, b):
                    break
            else:
                continue
            eab = M.ef(a, b)
            if len(eab) != 2 or len(M.VF[apex]) != 1:
                continue
            u = [x for x in eab if x != t][0]
            n1 = _normal(M.tri(t))
            n2 = _normal(M.tri(u))
            c = n1 @ n2 / max(np.linalg.norm(n1) * np.linalg.norm(n2), 1e-300)
            # never shrink a rim to a triangle
            if c < cos_fold and _loop_len(M, a) > 4:
                M.kill(t)
                stats["ear"] += 1
                changed = True
        ids = np.flatnonzero(M.alive)
        for t in ids[_min_angles(M.P, M.F[ids]) < BAD_MIN_DEG]:
            if not M.alive[t]:
                continue
            f = [int(x) for x in M.F[t]]
            pt = M.P[f]
            if _min_angle(pt) >= BAD_MIN_DEG:
                continue
            angs = _angles(pt)
            k = int(np.argmax(angs))
            apex, a, b = f[k], f[(k + 1) % 3], f[(k + 2) % 3]
            if angs[k] > CAP_DEG:
                if (M.is_bnd_edge(apex, a) and M.is_bnd_edge(apex, b) and len(M.ef(a, b)) == 2
                        and _pt_seg(M.P[apex], M.P[a], M.P[b]) <= tol):
                    M.kill(t)
                    stats["ear"] += 1
                    changed = True
                    continue
                eab = M.ef(a, b)
                if len(eab) == 2:
                    u = [x for x in eab if x != t][0]
                    g = [int(x) for x in M.F[u]]
                    w = [x for x in g if x != a and x != b][0]
                    if (w != apex and not M.ef(apex, w)
                            and _seg_seg(M.P[a], M.P[b], M.P[apex], M.P[w]) <= tol):
                        # t is (apex, a, b) in its own winding, u is (b, a, w)
                        new1 = np.array([apex, a, w])
                        new2 = np.array([apex, w, b])
                        worst_old = min(_min_angle(pt), _min_angle(M.P[g]))
                        worst_new = min(_min_angle(M.P[new1]), _min_angle(M.P[new2]))
                        n_old = _normal(pt) + _normal(M.P[g])
                        if worst_new > worst_old and all(_normal(M.P[x]) @ n_old > 0 for x in (new1, new2)):
                            M.set_face(t, new1)
                            M.set_face(u, new2)
                            stats["flip"] += 1
                            changed = True
                            continue
                if any(_try_collapse(M, apex, n, tol)
                       for n in sorted((a, b), key=lambda x: np.linalg.norm(M.P[x] - M.P[apex]))):
                    stats["collapse"] += 1
                    changed = True
                    continue
            for x, y in sorted(((f[i], f[j]) for i, j in ((0, 1), (1, 2), (2, 0))),
                               key=lambda e: np.linalg.norm(M.P[e[0]] - M.P[e[1]])):
                reach = max(tol, min(float(np.linalg.norm(M.P[x] - M.P[y])), 0.25 * med))
                if _try_collapse(M, x, y, reach) or _try_collapse(M, y, x, reach):
                    stats["collapse"] += 1
                    changed = True
                    break
        if not changed:
            break
    F2 = M.F[M.alive]
    used = np.unique(F2)
    remap = -np.ones(len(M.P), np.int64)
    remap[used] = np.arange(len(used))
    return M.P[used], remap[F2], used, stats


# ------------------------------------------------------ fins and crossings ---

def relax_folds_and_crossings(P, F, lam=0.5):
    """Untangle crossings and fins by relaxing only the vertices that carry them.

    Seeds are the vertices of self-intersecting faces and of fold edges; seeds
    within 1 mm of each other form one patch. Each patch is relaxed on its own:
    the defect vertices and their one-ring move a step towards the mean of
    their neighbours until the patch reads clean. Rim vertices stay put on the
    first attempt and may then slide towards the chord of their two rim
    neighbours -- a zig-zag fin in a rim cannot be cleared any other way, and
    the opening keeps its place. A patch that is not clean within the budget
    is put back exactly as it was.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    P0 = np.asarray(P, np.float64)
    P = P0.copy()
    F = np.asarray(F, np.int64)
    nv = len(P)
    _E, und, _key, inv, cnt = _edges(F, nv)
    bnd = np.zeros(nv, bool)
    bnd[und[cnt[inv] == 1].ravel()] = True
    nbr = [[] for _ in range(nv)]
    for a, b in np.unique(und, axis=0):
        nbr[a].append(b)
        nbr[b].append(a)
    rimnbr = [[] for _ in range(nv)]
    for a, b in und[cnt[inv] == 1]:
        rimnbr[a].append(b)
        rimnbr[b].append(a)
    cos_lim = np.cos(np.radians(FOLD_DEG))
    sx = self_intersections(P, F)
    seeds = set(np.unique(F[np.unique(sx)]).tolist()) if len(sx) else set()
    seeds |= _fold_verts(P, F, np.arange(len(F)), cos_lim)
    stats = dict(patches=0, fixed=0, reverted=0)
    if not seeds:
        return P, stats
    S = np.array(sorted(seeds))
    pairs = cKDTree(P[S]).query_pairs(1.0, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])) if len(pairs) else ([], ([], [])),
        shape=(len(S), len(S)),
    )
    n_patch, lab = connected_components(graph, directed=False)
    stats["patches"] = int(n_patch)
    ctree = cKDTree(P[F].mean(axis=1))
    for g in range(n_patch):
        seed = set(S[lab == g].tolist())
        pts = P[list(seed)]
        centre = pts.mean(axis=0)
        local = np.asarray(
            ctree.query_ball_point(centre, np.linalg.norm(pts - centre, axis=1).max() + 1.5), np.int64
        )

        def bad_verts():
            s = self_intersections(P, F[local])
            bv = set(np.unique(F[local][np.unique(s)]).tolist()) if len(s) else set()
            return bv | _fold_verts(P, F, local, cos_lim)

        ok = False
        for budget in RELAX_BUDGETS_MM:
            for slide in (False, True):
                active = set(seed)
                for _it in range(RELAX_MAX_ITER):
                    bv = bad_verts()
                    if not bv:
                        ok = True
                        break
                    active |= bv
                    ring = set(active)
                    for v in active:
                        ring.update(nbr[v])
                    mov = [v for v in ring if not bnd[v]]
                    tgt = [P[nbr[v]].mean(axis=0) for v in mov]
                    if slide:
                        for v in ring:
                            if bnd[v] and len(rimnbr[v]) == 2:
                                mov.append(v)
                                tgt.append(P[rimnbr[v]].mean(axis=0))
                    if not mov:
                        break
                    mov = np.asarray(mov, np.int64)
                    P[mov] += lam * (np.asarray(tgt) - P[mov])
                    if np.linalg.norm(P[mov] - P0[mov], axis=1).max() > budget:
                        break
                if ok:
                    break
                touched = set(active)
                for v in active:
                    touched.update(nbr[v])
                touched = np.fromiter(touched, np.int64)
                P[touched] = P0[touched]
            if ok:
                break
        stats["fixed" if ok else "reverted"] += 1
    return P, stats


# ------------------------------------------------------------ vtk wrappers ---

def _to_numpy(surface):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(surface)
    tri.PassVertsOff()
    tri.PassLinesOff()
    tri.Update()
    poly = tri.GetOutput()
    P = np.array(vtk_to_numpy(poly.GetPoints().GetData()), dtype=np.float64)
    conn = vtk_to_numpy(poly.GetPolys().GetConnectivityArray())
    return P, np.asarray(conn, np.int64).reshape(-1, 3), poly


def _from_numpy(P, F, source=None, keep_points=None):
    """Polydata from arrays, carrying ``source``'s point arrays for kept points."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    out = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetDataTypeToDouble()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(P, dtype=np.float64), deep=True))
    out.SetPoints(pts)
    cells = vtk.vtkCellArray()
    offsets = np.arange(0, 3 * len(F) + 1, 3, dtype=np.int64)
    cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                  numpy_to_vtkIdTypeArray(np.ascontiguousarray(F, dtype=np.int64).ravel(), deep=True))
    out.SetPolys(cells)
    if source is not None and keep_points is not None:
        spd = source.GetPointData()
        for i in range(spd.GetNumberOfArrays()):
            arr = spd.GetArray(i)
            if arr is None or arr.GetName() in (None, "Normals"):
                continue
            from vtk.util.numpy_support import vtk_to_numpy
            vals = vtk_to_numpy(arr)[keep_points]
            new = numpy_to_vtk(np.ascontiguousarray(vals), deep=True)
            new.SetName(arr.GetName())
            out.GetPointData().AddArray(new)
    return out


def polish_surface(surface, label="surface"):
    """Sliver pass, fin/crossing relaxation, sliver pass again -- or nothing.

    Returns ``(surface, report)``. The input comes back untouched when it has
    nothing to repair, and also when the repair would change what must not
    change (rims, components, genus, manifoldness) or would leave more folds
    or crossings than it found.
    """
    P, F, poly = _to_numpy(surface)
    before = surface_census(P, F)
    report = dict(before=before, after=before, changed=False)
    if not (before["folds"] or before["crossings"] or before["slivers"]
            or (_min_angles(P, F) < BAD_MIN_DEG).any()):
        return surface, report
    P1, F1, keep1, s1 = polish_slivers(P, F)
    P2, s2 = relax_folds_and_crossings(P1, F1)
    P3, F3, keep3, s3 = polish_slivers(P2, F1)
    after = surface_census(P3, F3)
    report.update(after=after, sliver=s1, relax=s2, sliver2=s3)
    kept = all(after[k] == before[k] for k in ("components", "loops", "nonmanifold_edges", "genus"))
    kept = kept and after["folds"] <= before["folds"] and after["crossings"] <= before["crossings"]
    kept = kept and after["slivers"] <= before["slivers"]
    if not kept:
        print(f"  Polish of the {label} refused: {before} -> {after}")
        return surface, report
    moved = float(np.linalg.norm(P2 - P1, axis=1).max()) if len(P1) else 0.0
    report.update(changed=True, relax_max_move_mm=moved)
    print(
        f"  Polished the {label}: slivers(<1deg) {before['slivers']}->{after['slivers']}, "
        f"folds {before['folds']}->{after['folds']}, crossings {before['crossings']}->"
        f"{after['crossings']} (ears {s1['ear'] + s3['ear']}, flips {s1['flip'] + s3['flip']}, "
        f"collapses {s1['collapse'] + s3['collapse']}, fin patches {s2['fixed']}/{s2['patches']}, "
        f"relax moved <= {moved:.3f} mm)"
    )
    return _from_numpy(P3, F3, poly, keep1[keep3]), report


def orient_outward(surface, label="surface"):
    """Wind every triangle so its normal points out of the lumen.

    The sign is read once for the whole surface from the fan-capped volume;
    the winding was already consistent (every shared edge is traversed both
    ways), so only that one bit was in question. Returns ``(surface, flipped)``.
    """
    import vtk
    P, F, _poly = _to_numpy(surface)
    if signed_volume(P, F) >= 0:
        return surface, False
    rev = vtk.vtkReverseSense()
    rev.SetInputData(surface)
    rev.ReverseCellsOn()
    rev.ReverseNormalsOn()
    rev.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(rev.GetOutput())
    print(f"  The {label} was wound inside out; reversed every triangle")
    return out, True
