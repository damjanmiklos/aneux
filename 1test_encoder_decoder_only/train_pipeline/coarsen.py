"""Mid/coarse template levels by sizing-field edge collapse, plus prolongation.

Why not decimation: quadric decimation minimises geometric error, and the sac
of a variable-density template is the flattest, most finely meshed region, so
it is the first to go.  On p131 the fine sac/parent edge ratio of 4.9 came out
as 0.83 at mid and 0.92 at coarse -- the sac ended up *coarser* than the
parent, and the density the template was built to carry was gone.

Here a level is a sizing field
    h_L(x) = max(h_fine(x), min(k_L * h_fine(x), 2 pi R_template(x) / N_min))
realised by half-edge collapses (shortest h-relative edge first) and Delaunay
flips.  The level keeps the density ratio, keeps at least ~N_min vertices
around every thin branch and every rim, and its vertex set is a subset of the
finer one, so the levels nest.  The core is C++ (`coarsen_core.cpp`) loaded
through ctypes, which works from any Python; `_coarsen_python` is the
line-by-line reference and the fallback when no compiler is available.

Prolongation projects each finer vertex onto the coarser surface and returns
the barycentric weights of the triangle it lands in.  Candidate triangles
facing away from the vertex's own normal are rejected, so a thin branch can
never borrow a displacement from its opposite wall -- the Euclidean kNN it
replaces did exactly that for 0.2-5 % of vertices.
"""
from __future__ import annotations

import ctypes
import heapq
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import warnings

import numpy as np
from scipy.spatial import cKDTree

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "coarsen_core.cpp")
_LIB_DIR = os.path.join(_HERE, "_build")
_LIB_NAME = "coarsen_core" + (".dll" if os.name == "nt" else ".so")

# collapse an edge shorter than ALPHA*h unless that makes an edge longer than
# BETA*min(h_a, h_b) (Botsch & Kobbelt use 4/5 and 4/3 with splits; collapse-only needs a
# slightly tighter upper bound)
ALPHA = 0.8
BETA = 1.35
MAX_NORMAL_TURN_DEG = 40.0
Q_MIN = 0.25
FLIP_NEW_NORMAL_DEG = 25.0
FLIP_DIHEDRAL_DEG = 30.0
# a new triangle may not turn further than this from the original normal at
# any of its corners (the per-collapse turn guard alone lets turns add up)
MAX_ANCHOR_TURN_DEG = 60.0
PASSES = 3

_lib = None
_lib_error = None


def _compile(out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = f"{out_path}.{os.getpid()}.tmp"
    if os.name == "nt":
        from setuptools._distutils import ccompiler
        from setuptools._distutils.sysconfig import customize_compiler

        cc = ccompiler.new_compiler()
        customize_compiler(cc)
        build = tempfile.mkdtemp(prefix="coarsen_build_")
        src = os.path.join(build, os.path.basename(_SRC))
        shutil.copyfile(_SRC, src)
        cwd = os.getcwd()
        try:
            # distutils puts objects next to an absolute source, so compile a
            # relative copy inside the scratch directory instead
            os.chdir(build)
            objs = cc.compile([os.path.basename(src)], extra_postargs=["/O2", "/std:c++17", "/EHsc"])
            cc.link_shared_object([os.path.join(build, o) for o in objs], tmp,
                                  export_symbols=["coarsen_mesh"], build_temp=build)
        finally:
            os.chdir(cwd)
            shutil.rmtree(build, ignore_errors=True)
    else:
        cxx = os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "g++"
        cmd = cxx.split() + ["-O2", "-std=c++17", "-shared", "-fPIC", _SRC, "-o", tmp]
        subprocess.run(cmd, check=True, capture_output=True)
    try:
        os.replace(tmp, out_path)
    except OSError:
        # another worker won the race and has the library loaded
        if not os.path.exists(out_path):
            raise
        try:
            os.remove(tmp)
        except OSError:
            pass


def _load():
    global _lib, _lib_error
    if _lib is not None or _lib_error is not None:
        return _lib
    path = os.path.join(_LIB_DIR, _LIB_NAME)
    try:
        if not os.path.exists(path) or os.path.getmtime(path) < os.path.getmtime(_SRC):
            _compile(path)
        lib = ctypes.CDLL(path)
        fn = lib.coarsen_mesh
        i64, dbl = ctypes.c_int64, ctypes.c_double
        p_d = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        p_i = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
        fn.argtypes = [i64, p_d, i64, p_i, p_d, p_d, dbl, dbl, dbl, dbl, i64, i64, i64, dbl, dbl, dbl,
                       p_i, p_i, p_i, p_i]
        fn.restype = i64
        _lib = lib
    except Exception as exc:  # no compiler, or a broken build
        _lib_error = exc
        warnings.warn(f"coarsen_core unavailable ({exc!r}); using the slow Python collapse")
    return _lib


def build_core():
    """Compile the C++ core now (call once before forking cache workers)."""
    return _load() is not None


# --------------------------------------------------------------------------
# reference implementation (identical decisions to coarsen_core.cpp)
# --------------------------------------------------------------------------

def _fn(V, tri):
    return np.cross(V[tri[1]] - V[tri[0]], V[tri[2]] - V[tri[0]])


def _q(V, tri):
    a, b, c = V[tri[0]], V[tri[1]], V[tri[2]]
    area2 = np.linalg.norm(np.cross(b - a, c - a))
    s = ((b - a) ** 2).sum() + ((c - b) ** 2).sum() + ((a - c) ** 2).sum()
    return 2.0 * np.sqrt(3.0) * area2 / max(s, 1e-30)


def _coarsen_python(V, F, h, N0, alpha, beta, cos_turn, q_min, min_rim, passes, do_flip,
                    cos_flip_new, cos_flip_dihedral, cos_anchor):
    F = F.copy()
    n, m = len(V), len(F)
    vf = [set() for _ in range(n)]
    for fi, (a, b, c) in enumerate(F):
        vf[a].add(fi); vf[b].add(fi); vf[c].add(fi)
    alive_f = np.ones(m, bool)
    alive_v = np.ones(n, bool)
    ecount = {}
    for a, b, c in F:
        for x, y in ((a, b), (b, c), (c, a)):
            k = (min(x, y), max(x, y))
            ecount[k] = ecount.get(k, 0) + 1
    if any(c > 2 for c in ecount.values()):
        raise ValueError("non-manifold input mesh")
    bnd = np.zeros(n, bool)
    badj = {}
    for (x, y), c in ecount.items():
        if c == 1:
            bnd[x] = bnd[y] = True
            badj.setdefault(x, []).append(y)
            badj.setdefault(y, []).append(x)
    loop = -np.ones(n, np.int64)
    rim_count = []
    for v0 in sorted(badj):
        if loop[v0] >= 0:
            continue
        lid = len(rim_count)
        stack, cnt = [v0], 0
        while stack:
            v = stack.pop()
            if loop[v] >= 0:
                continue
            loop[v] = lid
            cnt += 1
            stack.extend(badj[v])
        rim_count.append(cnt)

    def nbrs(v):
        s = set()
        for f in vf[v]:
            s.update(F[f].tolist())
        s.discard(v)
        return s

    def hedge(a, b):
        return 0.5 * (h[a] + h[b])

    def hmax_edge(a, b):
        return min(h[a], h[b])

    def try_collapse(b, a):
        fab = vf[a] & vf[b]
        if len(fab) not in (1, 2):
            return False, 0.0
        if bnd[b]:
            if not (bnd[a] and len(fab) == 1):
                return False, 0.0
            if rim_count[loop[b]] <= min_rim:
                return False, 0.0
        elif len(fab) == 1:
            return False, 0.0
        Na, Nb = nbrs(a), nbrs(b)
        opp = set()
        for f in fab:
            opp.update(F[f].tolist())
        opp -= {a, b}
        if (Na & Nb) != opp:
            return False, 0.0
        if not bnd[a] and not bnd[b] and len(Na | Nb) - 2 < 3:
            return False, 0.0
        for c in Nb:
            if c != a and np.linalg.norm(V[a] - V[c]) > beta * hmax_edge(a, c):
                return False, 0.0
        worst = 1.0
        for f in vf[b] - fab:
            tri = F[f].copy()
            n0 = _fn(V, tri)
            tri[tri == b] = a
            n1 = _fn(V, tri)
            l0, l1 = np.linalg.norm(n0), np.linalg.norm(n1)
            if l1 < 1e-14 or np.dot(n0, n1) < cos_turn * l0 * l1:
                return False, 0.0
            if any(np.dot(n1, N0[v]) < cos_anchor * l1 for v in tri):
                return False, 0.0
            qn = _q(V, tri)
            if qn < q_min and qn < _q(V, F[f]):
                return False, 0.0
            worst = min(worst, qn)
        return True, worst

    def do_collapse(b, a):
        fab = vf[a] & vf[b]
        for f in fab:
            alive_f[f] = False
            for v in F[f]:
                vf[v].discard(f)
        for f in list(vf[b]):
            F[f][F[f] == b] = a
            vf[a].add(f)
        vf[b].clear()
        alive_v[b] = False
        if bnd[b]:
            rim_count[loop[b]] -= 1

    def collapse_pass():
        heap = []

        def push(a, b):
            r = np.linalg.norm(V[a] - V[b]) / hedge(a, b)
            if r < alpha:
                heapq.heappush(heap, (r, min(a, b), max(a, b)))

        for f in np.nonzero(alive_f)[0]:
            for k in range(3):
                x, y = F[f][k], F[f][(k + 1) % 3]
                if x < y:
                    push(x, y)
                elif len(vf[x] & vf[y]) == 1:
                    push(y, x)
        nc = 0
        while heap:
            r, a, b = heapq.heappop(heap)
            if not (alive_v[a] and alive_v[b]) or not (vf[a] & vf[b]):
                continue
            if abs(np.linalg.norm(V[a] - V[b]) / hedge(a, b) - r) > 1e-12:
                continue
            ok1, q1 = try_collapse(b, a)
            ok2, q2 = try_collapse(a, b)
            if not (ok1 or ok2):
                continue
            if ok1 and (not ok2 or q1 >= q2):
                do_collapse(b, a)
                keep = a
            else:
                do_collapse(a, b)
                keep = b
            nc += 1
            for c in sorted(nbrs(keep)):
                push(keep, c)
        return nc

    def ang(p, x, y):
        u, w = V[x] - V[p], V[y] - V[p]
        c = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-30)
        return float(np.arccos(np.clip(c, -1.0, 1.0)))

    def flip_pass():
        nf = 0
        for _ in range(4):
            changed = 0
            for f1 in range(m):
                if not alive_f[f1]:
                    continue
                for k in range(3):
                    a, b, c = F[f1][k], F[f1][(k + 1) % 3], F[f1][(k + 2) % 3]
                    ef = vf[a] & vf[b]
                    if len(ef) != 2:
                        continue
                    f2 = (ef - {f1}).pop()
                    d = [v for v in F[f2] if v != a and v != b][0]
                    if d in nbrs(c):
                        continue
                    if np.linalg.norm(V[c] - V[d]) > beta * hmax_edge(c, d):
                        continue
                    if len(vf[a]) <= (2 if bnd[a] else 3) or len(vf[b]) <= (2 if bnd[b] else 3):
                        continue
                    if ang(c, a, b) + ang(d, a, b) <= np.pi + 1e-6:
                        continue
                    n1, n2 = _fn(V, F[f1]), _fn(V, F[f2])
                    t1, t2 = np.array([a, d, c]), np.array([d, b, c])
                    m1, m2 = _fn(V, t1), _fn(V, t2)
                    nn = n1 / np.linalg.norm(n1) + n2 / np.linalg.norm(n2)
                    c1 = np.dot(m1, nn) / (np.linalg.norm(m1) * np.linalg.norm(nn) + 1e-30)
                    c2 = np.dot(m2, nn) / (np.linalg.norm(m2) * np.linalg.norm(nn) + 1e-30)
                    if min(c1, c2) < cos_flip_new:
                        continue
                    if np.dot(n1, n2) < cos_flip_dihedral * np.linalg.norm(n1) * np.linalg.norm(n2):
                        continue
                    l1, l2 = np.linalg.norm(m1), np.linalg.norm(m2)
                    if any(np.dot(m1, N0[v]) < cos_anchor * l1 for v in t1) or                             any(np.dot(m2, N0[v]) < cos_anchor * l2 for v in t2):
                        continue
                    vf[a].discard(f2)
                    vf[b].discard(f1)
                    F[f1] = t1
                    F[f2] = t2
                    vf[c].add(f2)
                    vf[d].add(f1)
                    changed += 1
                    break
            nf += changed
            if not changed:
                break
        return nf

    stats = []
    for _ in range(passes):
        nc = collapse_pass()
        nfl = flip_pass() if do_flip else 0
        stats.append((nc, nfl))
        if nc == 0 and nfl == 0:
            break
    keep = np.nonzero(alive_v)[0]
    remap = -np.ones(n, np.int64)
    remap[keep] = np.arange(len(keep))
    return keep, remap[F[alive_f]], stats


def coarsen_by_sizing(points, faces, h, *, alpha=ALPHA, beta=BETA, max_normal_turn_deg=MAX_NORMAL_TURN_DEG,
                      q_min=Q_MIN, min_rim=6, passes=PASSES, flip=True, anchor_normals=None,
                      max_anchor_turn_deg=MAX_ANCHOR_TURN_DEG, backend="auto"):
    """Collapse `faces` towards the per-vertex target edge length `h`.

    Returns (keep, faces_out, stats): `keep` indexes the surviving input
    vertices (sorted), `faces_out` indexes into `keep`.  Rims keep at least
    `min_rim` vertices, and every rim vertex stays on its rim.  No new
    triangle may turn more than `max_anchor_turn_deg` from `anchor_normals`
    (default: this mesh's own vertex normals) at any of its corners.
    """
    V = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    F = np.ascontiguousarray(faces, dtype=np.int64).reshape(-1, 3)
    h = np.ascontiguousarray(h, dtype=np.float64).reshape(-1)
    if h.shape[0] != V.shape[0]:
        raise ValueError("h must hold one target edge length per vertex")
    if not np.all(np.isfinite(h)) or np.any(h <= 0):
        raise ValueError("h must be finite and positive")
    N0 = vertex_normals(V, F) if anchor_normals is None else np.asarray(anchor_normals, dtype=np.float64)
    N0 = np.ascontiguousarray(N0 / np.clip(np.linalg.norm(N0, axis=1, keepdims=True), 1e-30, None))
    args = (float(alpha), float(beta), float(np.cos(np.radians(max_normal_turn_deg))), float(q_min),
            int(min_rim), int(passes), int(bool(flip)),
            float(np.cos(np.radians(FLIP_NEW_NORMAL_DEG))), float(np.cos(np.radians(FLIP_DIHEDRAL_DEG))),
            float(np.cos(np.radians(max_anchor_turn_deg))))
    lib = _load() if backend in ("auto", "c") else None
    if backend == "c" and lib is None:
        raise RuntimeError(f"coarsen_core unavailable: {_lib_error!r}")
    if lib is None:
        return _coarsen_python(V, F, h, N0, *args)
    keep = np.zeros(V.shape[0], np.int64)
    f_out = np.zeros((F.shape[0], 3), np.int64)
    n_f_out = np.zeros(1, np.int64)
    stats = np.full(2 * max(int(passes), 1), -1, np.int64)
    nk = lib.coarsen_mesh(V.shape[0], V, F.shape[0], F, h, N0, *args, keep, f_out, n_f_out, stats)
    if nk < 0:
        raise ValueError({-1: "empty mesh", -2: "face index out of range", -3: "non-manifold input mesh"}.get(
            int(nk), f"coarsen_core error {nk}"))
    st = [(int(stats[2 * i]), int(stats[2 * i + 1])) for i in range(len(stats) // 2) if stats[2 * i] >= 0]
    return keep[:nk].copy(), f_out[: int(n_f_out[0])].copy(), st


def level_sizing(target_edge, r_template, k, n_min):
    """h_L = max(h, min(k h, 2 pi R / n_min)): the ratio is kept, thin branches keep n_min around."""
    h = np.asarray(target_edge, dtype=np.float64)
    r = np.asarray(r_template, dtype=np.float64)
    return np.maximum(h, np.minimum(float(k) * h, 2.0 * np.pi * r / float(n_min)))


# --------------------------------------------------------------------------
# prolongation
# --------------------------------------------------------------------------

def _closest_on_triangles(p, a, b, c):
    """Closest points on triangles (a,b,c) to p, as barycentric (u,v,w); all (N,3)."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    n = p.shape[0]
    bary = np.zeros((n, 3))
    done = np.zeros(n, bool)

    def put(mask, u, v, w):
        m = mask & ~done
        bary[m, 0] = u[m] if np.ndim(u) else u
        bary[m, 1] = v[m] if np.ndim(v) else v
        bary[m, 2] = w[m] if np.ndim(w) else w
        done[m] = True

    one, zero = np.ones(n), np.zeros(n)
    put((d1 <= 0) & (d2 <= 0), one, zero, zero)
    put((d3 >= 0) & (d4 <= d3), zero, one, zero)
    put((d6 >= 0) & (d5 <= d6), zero, zero, one)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = d1 / (d1 - d3)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1 - t, t, zero)
        t = d2 / (d2 - d6)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1 - t, zero, t)
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), zero, 1 - t, t)
        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        put(np.ones(n, bool), 1 - v - w, v, w)
    bary = np.nan_to_num(bary, nan=1.0 / 3.0)
    q = bary[:, :1] * a + bary[:, 1:2] * b + bary[:, 2:] * c
    return bary, q


def vertex_normals(points, faces):
    V = np.asarray(points, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    return vn / np.clip(np.linalg.norm(vn, axis=1, keepdims=True), 1e-30, None)


def barycentric_prolongation(coarse_pts, coarse_faces, fine_pts, fine_normals, k_candidates=16, min_cos=-0.5,
                             radius=None, trust_frac=0.1):
    """(idx, w) with idx (N,3) coarse vertices and w (N,3) barycentric weights.

    Each fine vertex goes to the closest point of a coarse triangle whose
    normal is not turned against the fine vertex normal (cos > min_cos).
    The opposite wall of a vessel sits near cos = -1; -0.5 rather than 0
    because a rim ear (a vertex with one sliver face) can have a normal
    more than 90 degrees off its own surface.  Both normals
    must follow the same face orientation, so pass `vertex_normals(fine
    points, fine faces)` for a mesh the coarse one was collapsed from.
    With `radius` (the local vessel radius per fine vertex), a triangle
    closer than `trust_frac * radius` is taken whatever its normal: the
    opposite wall is ~2R away, while a folded rim sliver in the template can
    carry a normal that points anywhere.
    Returns also the projection distance and the normal-guard misses.
    """
    Vc = np.asarray(coarse_pts, dtype=np.float64)
    Fc = np.asarray(coarse_faces, dtype=np.int64)
    P = np.asarray(fine_pts, dtype=np.float64)
    N = np.asarray(fine_normals, dtype=np.float64)
    a, b, c = Vc[Fc[:, 0]], Vc[Fc[:, 1]], Vc[Fc[:, 2]]
    fn = np.cross(b - a, c - a)
    fn /= np.clip(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30, None)
    k = int(min(k_candidates, Fc.shape[0]))
    _, cand = cKDTree((a + b + c) / 3.0).query(P, k=k, workers=-1)
    cand = np.asarray(cand).reshape(P.shape[0], k)
    best_d = np.full(P.shape[0], np.inf)
    best_f = np.full(P.shape[0], -1, np.int64)
    best_bary = np.zeros((P.shape[0], 3))
    any_d = np.full(P.shape[0], np.inf)
    any_f = np.full(P.shape[0], -1, np.int64)
    any_bary = np.zeros((P.shape[0], 3))
    for j in range(k):
        f = cand[:, j]
        bary, q = _closest_on_triangles(P, a[f], b[f], c[f])
        d = np.linalg.norm(P - q, axis=1)
        ok = np.einsum("ij,ij->i", fn[f], N) > min_cos
        upd = ok & (d < best_d)
        best_d[upd], best_f[upd], best_bary[upd] = d[upd], f[upd], bary[upd]
        upd = d < any_d
        any_d[upd], any_f[upd], any_bary[upd] = d[upd], f[upd], bary[upd]
    miss = best_f < 0
    if radius is not None:
        near = any_d < float(trust_frac) * np.asarray(radius, dtype=np.float64).reshape(-1)
        miss = miss | (near & (any_d < best_d))
    best_f[miss], best_bary[miss], best_d[miss] = any_f[miss], any_bary[miss], any_d[miss]
    w = np.clip(best_bary, 0.0, None)
    w /= np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    return Fc[best_f], w, best_d, miss


def build_levels(points, faces, target_edge, r_template, *, k_mid=2.0, k_coarse=3.5,
                 n_min_mid=8, n_min_coarse=6, min_rim_mid=8, min_rim_coarse=6, backend="auto"):
    """Nested mid and coarse levels of a fine template.

    Returns a dict with `keep_mid` (fine indices), `faces_mid`, `keep_coarse`
    (fine indices), `faces_coarse`, and the barycentric prolongation tables
    coarse->mid and mid->fine.
    """
    V = np.asarray(points, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    h_mid = level_sizing(target_edge, r_template, k_mid, n_min_mid)
    h_coarse = level_sizing(target_edge, r_template, k_coarse, n_min_coarse)
    n_fine = vertex_normals(V, F)
    keep_m, faces_m, st_m = coarsen_by_sizing(V, F, h_mid, min_rim=min_rim_mid, anchor_normals=n_fine,
                                              backend=backend)
    # coarse is anchored to the fine surface too, not to the mid one
    keep_c_in_m, faces_c, st_c = coarsen_by_sizing(V[keep_m], faces_m, h_coarse[keep_m], min_rim=min_rim_coarse,
                                                   anchor_normals=n_fine[keep_m], backend=backend)
    keep_c = keep_m[keep_c_in_m]
    R = np.asarray(r_template, dtype=np.float64)
    idx_f, w_f, d_f, miss_f = barycentric_prolongation(V[keep_m], faces_m, V, n_fine, radius=R)
    idx_m, w_m, d_m, miss_m = barycentric_prolongation(V[keep_c], faces_c, V[keep_m], n_fine[keep_m],
                                                       radius=R[keep_m])
    return {
        "keep_mid": keep_m, "faces_mid": faces_m, "stats_mid": st_m,
        "keep_coarse": keep_c, "faces_coarse": faces_c, "stats_coarse": st_c,
        "upsample_idx_mid": idx_m, "upsample_w_mid": w_m, "proj_dist_mid": d_m, "proj_miss_mid": miss_m,
        "upsample_idx_fine": idx_f, "upsample_w_fine": w_f, "proj_dist_fine": d_f, "proj_miss_fine": miss_f,
    }
