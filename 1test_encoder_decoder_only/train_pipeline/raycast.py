"""ICA-robust per-vertex radial GT r*(u, θ) via multi-hit ray-casting.

Rays start at the centerline sample and travel along the Bishop vertex normal.
Hits are accepted only if they pass a nearest-centerline (Voronoi) guard and a
cell-normal alignment check. Ambiguous multi-hits are masked rather than imputed.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from config import (
    R_STAR_AMBIGUOUS_MM,
    R_STAR_ARC_SLACK_MM,
    R_STAR_HIT_TOL,
    R_STAR_INWARD_MM,
    R_STAR_NORMAL_DOT,
    R_STAR_RING_SLACK,
    R_STAR_T_EPS_MM,
    R_STAR_T_MAX_MM,
    TUBE_RADIUS_MM,
)


def empty_r_star(n):
    """Zero r* with every vertex invalid."""
    n = int(n)
    return {
        "r_star": np.zeros(n, dtype=np.float64),
        "valid": np.zeros(n, dtype=bool),
        "ambiguous": np.zeros(n, dtype=bool),
        "dth": np.zeros(n, dtype=np.float64),
        "du": np.zeros(n, dtype=np.float64),
        "ring_med": np.zeros(n, dtype=np.float64),
    }


def pack_dense_centerline(dense_tracts):
    """Concatenate dense Bishop samples → (xyz, u, tract_id, arcs)."""
    xyz, u, tract, arcs = [], [], [], []
    for tid, dense in enumerate(dense_tracts):
        pts = np.asarray(dense["xyz"], dtype=np.float64)
        uu = np.asarray(dense["u"], dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] == 0:
            arcs.append(0.0)
            continue
        xyz.append(pts)
        u.append(uu)
        tract.append(np.full(pts.shape[0], tid, dtype=np.int64))
        arcs.append(float(dense.get("arc", 0.0)))
    if not xyz:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64), arcs
    return (
        np.concatenate(xyz, axis=0),
        np.concatenate(u, axis=0),
        np.concatenate(tract, axis=0),
        arcs,
    )


def max_ds_for_tract(arc_mm, n_len, ring_slack=R_STAR_RING_SLACK, arc_slack_mm=R_STAR_ARC_SLACK_MM):
    """Along-tract distance allowed between ray origin and hit's nearest CL sample."""
    n_len = max(int(n_len), 1)
    ring_ds = float(arc_mm) / float(max(n_len - 1, 1))
    return float(ring_slack) * ring_ds + float(arc_slack_mm)


def voronoi_ok_hit(nn_tract, nn_u, origin_tract, origin_u, arc_mm, max_ds_mm):
    """True if the hit's nearest dense CL sample is the same station as the ray."""
    if int(nn_tract) != int(origin_tract):
        return False
    ds = abs(float(nn_u) - float(origin_u)) * float(arc_mm)
    return ds <= float(max_ds_mm)


def choose_normal_sign(per_vertex_hits, normal_dot_min=R_STAR_NORMAL_DOT):
    """Pick a global GT-normal orientation; open ICA meshes are often flipped."""
    n_plus = 0
    n_minus = 0
    lo = float(normal_dot_min)
    for hits in per_vertex_hits:
        plus = False
        minus = False
        for h in hits:
            if not h["voronoi_ok"]:
                continue
            if h["normal_dot"] > lo:
                plus = True
            elif h["normal_dot"] < -lo:
                minus = True
        n_plus += int(plus)
        n_minus += int(minus)
    return 1.0 if n_plus >= n_minus else -1.0


MAX_R_STAR_HITS = 16


def _choose_normal_sign_arrays(ok, dots, n_hits, normal_dot_min=R_STAR_NORMAL_DOT):
    n = int(ok.shape[0])
    n_plus = 0
    n_minus = 0
    lo = float(normal_dot_min)
    for i in range(n):
        plus = False
        minus = False
        nh = int(n_hits[i])
        for j in range(nh):
            if not ok[i, j]:
                continue
            d = float(dots[i, j])
            if d > lo:
                plus = True
            elif d < -lo:
                minus = True
        n_plus += int(plus)
        n_minus += int(minus)
    return 1.0 if n_plus >= n_minus else -1.0


def select_r_star_from_hits(
    hits,
    *,
    t_eps=R_STAR_T_EPS_MM,
    ambiguous_mm=R_STAR_AMBIGUOUS_MM,
    normal_dot_min=R_STAR_NORMAL_DOT,
    normal_sign=1.0,
):
    """Pick one r* from candidate hits, or mask the vertex.

    Keeps the nearest outward hit (smallest t ≥ t_eps). If only inward hits
    exist, uses the one closest to the origin. Masks when two accepted hits
    are more than `ambiguous_mm` apart (overhang / double wall).

    Returns (r_star, valid, ambiguous). `ambiguous` is True only for the
    double-hit case so smoothness can unlock a crease; siphon/Voronoi misses
    stay `valid=False, ambiguous=False` and keep full Dirichlet.
    """
    ts = []
    lo = float(normal_dot_min)
    sign = float(normal_sign)
    for h in hits:
        if not h["voronoi_ok"]:
            continue
        if (sign * float(h["normal_dot"])) < lo:
            continue
        ts.append(float(h["t"]))
    if not ts:
        return 0.0, False, False
    arr = np.asarray(ts, dtype=np.float64)
    outward = arr[arr >= float(t_eps)]
    t_star = float(outward.min()) if outward.size else float(arr.max())
    if np.any(np.abs(arr - t_star) > float(ambiguous_mm)):
        return 0.0, False, True
    return float(max(t_star, float(t_eps))), True, False


def r_star_grid_stats(r_star, valid, branch_nl, n_radial):
    """Per-vertex wrapped Δθ r*, longitudinal Δu r*, and per-ring median r*."""
    r_star = np.asarray(r_star, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    n = int(r_star.shape[0])
    dth = np.zeros(n, dtype=np.float64)
    du = np.zeros(n, dtype=np.float64)
    ring_med = np.zeros(n, dtype=np.float64)
    n_radial = int(n_radial)
    if n_radial < 1:
        return dth, du, ring_med
    offset = 0
    for n_len in branch_nl:
        n_len = int(n_len)
        n_nodes = n_len * n_radial
        if n_len < 1 or offset + n_nodes > n:
            break
        sl = slice(offset, offset + n_nodes)
        R = r_star[sl].reshape(n_len, n_radial)
        V = valid[sl].reshape(n_len, n_radial)

        rp = np.roll(R, -1, axis=1)
        rm = np.roll(R, 1, axis=1)
        vp = np.roll(V, -1, axis=1)
        vm = np.roll(V, 1, axis=1)
        d1 = np.where(V & vp, np.abs(R - rp), 0.0)
        d2 = np.where(V & vm, np.abs(R - rm), 0.0)
        dth[sl] = np.maximum(d1, d2).reshape(-1)

        du_b = np.zeros((n_len, n_radial), dtype=np.float64)
        if n_len > 1:
            d_long = np.abs(R[:-1] - R[1:])
            ok = V[:-1] & V[1:]
            d_long = np.where(ok, d_long, 0.0)
            du_b[:-1] = np.maximum(du_b[:-1], d_long)
            du_b[1:] = np.maximum(du_b[1:], d_long)
        du[sl] = du_b.reshape(-1)

        med = np.zeros(n_len, dtype=np.float64)
        for i in range(n_len):
            vals = R[i, V[i]]
            med[i] = float(np.median(vals)) if vals.size else 0.0
        ring_med[sl] = np.repeat(med, n_radial)
        offset += n_nodes
    return dth, du, ring_med


def nearest_normal_offset_r_star(pos, normal, gt_pts, tube_radius=TUBE_RADIUS_MM):
    """Deprecated nearest-GT-vertex offset. Prefer `template_ray_r_star`.

    Kept so older probes can compare against the sac-wrong baseline of §7.1.
    Does not mark misses: every node is `valid=True`.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    normal = np.asarray(normal, dtype=np.float64).reshape(-1, 3)
    n = int(pos.shape[0])
    out = empty_r_star(n)
    if n == 0:
        return out
    gt_pts = np.asarray(gt_pts, dtype=np.float64).reshape(-1, 3) if gt_pts is not None else None
    if gt_pts is None or gt_pts.shape[0] == 0:
        return out
    nn = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = normal / np.clip(nn, 1e-12, None)
    _, idx = cKDTree(gt_pts).query(pos, k=1, workers=1)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    offset = ((gt_pts[idx] - pos) * normal).sum(axis=1)
    r_star = float(tube_radius) + offset
    out["r_star"] = r_star
    out["valid"] = np.ones(n, dtype=bool)
    out["ring_med"] = r_star.copy()
    return out


def stretch_distance_r_star(r_local, stretch):
    """r* = r_local + StretchDistance on template vertices (§7.1, §2.3).

    `StretchDistance` is the outward ray from the template to the GT. Missing
    or non-finite values stay `valid=False` (do not impute).
    """
    r_local = np.asarray(r_local, dtype=np.float64).reshape(-1)
    stretch = np.asarray(stretch, dtype=np.float64).reshape(-1)
    n = int(r_local.shape[0])
    out = empty_r_star(n)
    if n == 0:
        return out
    if stretch.shape[0] != n:
        return out
    ok = np.isfinite(stretch) & np.isfinite(r_local)
    out["r_star"] = np.where(ok, r_local + stretch, 0.0)
    out["valid"] = ok
    out["ring_med"] = out["r_star"].copy()
    return out


def mesh_r_star_edge_stats(pos, r_star, valid, edges):
    """Edge-based dθ / du / 1-ring median of r* for an unstructured mesh (§7.2).

    `dth` is the max |r*_i − r*_j| / edge_len over valid 1-ring neighbours
    (the StretchDistance gradient at the neck). `du` copies that physical
    gradient so Dirichlet sees it on either axis. `ring_med` is the median of
    r* over the vertex and its valid neighbours.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    r_star = np.asarray(r_star, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    n = int(r_star.shape[0])
    dth = np.zeros(n, dtype=np.float64)
    du = np.zeros(n, dtype=np.float64)
    ring_med = r_star.copy()
    if n == 0:
        return dth, du, ring_med
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2) if edges is not None else np.zeros((0, 2), dtype=np.int64)
    nbrs = [[] for _ in range(n)]
    seen = set()
    for a, b in edges:
        a = int(a)
        b = int(b)
        if a == b or a < 0 or b < 0 or a >= n or b >= n:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        nbrs[a].append(b)
        nbrs[b].append(a)
    acc_vals = [[] for _ in range(n)]
    for i in range(n):
        if valid[i]:
            acc_vals[i].append(float(r_star[i]))
        grad = 0.0
        pi = pos[i]
        ri = float(r_star[i])
        for j in nbrs[i]:
            if not (valid[i] and valid[j]):
                continue
            elen = float(np.linalg.norm(pos[j] - pi))
            if elen < 1e-8:
                continue
            g = abs(ri - float(r_star[j])) / elen
            if g > grad:
                grad = g
            acc_vals[i].append(float(r_star[j]))
        dth[i] = grad
        du[i] = grad
        if acc_vals[i]:
            ring_med[i] = float(np.median(np.asarray(acc_vals[i], dtype=np.float64)))
        elif not valid[i]:
            ring_med[i] = 0.0
    return dth, du, ring_med


def template_ray_r_star(
    pos,
    normal,
    r_local,
    gt_mesh,
    t_max=R_STAR_T_MAX_MM,
    t_inward=R_STAR_INWARD_MM,
    t_eps=R_STAR_T_EPS_MM,
    hit_tol=R_STAR_HIT_TOL,
    normal_dot_min=R_STAR_NORMAL_DOT,
    ambiguous_mm=R_STAR_AMBIGUOUS_MM,
):
    """Outward ray from each template vertex along its normal to the GT.

    `r* = r_local + t_hit` with `t_hit` the signed distance along the already
    outward template normal. Misses and grazes (`|n · n_cell|` too small) stay
    `valid=False`. Two accepted hits more than `ambiguous_mm` apart are
    `ambiguous=True` (neck / double wall). Invalid nodes are never flipped to
    `valid=True`.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    normal = np.asarray(normal, dtype=np.float64).reshape(-1, 3)
    r_local = np.asarray(r_local, dtype=np.float64).reshape(-1)
    n = int(pos.shape[0])
    out = empty_r_star(n)
    if n == 0 or gt_mesh is None:
        return out
    if r_local.shape[0] != n:
        r_local = np.zeros(n, dtype=np.float64)

    nn = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = normal / np.clip(nn, 1e-12, None)
    tree, cell_normals = _prepare_locator(gt_mesh)
    if tree is None:
        return out

    import vtk

    hit_points = vtk.vtkPoints()
    hit_cells = vtk.vtkIdList()
    t_max = float(t_max)
    t_inward = float(t_inward)
    lo = float(normal_dot_min)

    t_buf = np.full((n, MAX_R_STAR_HITS), np.nan, dtype=np.float64)
    ok_buf = np.zeros((n, MAX_R_STAR_HITS), dtype=bool)
    dot_buf = np.zeros((n, MAX_R_STAR_HITS), dtype=np.float64)
    n_hits = np.zeros(n, dtype=np.int32)
    for i in range(n):
        n_v = normal[i]
        origin = pos[i]
        p_start = origin - t_inward * n_v
        p_end = origin + t_max * n_v
        raw = _collect_hits(tree, p_start, p_end, hit_tol, hit_points, hit_cells)
        slot = 0
        for xyz, cid in raw:
            if slot >= MAX_R_STAR_HITS:
                break
            t = float(np.dot(xyz - origin, n_v))
            if t < -t_inward - 1e-6 or t > t_max + 1e-6:
                continue
            cn = _cell_normal(cell_normals, cid)
            cn_n = float(np.linalg.norm(cn))
            if cn_n > 1e-8:
                cn = cn / cn_n
                # Orientation-agnostic: GT files may still be wound inward (§2.3).
                nd = abs(float(np.dot(n_v, cn)))
                graze_ok = nd >= lo
            else:
                nd = 1.0
                graze_ok = True
            t_buf[i, slot] = t
            ok_buf[i, slot] = graze_ok
            dot_buf[i, slot] = nd
            slot += 1
        n_hits[i] = slot

    r_star = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    ambiguous = np.zeros(n, dtype=bool)
    for i in range(n):
        nh = int(n_hits[i])
        hits = [
            {
                "t": float(t_buf[i, j]),
                "voronoi_ok": bool(ok_buf[i, j]),
                "normal_dot": float(dot_buf[i, j]),
            }
            for j in range(nh)
        ]
        val, ok, amb = select_r_star_from_hits(
            hits,
            t_eps=t_eps,
            ambiguous_mm=ambiguous_mm,
            normal_dot_min=lo,
            normal_sign=1.0,
        )
        if ok:
            r_star[i] = float(r_local[i]) + float(val)
        valid[i] = ok
        ambiguous[i] = amb

    del tree, hit_points, hit_cells, cell_normals, t_buf, ok_buf, dot_buf
    out["r_star"] = r_star
    out["valid"] = valid
    out["ambiguous"] = ambiguous
    out["ring_med"] = r_star.copy()
    return out


def signed_distance_to_oriented_surface(query_pts, surf_pts, surf_normals):
    """Closest-vertex SDF; positive along the already-outward surface normal (sac)."""
    query_pts = np.asarray(query_pts, dtype=np.float64).reshape(-1, 3)
    surf_pts = np.asarray(surf_pts, dtype=np.float64).reshape(-1, 3)
    n = int(query_pts.shape[0])
    if n == 0 or surf_pts.shape[0] == 0:
        return np.zeros(n, dtype=np.float64)
    surf_normals = np.asarray(surf_normals, dtype=np.float64).reshape(-1, 3)
    if surf_normals.shape[0] != surf_pts.shape[0]:
        return np.zeros(n, dtype=np.float64)
    nn = np.linalg.norm(surf_normals, axis=1, keepdims=True)
    surf_normals = surf_normals / np.clip(nn, 1e-8, None)
    _, idx = cKDTree(surf_pts).query(query_pts, k=1, workers=1)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    delta = query_pts - surf_pts[idx]
    return np.einsum("ij,ij->i", surf_normals[idx], delta)


def transform_vessel_mesh(mesh, origin, rotation):
    """Apply the same COM-center + canonical rotation as the scaffold (in memory)."""
    import pyvista as pv

    if mesh is None:
        return None
    pv_mesh = pv.wrap(mesh)
    if pv_mesh.n_points == 0:
        return None
    if pv_mesh.n_cells > 0 and not bool(pv_mesh.is_all_triangles):
        pv_mesh = pv_mesh.triangulate()
    if pv_mesh.n_cells == 0:
        return None
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    pts = (np.asarray(pv_mesh.points, dtype=np.float64) - origin) @ rotation
    out = pv.PolyData(pts, pv_mesh.faces)
    pdata = getattr(pv_mesh, "point_data", None)
    if pdata is not None:
        for name in list(pdata.keys()):
            arr = np.asarray(pdata[name])
            if arr.shape[0] != pts.shape[0]:
                continue
            if arr.ndim == 2 and arr.shape[1] == 3:
                out.point_data[name] = arr @ rotation
            else:
                out.point_data[name] = arr
    return out


def closest_cell_normals(gt_mesh, query_pts):
    """Unit cell normals of the closest GT triangle to each query point."""
    import pyvista as pv

    query_pts = np.asarray(query_pts, dtype=np.float64).reshape(-1, 3)
    n = int(query_pts.shape[0])
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    mesh = pv.wrap(gt_mesh) if gt_mesh is not None else None
    if mesh is None or mesh.n_points == 0 or mesh.n_cells == 0:
        return np.zeros((n, 3), dtype=np.float64)
    if not bool(mesh.is_all_triangles):
        mesh = mesh.triangulate()
    try:
        mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
    except Exception:
        pass
    cn = np.asarray(getattr(mesh, "cell_normals", np.zeros((0, 3))), dtype=np.float64).reshape(-1, 3)
    if cn.shape[0] == 0:
        return np.zeros((n, 3), dtype=np.float64)
    try:
        cells = np.asarray(mesh.find_closest_cell(query_pts), dtype=np.int64).reshape(-1)
    except TypeError:
        cells = np.array([int(mesh.find_closest_cell(p)) for p in query_pts], dtype=np.int64)
    cells = np.clip(cells, 0, cn.shape[0] - 1)
    nrm = cn[cells]
    nn = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.clip(nn, 1e-8, None)
    nrm[nn.reshape(-1) <= 1e-8] = np.array([0.0, 0.0, 1.0])
    return nrm


def _prepare_locator(gt_mesh):
    import vtk

    import pyvista as pv

    pv_mesh = pv.wrap(gt_mesh)
    if pv_mesh.n_points == 0 or pv_mesh.n_cells == 0:
        return None, None
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(pv_mesh)
    normals_filter.ComputeCellNormalsOn()
    normals_filter.ComputePointNormalsOff()
    normals_filter.ConsistencyOn()
    normals_filter.SplittingOff()
    normals_filter.AutoOrientNormalsOff()
    normals_filter.Update()
    src = normals_filter.GetOutput()
    if src is None or src.GetNumberOfCells() == 0:
        return None, None
    # Own the output so the locator is not left dangling when the filter is GC'd.
    poly = vtk.vtkPolyData()
    poly.ShallowCopy(src)
    tree = vtk.vtkModifiedBSPTree()
    tree.SetDataSet(poly)
    tree.BuildLocator()
    cell_normals = poly.GetCellData().GetNormals()
    tree._keep_alive = poly
    return tree, cell_normals


def _cell_normal(cell_normals, cell_id):
    if cell_normals is None or cell_id < 0:
        return np.zeros(3, dtype=np.float64)
    if cell_id >= cell_normals.GetNumberOfTuples():
        return np.zeros(3, dtype=np.float64)
    return np.asarray(cell_normals.GetTuple3(cell_id), dtype=np.float64)


def _collect_hits(tree, p_start, p_end, tol, points, cell_ids):
    points.Reset()
    cell_ids.Reset()
    tree.IntersectWithLine(
        [float(p_start[0]), float(p_start[1]), float(p_start[2])],
        [float(p_end[0]), float(p_end[1]), float(p_end[2])],
        float(tol),
        points,
        cell_ids,
    )
    n_hit = int(points.GetNumberOfPoints())
    n_ids = int(cell_ids.GetNumberOfIds())
    hits = []
    for j in range(n_hit):
        xyz = np.asarray(points.GetPoint(j), dtype=np.float64)
        cid = int(cell_ids.GetId(j)) if j < n_ids else -1
        hits.append((xyz, cid))
    return hits


def compute_level_r_star(
    pos,
    normal,
    u,
    tract_id,
    branch_nl,
    n_radial,
    dense_tracts,
    gt_mesh,
    tube_radius=TUBE_RADIUS_MM,
    t_max=R_STAR_T_MAX_MM,
    t_inward=R_STAR_INWARD_MM,
    t_eps=R_STAR_T_EPS_MM,
    hit_tol=R_STAR_HIT_TOL,
    normal_dot_min=R_STAR_NORMAL_DOT,
    ambiguous_mm=R_STAR_AMBIGUOUS_MM,
):
    """Ray-cast r* for one scaffold level. Returns dict of numpy arrays."""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    normal = np.asarray(normal, dtype=np.float64).reshape(-1, 3)
    n = int(pos.shape[0])
    out = empty_r_star(n)
    if n == 0 or gt_mesh is None:
        return out

    nn = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = normal / np.clip(nn, 1e-12, None)
    origins = pos - float(tube_radius) * normal
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    tract_id = np.asarray(tract_id, dtype=np.int64).reshape(-1)
    branch_nl = [int(v) for v in np.asarray(branch_nl).reshape(-1).tolist()]
    n_radial = int(n_radial)

    cl_xyz, cl_u, cl_tract, arcs = pack_dense_centerline(dense_tracts)
    if cl_xyz.shape[0] == 0:
        return out
    cl_tree = cKDTree(cl_xyz)

    tree, cell_normals = _prepare_locator(gt_mesh)
    if tree is None:
        return out

    import vtk

    hit_points = vtk.vtkPoints()
    hit_cells = vtk.vtkIdList()

    t_max = float(t_max)
    t_inward = float(t_inward)
    n_tracts = max(len(arcs), 1)
    max_ds = np.array(
        [
            max_ds_for_tract(arcs[tid] if tid < len(arcs) else 0.0, branch_nl[tid] if tid < len(branch_nl) else 1)
            for tid in range(n_tracts)
        ],
        dtype=np.float64,
    )
    arc_arr = np.array([arcs[tid] if tid < len(arcs) else 0.0 for tid in range(n_tracts)], dtype=np.float64)

    t_buf = np.full((n, MAX_R_STAR_HITS), np.nan, dtype=np.float64)
    ok_buf = np.zeros((n, MAX_R_STAR_HITS), dtype=bool)
    dot_buf = np.zeros((n, MAX_R_STAR_HITS), dtype=np.float64)
    n_hits = np.zeros(n, dtype=np.int32)
    for i in range(n):
        n_v = normal[i]
        origin = origins[i]
        p_start = origin - t_inward * n_v
        p_end = origin + t_max * n_v
        raw = _collect_hits(tree, p_start, p_end, hit_tol, hit_points, hit_cells)
        tid = int(tract_id[i]) if i < tract_id.shape[0] else 0
        tid = min(max(tid, 0), n_tracts - 1)
        origin_u = float(u[i]) if i < u.shape[0] else 0.0
        slot = 0
        for xyz, cid in raw:
            if slot >= MAX_R_STAR_HITS:
                break
            t = float(np.dot(xyz - origin, n_v))
            if t < -t_inward - 1e-6 or t > t_max + 1e-6:
                continue
            _, nn_idx = cl_tree.query(xyz, k=1)
            nn_idx = int(nn_idx)
            v_ok = voronoi_ok_hit(
                cl_tract[nn_idx],
                cl_u[nn_idx],
                tid,
                origin_u,
                arc_arr[tid],
                max_ds[tid],
            )
            cn = _cell_normal(cell_normals, cid)
            cn_n = float(np.linalg.norm(cn))
            if cn_n > 1e-8:
                cn = cn / cn_n
            t_buf[i, slot] = t
            ok_buf[i, slot] = bool(v_ok)
            dot_buf[i, slot] = float(np.dot(n_v, cn))
            slot += 1
        n_hits[i] = slot

    sign = _choose_normal_sign_arrays(ok_buf, dot_buf, n_hits, normal_dot_min=normal_dot_min)
    r_star = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    ambiguous = np.zeros(n, dtype=bool)
    for i in range(n):
        nh = int(n_hits[i])
        hits = [
            {"t": float(t_buf[i, j]), "voronoi_ok": bool(ok_buf[i, j]), "normal_dot": float(dot_buf[i, j])}
            for j in range(nh)
        ]
        val, ok, amb = select_r_star_from_hits(
            hits,
            t_eps=t_eps,
            ambiguous_mm=ambiguous_mm,
            normal_dot_min=normal_dot_min,
            normal_sign=sign,
        )
        r_star[i] = val
        valid[i] = ok
        ambiguous[i] = amb

    del tree, hit_points, hit_cells, cell_normals, t_buf, ok_buf, dot_buf

    dth, du, ring_med = r_star_grid_stats(r_star, valid, branch_nl, n_radial)
    return {
        "r_star": r_star,
        "valid": valid,
        "ambiguous": ambiguous,
        "dth": dth,
        "du": du,
        "ring_med": ring_med,
    }
