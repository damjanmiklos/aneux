"""Uniform parent-vessel template with the aneurysm rebuilt from four spheres.

Per case:

1. Detection. The parent artery is the centerline tube whose inscribed radius
   has had the sac spike opened out along the graph. The sac and its neck are
   the compact patch of wall that stands off that tube, grown from the dome.
2. Four spheres. Seeded at the deepest medial balls of the sac and refined
   jointly (L-BFGS) so that the blended model -- parent tube smooth-union the
   four spheres -- passes through the sac wall without crossing it.
3. One implicit field. Parent tube (balls swept along the centerline, plus a
   short stub through each ostium plane) smooth-union the four spheres,
   sampled on a voxel grid. The smooth union is the "high smoothing" at the
   neck; a windowed-sinc pass removes the voxel staircase.
4. One marching-cubes surface, closed. Each ostium is opened with one planar
   cut limited to a short cylinder around its stub, so the rim is exactly the
   ground-truth ostium plane.
5. One uniform VMTK remesh (boundary included), then hard gates: one
   component, manifold, no bowties, genus 0, one rim per ostium, no
   self-intersections, outward winding.

There is no repair cascade: the field is built so that the surface is clean,
and a case that is not is reported, not patched. Typical cost is a few
seconds on one core; ``CaseModel.manifold`` is the part training can call again
with new spheres without repeating detection or the parent field.

Inputs are ``cleandata/uniformly_remeshed`` (mesh + ``.ostium_frames.npz``) and
``cleandata/original_centerline``. Meshes land in
``datatransform/template_creation/output_vessel_aneurysm`` next to a
``{id}.spheres.npz`` holding the four spheres and the sac vertex ids.
"""
import os
import sys

os.environ["VTK_OFFSCREEN"] = "1"
os.environ["EGL_PLATFORM"] = "surfaceless"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VTK_NUMBER_OF_THREADS"] = "1"
os.environ["VTK_SMP_MAX_THREADS"] = "1"

import argparse
import time

import numpy as np
import vtk
from scipy import ndimage
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from aneux_paths import (
    CLEANDATA_ORIGINAL_CENTERLINE,
    CLEANDATA_UNIFORM,
    TEMPLATE_OUTPUT_VESSEL_ANEURYSM as DEFAULT_OUTPUT_DIR,
)

from batch_run_log import (
    add_run_log_args,
    configure_batch_logging,
    finalize_run_logs,
    run_logged_case,
)
from surface_polish import self_intersections, signed_volume
from variable_remeshing import resolve_cut_frames
from vessel_pipeline import (
    DEFAULT_TARGET_EDGE_LENGTH,
    TemplateQualityError,
    _grow_solid_ball,
    add_shared_cli_args,
    read_polydata,
    run_batch,
    to_vtk_poly,
    save_polydata,
    square_frames_to_rims,
    with_dataset_id,
)

LOG_FOLDER = "vessel_aneurysm_logs"
N_SPHERES = 4

# Voxel of the implicit field. The surface is smoothed and remeshed at
# >= 0.3 mm afterwards, so 0.2 mm resolves the thinnest kept tube (0.35 mm
# radius, 3.5 voxels across) without paying for more.
DEFAULT_GRID_SPACING = 0.2
DEFAULT_MAX_GRID_SIZE = 320
# Thinnest tube the template carries. Ostia narrower than this on the ground
# truth are still opened, on a tube of this radius.
R_FLOOR_MM = 0.35
# Every sphere overlaps the parent or a connected sphere by at least this.
CONNECT_OVERLAP_MM = 0.3
# Smooth-union width between the sac spheres and the parent (and each other).
BLEND_MM = 0.9
# Windowed-sinc smoothing of the marching-cubes surface.
SMOOTH_PASSBAND = 0.03
SMOOTH_ITERS = 25
# Uniform edge: 0.6 of the narrowest rim radius, clamped to this range.
MIN_EDGE_MM = 0.3
EDGE_OVER_RIM_RADIUS = 0.6
REMESH_ITERS = 2
# Rims must sit on the ground-truth ostium planes. The clip puts them there;
# the remesher slides rim vertices along the rim, so anything it leaves off
# the plane is projected back and more than this is a failure.
PLANE_TOL_MM = 1e-3
REMESH_CONN_ITERS = 1


# ---------------------------------------------------------------------------
# Small mesh helpers (numpy only, no repair)
# ---------------------------------------------------------------------------

def _poly_arrays(poly):
    pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    polys = poly.GetPolys()
    if polys.GetNumberOfCells() == 0:
        return pts, np.zeros((0, 3), dtype=np.int64)
    conn = vtk_to_numpy(polys.GetConnectivityArray()).astype(np.int64)
    offs = vtk_to_numpy(polys.GetOffsetsArray()).astype(np.int64)
    if not np.all(np.diff(offs) == 3):
        raise TemplateQualityError("surface is not a triangle mesh")
    return pts, conn.reshape(-1, 3)


def _poly_from_arrays(pts, faces):
    poly = vtk.vtkPolyData()
    vpts = vtk.vtkPoints()
    vpts.SetDataTypeToDouble()
    vpts.SetData(numpy_to_vtk(np.ascontiguousarray(pts, dtype=np.float64), deep=True))
    poly.SetPoints(vpts)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    cells = vtk.vtkCellArray()
    offs = numpy_to_vtk(np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64), deep=True,
                        array_type=vtk.VTK_ID_TYPE)
    conn = numpy_to_vtk(faces.ravel(), deep=True, array_type=vtk.VTK_ID_TYPE)
    cells.SetData(offs, conn)
    poly.SetPolys(cells)
    return poly


def _compact(pts, faces):
    """Drop unused vertices."""
    used = np.unique(faces)
    remap = np.full(len(pts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return pts[used], remap[faces]


def _edge_table(faces):
    e = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    e.sort(axis=1)
    uniq, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    return uniq, inv.reshape(-1), cnt


def _vertex_components(n, edges):
    g = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)


def _boundary_loops(pts, faces):
    """Boundary loops as lists of vertex rings (assumes a manifold rim)."""
    uniq, _inv, cnt = _edge_table(faces)
    rim = uniq[cnt == 1]
    if len(rim) == 0:
        return []
    n = len(pts)
    ncomp, lab = _vertex_components(n, rim)
    rim_v = np.unique(rim)
    loops = []
    for c in np.unique(lab[rim_v]):
        loops.append(rim_v[lab[rim_v] == c])
    return loops


def _bowtie_count(faces, n):
    """Vertices whose incident faces form more than one fan."""
    # Faces around a vertex are connected through edges that contain the vertex.
    uniq, inv, _cnt = _edge_table(faces)
    nf = len(faces)
    fid = np.tile(np.arange(nf), 3)
    # For each (vertex, edge) incidence link faces sharing that edge.
    order = np.argsort(inv, kind="stable")
    inv_s = inv[order]
    f_s = fid[order]
    same = inv_s[1:] == inv_s[:-1]
    fa = f_s[:-1][same]
    fb = f_s[1:][same]
    ea = uniq[inv_s[:-1][same]]
    # node = (vertex, face); a fan component is a connected set of such nodes.
    nodes_v = faces.ravel()
    nodes_f = np.repeat(np.arange(nf), 3)
    key = nodes_v * nf + nodes_f
    sorter = np.argsort(key)
    key_s = key[sorter]

    def node(v, f):
        return sorter[np.searchsorted(key_s, v * nf + f)]

    a0 = node(ea[:, 0], fa)
    b0 = node(ea[:, 0], fb)
    a1 = node(ea[:, 1], fa)
    b1 = node(ea[:, 1], fb)
    src = np.concatenate((a0, a1))
    dst = np.concatenate((b0, b1))
    m = len(nodes_v)
    g = csr_matrix((np.ones(len(src)), (src, dst)), shape=(m, m))
    _nc, lab = connected_components(g, directed=False)
    fans = np.unique(np.stack((nodes_v, lab), axis=1), axis=0)
    per_v = np.bincount(fans[:, 0], minlength=n)
    return int(np.sum(per_v > 1))


def surface_report(pts, faces):
    """Topology and quality numbers for a triangle surface."""
    uniq, _inv, cnt = _edge_table(faces)
    n_used = len(np.unique(faces))
    ncomp, _lab = _vertex_components(len(pts), uniq)
    ncomp -= len(pts) - n_used
    loops = _boundary_loops(pts, faces)
    euler = n_used - len(uniq) + len(faces)
    genus = (2 * ncomp - len(loops) - euler) / 2.0
    e = pts[uniq[:, 0]] - pts[uniq[:, 1]]
    elen = np.sqrt(np.einsum("ij,ij->i", e, e))
    a = pts[faces[:, 1]] - pts[faces[:, 0]]
    b = pts[faces[:, 2]] - pts[faces[:, 0]]
    area2 = np.linalg.norm(np.cross(a, b), axis=1)
    l2 = (elen[_inv_face_edges(faces, uniq)] ** 2).sum(axis=1)
    # 1 for an equilateral triangle, 0 for a degenerate one.
    q = 2.0 * np.sqrt(3.0) * area2 / np.maximum(l2, 1e-30)
    return {
        "n_points": int(n_used),
        "n_faces": int(len(faces)),
        "components": int(ncomp),
        "loops": loops,
        "n_loops": len(loops),
        "nonmanifold_edges": int(np.sum(cnt > 2)),
        "bowties": _bowtie_count(faces, len(pts)),
        "genus": float(genus),
        "edge_mean": float(elen.mean()),
        "edge_cv": float(elen.std() / max(elen.mean(), 1e-12)),
        "edge_min": float(elen.min()),
        "edge_max": float(elen.max()),
        "q_min": float(q.min()),
        "q_p01": float(np.percentile(q, 1)),
        "area": float(0.5 * area2.sum()),
    }


def _inv_face_edges(faces, uniq):
    e = np.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1)
    e = np.sort(e, axis=2).reshape(-1, 2)
    key_u = uniq[:, 0] * (uniq.max() + 1) + uniq[:, 1]
    key_e = e[:, 0] * (uniq.max() + 1) + e[:, 1]
    return np.searchsorted(key_u, key_e).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

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


def load_case(v_file, centerline_path, dataset_id=None):
    mesh = read_polydata(v_file)
    if not os.path.isfile(centerline_path):
        raise TemplateQualityError(f"original centerline not found: {centerline_path}")
    cl = read_polydata(centerline_path)
    if "MaximumInscribedSphereRadius" not in cl.point_data:
        raise TemplateQualityError("centerline has no MaximumInscribedSphereRadius")
    pts, faces = _poly_arrays(_as_vtk(mesh))
    up, ur, edges = _dedup_centerline(
        np.asarray(cl.points, dtype=np.float64),
        np.asarray(cl.point_data["MaximumInscribedSphereRadius"], dtype=np.float64),
        np.asarray(cl.lines),
    )
    frames, _src = resolve_cut_frames(dataset_id, v_file)
    if not frames:
        raise TemplateQualityError("no ostium frames next to the input mesh")
    frames, _n = square_frames_to_rims(frames, mesh)
    return {
        "mesh": mesh,
        "pts": pts,
        "faces": faces,
        "up": up,
        "ur": ur,
        "cl_edges": edges,
        "frames": frames,
    }


def _as_vtk(mesh):
    if isinstance(mesh, vtk.vtkPolyData):
        return mesh
    out = vtk.vtkPolyData()
    out.ShallowCopy(mesh)
    return out


def wall_normals(pts, faces, up):
    """Area-weighted vertex normals, signed to point away from the centerline."""
    fn = np.cross(pts[faces[:, 1]] - pts[faces[:, 0]], pts[faces[:, 2]] - pts[faces[:, 0]])
    vn = np.zeros_like(pts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-30)
    sub = np.arange(0, len(pts), max(1, len(pts) // 3000))
    _d, i = cKDTree(up).query(pts[sub], workers=1)
    if np.median(np.einsum("ij,ij->i", pts[sub] - up[i], vn[sub])) < 0:
        vn = -vn
    return vn


# ---------------------------------------------------------------------------
# 1. Detection
# ---------------------------------------------------------------------------

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


def detect_aneurysm(pts, faces, up, r_parent):
    """Sac plus a short neck lip, and almost none of the parent wall.

    ``excess`` is how far a wall vertex stands off the parent tube. The dome
    is its maximum. The sac is the dome's connected level set at the level
    whose set is compact and high (the step before it spills along the
    parent); a geodesic lip of about half a parent radius picks up the neck.
    """
    t0 = time.perf_counter()
    dist, nearest = cKDTree(up).query(pts, k=1, workers=1)
    nearest = np.asarray(nearest, dtype=np.int64)
    excess = dist - r_parent[nearest]
    mesh_edges, _inv, _cnt = _edge_table(faces)
    adj = _adjacency(len(pts), mesh_edges)
    elen = np.linalg.norm(pts[mesh_edges[:, 0]] - pts[mesh_edges[:, 1]], axis=1)
    median_edge = float(np.median(elen))
    dome = int(np.argmax(excess))
    peak = float(excess[dome])
    r_dome = float(max(r_parent[nearest[dome]], 0.45))
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
    if int(core.sum()) < 30:
        raise TemplateQualityError(f"no aneurysm found (peak excess {peak:.2f} mm)")
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
        "seconds": time.perf_counter() - t0,
    }
    return mask, excess, info


# ---------------------------------------------------------------------------
# 2. Four spheres
# ---------------------------------------------------------------------------

def smin(a, b, k):
    """Quadratic smooth minimum; exact min where |a - b| >= k."""
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - 0.25 * k * h * h


def _parent_value(q, up, r_parent, tree, kn=8):
    _d, j = tree.query(q, k=min(kn, len(up)), workers=1)
    j = np.atleast_2d(j)
    return np.min(np.linalg.norm(q[:, None, :] - up[j], axis=2) - r_parent[j], axis=1)


def _fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    th = np.pi * (1.0 + 5 ** 0.5) * i
    return np.stack((np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)), axis=1)


def fit_aneurysm_spheres(pts, normals, mask, up, r_parent, blend=BLEND_MM, n_spheres=N_SPHERES):
    """Four spheres whose blended union with the parent tracks the sac wall.

    Objective, over the sac wall samples s and samples q on each sphere:
    mean F(s)^2 (the model surface passes through the wall) plus
    mean max(0, sd_wall(q))^2 on the exposed part of each sphere (the model
    does not cross the wall), where F is the same smooth union the surface is
    built from. Seeded from the deepest medial balls of the sac (greedy cover),
    refined jointly with L-BFGS on the 16 parameters. Cheap, not exhaustive.
    """
    t0 = time.perf_counter()
    sac_idx = np.flatnonzero(mask)
    sac = pts[sac_idx]
    step = max(1, len(sac) // 1500)
    S = np.ascontiguousarray(sac[::step])
    ptree = cKDTree(up)
    fp_S = _parent_value(S, up, r_parent, ptree)

    # Signed distance to the wall (positive outside) on a grid around the sac.
    lo = sac.min(axis=0) - 2.5
    hi = sac.max(axis=0) + 2.5
    vox = max(0.3, float(np.max(hi - lo)) / 60.0)
    axes = [np.arange(lo[i], hi[i] + vox, vox) for i in range(3)]
    shape = tuple(len(a) for a in axes)
    G = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    box = np.all((pts >= lo - 3.0) & (pts <= hi + 3.0), axis=1)
    wall_ids = np.flatnonzero(box)
    wtree = cKDTree(pts[wall_ids])
    d, j = wtree.query(G, k=1, workers=1)
    j = wall_ids[j]
    sign = np.sign(np.einsum("ij,ij->i", G - pts[j], normals[j]))
    sd = (d * np.where(sign == 0, 1.0, sign)).reshape(shape)
    owned = mask[j].reshape(shape)

    def sd_at(q):
        idx = ((q - lo) / vox).T
        return ndimage.map_coordinates(sd, idx, order=1, mode="nearest")

    # Medial candidates: local maxima of depth among voxels the sac owns.
    depth = np.where((sd < -0.2) & owned, -sd, 0.0)
    is_max = (depth > 0.3) & (depth >= ndimage.maximum_filter(depth, size=3) - 1e-9)
    cand = np.argwhere(is_max)
    if len(cand) == 0:
        k = np.unravel_index(int(np.argmax(-sd * owned)), shape)
        cand = np.array([k])
    cand_c = np.stack([axes[k][cand[:, k]] for k in range(3)], axis=1)
    cand_r = np.maximum(depth[tuple(cand.T)], 0.3)
    order = np.argsort(cand_r)[::-1][:200]
    cand_c, cand_r = cand_c[order], cand_r[order]

    def union(C, R, P, fp):
        f = fp
        for c, r in zip(C, R):
            f = smin(f, np.linalg.norm(P - c, axis=1) - r, blend)
        return f

    # Greedy cover of the sac wall, gain measured with the blended model.
    chosen = []
    base = fp_S.copy()
    for _ in range(n_spheres):
        best_gain, best_i = 0.0, None
        cur = np.mean(base ** 2)
        for i in range(len(cand_c)):
            if i in chosen:
                continue
            f = smin(base, np.linalg.norm(S - cand_c[i], axis=1) - cand_r[i], blend)
            gain = cur - np.mean(f ** 2)
            if gain > best_gain:
                best_gain, best_i = gain, i
        if best_i is None:
            break
        chosen.append(best_i)
        base = smin(base, np.linalg.norm(S - cand_c[best_i], axis=1) - cand_r[best_i], blend)
    C0 = [cand_c[i] for i in chosen]
    R0 = [cand_r[i] for i in chosen]
    while len(C0) < n_spheres:
        # Fewer medial maxima than spheres: split the largest ball.
        k = int(np.argmax(R0))
        jitter = np.array([0.3, -0.2, 0.25]) * R0[k] * (1 + len(C0)) / n_spheres
        C0.append(C0[k] + jitter)
        R0.append(0.7 * R0[k])
    C0 = np.asarray(C0, dtype=np.float64)
    R0 = np.asarray(R0, dtype=np.float64)

    U = _fib_sphere(40)
    r_min = 0.3

    def cost(x):
        C = x[: 3 * n_spheres].reshape(n_spheres, 3)
        R = x[3 * n_spheres:]
        f = union(C, R, S, fp_S)
        fit = np.mean(f * f)
        Q = (C[:, None, :] + R[:, None, None] * U[None]).reshape(-1, 3)
        out = np.maximum(sd_at(Q), 0.0)
        # Only the exposed part of a sphere is model surface.
        dQ = np.linalg.norm(Q[:, None, :] - C[None], axis=2) - R[None]
        own = np.repeat(np.arange(n_spheres), len(U))
        dQ[np.arange(len(Q)), own] = np.inf
        exposed = (dQ.min(axis=1) > 0.0) & (fp_Q_cache(Q) > 0.0)
        cross = np.mean((out * exposed) ** 2)
        small = np.sum(np.maximum(r_min - R, 0.0) ** 2)
        return fit + 2.0 * cross + 10.0 * small

    def fp_Q_cache(Q):
        return _parent_value(Q, up, r_parent, ptree, kn=4)

    x0 = np.concatenate((C0.ravel(), R0))
    res = minimize(cost, x0, method="L-BFGS-B", options={"maxiter": 60, "eps": 1e-3})
    x = res.x if np.isfinite(res.fun) and res.fun <= cost(x0) else x0
    C = x[: 3 * n_spheres].reshape(n_spheres, 3)
    R = np.maximum(x[3 * n_spheres:], r_min)
    C, n_moved = connect_spheres(C, R, up, r_parent, ptree)
    f_par = fp_S
    f_fin = union(C, R, S, fp_S)
    info = {
        "parent_rmse_mm": float(np.sqrt(np.mean(np.maximum(f_par, 0.0) ** 2))),
        "union_rmse_mm": float(np.sqrt(np.mean(f_fin ** 2))),
        "union_mae_mm": float(np.mean(np.abs(f_fin))),
        "iters": int(res.nit),
        "moved": n_moved,
        "seconds": time.perf_counter() - t0,
    }
    return C, R, info


def connect_spheres(C, R, up, r_parent, ptree=None, overlap=CONNECT_OVERLAP_MM):
    """Pull each sphere into the parent-connected union.

    A sphere is connected when it overlaps the parent tube (or a connected
    sphere) by ``overlap``. One that is not is moved straight toward whichever
    connected body is nearest until it overlaps it; nearest spheres go first.
    The fit nearly always satisfies this already; it is the guarantee that the
    four spheres never come out as an island the surface would drop.
    """
    ptree = ptree if ptree is not None else cKDTree(up)
    C = np.array(C, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    connected = np.zeros(len(C), dtype=bool)
    n_moved = 0
    for _ in range(len(C)):
        best = None
        for i in np.flatnonzero(~connected):
            # Gap to the parent (via the nearest few tube nodes) ...
            _d, j = ptree.query(C[i], k=min(8, len(up)))
            j = np.atleast_1d(j)
            gaps = np.linalg.norm(up[j] - C[i], axis=1) - r_parent[j] - R[i]
            k = int(np.argmin(gaps))
            cand = (gaps[k], up[j[k]], r_parent[j[k]])
            # ... or to a connected sphere.
            for m in np.flatnonzero(connected):
                g = np.linalg.norm(C[m] - C[i]) - R[m] - R[i]
                if g < cand[0]:
                    cand = (g, C[m], R[m])
            if best is None or cand[0] < best[1][0]:
                best = (i, cand)
        i, (gap, anchor, r_anchor) = best
        if gap > -overlap:
            v = C[i] - anchor
            dist = float(np.linalg.norm(v))
            want = max(r_anchor + R[i] - overlap, 0.0)
            C[i] = anchor + (v / dist if dist > 1e-9 else np.array([1.0, 0, 0])) * want
            n_moved += 1
        connected[i] = True
    return C, n_moved


# ---------------------------------------------------------------------------
# 3-5. Field, surface, openings, remesh
# ---------------------------------------------------------------------------

def _stamp(field, origin, g, centers, radii, reach_extra):
    """field = min(field, |x - c| - r) for every ball, in its own box."""
    nz, ny, nx = field.shape
    ox, oy, oz = origin
    for c, r in zip(centers, radii):
        reach = r + reach_extra
        i0 = max(0, int((c[0] - reach - ox) / g))
        i1 = min(nx, int((c[0] + reach - ox) / g) + 2)
        j0 = max(0, int((c[1] - reach - oy) / g))
        j1 = min(ny, int((c[1] + reach - oy) / g) + 2)
        k0 = max(0, int((c[2] - reach - oz) / g))
        k1 = min(nz, int((c[2] + reach - oz) / g) + 2)
        if i1 <= i0 or j1 <= j0 or k1 <= k0:
            continue
        dx = (ox + np.arange(i0, i1, dtype=np.float32) * g - c[0]) ** 2
        dy = (oy + np.arange(j0, j1, dtype=np.float32) * g - c[1]) ** 2
        dz = (oz + np.arange(k0, k1, dtype=np.float32) * g - c[2]) ** 2
        val = np.sqrt(dz[:, None, None] + dy[None, :, None] + dx[None, None, :]) - np.float32(r)
        sl = field[k0:k1, j0:j1, i0:i1]
        np.minimum(sl, val, out=sl)


def _sample_segments(p0, p1, r0, r1, step):
    """Balls every ``step`` along segments (p0 -> p1) with linear radii."""
    seg = p1 - p0
    length = np.linalg.norm(seg, axis=1)
    n = np.maximum(np.ceil(length / step).astype(np.int64), 1)
    idx = np.repeat(np.arange(len(p0)), n)
    t = (np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
    c = p0[idx] + t[:, None] * seg[idx]
    r = r0[idx] + t * (r1[idx] - r0[idx])
    return np.vstack((c, p1)), np.concatenate((r, r1))


class CaseModel:
    """Everything about one case that does not depend on the spheres.

    ``manifold(centers, radii)`` is the hot path: it copies the cached parent
    field, blends the spheres in, and builds the final uniform surface.
    """

    def __init__(self, up, r_parent, cl_edges, frames, bounds,
                 grid_spacing=DEFAULT_GRID_SPACING, max_grid_size=DEFAULT_MAX_GRID_SIZE,
                 target_edge_length=DEFAULT_TARGET_EDGE_LENGTH, blend=BLEND_MM):
        t0 = time.perf_counter()
        self.blend = float(blend)
        rp = np.maximum(r_parent, R_FLOOR_MM)
        up = np.asarray(up, dtype=np.float64)
        frames = [
            (np.asarray(fr["origin"], dtype=np.float64).reshape(3),
             np.asarray(fr["normal"], dtype=np.float64).reshape(3)
             / np.linalg.norm(np.asarray(fr["normal"], dtype=np.float64)),
             float(fr["radius"]))
            for fr in frames
        ]
        # Centerline beyond an ostium plane (and inside its tube) is dropped:
        # each opening is carried by its own straight stub, so the cut lands
        # on the ground-truth plane and nothing pokes through it.
        # Only a piece that ends in a centerline leaf is this vessel's own
        # continuation; another branch merely passing beyond the plane stays.
        keep = np.ones(len(up), dtype=bool)
        degree = np.bincount(cl_edges.ravel(), minlength=len(up)) if len(cl_edges) else np.zeros(len(up), int)
        adj_cl = _adjacency(len(up), cl_edges) if len(cl_edges) else None
        for o, n, r_gt in frames:
            rel = up - o
            s = rel @ n
            lat = np.linalg.norm(rel - s[:, None] * n, axis=1)
            beyond = (s > 0.0) & (s < 2.0 * r_gt + 2.0) & (lat < 1.5 * r_gt + 1.0)
            if adj_cl is None or not beyond.any():
                keep &= ~beyond
                continue
            sel = np.flatnonzero(beyond)
            _nc, lab = connected_components(adj_cl[sel][:, sel], directed=False)
            for c in np.unique(lab):
                ids = sel[lab == c]
                if np.any(degree[ids] <= 1):
                    keep[ids] = False
        if not keep.any():
            raise TemplateQualityError("centerline lies entirely beyond the ostium planes")
        e = cl_edges[keep[cl_edges[:, 0]] & keep[cl_edges[:, 1]]] if len(cl_edges) else cl_edges
        kept_ids = np.flatnonzero(keep)
        tree = cKDTree(up[kept_ids])
        self.ostia = []
        stub_c, stub_r = [], []
        for o, n, r_gt in frames:
            _d, j = tree.query(o)
            node = int(kept_ids[j])
            # Tube radius at the plane: the parent's, never wider than the rim.
            r_t = float(max(min(rp[node], max(r_gt, R_FLOOR_MM)), R_FLOOR_MM))
            back = max(1.0 * r_t, 0.6)
            out_len = max(1.0 * r_t, 0.6) + 2.0 * grid_spacing
            a = o - back * n
            b = o + out_len * n
            stub_c.append(np.stack((up[node], a)))
            stub_r.append(np.array([rp[node], r_t]))
            stub_c.append(np.stack((a, b)))
            stub_r.append(np.array([r_t, r_t]))
            self.ostia.append({"origin": o, "normal": n, "r": r_t, "out_len": out_len,
                               "gt_radius": r_gt})
        r_min = min(op["r"] for op in self.ostia)
        self.edge = float(np.clip(EDGE_OVER_RIM_RADIUS * r_min, MIN_EDGE_MM, target_edge_length))

        step0 = max(0.5 * float(grid_spacing), 0.08)
        if len(e):
            c, r = _sample_segments(up[e[:, 0]], up[e[:, 1]], rp[e[:, 0]], rp[e[:, 1]], step0)
        else:
            c, r = up[kept_ids], rp[kept_ids]
        for sc, sr in zip(stub_c, stub_r):
            c2, r2 = _sample_segments(sc[:1], sc[1:], sr[:1], sr[1:], step0)
            c = np.vstack((c, c2))
            r = np.concatenate((r, r2))
        # Balls closer than a quarter step to a kept one add nothing.
        key = np.round(c / (0.25 * step0)).astype(np.int64)
        _u, first = np.unique(key, axis=0, return_index=True)
        c, r = c[first], r[first]

        pad = self.blend + 1.0
        lo = np.minimum(np.asarray(bounds[0::2], dtype=np.float64), (c - r[:, None]).min(axis=0)) - pad
        hi = np.maximum(np.asarray(bounds[1::2], dtype=np.float64), (c + r[:, None]).max(axis=0)) + pad
        g = float(grid_spacing)
        if np.max(hi - lo) / g + 1 > max_grid_size:
            g = float(np.max(hi - lo)) / (max_grid_size - 1)
        self.g = g
        dims = np.ceil((hi - lo) / g).astype(np.int64) + 1
        self.origin = lo
        self.shape = (int(dims[2]), int(dims[1]), int(dims[0]))
        field = np.full(self.shape, 1.0e3, dtype=np.float32)
        _stamp(field, lo, g, c, r, reach_extra=self.blend + 2.0 * g)
        self.parent_field = field
        seed = kept_ids[int(np.argmax(rp[kept_ids]))]
        self.parent_seed = tuple(int(v) for v in np.round((up[seed] - lo) / g)[::-1])
        self.seconds_parent = time.perf_counter() - t0

    # -- field ---------------------------------------------------------------
    def field_with_spheres(self, centers, radii):
        field = self.parent_field.copy()
        g, lo, k = self.g, self.origin, self.blend
        for c, r in zip(np.asarray(centers, float), np.asarray(radii, float)):
            reach = r + k + 2.0 * g
            i0 = np.maximum(((c - reach - lo) / g).astype(int), 0)
            i1 = np.minimum(((c + reach - lo) / g).astype(int) + 2, np.array(self.shape[::-1]))
            if np.any(i1 <= i0):
                continue
            xs = lo[0] + np.arange(i0[0], i1[0], dtype=np.float32) * g - c[0]
            ys = lo[1] + np.arange(i0[1], i1[1], dtype=np.float32) * g - c[1]
            zs = lo[2] + np.arange(i0[2], i1[2], dtype=np.float32) * g - c[2]
            d = np.sqrt(zs[:, None, None] ** 2 + ys[None, :, None] ** 2 + xs[None, None, :] ** 2) - np.float32(r)
            sl = field[i0[2]:i1[2], i0[1]:i1[1], i0[0]:i1[0]]
            sl[...] = smin(sl, d, np.float32(k))
        for ax in range(3):
            idx = [slice(None)] * 3
            for end in (0, -1):
                idx[ax] = end
                border = field[tuple(idx)]
                np.maximum(border, np.float32(self.g), out=border)
        return field

    def _image(self, field):
        img = vtk.vtkImageData()
        img.SetDimensions(self.shape[2], self.shape[1], self.shape[0])
        img.SetOrigin(*self.origin)
        img.SetSpacing(self.g, self.g, self.g)
        arr = numpy_to_vtk(field.ravel(), deep=False)
        arr.SetName("f")
        img.GetPointData().SetScalars(arr)
        img._keep = field  # keep the buffer alive
        return img

    def _contour(self, field):
        fe = vtk.vtkFlyingEdges3D()
        fe.SetInputData(self._image(field))
        fe.SetValue(0, 0.0)
        fe.ComputeNormalsOff()
        fe.ComputeGradientsOff()
        fe.ComputeScalarsOff()
        fe.Update()
        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetInputConnection(fe.GetOutputPort())
        conn.SetExtractionModeToLargestRegion()
        conn.Update()
        pts, faces = _poly_arrays(conn.GetOutput())
        return _compact(pts, faces)

    def _solid_from_parent(self, field):
        """Keep the inside component holding the parent; push the rest outside."""
        inside = field < 0.0
        lab, n = ndimage.label(inside)
        if n <= 1:
            return field
        keep = int(lab[self.parent_seed])
        if keep == 0:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            keep = int(np.argmax(sizes))
        field[(lab > 0) & (lab != keep)] = np.float32(0.5 * self.g)
        return field

    def _untangle(self, field):
        """Cut handles: regrow the solid from the parent with simple voxels only."""
        inside = field < 0.0
        idx = np.argwhere(inside)
        lo = np.maximum(idx.min(axis=0) - 1, 0)
        hi = idx.max(axis=0) + 2
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        sub = field[box]
        solid = ndimage.binary_fill_holes(inside[box])
        filled = solid & ~inside[box]
        depth = np.array(sub, copy=True)
        s = tuple(int(self.parent_seed[k] - lo[k]) for k in range(3))
        if all(0 <= s[k] < depth.shape[k] for k in range(3)) and solid[s]:
            depth[s] = np.float32(-1.0e9)
        grown = _grow_solid_ball(solid, depth)
        cut = solid & ~grown
        g2 = np.float32(0.25 * self.g)
        sub[filled] = -g2
        sub[cut] = np.maximum(-sub[cut], g2)
        return int(cut.sum()), int(filled.sum())

    def closed_surface(self, field):
        field = self._solid_from_parent(field)
        pts, faces = self._contour(field)
        rep = surface_report(pts, faces)
        if rep["genus"] > 0 or rep["nonmanifold_edges"] or rep["bowties"]:
            n_cut, n_fill = self._untangle(field)
            pts, faces = self._contour(field)
            rep2 = surface_report(pts, faces)
            print(f"  Untangle: genus {rep['genus']:g} -> {rep2['genus']:g} "
                  f"(cut {n_cut}, filled {n_fill} voxels)")
            rep = rep2
        if rep["genus"] != 0 or rep["n_loops"] or rep["nonmanifold_edges"] or rep["bowties"]:
            raise TemplateQualityError(
                f"closed surface is not a sphere: genus {rep['genus']:g}, loops {rep['n_loops']}, "
                f"nonmanifold {rep['nonmanifold_edges']}, bowties {rep['bowties']}"
            )
        return pts, faces

    def smooth(self, pts, faces):
        sm = vtk.vtkWindowedSincPolyDataFilter()
        sm.SetInputData(_poly_from_arrays(pts, faces))
        sm.SetNumberOfIterations(SMOOTH_ITERS)
        sm.SetPassBand(SMOOTH_PASSBAND)
        sm.BoundarySmoothingOff()
        sm.FeatureEdgeSmoothingOff()
        sm.NonManifoldSmoothingOn()
        sm.NormalizeCoordinatesOn()
        sm.Update()
        return vtk_to_numpy(sm.GetOutput().GetPoints().GetData()).astype(np.float64)

    def decimate(self, pts, faces):
        """Thin the marching-cubes mesh to about twice the final face count.

        VMTK's remesher costs time per input triangle, and the 0.2 mm voxel
        surface has ~10x more than the template needs.
        """
        a = pts[faces[:, 1]] - pts[faces[:, 0]]
        b = pts[faces[:, 2]] - pts[faces[:, 0]]
        area = 0.5 * float(np.linalg.norm(np.cross(a, b), axis=1).sum())
        target = 2.0 * area / (0.433 * self.edge ** 2)
        if target >= 0.8 * len(faces):
            return pts, faces
        dec = vtk.vtkQuadricDecimation()
        dec.SetInputData(_poly_from_arrays(pts, faces))
        dec.SetTargetReduction(1.0 - target / len(faces))
        dec.VolumePreservationOn()
        dec.Update()
        p, f = _compact(*_poly_arrays(dec.GetOutput()))
        rep = surface_report(p, f)
        if rep["genus"] != 0 or rep["n_loops"] or rep["nonmanifold_edges"] or rep["bowties"] \
                or rep["components"] != 1:
            return pts, faces  # keep the dense surface; slower, still correct
        return p, f

    def open_ostia(self, pts, faces):
        """Open each ostium exactly on its ground-truth plane.

        For one ostium, the part of the surface to delete is the connected
        piece beyond the plane that holds the stub tip -- found on the mesh,
        not by a region in space, so a neighbouring vessel passing close to a
        small ostium is never touched. The cut is a clip on the signed plane
        distance, which is linear along every edge, so the new rim vertices
        lie on the plane to rounding.
        """
        for op in self.ostia:
            o, n, r = op["origin"], op["normal"], op["r"]
            s = (pts - o) @ n
            lat = np.linalg.norm((pts - o) - s[:, None] * n, axis=1)
            cand = (s > 0.0) & (s < op["out_len"] + r + 3.0 * self.g) & (lat < 3.0 * r + 1.0)
            if not cand.any():
                raise TemplateQualityError("an ostium stub is missing from the surface")
            ids = np.flatnonzero(cand)
            tip = o + op["out_len"] * n
            seed = int(ids[np.argmin(np.linalg.norm(pts[ids] - tip, axis=1))])
            edges, _inv, _cnt = _edge_table(faces)
            comp = _component_with_seed(cand, _adjacency(len(pts), edges), seed)
            value = np.where(comp | (s <= 0.0), s, -1.0e3)
            poly = _poly_from_arrays(pts, faces)
            arr = numpy_to_vtk(np.ascontiguousarray(value), deep=True)
            arr.SetName("s")
            poly.GetPointData().SetScalars(arr)
            clip = vtk.vtkClipPolyData()
            clip.SetInputData(poly)
            clip.SetValue(0.0)
            clip.InsideOutOn()
            clip.Update()
            tri = vtk.vtkTriangleFilter()
            tri.SetInputConnection(clip.GetOutputPort())
            tri.Update()
            pts, faces = _poly_arrays(tri.GetOutput())
            pts, faces = _weld(pts, faces)
        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetInputData(_poly_from_arrays(pts, faces))
        conn.SetExtractionModeToLargestRegion()
        conn.Update()
        p, f = _compact(*_poly_arrays(conn.GetOutput()))
        # Clip vertices that landed a hair from old ones make slivers; weld
        # them, keeping the rim vertex so the rim stays on its plane.
        return _collapse_short(p, f, 0.15 * self.edge)

    def remesh(self, pts, faces):
        out = remesh_surface_isotropically(
            _poly_from_arrays(pts, faces), self.edge,
            n_iter=REMESH_ITERS, connectivity_iter=REMESH_CONN_ITERS,
        )
        p, f = _compact(*_poly_arrays(out))
        return self.snap_rims(p, f)

    def snap_rims(self, pts, faces):
        """Project every rim vertex onto the plane of the ostium it belongs to."""
        loops = _boundary_loops(pts, faces)
        self.rim_drift_mm = 0.0
        if len(loops) != len(self.ostia):
            return pts, faces  # check() reports it
        from scipy.optimize import linear_sum_assignment
        cen = np.array([pts[l].mean(axis=0) for l in loops])
        org = np.array([op["origin"] for op in self.ostia])
        ri, ci = linear_sum_assignment(np.linalg.norm(cen[:, None] - org[None], axis=2))
        pts = pts.copy()
        for li, oi in zip(ri, ci):
            op, ids = self.ostia[oi], loops[li]
            s = (pts[ids] - op["origin"]) @ op["normal"]
            if np.abs(s).max() > 0.25 * self.edge:
                continue  # not this ostium's rim; check() reports it
            self.rim_drift_mm = max(self.rim_drift_mm, float(np.abs(s).max()))
            pts[ids] -= s[:, None] * op["normal"]
        return pts, faces

    def check(self, pts, faces, label="template"):
        rep = surface_report(pts, faces)
        issues = []
        if rep["components"] != 1:
            issues.append(f"{rep['components']} components")
        if rep["nonmanifold_edges"]:
            issues.append(f"{rep['nonmanifold_edges']} non-manifold edges")
        if rep["bowties"]:
            issues.append(f"{rep['bowties']} bowtie vertices")
        if rep["genus"] != 0:
            issues.append(f"genus {rep['genus']:g}")
        if rep["n_loops"] != len(self.ostia):
            issues.append(f"{rep['n_loops']} openings for {len(self.ostia)} ostia")
        else:
            cen = np.array([pts[l].mean(axis=0) for l in rep["loops"]])
            org = np.array([op["origin"] for op in self.ostia])
            d = np.linalg.norm(cen[:, None] - org[None], axis=2)
            from scipy.optimize import linear_sum_assignment
            ri, ci = linear_sum_assignment(d)
            lim = np.array([op["r"] + 1.0 for op in self.ostia])[ci]
            if np.any(d[ri, ci] > lim):
                issues.append("an opening is not at its ostium")
            off = 0.0
            for li, oi in zip(ri, ci):
                op = self.ostia[oi]
                off = max(off, float(np.abs((pts[rep["loops"][li]] - op["origin"]) @ op["normal"]).max()))
            rep["plane_mm"] = off
            rep["loop_ostium"] = dict(zip(ri.tolist(), ci.tolist()))
            if off > PLANE_TOL_MM:
                issues.append(f"a rim is {off:.3f} mm off its ostium plane")
        if rep["edge_min"] < 0.05 * self.edge:
            issues.append(f"edge {rep['edge_min']:.4f} mm")
        if rep["q_min"] < 0.05:
            issues.append(f"sliver triangle (q={rep['q_min']:.3f})")
        if not issues:
            n_x = len(self_intersections(pts, faces))
            if n_x:
                issues.append(f"{n_x} self-intersecting triangle pairs")
            rep["crossings"] = n_x
        rep["issues"] = issues
        return rep

    def manifold(self, centers, radii, verbose=True):
        """Uniform open surface of parent tube smooth-union the spheres."""
        t = {}
        t0 = time.perf_counter()
        field = self.field_with_spheres(centers, radii)
        t["field"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pts, faces = self.closed_surface(field)
        idx = np.round((np.asarray(centers, float) - self.origin) / self.g).astype(int)[:, ::-1]
        idx = np.clip(idx, 0, np.array(self.shape) - 1)
        if np.any(field[tuple(idx.T)] >= 0.0):
            raise TemplateQualityError("a sac sphere is not part of the surface")
        t["contour"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pts = self.smooth(pts, faces)
        t["smooth"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pts, faces = self.decimate(pts, faces)
        t["decimate"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pts, faces = self.open_ostia(pts, faces)
        t["open"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        rpts, rfaces = self.remesh(pts, faces)
        t["remesh"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        rep = self.check(rpts, rfaces)
        if rep["issues"]:
            # One retry: a remesh that tore is usually fine with more passes.
            if verbose:
                print(f"  Remesh gate: {'; '.join(rep['issues'])}; retrying")
            rpts2, rfaces2 = self.snap_rims(*_compact(*_poly_arrays(remesh_surface_isotropically(
                _poly_from_arrays(pts, faces), 0.97 * self.edge,
                n_iter=6, connectivity_iter=6))))
            rep2 = self.check(rpts2, rfaces2)
            if not rep2["issues"]:
                rpts, rfaces, rep = rpts2, rfaces2, rep2
        t["check"] = time.perf_counter() - t0
        if rep["issues"]:
            raise TemplateQualityError("template failed: " + "; ".join(rep["issues"]))
        if signed_volume(rpts, rfaces) < 0:
            rfaces = rfaces[:, ::-1].copy()
        rep["timing"] = t
        rep["rim_drift_mm"] = self.rim_drift_mm
        return rpts, rfaces, rep


def remesh_surface_isotropically(surface, target_edge_length, n_iter=REMESH_ITERS,
                                 connectivity_iter=REMESH_CONN_ITERS):
    """VMTK isotropic remesh that also resamples the rims (clip edges are uneven)."""
    from vmtk import vmtkscripts

    remesher = vmtkscripts.vmtkSurfaceRemeshing()
    remesher.Surface = surface
    remesher.ElementSizeMode = "edgelength"
    remesher.TargetEdgeLength = float(target_edge_length)
    remesher.PreserveBoundaryEdges = 0
    remesher.NumberOfIterations = int(n_iter)
    remesher.NumberOfConnectivityOptimizationIterations = int(connectivity_iter)
    remesher.CollapseAngleThreshold = 0.2
    remesher.Execute()
    return to_vtk_poly(remesher.Surface)


def _same_rim_edge(a, b, uq, cn):
    k = np.flatnonzero((uq[:, 0] == min(a, b)) & (uq[:, 1] == max(a, b)))
    return bool(len(k) and cn[k[0]] == 1)


def _weld(pts, faces):
    """Merge coincident points (the clip emits one new point per cut edge and cell)."""
    key, inv = np.unique(np.round(pts, 9), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    first = np.zeros(len(key), dtype=np.int64)
    first[inv[::-1]] = np.arange(len(pts))[::-1]
    faces = inv[faces]
    good = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return _compact(pts[first], faces[good])


def _collapse_short(pts, faces, tol):
    """Weld vertex pairs closer than ``tol`` along an edge (clip leftovers)."""
    for _ in range(3):
        uniq, _inv, _cnt = _edge_table(faces)
        d = np.linalg.norm(pts[uniq[:, 0]] - pts[uniq[:, 1]], axis=1)
        short = uniq[d < tol]
        if len(short) == 0:
            break
        # Greedy independent set of short edges.
        used = np.zeros(len(pts), dtype=bool)
        target = np.arange(len(pts))
        rim = np.zeros(len(pts), dtype=bool)
        uq, _i, cn = _edge_table(faces)
        rim[uq[cn == 1].ravel()] = True
        for a, b in short:
            if used[a] or used[b] or (rim[a] and rim[b] and not _same_rim_edge(a, b, uq, cn)):
                continue
            used[a] = used[b] = True
            if rim[b] and not rim[a]:
                a, b = b, a
            target[b] = a
        faces = target[faces]
        good = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
        faces = faces[good]
        pts, faces = _compact(pts, faces)
    return pts, faces


# ---------------------------------------------------------------------------
# Per-case entry
# ---------------------------------------------------------------------------

def to_vtk_surface(pts, faces):
    poly = _poly_from_arrays(pts, faces)
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputData(poly)
    nrm.ComputePointNormalsOn()
    nrm.ComputeCellNormalsOff()
    nrm.SplittingOff()
    nrm.ConsistencyOff()
    nrm.AutoOrientNormalsOff()
    nrm.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(nrm.GetOutput())
    return out


def prepare_case(dataset_id, v_file, centerline_path, grid_spacing=DEFAULT_GRID_SPACING,
                 max_grid_size=DEFAULT_MAX_GRID_SIZE, target_edge_length=DEFAULT_TARGET_EDGE_LENGTH):
    """Detection, sphere fit and the parent field: everything before the surface."""
    case = load_case(v_file, centerline_path, dataset_id)
    pts, faces, up = case["pts"], case["faces"], case["up"]
    r_par = parent_radius(up, case["ur"], case["cl_edges"])
    mask, excess, det = detect_aneurysm(pts, faces, up, r_par)
    normals = wall_normals(pts, faces, up)
    centers, radii, fit = fit_aneurysm_spheres(pts, normals, mask, up, r_par)
    model = CaseModel(up, r_par, case["cl_edges"], case["frames"], case["mesh"].bounds,
                      grid_spacing=grid_spacing, max_grid_size=max_grid_size,
                      target_edge_length=target_edge_length)
    return case, mask, det, centers, radii, fit, model


@with_dataset_id
def process_vessel_aneurysm_dataset(
    dataset_id,
    v_file,
    output_dir,
    centerline_path,
    target_edge_length=DEFAULT_TARGET_EDGE_LENGTH,
    grid_spacing=DEFAULT_GRID_SPACING,
    max_grid_size=DEFAULT_MAX_GRID_SIZE,
):
    """Detect the sac, fit four spheres, and write one uniform manifold."""
    t_all = time.perf_counter()
    print(f"\n=========================================\nVessel+aneurysm template: {dataset_id}")
    case, mask, det, centers, radii, fit, model = prepare_case(
        dataset_id, v_file, centerline_path, grid_spacing, max_grid_size, target_edge_length
    )
    print(
        f"  Sac: {100 * det['frac']:.1f}% of vertices, diameter {det['diameter_mm']:.1f} mm, "
        f"peak {det['peak_mm']:.2f} mm, neck lip {100 * det['neck_frac']:.0f}% "
        f"({det['seconds']:.2f}s)"
    )
    print(
        f"  Spheres: R={np.round(radii, 2).tolist()} mm, sac RMSE {fit['parent_rmse_mm']:.2f} -> "
        f"{fit['union_rmse_mm']:.2f} mm (MAE {fit['union_mae_mm']:.2f}), "
        f"{fit['iters']} iters, {fit['moved']} pulled in ({fit['seconds']:.2f}s)"
    )
    print(
        f"  Parent field {model.shape[::-1]} at {model.g:.3f} mm, {len(model.ostia)} ostia, "
        f"edge {model.edge:.3f} mm ({model.seconds_parent:.2f}s)"
    )
    pts, faces, rep = model.manifold(centers, radii)
    tm = rep["timing"]
    print(
        "  Surface: " + ", ".join(f"{k} {v:.2f}s" for k, v in tm.items())
        + f" -> {rep['n_points']} pts, edge {rep['edge_mean']:.3f} mm (CV {rep['edge_cv']:.2f}), "
        f"q_min {rep['q_min']:.2f}, {rep['n_loops']} openings on their planes "
        f"(remesh drift {rep['rim_drift_mm']:.3f} mm, snapped), genus {rep['genus']:g}"
    )
    gt_area = float(surface_report(case["pts"], case["faces"])["area"])
    ratio = rep["area"] / max(gt_area, 1e-9)
    if not 0.5 <= ratio <= 1.5:
        raise TemplateQualityError(f"template area is {ratio:.2f}x the vessel")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{dataset_id}.vtp")
    save_polydata(to_vtk_surface(pts, faces), out_file)
    np.savez(
        os.path.join(output_dir, f"{dataset_id}.spheres.npz"),
        centers=centers,
        radii=radii,
        sac_vertex_ids=np.flatnonzero(mask).astype(np.int32),
        edge_length=np.float64(model.edge),
    )
    print(f"  Saved {out_file} (area {ratio:.2f}x GT) in {time.perf_counter() - t_all:.2f}s")
    return out_file


def _process_one(dataset_id, v_file, args):
    def work():
        centerline_path = os.path.join(args.centerline_dir, f"{dataset_id}.vtp")
        return process_vessel_aneurysm_dataset(
            dataset_id=dataset_id,
            v_file=v_file,
            output_dir=args.output_dir,
            centerline_path=centerline_path,
            target_edge_length=args.target_edge_length,
            grid_spacing=args.grid_spacing,
            max_grid_size=args.max_grid_size,
        )

    return run_logged_case(
        dataset_id, v_file, args, work,
        log_folder_name=LOG_FOLDER, default_output_dir=DEFAULT_OUTPUT_DIR,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Uniform template of the parent vessel with the aneurysm as four spheres"
    )
    add_shared_cli_args(parser, DEFAULT_OUTPUT_DIR, default_workers=os.cpu_count() or 4,
                        include_remesh_grid=True)
    parser.add_argument("--centerline-dir", type=str, default=CLEANDATA_ORIGINAL_CENTERLINE,
                        help="Directory of original_centerline {id}.vtp files")
    add_run_log_args(parser, LOG_FOLDER)
    parser.set_defaults(
        from_folder=True,
        vessel_dir=CLEANDATA_UNIFORM,
        grid_spacing=DEFAULT_GRID_SPACING,
        max_grid_size=DEFAULT_MAX_GRID_SIZE,
        case_timeout=600.0,
    )
    args = parser.parse_args()
    extra_log, on_worker_result = configure_batch_logging(
        args, LOG_FOLDER, DEFAULT_OUTPUT_DIR, keep_all_transcripts=False
    )
    extra = [
        "--target-edge-length", str(args.target_edge_length),
        "--grid-spacing", str(args.grid_spacing),
        "--max-grid-size", str(args.max_grid_size),
        "--centerline-dir", args.centerline_dir,
        "--from-folder",
    ] + extra_log
    try:
        run_batch(os.path.abspath(__file__), _process_one, args, extra,
                  on_worker_result=on_worker_result)
    finally:
        finalize_run_logs(args)


if __name__ == "__main__":
    main()
