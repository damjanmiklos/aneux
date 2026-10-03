"""Soft aneurysm membership on the ground-truth wall.

The detector in ``aneurysm_detection`` labels the dome and a short neck
lip. Membership is 1 on that patch and falls with a cosine over one parent
radius of geodesic distance past the lip, so the loss has no step at the neck.
Template vertices copy the value of the ground-truth face their ray hits; the
predicted mesh is never searched for an aneurysm.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def geodesic_distance_to_mask(pts, faces, mask):
    """Geodesic distance (mm) from each vertex to the nearest masked vertex."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    n = int(pts.shape[0])
    dist = np.full(n, np.inf, dtype=np.float64)
    if n == 0 or faces.shape[0] == 0 or not bool(mask.any()):
        return dist
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    keep = (
        (edges[:, 0] >= 0)
        & (edges[:, 1] < n)
        & (edges[:, 0] != edges[:, 1])
    )
    edges = edges[keep]
    if edges.shape[0] == 0:
        dist[mask] = 0.0
        return dist
    length = np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1)
    src = np.concatenate((edges[:, 0], edges[:, 1]))
    dst = np.concatenate((edges[:, 1], edges[:, 0]))
    weight = np.concatenate((length, length))
    graph = csr_matrix((weight, (src, dst)), shape=(n, n))
    seeds = np.flatnonzero(mask)
    return np.asarray(
        dijkstra(graph, directed=False, indices=seeds, min_only=True),
        dtype=np.float64,
    )


def membership_from_mask(pts, faces, mask, decay_mm):
    """1 on ``mask``, cosine to 0 over ``decay_mm`` of geodesic distance."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    n = int(pts.shape[0])
    decay = float(decay_mm)
    if n == 0 or mask.shape[0] != n or decay <= 0.0:
        return np.zeros(n, dtype=np.float64)
    dist = geodesic_distance_to_mask(pts, faces, mask)
    m = np.zeros(n, dtype=np.float64)
    on = mask | (np.isfinite(dist) & (dist <= 1e-8))
    m[on] = 1.0
    band = (~on) & np.isfinite(dist) & (dist < decay)
    x = dist[band] / decay
    m[band] = 0.5 * (1.0 + np.cos(np.pi * x))
    return m


def _detector():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from aneux_paths import TEMPLATE_DIR

    template_dir = os.path.abspath(TEMPLATE_DIR)
    if template_dir not in sys.path:
        sys.path.insert(0, template_dir)
    from aneurysm_detection import (
        AneurysmDetectionError,
        _dedup_centerline,
        detect_aneurysm,
        parent_radius,
    )

    return AneurysmDetectionError, _dedup_centerline, detect_aneurysm, parent_radius


def wall_sac_membership(pts, faces, cl_pts, cl_rad, cl_lines):
    """Membership on the GT wall, or None when the centerline has no radius.

    Failures of detection leave the loss unweighted (the caller treats None
    as "no aneurysm field") and are reported, not fatal: one vessel must not
    stop the cache.
    """
    if cl_rad is None:
        return None
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if pts.shape[0] == 0 or faces.shape[0] == 0:
        return None
    if int(faces.min()) < 0 or int(faces.max()) >= pts.shape[0]:
        return None
    cl_pts = np.asarray(cl_pts, dtype=np.float64).reshape(-1, 3)
    cl_rad = np.asarray(cl_rad, dtype=np.float64).reshape(-1)
    cl_lines = np.asarray(cl_lines, dtype=np.int64).reshape(-1)
    if cl_pts.shape[0] == 0 or cl_rad.shape[0] != cl_pts.shape[0] or cl_lines.size == 0:
        return None
    try:
        quality_error, dedup, detect, opened_radius = _detector()
        up, _ur, edges = dedup(cl_pts, cl_rad, cl_lines)
        r_parent = opened_radius(up, _ur, edges)
        mask, _excess, info = detect(pts, faces, up, r_parent)
    except quality_error as exc:
        warnings.warn(f"aneurysm membership skipped: {exc}", RuntimeWarning, stacklevel=2)
        return None
    except Exception as exc:
        warnings.warn(f"aneurysm membership skipped: {exc}", RuntimeWarning, stacklevel=2)
        return None
    core = np.asarray(info["core"], dtype=bool)
    nearest = np.asarray(info["nearest"], dtype=np.int64).reshape(-1)
    if int(core.sum()) == 0 or nearest.shape[0] != pts.shape[0]:
        return None
    r_neck = float(np.median(r_parent[nearest[core]]))
    # One parent radius past the lip. The lip itself stays at 1.
    decay = float(np.clip(r_neck, 0.35, 8.0))
    return membership_from_mask(pts, faces, mask, decay)
