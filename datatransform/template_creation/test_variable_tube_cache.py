"""The parent-tube cache in variable_remeshing.py.

A cached tube must come back exactly as it was built, and anything that would
build a different tube must miss. Meshes are explicit triangles: ``pv.Sphere``
aborts this machine's interpreter (see test_remeshing_repairs.py).
"""
import os
import sys

import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import variable_remeshing as vr


def _strip(n=12, z=0.0):
    """A band of triangles carrying a point array, like an open tube wall."""
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.column_stack([np.cos(ang), np.sin(ang)])
    pts = np.vstack([
        np.column_stack([ring, np.full(n, z)]),
        np.column_stack([ring, np.full(n, z + 1.0)]),
    ])
    faces = []
    for k in range(n):
        a, b = k, (k + 1) % n
        faces.extend([3, a, b, n + b, 3, a, n + b, n + a])
    mesh = pv.PolyData(pts, np.array(faces, dtype=np.int64))
    mesh.point_data["MaximumInscribedSphereRadius"] = np.linspace(0.5, 1.0, len(pts))
    return mesh


def _built(vessel):
    return {
        "open_base_surface": _strip(),
        "branched_centerline": _strip(z=3.0),
        "anatomical_profiles": [{"center": np.array([0.0, 0.0, 1.0]), "radius": 1.0}],
        "n_clipped": 2,
        "vessel_mesh": vessel,
    }


def test_a_stored_tube_comes_back_exactly(tmp_path):
    vessel = _strip(z=-5.0)
    built = _built(vessel)
    vr.store_cached_tube(str(tmp_path), "k", built, vessel)
    got = vr.load_cached_tube(str(tmp_path), "k", vessel)
    assert got is not None
    for name in ("open_base_surface", "branched_centerline"):
        a, b = pv.wrap(built[name]), pv.wrap(got[name])
        assert np.array_equal(a.points, b.points)
        assert np.array_equal(a.faces, b.faces)
        assert np.array_equal(
            a.point_data["MaximumInscribedSphereRadius"],
            b.point_data["MaximumInscribedSphereRadius"],
        )
    assert got["n_clipped"] == 2
    assert np.array_equal(got["anatomical_profiles"][0]["center"], [0.0, 0.0, 1.0])
    # The vessel was not repaired, so the caller's own mesh is handed back.
    assert got["vessel_mesh"] is vessel


def test_a_repaired_vessel_is_stored_with_the_tube(tmp_path):
    vessel = _strip(z=-5.0)
    built = _built(_strip(z=-9.0))
    vr.store_cached_tube(str(tmp_path), "k", built, vessel)
    got = vr.load_cached_tube(str(tmp_path), "k", vessel)
    assert got["vessel_mesh"] is not vessel
    assert np.array_equal(pv.wrap(got["vessel_mesh"]).points, built["vessel_mesh"].points)


def test_a_torn_entry_is_a_miss(tmp_path):
    vessel = _strip(z=-5.0)
    vr.store_cached_tube(str(tmp_path), "k", _built(vessel), vessel)
    os.remove(os.path.join(tmp_path, "k", "open_base_surface.vtp"))
    assert vr.load_cached_tube(str(tmp_path), "k", vessel) is None
    assert vr.load_cached_tube(str(tmp_path), "never-stored", vessel) is None


def test_anything_that_changes_the_tube_changes_the_key(tmp_path):
    src = tmp_path / "v.vtp"
    src.write_bytes(b"mesh one")
    frames = [{"origin": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0]), "radius": 1.0}]
    params = dict(extension_length=5.0, sample_spacing=0.1, grid_spacing=0.08,
                  max_grid_size=250, skip_mc_decimate=False)
    base = vr.tube_cache_key(str(src), frames, params)
    assert vr.tube_cache_key(str(src), frames, params) == base

    assert vr.tube_cache_key(str(src), None, params) != base
    moved = [dict(frames[0], radius=1.01)]
    assert vr.tube_cache_key(str(src), moved, params) != base
    assert vr.tube_cache_key(str(src), frames, dict(params, grid_spacing=0.07)) != base
    assert vr.tube_cache_key(str(src), frames, dict(params, skip_mc_decimate=True)) != base
    src.write_bytes(b"mesh two")
    assert vr.tube_cache_key(str(src), frames, params) != base
