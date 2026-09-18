"""Repairs added after the 2026-09-17 batch run.

Every test here builds its mesh from explicit triangles rather than
``pyvista.Sphere``: on this machine's VTK build ``pv.Sphere`` aborts the
interpreter, which takes the rest of the session's tests with it.
"""
import os
import sys

import numpy as np
import pytest
import vtk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vessel_pipeline import (
    CLIP_MAX_AREA_LOSS_FRACTION,
    cap_surface,
    MAX_FLOW_EXTENSION_LAYERS,
    MIN_EDGE_LENGTH_MM,
    MIN_OPENING_LOOP_POINTS,
    TemplateQualityError,
    _flow_extension_layer_estimate,
    _is_disc,
    _keep_region_with_point,
    _opening_clip_height,
    _polydata_from_triangles,
    _triangle_edge_lengths,
    _triangle_points_faces,
    add_flow_extensions,
    boundary_loop_radii,
    fan_fill_small_loops,
    finalize_surface,
    force_manifold_triangles,
    inspect_surface_topology,
    original_cell_mask,
    patch_wall_pinholes,
    trim_extension_patches,
    uncap_closed_surface,
    weld_degenerate_vertices,
)


def open_tube(radius=1.0, length=6.0, n_sides=40, n_rings=30, axis=2, center=(0.0, 0.0, 0.0)):
    """Triangulated open cylinder along ``axis``, both ends free."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    zs = np.linspace(-0.5 * length, 0.5 * length, n_rings)
    pts = np.zeros((n_rings * n_sides, 3), dtype=np.float64)
    other = [a for a in range(3) if a != axis]
    for i, z in enumerate(zs):
        block = pts[i * n_sides : (i + 1) * n_sides]
        block[:, other[0]] = radius * np.cos(theta)
        block[:, other[1]] = radius * np.sin(theta)
        block[:, axis] = z
    pts += np.asarray(center, dtype=np.float64)
    faces = []
    for i in range(n_rings - 1):
        for j in range(n_sides):
            a = i * n_sides + j
            b = i * n_sides + (j + 1) % n_sides
            c = (i + 1) * n_sides + j
            d = (i + 1) * n_sides + (j + 1) % n_sides
            faces.append((a, b, d))
            faces.append((a, d, c))
    return _polydata_from_triangles(pts, np.asarray(faces, dtype=np.int64))


def capped_tube(**kwargs):
    """``open_tube`` closed with the same capper the pipeline uses."""
    return cap_surface(open_tube(**kwargs), displacement=0.0)


def _interior_edge(faces):
    """An edge used by exactly two triangles, so a third makes it non-manifold."""
    usage = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (int(u), int(v)) if u <= v else (int(v), int(u))
            usage[key] = usage.get(key, 0) + 1
    return next(edge for edge, n in usage.items() if n == 2)


def test_weld_removes_micron_edges_without_moving_the_surface():
    tube = open_tube()
    _p, pts, faces = _triangle_points_faces(tube)
    area_before = float(vtk_area(tube))
    # Split one vertex into two points 2 nm apart, as the originals do.
    pts = np.vstack([pts, pts[0] + np.array([2e-6, 0.0, 0.0])])
    faces = np.vstack([faces, np.array([[0, len(pts) - 1, int(faces[0][1])]])])
    dirty = _polydata_from_triangles(pts, faces)
    assert inspect_surface_topology(dirty)["min_edge"] < MIN_EDGE_LENGTH_MM

    welded, shortest = weld_degenerate_vertices(dirty)
    assert shortest >= MIN_EDGE_LENGTH_MM
    assert inspect_surface_topology(welded)["min_edge"] >= MIN_EDGE_LENGTH_MM
    assert vtk_area(welded) == pytest.approx(area_before, rel=1e-4)


def vtk_area(surface):
    mass = vtk.vtkMassProperties()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(surface)
    tri.Update()
    mass.SetInputData(tri.GetOutput())
    mass.Update()
    return mass.GetSurfaceArea()


def test_force_manifold_cuts_a_sheet_sharing_an_edge():
    tube = open_tube()
    _p, pts, faces = _triangle_points_faces(tube)
    a, b = _interior_edge(faces)
    # A third, full-size triangle on the (a, b) edge: too big for the flap rule.
    apex = pts[a] + np.array([0.0, 0.0, 2.0])
    pts = np.vstack([pts, apex])
    faces = np.vstack([faces, np.array([[a, b, len(pts) - 1]])])
    dirty = _polydata_from_triangles(pts, faces)
    assert inspect_surface_topology(dirty)["n_nonmanifold"] == 1

    fixed, n_dropped = force_manifold_triangles(dirty)
    assert n_dropped >= 1
    assert inspect_surface_topology(fixed)["n_nonmanifold"] == 0


def test_patch_wall_pinholes_closes_a_puncture_and_keeps_both_ostia():
    tube = open_tube()
    _p, pts, faces = _triangle_points_faces(tube)
    # Punch a hole in the middle of the wall.
    mid = len(faces) // 2
    keep = np.ones(len(faces), dtype=bool)
    keep[mid] = False
    punctured = _polydata_from_triangles(pts, faces[keep])
    assert len(boundary_loop_radii(punctured)) == 3

    patched, n_filled = patch_wall_pinholes(punctured)
    assert n_filled == 1
    loops = boundary_loop_radii(patched)
    assert len(loops) == 2
    assert all(radius > 0.5 for radius, _n, _bary in loops)


def test_patch_wall_pinholes_treats_a_four_point_rim_as_a_pinhole():
    """A 4-point rim is what makes vmtkFlowExtensions extrude millions of layers."""
    tube = open_tube()
    _p, pts, faces = _triangle_points_faces(tube)
    keep = np.ones(len(faces), dtype=bool)
    keep[len(faces) // 2] = False
    punctured = _polydata_from_triangles(pts, faces[keep])
    small = [lp for lp in boundary_loop_radii(punctured) if lp[1] < MIN_OPENING_LOOP_POINTS]
    assert small, "expected the single-triangle hole to have fewer than 6 rim points"
    patched, n_filled = patch_wall_pinholes(patched_input := punctured, min_radius=0.0)
    assert n_filled == 1
    assert len(boundary_loop_radii(patched)) == 2
    assert patched_input is not patched


def test_flow_extension_layer_estimate_flags_a_degenerate_rim():
    tube = open_tube()
    assert _flow_extension_layer_estimate(tube, 5.0) < MAX_FLOW_EXTENSION_LAYERS
    _p, pts, faces = _triangle_points_faces(tube)
    keep = np.ones(len(faces), dtype=bool)
    keep[len(faces) // 2] = False
    tiny_rim = _polydata_from_triangles(pts * 1e-4, faces[keep])
    assert _flow_extension_layer_estimate(tiny_rim, 5.0) > MAX_FLOW_EXTENSION_LAYERS


def test_add_flow_extensions_refuses_a_runaway_rim():
    tube = open_tube(radius=1e-4, length=1e-3, n_sides=8, n_rings=4)
    with pytest.raises(TemplateQualityError, match="flow-extension layers"):
        add_flow_extensions(tube, extension_length=5.0)


def test_original_cell_mask_separates_the_extension():
    tube = open_tube()
    extended = add_flow_extensions(tube, extension_length=3.0)
    _ext, mask = original_cell_mask(extended, tube)
    assert mask.sum() == tube.GetNumberOfCells()
    assert (~mask).sum() > 0


def test_trim_extension_patches_removes_an_oblique_tube():
    """The cutter's cylinder misses a tube that leans away from the tangent."""
    tube = open_tube(radius=1.0, length=6.0)
    extended = add_flow_extensions(tube, extension_length=4.0)
    area_tube = vtk_area(tube)
    assert vtk_area(extended) > 1.5 * area_tube

    frames = [
        (np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, 1.0]), 1.0),
        (np.array([0.0, 0.0, -3.0]), np.array([0.0, 0.0, -1.0]), 1.0),
    ]
    trimmed, n_trimmed = trim_extension_patches(extended, tube, frames)
    assert n_trimmed == 2
    assert vtk_area(trimmed) == pytest.approx(area_tube, rel=0.02)


def test_trimmed_clip_height_is_short_enough_for_a_siphon():
    """A 12 mm cutter reaches the other limb of a tortuous vessel; 3 mm does not."""
    assert _opening_clip_height(1.66, extension_length=5.0) > 10.0
    assert _opening_clip_height(1.66, extension_length=5.0, trimmed=True) < 4.0
    assert _opening_clip_height(0.1, trimmed=True) >= 0.75


def test_keep_region_with_point_drops_a_severed_limb():
    body = open_tube(radius=1.0, length=6.0)
    limb = open_tube(radius=1.0, length=6.0, center=(20.0, 0.0, 0.0))
    append = vtk.vtkAppendPolyData()
    append.AddInputData(body)
    append.AddInputData(limb)
    append.Update()
    both = append.GetOutput()

    kept = _keep_region_with_point(both, (0.0, 0.0, 0.0))
    assert kept.GetNumberOfPoints() < both.GetNumberOfPoints()
    bounds = kept.GetBounds()
    assert bounds[1] < 10.0


def test_uncap_closed_surface_opens_a_capped_tube():
    closed = capped_tube(radius=1.0, length=6.0)
    assert len(boundary_loop_radii(closed)) == 0

    opened = uncap_closed_surface(closed)
    loops = boundary_loop_radii(opened)
    assert len(loops) == 2
    assert all(radius == pytest.approx(1.0, rel=0.05) for radius, _n, _b in loops)


def test_uncap_closed_surface_leaves_an_open_vessel_alone():
    tube = open_tube()
    out = uncap_closed_surface(tube)
    assert out.GetNumberOfCells() == tube.GetNumberOfCells()


def test_is_disc_accepts_a_fan_and_rejects_a_closed_shell():
    fan = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4]], dtype=np.int64)
    assert _is_disc(fan)
    tetra = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64)
    assert not _is_disc(tetra)


def test_clip_area_loss_budget_is_tight_enough_to_catch_a_lost_branch():
    # The 2026-09-17 failures lost 12-30% of the surface to a single cut.
    assert CLIP_MAX_AREA_LOSS_FRACTION <= 0.15


def test_fan_fill_closes_a_loop_the_capper_cannot_walk():
    """The vmtk capper bails on rims with a branching vertex; the fan never does."""
    tube = open_tube()
    _p, pts, faces = _triangle_points_faces(tube)
    keep = np.ones(len(faces), dtype=bool)
    keep[len(faces) // 2] = False
    punctured = _polydata_from_triangles(pts, faces[keep])
    assert len(boundary_loop_radii(punctured)) == 3

    filled, n_filled = fan_fill_small_loops(punctured)
    assert n_filled == 1
    loops = boundary_loop_radii(filled)
    assert len(loops) == 2
    assert all(radius > 0.5 for radius, _n, _bary in loops)


def test_fan_fill_leaves_the_ostia_open():
    tube = open_tube()
    filled, n_filled = fan_fill_small_loops(tube)
    assert n_filled == 0
    assert len(boundary_loop_radii(filled)) == 2


def test_finalize_surface_repairs_what_the_quality_gate_checks():
    """A surface with a puncture, a stuck sheet and a micron edge must come out clean."""
    tube = open_tube()
    _p, pts, faces = _triangle_points_faces(tube)
    keep = np.ones(len(faces), dtype=bool)
    keep[len(faces) // 2] = False
    faces = faces[keep]
    a, b = _interior_edge(faces)
    pts = np.vstack([pts, pts[a] + np.array([0.0, 0.0, 2.0]), pts[0] + np.array([2e-6, 0.0, 0.0])])
    faces = np.vstack(
        [
            faces,
            np.array([[a, b, len(pts) - 2]]),
            np.array([[0, len(pts) - 1, int(faces[0][1])]]),
        ]
    )
    dirty = _polydata_from_triangles(pts, faces)
    topo = inspect_surface_topology(dirty)
    assert topo["n_nonmanifold"] > 0
    assert topo["min_edge"] < MIN_EDGE_LENGTH_MM

    final, n_regions = finalize_surface(dirty)
    topo = inspect_surface_topology(final)
    assert n_regions == 1
    assert topo["n_nonmanifold"] == 0
    assert topo["min_edge"] >= MIN_EDGE_LENGTH_MM
    loops = boundary_loop_radii(final)
    assert len(loops) == 2
    assert all(radius > 0.5 for radius, _n, _bary in loops)


def test_repairs_did_not_relax_the_training_objectives():
    """The robustness work must not cost edge uniformity, detail or the gates."""
    import remeshing as rm

    assert rm.DEFAULT_GT_EDGE_LENGTH_MM == 0.15
    assert rm.GT_REMESH_N_ITER == 20
    # Light Taubin only: pass band near 2 barely filters, and few iterations.
    assert rm.GT_TAUBIN_PASS_BAND >= 1.4
    assert rm.GT_TAUBIN_ITER <= 8
    # The area gates still bracket the original closely.
    assert rm.GT_MIN_AREA_RATIO >= 0.88
    assert rm.GT_MAX_AREA_RATIO <= 1.20
    # Welding must stay far below the target edge so it cannot smooth anything.
    from vessel_pipeline import WELD_TOLERANCE_MM, WALL_PINHOLE_RADIUS_MM

    assert WELD_TOLERANCE_MM <= rm.DEFAULT_GT_EDGE_LENGTH_MM / 100.0
    # Only sub-ostium holes may be patched shut.
    assert WALL_PINHOLE_RADIUS_MM <= 0.2


def test_prepare_gt_surface_still_keeps_the_original_tessellation():
    """Repairs may remove debris; they may not resample the aneurysm wall."""
    import remeshing as rm

    tube = open_tube(n_sides=60, n_rings=60)
    prepared = rm.prepare_gt_surface(tube)
    assert prepared.GetNumberOfPoints() == tube.GetNumberOfPoints()
    assert vtk_area(prepared) == pytest.approx(vtk_area(tube), rel=1e-6)


def _pinch_two_holes(n_sides=40, n_rings=30):
    """Open tube with two vertex fans removed whose links share one vertex.

    The two holes then meet at that vertex, which carries four free edges. VMTK's
    boundary extractor bails on such a rim ("Can't find adjacent point") and a
    cycle walk cannot traverse it, so this is the shape that used to reach
    ``assert_template_quality`` as a surviving pinhole.
    """
    tube = open_tube(n_sides=n_sides, n_rings=n_rings)
    _poly, pts, faces = _triangle_points_faces(tube)
    u = 10 * n_sides + 5
    w = 12 * n_sides + 7
    kept = np.asarray([f for f in faces.tolist() if u not in f and w not in f], dtype=np.int64)
    return _polydata_from_triangles(pts, kept)


def test_collapse_closes_a_branching_pinhole_rim():
    from vessel_pipeline import (
        boundary_loop_radii,
        collapse_small_boundary_components,
        count_connected_regions,
        inspect_surface_topology,
    )

    holed = _pinch_two_holes()
    loops = boundary_loop_radii(holed)
    # two real ends plus the two pinched holes
    assert len(loops) == 4

    fixed, n = collapse_small_boundary_components(holed, min_radius=0.5)
    assert n == 1, "the pinched pair is one free-edge component"
    after = boundary_loop_radii(fixed)
    assert len(after) == 2, f"expected only the two tube ends, got {after}"
    assert inspect_surface_topology(fixed)["n_nonmanifold"] == 0
    assert count_connected_regions(fixed) == 1


def test_close_wall_pinholes_leaves_no_pinhole_on_a_pinched_rim():
    """Whichever repair gets there first, nothing sub-ostium may survive.

    Whether the fan can walk a pinched rim depends on the tessellation -- on some
    it recovers after the ear drop, on others it merges the two holes into a
    bigger one. close_wall_pinholes has to converge either way, which is what the
    batch run needs and what a single repair could not promise.
    """
    from vessel_pipeline import (
        _is_wall_pinhole,
        boundary_loop_radii,
        close_wall_pinholes,
    )

    holed = _pinch_two_holes()
    assert len(boundary_loop_radii(holed)) == 4

    fixed, n = close_wall_pinholes(holed, min_radius=0.5, label="test surface")
    assert n >= 1
    after = boundary_loop_radii(fixed)
    assert len(after) == 2, f"expected only the two tube ends, got {after}"
    assert not [lp for lp in after if _is_wall_pinhole(lp, 0.5)]


def test_collapse_refuses_to_close_the_last_openings():
    """A pinhole patch may never cost the mesh its anatomical ostia."""
    from vessel_pipeline import boundary_loop_radii, collapse_small_boundary_components

    tube = open_tube(n_sides=40, n_rings=30)
    # min_radius above the tube radius makes every loop look like a pinhole
    fixed, n = collapse_small_boundary_components(tube, min_radius=5.0)
    assert n == 0
    assert len(boundary_loop_radii(fixed)) == 2


def _tube_with_a_narrow_end(end_radius=0.18):
    """Open tube whose far end is narrowed to a sub-threshold opening.

    0.18 mm is below WALL_PINHOLE_RADIUS_MM but above the 0.128-0.199 mm band
    the old run's leftover rims occupied, and the smallest real ostium measured
    on this dataset was 0.205 mm -- so radius alone cannot tell them apart.
    """
    tube = open_tube(radius=1.0, length=6.0, n_sides=40, n_rings=30)
    _poly, pts, faces = _triangle_points_faces(tube)
    z = pts[:, 2]
    top = z > (z.max() - 1e-9)
    scale = end_radius / 1.0
    pts[top, 0] *= scale
    pts[top, 1] *= scale
    return _polydata_from_triangles(pts, faces)


def _profiles_for(surface):
    from vessel_pipeline import boundary_loop_radii

    return [
        {"barycenter": np.asarray(bary, dtype=np.float64), "radius": float(r)}
        for r, _n, bary in boundary_loop_radii(surface)
    ]


def test_a_real_ostium_below_the_pinhole_radius_is_not_closed():
    """Objective (c): a genuine opening may never be welded shut."""
    from vessel_pipeline import boundary_loop_radii, close_wall_pinholes

    surf = _tube_with_a_narrow_end(end_radius=0.18)
    loops = boundary_loop_radii(surf)
    narrow = min(lp[0] for lp in loops)
    assert narrow < 0.2, "the fixture must sit below the pinhole radius"

    profiles = _profiles_for(surf)
    fixed, n = close_wall_pinholes(surf, label="test surface", profiles=profiles)
    assert n == 0, "a loop at an anatomical profile must be left alone"
    assert len(boundary_loop_radii(fixed)) == len(loops)


def test_debris_is_still_closed_when_no_profile_claims_it():
    """Protection is not a blanket amnesty: unclaimed holes still get closed."""
    from vessel_pipeline import boundary_loop_radii, close_wall_pinholes

    holed = _pinch_two_holes()
    loops = boundary_loop_radii(holed)
    assert len(loops) == 4
    # the two tube ends are the anatomy; the pinched pair is debris
    profiles = [p for p in _profiles_for(holed) if p["radius"] > 0.5]
    assert len(profiles) == 2

    fixed, n = close_wall_pinholes(
        holed, min_radius=0.5, label="test surface", profiles=profiles
    )
    assert n >= 1
    assert len(boundary_loop_radii(fixed)) == 2


def test_profile_protection_needs_the_loop_to_be_at_the_profile():
    """Protection is by position, not by merely having profiles around."""
    from vessel_pipeline import _loop_at_a_profile

    profiles = [{"barycenter": np.zeros(3), "radius": 0.3}]
    assert _loop_at_a_profile(np.zeros(3), profiles)
    assert not _loop_at_a_profile(np.asarray([5.0, 0.0, 0.0]), profiles)
    assert not _loop_at_a_profile(np.zeros(3), [])


def test_debris_next_to_an_ostium_is_not_protected():
    """Proximity alone must not shield a 4-point micron rim beside a real ostium.

    This is the shape that survived the uncap stage on SNF00000365_01
    (r=0.004 mm, 4 points) because it sat within the protection radius of a
    genuine opening.
    """
    from vessel_pipeline import _loop_at_a_profile

    profiles = [{"barycenter": np.zeros(3), "radius": 1.0}]
    near = np.asarray([0.3, 0.0, 0.0])
    # right size and a real rim -> anatomy
    assert _loop_at_a_profile(near, profiles, radius=0.9, n_points=40)
    # right place, far too small -> debris
    assert not _loop_at_a_profile(near, profiles, radius=0.004, n_points=40)
    # right place and size, but no rim to speak of -> debris
    assert not _loop_at_a_profile(near, profiles, radius=0.9, n_points=4)


def test_a_small_but_real_ostium_is_still_protected():
    """An ostium below the pinhole radius must survive on the profile's word.

    The smallest real opening measured on this dataset was 0.205 mm against a
    0.20 mm threshold, so the margin is 2.5% and the next mesh may well fall the
    other side of it. 0.19 mm is that mesh.
    """
    from vessel_pipeline import _is_wall_pinhole, _loop_at_a_profile

    profiles = [{"barycenter": np.zeros(3), "radius": 0.19}]
    loop = (0.19, 14, np.zeros(3))
    assert _loop_at_a_profile(loop[2], profiles, radius=loop[0], n_points=loop[1])
    assert not _is_wall_pinhole(loop, 0.20, profiles)
    # and without the anatomy backing it, the same loop is closable debris
    assert _is_wall_pinhole(loop, 0.20, None)


def test_finalize_does_not_reintroduce_holes_it_just_repaired():
    """vtkFillHolesFilter is out of the loop; the loop must not spin.

    On C0010 it turned an nm=0 surface into nm=5 while closing no holes at all,
    and the next pass removed exactly those triangles again -- four identical
    passes in a row.
    """
    import inspect

    from vessel_pipeline import finalize_surface

    src = inspect.getsource(finalize_surface)
    assert "fill_pinholes" not in src, "the hole filler must stay out of the repair loop"
    assert "seen" in src, "the loop must detect a repeated state"


def test_finalize_returns_the_best_surface_not_the_last():
    from vessel_pipeline import _finalize_defects, finalize_surface

    holed = _pinch_two_holes()
    fixed, n_regions = finalize_surface(holed)
    assert n_regions == 1
    # whatever path it took, what comes back must be no worse than clean
    assert _finalize_defects(fixed, None, n_regions)[0] == 0
