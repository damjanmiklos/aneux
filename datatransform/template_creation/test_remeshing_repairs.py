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
    assert_template_quality,
    compute_template_local_radii,
    opening_clip_frames,
    _flow_extension_layer_estimate,
    _is_disc,
    _keep_region_with_point,
    _opening_clip_height,
    _polydata_from_triangles,
    _triangle_edge_lengths,
    _triangle_points_faces,
    add_flow_extensions,
    boundary_loop_radii,
    count_bowtie_vertices,
    drop_hanging_pieces,
    drop_pillow_triangles,
    fan_fill_small_loops,
    finalize_surface,
    force_manifold_triangles,
    inspect_surface_topology,
    measure_open_profiles,
    stamp_polyball_image,
    surface_genus,
    untangle_polyball_field,
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
    # This used to pin 20/20 on the assumption that more remesh iterations buy
    # quality. Measured, they cost it: VMTK's relocation oscillates instead of
    # converging, and 20/20 grows p129's area 5.45x at CV 2.3181 while 6/10
    # holds it at 1.024x and CV 0.3952. So the objective is not "many
    # iterations", it is "the iteration counts the rest of the pipeline is
    # validated at", and raising them is the regression to guard against.
    import vessel_pipeline as _vp

    assert rm.GT_REMESH_N_ITER == _vp.REMESH_N_ITER
    assert rm.GT_REMESH_CONNECTIVITY_ITER == _vp.REMESH_CONNECTIVITY_ITER
    assert rm.GT_REMESH_N_ITER <= 6
    # Every fallback must ask for *less* work than the attempt that diverged.
    assert all(a[0] < rm.GT_REMESH_N_ITER for a in _vp.REMESH_ITER_FALLBACKS)
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


def _component_and_loop_radii(surface):
    """Radius of each free-edge component, and of each boundary loop."""
    from vessel_pipeline import (
        _boundary_component_extent,
        _free_edge_components,
        boundary_loop_radii,
    )

    _poly, pts, faces = _triangle_points_faces(surface)
    comps = _free_edge_components(pts, faces)
    comp_radii = [_boundary_component_extent(pts, ids)[1] for ids in comps]
    loop_radii = [lp[0] for lp in boundary_loop_radii(surface)]
    return comp_radii, loop_radii


def test_a_pinched_pair_hides_inside_one_free_edge_component():
    """Documents why the per-loop pass exists.

    Two punctures meeting at a vertex share a free-edge component, and that
    component measures larger than either hole. A threshold that catches the
    holes therefore misses the component -- on ANSYS_UNIGE_30_614 the component
    was fused to an ostium rim and an r=0.08 mm hole reached the quality gate.
    """
    comp_radii, loop_radii = _component_and_loop_radii(_pinch_two_holes())
    small_loops = sorted(loop_radii)[:2]
    fused = min(r for r in comp_radii if r > max(small_loops))
    assert fused > max(small_loops), (
        f"the fused component {fused:.3f} must measure larger than its holes "
        f"{small_loops}"
    )


def test_loop_collapse_reaches_what_component_collapse_cannot():
    from vessel_pipeline import (
        _is_wall_pinhole,
        boundary_loop_radii,
        collapse_pinhole_loops,
        collapse_small_boundary_components,
        count_connected_regions,
        inspect_surface_topology,
    )

    holed = _pinch_two_holes()
    comp_radii, loop_radii = _component_and_loop_radii(holed)
    small_loops = sorted(loop_radii)[:2]
    fused = min(r for r in comp_radii if r > max(small_loops))
    # a threshold that catches both holes but not the component they share
    cutoff = 0.5 * (max(small_loops) + fused)

    _unchanged, n_comp = collapse_small_boundary_components(holed, min_radius=cutoff)
    assert n_comp == 0, "the component pass cannot see these holes"

    fixed, n_loop = collapse_pinhole_loops(holed, min_radius=cutoff)
    assert n_loop == 2, "the per-loop pass closes both"
    assert not [lp for lp in boundary_loop_radii(fixed) if _is_wall_pinhole(lp, cutoff)]
    assert inspect_surface_topology(fixed)["n_nonmanifold"] == 0
    assert count_connected_regions(fixed) == 1


def test_loop_collapse_also_refuses_to_close_the_last_openings():
    from vessel_pipeline import boundary_loop_radii, collapse_pinhole_loops

    tube = open_tube(n_sides=40, n_rings=30)
    fixed, n = collapse_pinhole_loops(tube, min_radius=5.0, label="test surface")
    assert n == 0
    assert len(boundary_loop_radii(fixed)) == 2


def _scaled_copy(surface, factor):
    """A surface with `factor` times the area, standing in for a diverged remesh."""
    import pyvista as pv

    from vessel_pipeline import to_vtk_poly

    grown = pv.wrap(to_vtk_poly(surface)).copy()
    grown.points = grown.points * float(factor) ** 0.5
    return to_vtk_poly(grown)


def test_remesh_keeps_the_configured_iterations_when_they_converge():
    """The 600+ cases that already remesh cleanly must not change behaviour."""
    import vessel_pipeline as vp

    tube = open_tube(n_sides=40, n_rings=30)
    calls = []

    def fake(surface, target_edge_length, n_iter, connectivity_iter, collapse_angle=None):
        calls.append((n_iter, connectivity_iter))
        return vp.to_vtk_poly(surface)

    original = vp.remesh_surface_isotropically
    vp.remesh_surface_isotropically = fake
    try:
        vp.remesh_surface_verified(
            tube, target_edge_length=0.15, n_iter=20, connectivity_iter=20
        )
    finally:
        vp.remesh_surface_isotropically = original

    assert calls == [(20, 20)], "a converging remesh must not be retried"


def test_remesh_backs_off_when_the_configured_iterations_diverge():
    """vmtkSurfaceRemeshing reports no error when it folds the surface."""
    import vessel_pipeline as vp

    tube = open_tube(n_sides=40, n_rings=30)
    calls = []

    def fake(surface, target_edge_length, n_iter, connectivity_iter, collapse_angle=None):
        calls.append((n_iter, connectivity_iter))
        # The configured count diverges the way 20/20 does on p129; backing off
        # to fewer iterations is what recovers it.
        if n_iter >= vp.REMESH_N_ITER:
            return _scaled_copy(surface, 5.45)
        return vp.to_vtk_poly(surface)

    original = vp.remesh_surface_isotropically
    vp.remesh_surface_isotropically = fake
    try:
        out = vp.remesh_surface_verified(
            tube,
            target_edge_length=0.15,
            n_iter=vp.REMESH_N_ITER,
            connectivity_iter=vp.REMESH_CONNECTIVITY_ITER,
        )
    finally:
        vp.remesh_surface_isotropically = original

    # It must retry downwards, never upwards.
    assert len(calls) > 1, calls
    assert calls[0][0] == vp.REMESH_N_ITER, calls
    assert [c[0] for c in calls] == sorted((c[0] for c in calls), reverse=True), calls
    import pyvista as pv

    ratio = pv.wrap(out).area / pv.wrap(vp.to_vtk_poly(tube)).area
    assert ratio <= vp.REMESH_MAX_AREA_DRIFT, ratio


def test_remesh_failure_is_reported_as_a_remesh_failure():
    """The old message blamed flow extensions for damage done here."""
    import pytest

    import vessel_pipeline as vp

    tube = open_tube(n_sides=40, n_rings=30)

    def fake(surface, target_edge_length, n_iter, connectivity_iter, collapse_angle=None):
        return _scaled_copy(surface, 4.0)

    original = vp.remesh_surface_isotropically
    vp.remesh_surface_isotropically = fake
    try:
        with pytest.raises(vp.TemplateQualityError) as excinfo:
            vp.remesh_surface_verified(
                tube, target_edge_length=0.15, n_iter=20, connectivity_iter=20
            )
    finally:
        vp.remesh_surface_isotropically = original

    message = str(excinfo.value)
    assert "remesh" in message.lower()
    assert "flow extension" not in message.lower()

def test_fan_lid_is_not_coplanar_with_the_rim_it_closes():
    """A flat lid hands the Delaunay step the degeneracy caps exist to avoid.

    ``cap_surface`` falls back to fanning when vtkvmtkCapPolyData cannot walk a
    rim, and that fallback only ever runs on damaged meshes -- the ones whose
    centerlines hang. Putting the apex in the rim plane there would reintroduce
    coplanar points on exactly those cases.
    """
    import vessel_pipeline as vp

    tube = open_tube(n_sides=40, n_rings=30)
    closed = vp.fan_fill_every_loop(tube)
    assert vp.extract_boundary_loops(closed).GetNumberOfCells() == 0

    _poly, pts, _faces = _triangle_points_faces(tube)
    # The rim of an open tube lies in a plane of constant z; a domed lid must not.
    rim_z = np.unique(np.round(pts[:, 2], 6))
    _poly2, pts2, _faces2 = _triangle_points_faces(closed)
    added = pts2[len(pts):]
    assert len(added) > 0
    assert np.all(added[:, 2] < rim_z.min() - 1e-9) or np.all(
        added[:, 2] > rim_z.max() + 1e-9
    ) or np.any(
        np.abs(added[:, 2][:, None] - rim_z[None, :]).min(axis=1) > 1e-6
    ), added


def test_fan_lid_domes_away_from_the_lumen():
    """Outwards, so the lid reads as a cap rather than a dent in the vessel."""
    import vessel_pipeline as vp

    tube = open_tube(n_sides=40, n_rings=30)
    _poly, pts, _faces = _triangle_points_faces(tube)
    centroid = pts.mean(axis=0)
    rim_reach = np.abs(pts[:, 2] - centroid[2]).max()
    closed = vp.fan_fill_every_loop(tube)
    _poly2, pts2, _faces2 = _triangle_points_faces(closed)
    added = pts2[len(pts):]
    assert len(added) > 0
    # Strictly beyond the rim, not merely off-plane: a lid pushed the other way
    # would also be non-coplanar but would dent the lumen it is meant to close.
    assert np.all(np.abs(added[:, 2] - centroid[2]) > rim_reach + 1e-9), (
        added, rim_reach
    )


def _grid_sheet(n=6, spacing=1.0, z=0.0, first_id=0):
    """A flat (n x n) triangulated patch; ids start at ``first_id``."""
    pts, faces = [], []
    for i in range(n):
        for j in range(n):
            pts.append((i * spacing, j * spacing, z))
    for i in range(n - 1):
        for j in range(n - 1):
            a = first_id + i * n + j
            b = a + 1
            c = a + n
            d = c + 1
            faces.append((a, c, b))
            faces.append((b, c, d))
    return np.asarray(pts, dtype=float), faces


def test_collapse_tiny_edges_removes_a_sliver():
    """The whole point: an edge microns long must not reach the remesher."""
    from vessel_pipeline import collapse_tiny_edges, count_connected_regions

    pts, faces = _grid_sheet()
    # Pull one interior vertex onto its neighbour, leaving a 1e-6 mm edge.
    victim, anchor = 14, 15
    pts[victim] = pts[anchor] + np.array([1e-6, 0.0, 0.0])
    surf = _polydata_from_triangles(pts, np.asarray(faces, dtype=np.int64))

    _p, before_pts, before_faces = _triangle_points_faces(surf)
    assert _triangle_edge_lengths(before_pts, before_faces).min() < 1e-5

    fixed = collapse_tiny_edges(surf, floor=1e-3)
    _p, after_pts, after_faces = _triangle_points_faces(fixed)
    assert _triangle_edge_lengths(after_pts, after_faces).min() >= 1e-3
    assert inspect_surface_topology(fixed)["n_nonmanifold"] == 0
    assert count_connected_regions(fixed) == 1


def test_collapse_tiny_edges_leaves_a_clean_surface_alone():
    """No edge under the floor means nothing may move."""
    from vessel_pipeline import collapse_tiny_edges, count_connected_regions

    pts, faces = _grid_sheet()
    surf = _polydata_from_triangles(pts, np.asarray(faces, dtype=np.int64))
    fixed = collapse_tiny_edges(surf, floor=1e-3)

    _p, a, fa = _triangle_points_faces(surf)
    _p, b, fb = _triangle_points_faces(fixed)
    assert len(a) == len(b)
    assert len(fa) == len(fb)
    assert np.allclose(np.sort(a, axis=0), np.sort(b, axis=0))


def test_collapse_tiny_edges_does_not_fuse_two_sheets_touching_at_a_point():
    """The link condition, which is the reason this is not a point merge.

    Two sheets that meet at a single vertex share that vertex's neighbours.
    Folding an edge into it would zip them together along a seam -- the exact
    damage the tolerance merge used to do, and the reason a naive collapse took
    one surface from 5 non-manifold edges to 9. Here the collapse must decline.
    """
    from vessel_pipeline import collapse_tiny_edges, count_connected_regions

    lower_pts, lower_faces = _grid_sheet(z=0.0, first_id=0)
    upper_pts, upper_faces = _grid_sheet(z=1.0, first_id=len(lower_pts))
    pts = np.vstack([lower_pts, upper_pts])
    faces = lower_faces + upper_faces

    # Pinch: drag one upper vertex down onto a lower one so the two sheets meet
    # at that single point, then put a sliver edge across the pinch.
    pinch_low = 14
    pinch_high = len(lower_pts) + 14
    pts[pinch_high] = pts[pinch_low]
    pts[len(lower_pts) + 15] = pts[pinch_low] + np.array([1e-6, 0.0, 0.0])

    surf = _polydata_from_triangles(pts, np.asarray(faces, dtype=np.int64))
    before = count_connected_regions(surf)
    fixed = collapse_tiny_edges(surf, floor=1e-3)

    assert count_connected_regions(fixed) == before, "the sheets were fused"
    assert (
        inspect_surface_topology(fixed)["n_nonmanifold"]
        <= inspect_surface_topology(surf)["n_nonmanifold"]
    ), "the collapse added non-manifold edges"


def _with_pillow(copies=2):
    """open_tube with a triangle on three wall vertices stacked ``copies`` times.

    Alternate copies are flipped, so two of them are back to back: a closed
    zero-volume surface touching the wall only at its corners, as the uncap
    left on SNF00000360_01_1.
    """
    tube = open_tube()
    _poly, pts, faces = _triangle_points_faces(tube, clean=False)
    a, b, c = 5 * 40 + 3, 5 * 40 + 9, 9 * 40 + 6
    extra = [(a, b, c) if k % 2 == 0 else (c, b, a) for k in range(copies)]
    return tube, _polydata_from_triangles(pts, np.vstack([faces, extra]))


def test_a_pillow_is_dropped_whole():
    tube, pillowed = _with_pillow()
    out, n = drop_pillow_triangles(pillowed)
    assert n == 1
    assert out.GetNumberOfCells() == tube.GetNumberOfCells()
    assert len(boundary_loop_radii(out)) == 2


def test_an_odd_copy_is_kept_as_a_face():
    tube, pillowed = _with_pillow(copies=3)
    out, n = drop_pillow_triangles(pillowed)
    assert n == 1
    assert out.GetNumberOfCells() == tube.GetNumberOfCells() + 1


def test_a_surface_without_a_pillow_is_returned_as_it_was():
    tube = open_tube()
    out, n = drop_pillow_triangles(tube)
    assert n == 0
    assert out is tube


def _with_hanging(size=0.3, closed=True):
    """open_tube with a small sheet joined to the wall by one vertex only.

    ``closed`` gives a tetrahedron on a mid-wall vertex, the bubble C0058 took
    into the remesh. Otherwise it is a four-triangle disc on a rim vertex, the
    clip flap ANSYS_UNIGE_17_10 did, whose own boundary reads as an opening.
    """
    tube = open_tube()
    _poly, pts, faces = _triangle_points_faces(tube, clean=False)
    n = len(pts)
    if closed:
        w = 15 * 40
        tip = pts[w] + size * np.array([[1.0, 0.0, 0.0], [1.0, 0.8, 0.0], [1.0, 0.4, 0.8]])
        a, b, c = n, n + 1, n + 2
        extra = [(w, a, b), (w, b, c), (w, c, a), (a, c, b)]
    else:
        w = 0
        centre = pts[w] + size * np.array([0.5, 0.0, -0.5])
        ring = [centre + 0.5 * size * np.array([np.cos(t), 0.0, np.sin(t)]) for t in (0.8, 2.4, 4.0)]
        tip = np.vstack([centre] + ring)
        ctr, r1, r2, r3 = n, n + 1, n + 2, n + 3
        extra = [(ctr, w, r1), (ctr, r1, r2), (ctr, r2, r3), (ctr, r3, w)]
    pts = np.vstack([pts, tip])
    return tube, _polydata_from_triangles(pts, np.vstack([faces, extra]))


def test_a_vertex_joined_sheet_is_a_bowtie():
    tube, hung = _with_hanging()
    _p, pts, faces = _triangle_points_faces(tube)
    assert count_bowtie_vertices(faces, len(pts)) == 0
    _p, pts, faces = _triangle_points_faces(hung)
    assert count_bowtie_vertices(faces, len(pts)) == 1
    assert inspect_surface_topology(hung)["n_bowtie"] == 1
    assert inspect_surface_topology(hung)["n_nonmanifold"] == 0


def test_a_hanging_bubble_is_dropped():
    tube, hung = _with_hanging(closed=True)
    out, n = drop_hanging_pieces(hung)
    assert n == 1
    assert out.GetNumberOfCells() == tube.GetNumberOfCells()
    assert inspect_surface_topology(out)["n_bowtie"] == 0
    assert len(boundary_loop_radii(out)) == 2


def test_a_hanging_rim_flap_is_dropped_and_its_loop_with_it():
    tube, hung = _with_hanging(closed=False)
    assert inspect_surface_topology(hung)["n_bowtie"] == 1
    out, n = drop_hanging_pieces(hung)
    assert n == 1
    assert out.GetNumberOfCells() == tube.GetNumberOfCells()
    assert len(boundary_loop_radii(out)) == 2


def test_a_hanging_piece_big_enough_to_be_anatomy_is_kept():
    _tube, hung = _with_hanging(size=2.0)
    out, n = drop_hanging_pieces(hung)
    assert n == 0
    assert out.GetNumberOfCells() == hung.GetNumberOfCells()


def test_a_surface_without_a_hanging_piece_is_returned_as_it_was():
    tube = open_tube()
    out, n = drop_hanging_pieces(tube)
    assert n == 0
    assert out.GetNumberOfCells() == tube.GetNumberOfCells()
    assert out.GetNumberOfPoints() == tube.GetNumberOfPoints()
    # Not even the clean that finds the pieces may touch it.
    _p, pts, faces = _triangle_points_faces(tube, clean=False)
    _q, out_pts, out_faces = _triangle_points_faces(out, clean=False)
    assert np.array_equal(pts, out_pts) and np.array_equal(faces, out_faces)


def test_remesh_ladder_moves_past_a_rung_that_adds_a_bowtie():
    """ANSYS_UNIGE_09: collapse-on bowtied a clean input, collapse-off did not."""
    import vessel_pipeline as vp

    tube, hung = _with_hanging()
    calls = []

    def fake(surface, target_edge_length, n_iter, connectivity_iter, collapse_angle=None):
        calls.append(collapse_angle)
        if collapse_angle == vp.REMESH_COLLAPSE_ANGLE:
            return vp.to_vtk_poly(hung)
        return vp.to_vtk_poly(surface)

    original = vp.remesh_surface_isotropically
    vp.remesh_surface_isotropically = fake
    try:
        out = vp.remesh_surface_verified(tube, target_edge_length=0.15, n_iter=20, connectivity_iter=20)
    finally:
        vp.remesh_surface_isotropically = original

    assert calls == [vp.REMESH_COLLAPSE_ANGLE, vp.REMESH_COLLAPSE_ANGLE_OFF], calls
    assert inspect_surface_topology(out)["n_bowtie"] == 0


def test_remesh_ladder_does_not_blame_the_remesher_for_an_input_bowtie():
    """A pinch too big to drop is the input's; the first rung still stands."""
    import vessel_pipeline as vp

    _tube, hung = _with_hanging(size=2.0)
    calls = []

    def fake(surface, target_edge_length, n_iter, connectivity_iter, collapse_angle=None):
        calls.append(collapse_angle)
        return vp.to_vtk_poly(surface)

    original = vp.remesh_surface_isotropically
    vp.remesh_surface_isotropically = fake
    try:
        vp.remesh_surface_verified(hung, target_edge_length=0.15, n_iter=20, connectivity_iter=20)
    finally:
        vp.remesh_surface_isotropically = original

    assert calls == [vp.REMESH_COLLAPSE_ANGLE], calls


def test_frames_can_vouch_for_an_opening_the_pinhole_filter_drops():
    """ANSYS_UNIGE_27 lost a real 0.207 mm outlet to the 0.2 mm seed filter.

    A big tube and a thin one, 0.12 mm across its ends: by size alone the thin
    one's rims read as pinholes. A caller that knows every rim is real (the GT
    frames count them) turns the filter off and keeps all four.
    """
    big = open_tube(radius=1.0, n_sides=40, n_rings=20)
    thin = open_tube(radius=0.12, length=2.0, n_sides=16, n_rings=8, center=(5.0, 0.0, 0.0))
    _p, p1, f1 = _triangle_points_faces(big, clean=False)
    _p, p2, f2 = _triangle_points_faces(thin, clean=False)
    both = _polydata_from_triangles(np.vstack([p1, p2]), np.vstack([f1, f2 + len(p1)]))
    assert len(measure_open_profiles(both)) == 2
    kept = measure_open_profiles(both, min_radius=0.0)
    assert len(kept) == 4
    assert sorted(round(p["radius"], 2) for p in kept)[:2] == [0.12, 0.12]


def _polyball_surface(centres, radius, untangle):
    """Marching cubes of a sphere chain, optionally untangled first."""
    from vtk.util.numpy_support import vtk_to_numpy

    centres = np.asarray(centres, dtype=np.float64)
    spacing = 0.1
    lo = centres.min(axis=0) - radius - 0.5
    hi = centres.max(axis=0) + radius + 0.5
    dims = [int(np.ceil((b - a) / spacing)) + 1 for a, b in zip(lo, hi)]
    bounds = [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]
    image = stamp_polyball_image(centres, np.full(len(centres), radius), bounds, dims, spacing)
    n_cut = 0
    if untangle:
        field = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(dims[2], dims[1], dims[0])
        n_cut, _n_filled = untangle_polyball_field(field, spacing)
        image.GetPointData().GetScalars().Modified()
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(image)
    mc.SetValue(0, 0.0)
    mc.Update()
    return mc.GetOutput(), n_cut


def _ring(n=60, radius=2.0):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(t), radius * np.sin(t), np.zeros(n)))


def test_surface_genus_counts_handles_and_ignores_rims():
    torus, _n = _polyball_surface(_ring(), 0.5, untangle=False)
    assert surface_genus(torus) == 1.0
    assert surface_genus(open_tube(radius=1.0, n_sides=24, n_rings=8)) == 0.0


def test_untangle_cuts_the_tunnel_where_two_branches_fuse():
    """Two ends of a sphere chain that just overlap: p402's fused branches in miniature."""
    # 0.84 mm between the end centres, so the two r=0.5 spheres share a lens.
    ends_touching = _ring()[:-3]
    fused, _n = _polyball_surface(ends_touching, 0.5, untangle=False)
    assert surface_genus(fused) == 1.0
    surface, n_cut = _polyball_surface(ends_touching, 0.5, untangle=True)
    assert n_cut > 0
    assert surface_genus(surface) == 0.0
    # The cut is the lens where the ends overlap (0.27 mm across), not a slice
    # through the tube, which would be ~78 voxels at r=0.5 and 0.1 mm.
    assert n_cut < 50


def test_untangle_leaves_a_tree_alone():
    chain = np.column_stack((np.linspace(0.0, 4.0, 41), np.zeros(41), np.zeros(41)))
    before, _n = _polyball_surface(chain, 0.5, untangle=False)
    after, n_cut = _polyball_surface(chain, 0.5, untangle=True)
    assert n_cut == 0
    assert before.GetNumberOfPoints() == after.GetNumberOfPoints()
    assert surface_genus(after) == 0.0


def test_quality_gate_refuses_a_surface_with_a_tunnel():
    torus, _n = _polyball_surface(_ring(), 0.5, untangle=False)
    with pytest.raises(TemplateQualityError, match="genus 1"):
        assert_template_quality(torus, context="torus")


def _centerline(tracts):
    """Polyline centerline, one cell per (points, radii) tract."""
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    radii = vtk.vtkDoubleArray()
    radii.SetName("MaximumInscribedSphereRadius")
    for pts, r in tracts:
        ids = vtk.vtkIdList()
        for p, rv in zip(pts, r):
            ids.InsertNextId(points.InsertNextPoint(*map(float, p)))
            radii.InsertNextValue(float(rv))
        lines.InsertNextCell(ids)
    cl = vtk.vtkPolyData()
    cl.SetPoints(points)
    cl.SetLines(lines)
    cl.GetPointData().AddArray(radii)
    return cl


def test_template_radius_comes_from_the_ball_the_wall_is_on():
    """p388: a thick wall next to a thin branch's first points keeps the thick radius."""
    xs = np.linspace(0.0, 10.0, 51)
    trunk = (np.column_stack((xs, np.zeros(51), np.zeros(51))), np.full(51, 2.5))
    ys = np.linspace(0.0, 6.0, 31)
    branch = (np.column_stack((np.full(31, 5.0), ys, np.zeros(31))), np.full(31, 0.35))
    cl = _centerline([trunk, branch])
    points = vtk.vtkPoints()
    # On the trunk wall 2 mm from the branch (the nearest centerline there is
    # the branch's, 2.0 mm against the trunk's 2.5); on the branch wall outside
    # the trunk; and midway along the trunk where the radius tapers.
    for p in ((3.0, 2.5, 0.0), (5.0, 4.0, 0.35), (8.0, 0.0, -2.5)):
        points.InsertNextPoint(*p)
    probe = vtk.vtkPolyData()
    probe.SetPoints(points)
    r = compute_template_local_radii(probe, cl)
    assert r[0] == pytest.approx(2.5, abs=1e-6)
    assert r[1] == pytest.approx(0.35, abs=1e-6)
    assert r[2] == pytest.approx(2.5, abs=1e-6)


def test_template_radius_interpolates_along_a_tapering_segment():
    xs = np.array([0.0, 4.0])
    cl = _centerline([(np.column_stack((xs, np.zeros(2), np.zeros(2))), np.array([2.0, 1.0]))])
    points = vtk.vtkPoints()
    # Over the middle of the segment. The ball that touches a cone's wall sits a
    # little upstream of the foot of the perpendicular (t = 0.4, r = 1.6 here),
    # so the answer is the interpolated radius, not either end's.
    points.InsertNextPoint(2.0, 1.5, 0.0)
    probe = vtk.vtkPolyData()
    probe.SetPoints(points)
    r = compute_template_local_radii(probe, cl)[0]
    assert r == pytest.approx(1.6, abs=1e-6)


def test_opening_axis_ignores_a_last_cell_lying_across_the_rim():
    """UPF_P0258's inlet: the trace ends in a 2-point cell in the rim plane."""
    stub = (np.array([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]]), np.full(2, 1.0))
    zs = np.linspace(0.0, -10.0, 101)
    trunk = (np.column_stack((np.full(101, 0.08), np.zeros(101), zs)), np.full(101, 1.0))
    # A branch leaving right at the opening and running out past its plane.
    ts = np.linspace(0.0, 1.0, 51)
    branch = (np.column_stack((0.08 + 3.0 * ts, np.zeros(51), 5.0 * ts)), np.full(51, 0.4))
    cl = _centerline([stub, trunk, branch])
    profile = {"barycenter": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0]), "radius": 1.0}
    (_origin, outward, _radius), = opening_clip_frames(cl, [profile])
    assert float(np.dot(outward, [0.0, 0.0, 1.0])) > 0.99


def test_a_squared_frame_takes_its_radius_from_the_rim_too():
    """A frame whose axis lay across its rim got its radius off the same point.

    p531's frame 7 carried a 0.585 mm inscribed radius into a 0.391 mm rim, and
    the cutter sized from it swallowed the ostium next door. A frame narrower
    than its rim keeps its own radius: only the foreign excess is dropped.
    """
    from vessel_pipeline import square_frames_to_rims

    tube = open_tube(radius=0.4, length=6.0, n_sides=40, n_rings=30)
    _poly, pts, _faces = _triangle_points_faces(tube)
    top = np.array([0.0, 0.0, pts[:, 2].max()])
    bottom = np.array([0.0, 0.0, pts[:, 2].min()])
    frames = [
        {"origin": top, "normal": np.array([1.0, 0.0, 0.0]), "radius": 0.6},
        {"origin": bottom, "normal": np.array([0.0, 1.0, 0.05]), "radius": 0.3},
        {"origin": top, "normal": np.array([0.0, 1.0, -0.05]), "radius": 0.3},
    ]
    squared, n = square_frames_to_rims(frames, tube)
    assert n == 3
    # Outward is read off the wall, not the old normal: +z at the top even
    # though the bottom frame's old normal leaned the other way.
    assert squared[0]["normal"][2] == pytest.approx(1.0, abs=1e-6)
    assert squared[1]["normal"][2] == pytest.approx(-1.0, abs=1e-6)
    assert squared[2]["normal"][2] == pytest.approx(1.0, abs=1e-6)
    assert squared[0]["radius"] == pytest.approx(0.4, abs=0.01)
    assert squared[1]["radius"] == pytest.approx(0.3)
    assert frames[0]["radius"] == 0.6


def test_a_centerline_end_is_trimmed_only_at_its_own_opening():
    """C0048: two side-by-side outlets must not trim each other's branch."""
    from vessel_pipeline import clip_centerline_at_profiles

    xs = np.linspace(0.0, 15.0, 151)
    a = (np.column_stack((xs, np.zeros_like(xs), np.zeros_like(xs))), np.full(xs.size, 0.5))
    xb = np.linspace(0.0, 14.0, 141)
    b = (np.column_stack((xb, np.full(xb.size, 2.0), np.zeros_like(xb))), np.full(xb.size, 0.5))
    profiles = [
        {"barycenter": np.array([10.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0]), "radius": 0.5},
        {"barycenter": np.array([9.0, 2.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0]), "radius": 0.5},
    ]
    out = clip_centerline_at_profiles(_centerline([a, b]), profiles, extension_length=5.0)
    pts = np.array([out.GetPoint(i) for i in range(out.GetNumberOfPoints())])
    on_a = pts[np.abs(pts[:, 1]) < 1e-9]
    on_b = pts[np.abs(pts[:, 1] - 2.0) < 1e-9]
    # Tract a ends within reach of b's opening and outside its plane, so it
    # used to be cut back to x=9 as well.
    assert on_a[:, 0].max() == pytest.approx(10.0, abs=0.11)
    assert on_b[:, 0].max() == pytest.approx(9.0, abs=0.11)
