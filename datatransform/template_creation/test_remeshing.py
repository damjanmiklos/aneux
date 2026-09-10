"""GT remesh: original vessel, constant edge length, planar ostia."""
import os
import sys

import numpy as np
import pyvista as pv
import pytest
import vtk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from remeshing import (
    DEFAULT_GT_EDGE_LENGTH_MM,
    GT_MAX_AREA_RATIO,
    GT_MIN_AREA_RATIO,
    GT_REMESH_N_ITER,
    GT_TAUBIN_ITER,
    GT_TAUBIN_PASS_BAND,
    assert_gt_remesh_scale,
    parse_args,
    prepare_gt_surface,
)
from vessel_pipeline import (
    TemplateQualityError,
    _drop_small_fragments,
    _opening_clip_height,
    _triangle_edge_lengths,
    _triangle_points_faces,
    assert_template_quality,
    clip_one_opening_pipe_section,
    count_connected_regions,
    extract_boundary_loops,
    fill_pinholes,
    inspect_openings,
    inspect_surface_topology,
    remesh_surface_isotropically,
    remove_spurious_openings,
    sanitize_vessel_for_vmtk,
    tessellation_looks_original,
    to_vtk_poly,
)


def _capped_tube(p0, p1, radius=2.0, n_sides=24):
    line = pv.Line(p0, p1)
    tube = to_vtk_poly(line.tube(radius=radius, n_sides=n_sides, capping=True))
    return fill_pinholes(tube, hole_size=20.0)


def _loop_points(surface):
    loops = extract_boundary_loops(surface)
    out = []
    for i in range(loops.GetNumberOfCells()):
        cell = loops.GetCell(i)
        n = cell.GetNumberOfPoints()
        pts = np.array([cell.GetPoints().GetPoint(j) for j in range(n)], dtype=np.float64)
        out.append(pts)
    return out


def test_gt_smoothing_is_weaker_than_template():
    assert GT_TAUBIN_PASS_BAND >= 1.4
    assert GT_TAUBIN_ITER <= 8
    assert DEFAULT_GT_EDGE_LENGTH_MM == 0.15
    assert GT_REMESH_N_ITER == 20


def test_gt_scale_rejects_lost_sac():
    original = pv.Sphere(radius=5.0, theta_resolution=40, phi_resolution=40)
    shrunk = pv.Sphere(radius=4.0, theta_resolution=40, phi_resolution=40)
    with pytest.raises(TemplateQualityError, match="aneurysm or a branch"):
        assert_gt_remesh_scale(shrunk, original, context="case")


def test_gt_scale_rejects_kept_extensions():
    original = pv.Sphere(radius=5.0, theta_resolution=40, phi_resolution=40)
    bloated = pv.Sphere(radius=6.5, theta_resolution=40, phi_resolution=40)
    with pytest.raises(TemplateQualityError, match="flow extensions"):
        assert_gt_remesh_scale(bloated, original, context="case")


def test_gt_scale_accepts_near_identity():
    original = pv.Sphere(radius=5.0, theta_resolution=40, phi_resolution=40)
    assert_gt_remesh_scale(original, original, context="case")
    assert GT_MIN_AREA_RATIO < 1.0 < GT_MAX_AREA_RATIO


def test_prepare_gt_surface_does_not_decimate_originals():
    mesh = pv.Sphere(radius=2.0, theta_resolution=160, phi_resolution=160).triangulate()
    assert tessellation_looks_original(mesh)
    n0 = mesh.n_points
    assert n0 > 20000
    kept = prepare_gt_surface(mesh)
    sanitised = sanitize_vessel_for_vmtk(mesh)
    assert kept.GetNumberOfPoints() >= 0.85 * n0
    assert sanitised.GetNumberOfPoints() < kept.GetNumberOfPoints()


def test_isotropic_remesh_uses_constant_edge_length():
    tube = pv.Cylinder(
        center=(0, 0, 10),
        direction=(0, 0, 1),
        radius=2.0,
        height=20.0,
        capping=False,
    ).triangulate()
    remeshed = remesh_surface_isotropically(tube, target_edge_length=0.5)
    topo = inspect_surface_topology(remeshed)
    assert topo["n_triangles"] > 100
    _poly, pts, faces = _triangle_points_faces(remeshed)
    med = float(np.median(_triangle_edge_lengths(pts, faces)))
    assert 0.30 < med < 0.75


def test_pipe_section_then_edge_remesh_keeps_planar_rim():
    tube = _capped_tube((0, 0, 0), (0, 0, 20), radius=2.0)
    origin = np.array([0.0, 0.0, 0.0])
    outward = np.array([0.0, 0.0, -1.0])
    opened, ok = clip_one_opening_pipe_section(tube, origin, outward, 2.0, origin)
    assert ok
    remeshed = remesh_surface_isotropically(opened, target_edge_length=0.4)
    loops = _loop_points(remeshed)
    assert len(loops) >= 1
    rim = min(loops, key=lambda p: np.linalg.norm(p.mean(axis=0) - origin))
    axial = (rim - origin) @ outward
    assert float(np.std(axial)) < 0.20


def test_cli_defaults_to_originals_and_cleandata():
    args = parse_args([])
    assert args.target_edge_length == DEFAULT_GT_EDGE_LENGTH_MM
    vessel_dir = os.path.normpath(args.vessel_dir).replace("\\", "/")
    assert vessel_dir.endswith("vessels/original")
    out_dir = os.path.normpath(args.output_dir).replace("\\", "/")
    assert out_dir.endswith("cleandata/uniformly_remeshed") or out_dir.endswith(
        "cleandata\\uniformly_remeshed"
    )


def _open_tube(p0, p1, radius=2.0, n_sides=32, n_along=40):
    """Open cylinder with enough axial samples to punch a clean wall hole."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    line = pv.Line(tuple(p0), tuple(p1), resolution=int(n_along))
    tube = to_vtk_poly(line.tube(radius=float(radius), n_sides=int(n_sides), capping=True))
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    axis = axis / max(length, 1e-12)
    inset = min(0.3, 0.05 * length)
    for origin, normal in ((p0 + inset * axis, axis), (p1 - inset * axis, -axis)):
        plane = vtk.vtkPlane()
        plane.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
        plane.SetNormal(float(normal[0]), float(normal[1]), float(normal[2]))
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(tube)
        clipper.SetClipFunction(plane)
        clipper.InsideOutOff()
        clipper.Update()
        tube = to_vtk_poly(clipper.GetOutput())
    return tube


def _profiles_from_ends(p0, p1, radius):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    axis = p1 - p0
    axis = axis / np.linalg.norm(axis)
    return [
        {"index": 0, "barycenter": p0, "normal": -axis, "radius": float(radius)},
        {"index": 1, "barycenter": p1, "normal": axis, "radius": float(radius)},
    ]


def _punch_wall_hole(surface, center, hole_radius):
    sphere = vtk.vtkSphere()
    sphere.SetCenter(float(center[0]), float(center[1]), float(center[2]))
    sphere.SetRadius(float(hole_radius))
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(to_vtk_poly(surface))
    clipper.SetClipFunction(sphere)
    clipper.InsideOutOff()
    clipper.Update()
    punched = to_vtk_poly(clipper.GetOutput())
    assert punched.GetNumberOfCells() > 20
    return punched


def test_clip_height_covers_fixed_flow_extension():
    """Thin outlets used 7R which is shorter than a 5 mm flow extension."""
    assert _opening_clip_height(0.45) < 5.0
    assert _opening_clip_height(0.45, extension_length=5.0) >= 5.0
    assert _opening_clip_height(2.0, extension_length=5.0) >= _opening_clip_height(2.0)


def test_pipe_section_removes_long_extension_stub():
    """A 5 mm stub on a thin tube must not survive as a second component."""
    radius = 0.5
    tube = _open_tube((0, 0, 0), (0, 0, 25), radius=radius, n_sides=24, n_along=50)
    origin = np.array([0.0, 0.0, 20.0])
    outward = np.array([0.0, 0.0, 1.0])
    clipped, ok = clip_one_opening_pipe_section(
        tube, origin, outward, radius, origin, extension_length=5.0
    )
    assert ok
    assert count_connected_regions(clipped) == 1
    bounds = clipped.GetBounds()
    assert bounds[5] < 21.0
    openings = inspect_openings(clipped)
    assert len(openings) == 2


def test_drop_small_fragments_drops_extension_sized_stub():
    """Leftover extensions can be ~8% of the mesh; 5% of total was too timid."""
    main = _open_tube((0, 0, 0), (0, 0, 20), radius=2.0, n_sides=24, n_along=40)
    stub = _open_tube((10, 0, 0), (12, 0, 0), radius=0.4, n_sides=8, n_along=6)
    combined = pv.wrap(main).merge(pv.wrap(stub))
    n_before = count_connected_regions(combined)
    assert n_before == 2
    cleaned = _drop_small_fragments(combined)
    assert count_connected_regions(cleaned) == 1
    assert cleaned.GetNumberOfPoints() > 0.7 * main.GetNumberOfPoints()


def test_remove_spurious_openings_fills_wall_hole():
    """C0005-style leftover: true ostia plus an extra hole in the wall."""
    p0, p1 = (0.0, 0.0, 0.0), (0.0, 0.0, 20.0)
    radius = 2.0
    tube = _open_tube(p0, p1, radius=radius)
    assert len(inspect_openings(tube)) == 2
    punched = _punch_wall_hole(tube, (radius, 0.0, 10.0), 0.8)
    n_punched = len(inspect_openings(punched))
    assert n_punched == 3
    profiles = _profiles_from_ends(p0, p1, radius)
    with pytest.raises(TemplateQualityError, match="openings"):
        assert_template_quality(punched, n_expected_openings=2, context="hole")
    filled, n_filled = remove_spurious_openings(punched, profiles)
    assert n_filled == 1
    openings = inspect_openings(filled)
    assert len(openings) == 2
    assert_template_quality(filled, n_expected_openings=2, context="hole")


def test_remove_spurious_openings_fills_leftover_rim_near_ostium():
    """Extra loop a few mm from a real ostium must not steal that ostium."""
    p0, p1 = (0.0, 0.0, 0.0), (0.0, 0.0, 20.0)
    radius = 2.0
    tube = _open_tube(p0, p1, radius=radius)
    punched = _punch_wall_hole(tube, (radius, 0.0, 5.0), 0.6)
    assert len(inspect_openings(punched)) == 3
    profiles = _profiles_from_ends(p0, p1, radius)
    filled, n_filled = remove_spurious_openings(punched, profiles)
    assert n_filled == 1
    openings = inspect_openings(filled)
    assert len(openings) == 2
    ends = [np.asarray(op["center"]) for op in openings]
    assert min(np.linalg.norm(c - np.asarray(p0)) for c in ends) < 1.0
    assert min(np.linalg.norm(c - np.asarray(p1)) for c in ends) < 1.0


def test_remove_spurious_openings_keeps_real_third_branch():
    """A genuine extra ostium listed in the profiles must stay open."""
    p0, p1 = (0.0, 0.0, 0.0), (0.0, 0.0, 20.0)
    radius = 2.0
    tube = _open_tube(p0, p1, radius=radius)
    punched = _punch_wall_hole(tube, (radius, 0.0, 10.0), 0.8)
    openings = inspect_openings(punched)
    assert len(openings) == 3
    profiles = _profiles_from_ends(p0, p1, radius)
    extra = min(
        openings,
        key=lambda op: abs(float(op["center"][2]) - 10.0),
    )
    profiles = profiles + [
        {
            "index": 2,
            "barycenter": np.asarray(extra["center"], dtype=np.float64),
            "normal": np.array([1.0, 0.0, 0.0]),
            "radius": float(extra["radius"]),
        }
    ]
    filled, n_filled = remove_spurious_openings(punched, profiles)
    assert n_filled == 0
    assert len(inspect_openings(filled)) == 3
