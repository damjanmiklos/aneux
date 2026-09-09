"""Pipe-section ostium cuts: planar rims, no neighbour slicing."""
import numpy as np
import pyvista as pv
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from vessel_pipeline import (
    clip_flow_extensions_and_uncap,
    clip_one_opening_pipe_section,
    extra_opening_spheres,
    extract_boundary_loops,
    fill_pinholes,
    opening_clip_frames,
    to_vtk_poly,
)


def _polyline_with_misr(points, radius):
    pts = np.asarray(points, dtype=np.float64)
    poly = vtk.vtkPolyData()
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(np.ascontiguousarray(pts), deep=True))
    poly.SetPoints(vtk_pts)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(len(pts))
    for i in range(len(pts)):
        lines.InsertCellPoint(i)
    poly.SetLines(lines)
    misr = vtk.vtkDoubleArray()
    misr.SetName("MaximumInscribedSphereRadius")
    misr.SetNumberOfTuples(len(pts))
    for i in range(len(pts)):
        misr.SetTuple1(i, float(radius))
    poly.GetPointData().AddArray(misr)
    return poly


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


def test_extra_spheres_are_outboard_only():
    cl = _polyline_with_misr([(0, 0, 0), (0, 0, 10)], radius=2.0)
    profiles = [
        {"index": 0, "barycenter": np.array([0.0, 0.0, 0.0]), "normal": np.array([0.0, 0.0, -1.0]), "radius": 2.0},
        {"index": 1, "barycenter": np.array([0.0, 0.0, 10.0]), "normal": np.array([0.0, 0.0, 1.0]), "radius": 2.0},
    ]
    pts, radii = extra_opening_spheres(cl, profiles)
    assert len(pts) >= 4
    assert np.allclose(radii, 2.0)
    z = pts[:, 2]
    assert np.any(z < -0.1)
    assert np.any(z > 10.1)
    assert not np.any((z > 0.5) & (z < 9.5))


def test_pipe_section_cut_is_planar_not_a_cap():
    tube = _capped_tube((0, 0, 0), (0, 0, 20), radius=2.0)
    origin = np.array([0.0, 0.0, 0.0])
    outward = np.array([0.0, 0.0, -1.0])
    opened, ok = clip_one_opening_pipe_section(tube, origin, outward, 2.0, origin)
    assert ok
    loops = _loop_points(opened)
    assert len(loops) >= 1
    rim = min(loops, key=lambda p: np.linalg.norm(p.mean(axis=0) - origin))
    axial = (rim - origin) @ outward
    assert float(np.std(axial)) < 0.15
    assert float(np.mean(np.abs(axial))) < 0.25


def test_pipe_cut_does_not_slice_neighbour_branch():
    main = pv.wrap(_capped_tube((0, 0, 0), (0, 0, 20), radius=2.0))
    neighbour = pv.wrap(_capped_tube((8, 0, 0), (8, 0, 20), radius=1.5))
    merged = to_vtk_poly(main.merge(neighbour))
    n_before = merged.GetNumberOfPoints()
    opened, ok = clip_one_opening_pipe_section(
        merged,
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        2.0,
        np.array([0.0, 0.0, 10.0]),
    )
    assert ok
    assert opened.GetNumberOfPoints() > 0.6 * n_before
    pts = np.array([opened.GetPoint(i) for i in range(opened.GetNumberOfPoints())])
    assert np.any(pts[:, 0] > 6.0)


def test_uncap_uses_centerline_frames():
    cl = _polyline_with_misr([(0, 0, 0), (0, 0, 20)], radius=2.0)
    profiles = [
        {"index": 0, "barycenter": np.array([0.0, 0.0, 0.0]), "normal": np.array([0.0, 0.0, -1.0]), "radius": 2.0},
        {"index": 1, "barycenter": np.array([0.0, 0.0, 20.0]), "normal": np.array([0.0, 0.0, 1.0]), "radius": 2.0},
    ]
    frames = opening_clip_frames(cl, profiles)
    assert len(frames) == 2
    assert abs(frames[0][1][2] + 1.0) < 1e-6
    assert abs(frames[1][1][2] - 1.0) < 1e-6
    closed = _capped_tube((0, 0, 0), (0, 0, 20), radius=2.0)
    opened, n = clip_flow_extensions_and_uncap(closed, profiles, extension_length=5.0, centerline=cl)
    assert n == 2
    assert len(_loop_points(opened)) >= 2


if __name__ == "__main__":
    test_extra_spheres_are_outboard_only()
    test_pipe_section_cut_is_planar_not_a_cap()
    test_pipe_cut_does_not_slice_neighbour_branch()
    test_uncap_uses_centerline_frames()
    print("ok")
