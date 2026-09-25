"""Trace one Voronoi centerline in a child process.

vmtkCenterlines tetrahedralises the whole capped surface before it traces
anything, and on six of the SNF vessels that step never comes back. It is not
an exception that can be caught, it is a hang, so the only way to put a clock
on it is to run it somewhere that can be killed. The caller falls back to the
un-extended surface, which traces those same six in 4-15 s.

Called as::

    python centerline_worker.py <surface.vtp> <seeds.json> <out.vtp>
"""
import json
import sys


def main(argv):
    surface_path, seeds_path, out_path = argv[1:4]

    import pyvista as pv

    import vessel_pipeline as vp

    seeds = json.loads(open(seeds_path, encoding="utf-8").read())
    surface = vp.to_vtk_poly(vp.read_polydata(surface_path))
    centerline = vp.extract_voronoi_centerlines(
        surface, seeds["source"], seeds["target"]
    )
    vp.save_polydata(centerline, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
