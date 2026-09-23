"""The supervision contract on a finished variable template, and the density measure.

Meshes are explicit triangles: ``pv.Sphere`` aborts this machine's interpreter
(see test_remeshing_repairs.py).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import variable_remeshing as vr
import vessel_pipeline as vp
from test_variable_tube_cache import _strip


def _supervised(r=0.5, s=1.0, t=0.2):
    mesh = _strip()
    n = mesh.n_points
    return vr.attach_template_supervision_arrays(
        mesh, np.full(n, r), np.full(n, s), target_edge=np.full(n, t)
    )


def test_a_template_inside_the_contract_passes():
    stats = vr.assert_template_supervision_arrays(_supervised())
    assert stats["StretchDistance"] == (1.0, 1.0)


def test_a_stretch_at_the_ray_ceiling_passes():
    vr.assert_template_supervision_arrays(_supervised(r=0.4, s=3.5 * 0.4))


@pytest.mark.parametrize("kwargs, what", [
    (dict(s=-0.2), "negative"),
    (dict(r=0.4, s=3.5 * 0.4 + 0.01), "3.5 R ceiling"),
    (dict(r=0.0), "not positive"),
    (dict(t=0.6), "TargetEdgeLength outside"),
    (dict(t=0.001), "TargetEdgeLength outside"),
])
def test_a_value_the_raycast_cannot_write_is_refused(kwargs, what):
    with pytest.raises(vp.TemplateQualityError, match=what):
        vr.assert_template_supervision_arrays(_supervised(**kwargs))


def test_the_edge_bound_follows_the_base_edge_asked_for():
    vr.assert_template_supervision_arrays(_supervised(t=0.6), base_edge=0.8)


def test_density_counts_edges_within_twice_the_target():
    # _strip's edges run 0.52 (ring), 1.0 (axial) and 1.13 (diagonal) long.
    assert vp.density_within_x2(_strip(), _supervised(t=0.8)) == 1.0
    assert vp.density_within_x2(_strip(), _supervised(t=0.2)) == 0.0
    # No target array on the reference: nothing to judge, so nothing fails.
    assert vp.density_within_x2(_strip(), _strip()) == 1.0
