/*
 * Tight C++ loop around vtkCellLocator::IntersectWithLine.
 *
 * Semantics match vessel_pipeline.compute_raycast_stretch_distances:
 * inward probe max(1.5, 3.5 R) with the GT-normal side test (the 0.4 mm
 * veto only when no GT normals), outward max_ray, 0.10 < d <= 3.5 R,
 * dot > 0.2.
 *
 * Built against the hemomesh conda VTK. Do not import this from remeshing.py.
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NOMINMAX

#include <Python.h>
#include <numpy/arrayobject.h>

#include "vtkCellLocator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

vtkCellLocator *locator_from_addr(unsigned long long addr) {
  return reinterpret_cast<vtkCellLocator *>(static_cast<uintptr_t>(addr));
}

PyObject *as_c_double2d(PyObject *obj, int ncols, const char *name) {
  PyObject *arr = PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (arr == nullptr) {
    return nullptr;
  }
  if (PyArray_NDIM(reinterpret_cast<PyArrayObject *>(arr)) != 2 ||
      PyArray_DIM(reinterpret_cast<PyArrayObject *>(arr), 1) != ncols) {
    Py_DECREF(arr);
    PyErr_Format(PyExc_ValueError, "%s must be (N, %d) float64 C-contiguous", name, ncols);
    return nullptr;
  }
  return arr;
}

PyObject *as_c_double1d(PyObject *obj, const char *name) {
  PyObject *arr = PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (arr == nullptr) {
    return nullptr;
  }
  if (PyArray_NDIM(reinterpret_cast<PyArrayObject *>(arr)) != 1) {
    Py_DECREF(arr);
    PyErr_Format(PyExc_ValueError, "%s must be (N,) float64 C-contiguous", name);
    return nullptr;
  }
  return arr;
}

} // namespace

static PyObject *compute(PyObject * /*self*/, PyObject *args) {
  unsigned long long loc_addr = 0;
  PyObject *pts_obj = nullptr;
  PyObject *out_obj = nullptr;
  PyObject *r_obj = nullptr;
  PyObject *nrm_obj = nullptr;
  double max_ray = 25.0;
  double tol = 1e-4;
  if (!PyArg_ParseTuple(args, "KOOOO|dd", &loc_addr, &pts_obj, &out_obj, &r_obj, &nrm_obj, &max_ray,
                        &tol)) {
    return nullptr;
  }
  if (loc_addr == 0) {
    PyErr_SetString(PyExc_ValueError, "locator address is null");
    return nullptr;
  }

  PyObject *pts_arr = as_c_double2d(pts_obj, 3, "points");
  if (pts_arr == nullptr) {
    return nullptr;
  }
  PyObject *out_arr = as_c_double2d(out_obj, 3, "outward");
  if (out_arr == nullptr) {
    Py_DECREF(pts_arr);
    return nullptr;
  }

  npy_intp n_pts = PyArray_DIM(reinterpret_cast<PyArrayObject *>(pts_arr), 0);
  if (PyArray_DIM(reinterpret_cast<PyArrayObject *>(out_arr), 0) != n_pts) {
    Py_DECREF(pts_arr);
    Py_DECREF(out_arr);
    PyErr_SetString(PyExc_ValueError, "points and outward must have the same length");
    return nullptr;
  }

  PyObject *r_arr = nullptr;
  if (r_obj != Py_None) {
    r_arr = as_c_double1d(r_obj, "r_template");
    if (r_arr == nullptr) {
      Py_DECREF(pts_arr);
      Py_DECREF(out_arr);
      return nullptr;
    }
    if (PyArray_DIM(reinterpret_cast<PyArrayObject *>(r_arr), 0) != n_pts) {
      Py_DECREF(pts_arr);
      Py_DECREF(out_arr);
      Py_DECREF(r_arr);
      PyErr_SetString(PyExc_ValueError, "r_template must have one value per point");
      return nullptr;
    }
  }

  PyObject *nrm_arr = nullptr;
  npy_intp n_gt = 0;
  if (nrm_obj != Py_None) {
    nrm_arr = as_c_double2d(nrm_obj, 3, "gt_cell_normals");
    if (nrm_arr == nullptr) {
      Py_DECREF(pts_arr);
      Py_DECREF(out_arr);
      Py_XDECREF(r_arr);
      return nullptr;
    }
    n_gt = PyArray_DIM(reinterpret_cast<PyArrayObject *>(nrm_arr), 0);
  }

  npy_intp dist_dims[1] = {n_pts};
  PyObject *dist_arr = PyArray_ZEROS(1, dist_dims, NPY_DOUBLE, 0);
  if (dist_arr == nullptr) {
    Py_DECREF(pts_arr);
    Py_DECREF(out_arr);
    Py_XDECREF(r_arr);
    Py_XDECREF(nrm_arr);
    return nullptr;
  }

  const double *pts = reinterpret_cast<const double *>(
      PyArray_DATA(reinterpret_cast<PyArrayObject *>(pts_arr)));
  const double *outward = reinterpret_cast<const double *>(
      PyArray_DATA(reinterpret_cast<PyArrayObject *>(out_arr)));
  const double *radii =
      r_arr == nullptr
          ? nullptr
          : reinterpret_cast<const double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(r_arr)));
  const double *gtn =
      nrm_arr == nullptr
          ? nullptr
          : reinterpret_cast<const double *>(
                PyArray_DATA(reinterpret_cast<PyArrayObject *>(nrm_arr)));
  double *dist =
      reinterpret_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(dist_arr)));

  vtkCellLocator *loc = locator_from_addr(loc_addr);

  // Release the GIL: the locator is not shared with Python during this call.
  Py_BEGIN_ALLOW_THREADS

  double t = 0.0;
  double x[3] = {0.0, 0.0, 0.0};
  double pcoords[3] = {0.0, 0.0, 0.0};
  int subId = 0;
  vtkIdType cellId = 0;
  double p_in[3];
  double p_end[3];

  for (npy_intp i = 0; i < n_pts; ++i) {
    const double *p = pts + 3 * i;
    const double *n = outward + 3 * i;
    const double r_local = radii == nullptr ? 1.0 : radii[i];

    // Reaches as far inward as an outward hit is allowed to be accepted.
    const double probe = std::max(1.5, 3.5 * r_local);
    p_in[0] = p[0] - n[0] * probe;
    p_in[1] = p[1] - n[1] * probe;
    p_in[2] = p[2] - n[2] * probe;
    const int hit_inward = loc->IntersectWithLine(p, p_in, tol, t, x, pcoords, subId, cellId);
    if (hit_inward) {
      const double dx = x[0] - p[0];
      const double dy = x[1] - p[1];
      const double dz = x[2] - p[2];
      const double d_inward = std::sqrt(dx * dx + dy * dy + dz * dz);
      bool decided = false;
      if (gtn != nullptr && cellId >= 0 && cellId < n_gt) {
        // A wall facing back at us is this point's own near wall, so the tube
        // is outside the GT here and there is no outward stretch to find. A
        // wall facing away is the far side of the lumen, so the point is
        // inside however close that far side happens to be.
        const double *g = gtn + 3 * cellId;
        decided = true;
        if (n[0] * g[0] + n[1] * g[1] + n[2] * g[2] > 0.2) {
          dist[i] = 0.0;
          continue;
        }
      }
      if (!decided && d_inward < 0.4) {
        dist[i] = 0.0;
        continue;
      }
    }

    p_end[0] = p[0] + n[0] * max_ray;
    p_end[1] = p[1] + n[1] * max_ray;
    p_end[2] = p[2] + n[2] * max_ray;
    const int hit = loc->IntersectWithLine(p, p_end, tol, t, x, pcoords, subId, cellId);
    if (!hit) {
      continue;
    }
    const double dx = x[0] - p[0];
    const double dy = x[1] - p[1];
    const double dz = x[2] - p[2];
    const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (!(d > 0.10 && d <= 3.5 * r_local)) {
      continue;
    }
    if (gtn != nullptr) {
      if (cellId < 0 || cellId >= n_gt) {
        continue;
      }
      const double *g = gtn + 3 * cellId;
      const double dot = n[0] * g[0] + n[1] * g[1] + n[2] * g[2];
      if (dot > 0.2) {
        dist[i] = d;
      }
    } else {
      dist[i] = d;
    }
  }

  Py_END_ALLOW_THREADS

  Py_DECREF(pts_arr);
  Py_DECREF(out_arr);
  Py_XDECREF(r_arr);
  Py_XDECREF(nrm_arr);
  return dist_arr;
}

static PyMethodDef methods[] = {
    {"compute", compute, METH_VARARGS,
     "Raycast stretch distances with vtkCellLocator::IntersectWithLine."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "stretch_raycast",
    "C loop around vtkCellLocator for variable-remesh stretch.",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit_stretch_raycast(void) {
  import_array();
  return PyModule_Create(&moduledef);
}
