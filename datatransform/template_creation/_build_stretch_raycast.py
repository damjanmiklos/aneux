"""Compile stretch_raycast against the active env's VTK (hemomesh on this machine)."""
from __future__ import annotations

import os
import sys

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution


HERE = os.path.dirname(os.path.abspath(__file__))


def vtk_prefix():
    conda = sys.prefix
    inc = os.path.join(conda, "Library", "include", "vtk-9.2")
    lib = os.path.join(conda, "Library", "lib")
    if os.path.isdir(inc) and os.path.isdir(lib):
        return inc, lib
    inc = os.path.join(conda, "include", "vtk-9.2")
    lib = os.path.join(conda, "lib")
    return inc, lib


def extension():
    vtk_inc, vtk_lib = vtk_prefix()
    if not os.path.isfile(os.path.join(vtk_inc, "vtkCellLocator.h")):
        raise FileNotFoundError(f"vtkCellLocator.h not found under {vtk_inc}")
    compile_args = []
    if os.name == "nt":
        compile_args = ["/O2", "/std:c++17", "/EHsc", "/DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"]
    else:
        compile_args = ["-O3", "-std=c++17", "-fPIC", "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"]
    libs = [
        "vtkCommonDataModel-9.2",
        "vtkCommonCore-9.2",
        "vtkCommonMath-9.2",
        "vtkCommonTransforms-9.2",
        "vtkCommonMisc-9.2",
        "vtkCommonSystem-9.2",
        "vtksys-9.2",
    ]
    return Extension(
        "stretch_raycast",
        sources=[os.path.join(HERE, "stretch_raycast.cpp")],
        include_dirs=[np.get_include(), vtk_inc, os.path.join(sys.prefix, "Include")],
        library_dirs=[vtk_lib, os.path.join(sys.prefix, "libs")],
        libraries=libs,
        language="c++",
        extra_compile_args=compile_args,
    )


class _BuildExt(build_ext):
    def get_ext_filename(self, ext_name):
        return super().get_ext_filename(ext_name)


def build_inplace():
    ext = extension()
    dist = Distribution({"name": "stretch_raycast", "ext_modules": [ext]})
    dist.script_name = "setup.py"
    cmd = _BuildExt(dist)
    cmd.inplace = True
    cmd.build_temp = os.path.join(HERE, "build", "temp")
    cmd.build_lib = HERE
    cmd.ensure_finalized()
    cmd.run()
    print("built", os.path.join(HERE, cmd.get_ext_filename("stretch_raycast")))


if __name__ == "__main__":
    os.chdir(HERE)
    build_inplace()
