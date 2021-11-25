#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Install TorchSWE.

Notes
-----
Configuration of the package/project has been migrated to setup.cfg. Only extension modules (i.e.,
those require being compiled first) remain here.
"""
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize, build_ext

exts = [
    Extension(
        name="torchswe.kernels.cython", sources=["torchswe/kernels/cython_kernels.pyx"],
        include_dirs=[numpy.get_include()], language="c",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        name="torchswe.kernels.cupy", sources=["torchswe/kernels/cupy_kernels.pyx"],
        include_dirs=[numpy.get_include()], language="c",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        name="torchswe.bcs._cython_outflow", sources=["torchswe/bcs/cython_outflow.pyx"],
        include_dirs=[numpy.get_include()], language="c",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        name="torchswe.bcs._cython_linear_extrap",
        sources=["torchswe/bcs/cython_linear_extrap.pyx"],
        include_dirs=[numpy.get_include()], language="c",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        name="torchswe.bcs._cython_const_val",
        sources=["torchswe/bcs/cython_const_val.pyx"],
        include_dirs=[numpy.get_include()], language="c",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        name="torchswe.bcs._cupy_outflow", sources=["torchswe/bcs/cupy_outflow.pyx"],
        include_dirs=[numpy.get_include()], language="c",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        name="torchswe.bcs._cupy_linear_extrap", sources=["torchswe/bcs/cupy_linear_extrap.pyx"],
        include_dirs=[numpy.get_include()], language="c",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        name="torchswe.bcs._cupy_const_val", sources=["torchswe/bcs/cupy_const_val.pyx"],
        include_dirs=[numpy.get_include()], language="c",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        name="torchswe.bcs._cupy_inflow", sources=["torchswe/bcs/cupy_inflow.pyx"],
        include_dirs=[numpy.get_include()], language="c",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
]

setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(
        exts,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False
        },
    )
)
