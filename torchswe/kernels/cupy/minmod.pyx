#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Minmod implementation for Cuda through CuPy.
"""
import cupy
cimport cython

cdef _minmod_slope_kernel = cupy.ElementwiseKernel(
    "T s1, T s2, T s3, T theta, T dx",
    "T slp",
    """
    T denominator = s3 - s2;
    slp = (s2 - s1) / denominator;
    slp = min(slp*theta, (1.0 + slp) / 2.0);
    slp = min(slp, theta);
    slp = max(slp, 0.);
    slp *= denominator;
    slp /= dx;
    """,
    "minmod_slope_kernel",
)


@cython.boundscheck(False)  # deactivate bounds checking
def minmod_slope(object states, double theta):
    """Calculate the slope of using minmod limiter.

    Arguments
    ---------
    states : torchswe.utils.data.State
        The instance of States holding quantities.
    theta : float
        The parameter adjusting the dissipation.

    Returns
    -------
    states : torchswe.utils.data.State
        The same object as the input.
    """

    # alias
    cdef Py_ssize_t ngh = states.ngh
    cdef double dx = states.domain.x.delta
    cdef double dy = states.domain.y.delta
    Q = states.Q  # pylint: disable=invalid-name

    # kernels
    states.slpx = _minmod_slope_kernel(
        Q[:, ngh:-ngh, :-ngh], Q[:, ngh:-ngh, 1:-1], Q[:, ngh:-ngh, ngh:], theta, dx)
    states.slpy = _minmod_slope_kernel(
        Q[:, :-ngh, ngh:-ngh], Q[:, 1:-1, ngh:-ngh], Q[:, ngh:, ngh:-ngh], theta, dy)

    return states
