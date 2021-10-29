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

_kernel = cupy.ElementwiseKernel(
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
    "kernel"
)


def minmod_slope(states, theta):
    """Calculate the slope of using minmod limiter.

    Arguments
    ---------
    states : torchswe.utils.data.State
        The instance of States holding quantities.
    theta : float
        The parameter adjusting the dissipation.

    Returns
    -------
    slpx : cupy.ndarray with shape (3, ny, nx+2)
    slpy : cupy.ndarray with shape (3, ny+2, nx)
    """

    # alias
    ngh = states.ngh
    Q = states.Q
    dx, dy = states.domain.x.delta, states.domain.y.delta

    # kernels
    slpx = _kernel(Q[:, ngh:-ngh, :-ngh], Q[:, ngh:-ngh, 1:-1], Q[:, ngh:-ngh, ngh:], theta, dx)
    slpy = _kernel(Q[:, :-ngh, ngh:-ngh], Q[:, 1:-1, ngh:-ngh], Q[:, ngh:, ngh:-ngh], theta, dy)

    return slpx, slpy
