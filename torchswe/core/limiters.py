#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Slope/flux limiters.
"""
import numpy


def minmod_slope(quantity, delta_x, delta_y, n_ghost, theta, tol=1e-12):
    """Calculate minmod slopes in x- and y-direction of quantity.

    Arguments
    ---------
    quantity : (Ny+2*n_ghost, Nx+2*n_ghost) numpy.ndarray
        Target quantity. For example, w, hu, or hv.
    delta_x, delta_y : float
        Cell size in x- and y-direction. Currently only supports constant cell sizes.
    n_ghost : int
        Number of ghost cell layers outsied each boundary.
    theta: float
        Parameter to controll oscillation and dispassion. 1 <= theta <= 2.
    tol : float
        To control how small can be treat as zero.

    Return:
    -------
    slope_x : (Ny, Nx+2) numpy.ndarray
        Slopes in x-direction, including one layer of ghost cells in x-direction.
    slope_y : (Ny+2, Nx) numpy.ndarray
        Slopes in y-direction, including one layer of ghost cells in y-direction.
    """

    # x-direction
    denominator = quantity[:, 2:]-quantity[:, 1:-1]  # q_{i+1} - q_{i}
    slope_x = numpy.divide(
        quantity[:, 1:-1]-quantity[:, :-2], denominator, out=numpy.zeros_like(denominator),
        where=numpy.logical_not(numpy.isclose(denominator, 0., atol=tol))
    )  # inf and NaN from division will just be zeros
    slope_x = numpy.maximum(numpy.minimum(theta*slope_x, (1.+slope_x)/2., theta), 0.)
    slope_x *= denominator
    slope_x /= delta_x
    slope_x[numpy.isclose(slope_x, 0., atol=tol)] = 0.  # hard-code to zero for slope < tol

    # y-direction
    denominator = quantity[2:, :]-quantity[1:-1, :]  # q_{i+1} - q_{i}
    slope_y = numpy.divide(
        quantity[1:-1, :]-quantity[:-2, :], denominator, out=numpy.zeros_like(denominator),
        where=numpy.logical_not(numpy.isclose(denominator, 0., atol=tol))
    )  # inf and NaN from division will just be zeros
    slope_y = numpy.maximum(numpy.minimum(theta*slope_y, (1.+slope_y)/2., theta), 0.)
    slope_y *= denominator
    slope_y /= delta_y
    slope_y[numpy.isclose(slope_y, 0., atol=tol)] = 0.  # hard-code to zero for slope < tol

    return \
        slope_x[n_ghost:-n_ghost, n_ghost-1:-n_ghost+1], \
        slope_y[n_ghost-1:-n_ghost+1, n_ghost:-n_ghost]
