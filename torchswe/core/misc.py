#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Miscellaneous functions.
"""
import numpy


class CFLWarning(Warning):
    """A category of Warning for custome controll of warning action."""
    pass  # pylint: disable=unnecessary-pass


def decompose_variables(w, hu, hv, topo_elev, epsilon):
    """Decompose conservative variables to dpeth and velocity.

    Notes
    -----
    Depending on the use case, their shapes can be either (Ny, Nx), (Ny+1, Nx), or (Ny, Nx+1). And
    all of them should have the same shape.

    Arguments
    ---------
    w, hu, hv : numpy.ndarray
        Conservative quantities.
    topo_elev : numpy.ndarray
        Topography elevation.
    epsilon : float
        A very small number to avoid division by zero.

    Returns
    -------
    depth, u_vel, v_vel : numpy.ndarray
        Dpth, u-velocity, and v-velocity.
    """

    depth = w - topo_elev

    h_4 = numpy.pow(depth, 4)
    denominator = numpy.sqrt(h_4+numpy.maximum(h_4, epsilon))

    u_vel = depth * hu * 1.4142135623730951 / denominator
    v_vel = depth * hv * 1.4142135623730951 / denominator

    return depth, u_vel, v_vel


def local_speed(vel_minus, vel_plus, depth_minus, depth_plus, gravity):
    """Calculate local speed on the two sides of cell faces.

    Notes
    -----
    If the target faces are normal to x-axis, then the shape of these arrays should be (Ny, Nx+1).
    If they are normal to y-axis, then the shape should be (Ny+1, Nx).

    Arguments
    ---------
    vel_minus, vel_plus : numpy.ndarray
        The velocity component normal to the target cell faces and on the two sides of the faces.
    depth_minus, depth_plus : numpy.ndarray
        The depth on the two sides of cell faces.

    Returns
    -------
    a_minus, a_plus : numpy.ndarray
        Local speeds on the two sides of cell faces.
    """

    sqrt_gh_minus = numpy.sqrt(gravity*depth_minus)
    sqrt_gh_plus = numpy.sqrt(gravity*depth_plus)
    a_minus = numpy.minimum(numpy.minimum(vel_plus-sqrt_gh_plus, vel_minus-sqrt_gh_minus), 0.)
    a_plus = numpy.maximum(numpy.maximum(vel_plus+sqrt_gh_plus, vel_minus+sqrt_gh_minus), 0.)

    return a_minus, a_plus
