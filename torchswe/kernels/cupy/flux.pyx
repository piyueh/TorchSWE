#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Functions for calculating discontinuous flux.
"""
import cupy
cimport cython


cdef get_discontinuous_flux_x = cupy.ElementwiseKernel(
    "T hu, T h, T u, T v, float64 grav2",
    "T f0, T f1, T f2",
    """
        f0 = hu;
        f1 = hu * u + grav2 * h * h;
        f2 = hu * v;
    """,
    "get_discontinuous_flux_x"
)


cdef get_discontinuous_flux_y = cupy.ElementwiseKernel(
    "T hv, T h, T u, T v, float64 grav2",
    "T f0, T f1, T f2",
    """
        f0 = hv;
        f1 = u * hv;
        f2 = hv * v + grav2 * h * h;
    """,
    "get_discontinuous_flux_y"
)


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
def get_discontinuous_flux(object states, double gravity):
    """Calculting the discontinuous fluxes on the both sides at cell faces.

    Arguments
    ---------
    states : torchswe.utils.data.States
    gravity : float

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Changed inplace. Returning it just for coding style.
    """

    cdef double grav2 = gravity / 2.

    x = states.face.x
    xm = x.minus
    xp = x.plus

    y = states.face.y
    ym = y.minus
    yp = y.plus

    # face normal to x-direction: [hu, hu^2 + g(h^2)/2, huv]
    get_discontinuous_flux_x(xm.Q[1], xm.U[0], xm.U[1], xm.U[2], grav2, xm.F[0], xm.F[1], xm.F[2])
    get_discontinuous_flux_x(xp.Q[1], xp.U[0], xp.U[1], xp.U[2], grav2, xp.F[0], xp.F[1], xp.F[2])

    # face normal to y-direction: [hv, huv, hv^2+g(h^2)/2]
    get_discontinuous_flux_y(ym.Q[2], ym.U[0], ym.U[1], ym.U[2], grav2, ym.F[0], ym.F[1], ym.F[2])
    get_discontinuous_flux_y(yp.Q[2], yp.U[0], yp.U[1], yp.U[2], grav2, yp.F[0], yp.F[1], yp.F[2])

    return states


cdef central_scheme_kernel = cupy.ElementwiseKernel(
    "T ma, T pa, T mf, T pf, T mq, T pq",
    "T flux",
    """
        T denominator = pa - ma;
        T coeff = pa * ma;
        flux = (pa * mf - ma * pf + coeff * (pq - mq)) / denominator;
        if (flux != flux) flux = 0.0;
    """,
    "central_scheme_kernel"
)


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
def central_scheme(object states):
    """A central scheme to calculate numerical flux at interfaces.

    Arguments
    ---------
    states : torchswe.utils.data.States

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Updated in-place. Returning it just for coding style.
    """

    x = states.face.x
    xm = x.minus
    xp = x.plus

    y = states.face.y
    ym = y.minus
    yp = y.plus

    central_scheme_kernel(xm.a, xp.a, xm.F, xp.F, xm.Q, xp.Q, x.H)
    central_scheme_kernel(ym.a, yp.a, ym.F, yp.F, ym.Q, yp.Q, y.H)

    return states
