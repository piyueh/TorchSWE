#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Functions for calculating discontinuous flux.
"""
from ..utils.data import States, Topography


def get_discontinuous_flux(states: States, topo: Topography, gravity: float) -> States:
    """Calculting the discontinuous fluxes on the both sides at cell faces.

    Arguments
    ---------
    states : torchswe.utils.data.States
    topo : torchswe.utils.data.Topography
    gravity : float

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Changed inplace. Returning it just for coding style.

    Notes
    -----
    When calculating (w-z)^2, it seems using w*w-w*z-z*w+z*z has smaller rounding errors. Not sure
    why. But it worth more investigation. This is apparently slower, though with smaller errors.
    """

    grav2 = gravity / 2.
    topoxface2 = topo.xface * topo.xface
    topoyface2 = topo.yface * topo.yface

    # face normal to x-direction, minus side; the scheme requires reconstruct hu
    # --------------------------------------------------------------------------
    states.face.x.minus.hu = states.face.x.minus.h * states.face.x.minus.u

    # flux for continuity eq at x-direction is hu
    states.face.x.minus.flux.w = states.face.x.minus.hu

    # flus for x-momentum eq at x-direction is hu^2 + (g*(w-z)^2)/2
    states.face.x.minus.flux.hu = states.face.x.minus.hu * states.face.x.minus.u + \
        grav2 * (
            states.face.x.minus.w * states.face.x.minus.w - states.face.x.minus.w * topo.xface -
            topo.xface * states.face.x.minus.w + topoxface2)

    # flux for y-momentum eq at x-direction is huv
    states.face.x.minus.flux.hv = states.face.x.minus.hu * states.face.x.minus.v

    # face normal to x-direction, plus side; the scheme requires reconstruct hu
    # --------------------------------------------------------------------------
    states.face.x.plus.hu = states.face.x.plus.h * states.face.x.plus.u

    # flux for continuity eq at x-direction is hu
    states.face.x.plus.flux.w = states.face.x.plus.hu

    # flus for x-momentum eq at x-direction is hu^2 + (g*(w-z)^2)/2
    states.face.x.plus.flux.hu = states.face.x.plus.hu * states.face.x.plus.u + \
        grav2 * (
            states.face.x.plus.w * states.face.x.plus.w - states.face.x.plus.w * topo.xface -
            topo.xface * states.face.x.plus.w + topoxface2)

    # flux for y-momentum eq at x-direction is huv
    states.face.x.plus.flux.hv = states.face.x.plus.hu * states.face.x.plus.v

    # face normal to y-direction, minus side; the scheme requires reconstruct hv
    # --------------------------------------------------------------------------
    states.face.y.minus.hv = states.face.y.minus.h * states.face.y.minus.v

    # flux for continuity eq at y-direction is hv
    states.face.y.minus.flux.w = states.face.y.minus.hv

    # flux for x-momentum eq at y-direction is huv
    states.face.y.minus.flux.hu = states.face.y.minus.u * states.face.y.minus.hv

    # flus for y-momentum eq at y-direction is hv^2 + (g*(w-z)^2)/2
    states.face.y.minus.flux.hv = states.face.y.minus.hv * states.face.y.minus.v + \
        grav2 * (
            states.face.y.minus.w * states.face.y.minus.w - states.face.y.minus.w * topo.yface -
            topo.yface * states.face.y.minus.w + topoyface2)

    # face normal to y-direction, plus side; the scheme requires reconstruct hv
    # --------------------------------------------------------------------------
    states.face.y.plus.hv = states.face.y.plus.h * states.face.y.plus.v

    # flux for continuity eq at y-direction is hv
    states.face.y.plus.flux.w = states.face.y.plus.hv

    # flux for x-momentum eq at y-direction is huv
    states.face.y.plus.flux.hu = states.face.y.plus.u * states.face.y.plus.hv

    # flus for y-momentum eq at y-direction is hv^2 + (g*(w-z)^2)/2
    states.face.y.plus.flux.hv = states.face.y.plus.hv * states.face.y.plus.v + \
        grav2 * (
            states.face.y.plus.w * states.face.y.plus.w - states.face.y.plus.w * topo.yface -
            topo.yface * states.face.y.plus.w + topoyface2)

    return states
