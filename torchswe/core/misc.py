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
from ..utils.data import States, Topography


class CFLWarning(Warning):
    """A category of Warning for custome controll of warning action."""
    pass  # pylint: disable=unnecessary-pass


def decompose_variables(states: States, topo: Topography, epsilon: float) -> States:
    """Decompose conservative variables an the both sides of cell faces to dpeth and velocity.

    Arguments
    ---------
    states : torchswe.utils.data.States
    topo : torchswe.utils.data.Topography
    epsilon : float

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Changed inplace. Returning it just for coding style.
    """

    # squared root of 2
    sqrt2 = 1.4142135623730951

    def get_uv(h, hu, hv):
        # pylint: disable=invalid-name
        h4 = numpy.power(h, 4)
        coeff = h * sqrt2 / numpy.sqrt(h4+numpy.maximum(h4, epsilon))
        return coeff * hu, coeff * hv

    # all dpeths
    states.face.x.minus.h = states.face.x.minus.w - topo.xface
    states.face.x.plus.h = states.face.x.plus.w - topo.xface
    states.face.y.minus.h = states.face.y.minus.w - topo.yface
    states.face.y.plus.h = states.face.y.plus.w - topo.yface

    # hu & hv on the minus side of x-faces
    states.face.x.minus.u, states.face.x.minus.v = get_uv(
        states.face.x.minus.h, states.face.x.minus.hu, states.face.x.minus.hv)

    # hu & hv on the plus side of x-faces
    states.face.x.plus.u, states.face.x.plus.v = get_uv(
        states.face.x.plus.h, states.face.x.plus.hu, states.face.x.plus.hv)

    # hu & hv on the minus side of x-faces
    states.face.y.minus.u, states.face.y.minus.v = get_uv(
        states.face.y.minus.h, states.face.y.minus.hu, states.face.y.minus.hv)

    # hu & hv on the plus side of x-faces
    states.face.y.plus.u, states.face.y.plus.v = get_uv(
        states.face.y.plus.h, states.face.y.plus.hu, states.face.y.plus.hv)

    return states


def get_local_speed(states: States, gravity: float) -> States:
    """Calculate local speeds on the two sides of cell faces.

    Arguments
    ---------
    states : torchswe.utils.data.States
    gravity : float
        Gravity in m / s^2.

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Changed inplace. Returning it just for coding style.
    """

    # faces normal to x-direction
    sqrt_gh_plus = numpy.sqrt(gravity*states.face.x.plus.h)
    sqrt_gh_minus = numpy.sqrt(gravity*states.face.x.minus.h)

    states.face.x.plus.a = numpy.maximum(numpy.maximum(
        states.face.x.plus.u+sqrt_gh_plus, states.face.x.minus.u+sqrt_gh_minus), 0.)

    states.face.x.minus.a = numpy.minimum(numpy.minimum(
        states.face.x.plus.u-sqrt_gh_plus, states.face.x.minus.u-sqrt_gh_minus), 0.)

    # faces normal to y-direction
    sqrt_gh_plus = numpy.sqrt(gravity*states.face.y.plus.h)
    sqrt_gh_minus = numpy.sqrt(gravity*states.face.y.minus.h)

    states.face.y.plus.a = numpy.maximum(numpy.maximum(
        states.face.y.plus.v+sqrt_gh_plus, states.face.y.minus.v+sqrt_gh_minus), 0.)

    states.face.y.minus.a = numpy.minimum(numpy.minimum(
        states.face.y.plus.v-sqrt_gh_plus, states.face.y.minus.v-sqrt_gh_minus), 0.)

    return states


def write_states(states: States, fname: str):
    """Write flatten states to a .npz file for debug."""

    keys = ["w", "hu", "hv"]
    keys2 = ["h", "u", "v", "a"]

    data = {}
    data.update({"q_{}".format(k): states.q[k] for k in keys})
    data.update({"src_{}".format(k): states.src[k] for k in keys})
    data.update({"slp_x_{}".format(k): states.slp.x[k] for k in keys})
    data.update({"slp_y_{}".format(k): states.slp.y[k] for k in keys})
    data.update({"face_x_minus_{}".format(k): states.face.x.minus[k] for k in keys})
    data.update({"face_x_plus_{}".format(k): states.face.x.plus[k] for k in keys})
    data.update({"face_y_minus_{}".format(k): states.face.y.minus[k] for k in keys})
    data.update({"face_y_plus_{}".format(k): states.face.y.plus[k] for k in keys})
    data.update({"face_x_minus_{}".format(k): states.face.x.minus[k] for k in keys2})
    data.update({"face_x_plus_{}".format(k): states.face.x.plus[k] for k in keys2})
    data.update({"face_y_minus_{}".format(k): states.face.y.minus[k] for k in keys2})
    data.update({"face_y_plus_{}".format(k): states.face.y.plus[k] for k in keys2})
    data.update({"face_x_minus_flux_{}".format(k): states.face.x.minus.flux[k] for k in keys})
    data.update({"face_x_plus_flux_{}".format(k): states.face.x.plus.flux[k] for k in keys})
    data.update({"face_y_minus_flux_{}".format(k): states.face.y.minus.flux[k] for k in keys})
    data.update({"face_y_plus_flux_{}".format(k): states.face.y.plus.flux[k] for k in keys})
    data.update({"face_x_num_flux_{}".format(k): states.face.x.num_flux[k] for k in keys})
    data.update({"face_y_num_flux_{}".format(k): states.face.y.num_flux[k] for k in keys})
    data.update({"rhs_{}".format(k): states.rhs[k] for k in keys})

    numpy.savez(fname, **data)
