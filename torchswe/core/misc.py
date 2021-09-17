#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Miscellaneous functions.
"""
from torchswe import nplike as _nplike
from torchswe.utils.data import WHUHVModel as _WHUHVModel
from torchswe.utils.data import States as _States
from torchswe.utils.data import Topography as _Topography


class CFLWarning(Warning):
    """A category of Warning for custome controll of warning action."""
    pass  # pylint: disable=unnecessary-pass


def decompose_variables(states: _States, topo: _Topography, epsilon: float) -> _States:
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
        h4 = _nplike.power(h, 4)
        coeff = h * sqrt2 / _nplike.sqrt(h4+_nplike.maximum(h4, _nplike.array(epsilon)))
        return coeff * hu, coeff * hv

    # all dpeths
    states.face.x.minus.h = states.face.x.minus.w - topo.xfcenters
    states.face.x.plus.h = states.face.x.plus.w - topo.xfcenters
    states.face.y.minus.h = states.face.y.minus.w - topo.yfcenters
    states.face.y.plus.h = states.face.y.plus.w - topo.yfcenters

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


def get_local_speed(states: _States, gravity: float) -> _States:
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
    sqrt_gh_plus = _nplike.sqrt(gravity*states.face.x.plus.h)
    sqrt_gh_minus = _nplike.sqrt(gravity*states.face.x.minus.h)

    # for convenience
    zero = _nplike.array(0.)

    states.face.x.plus.a = _nplike.maximum(_nplike.maximum(
        states.face.x.plus.u+sqrt_gh_plus, states.face.x.minus.u+sqrt_gh_minus), zero)

    states.face.x.minus.a = _nplike.minimum(_nplike.minimum(
        states.face.x.plus.u-sqrt_gh_plus, states.face.x.minus.u-sqrt_gh_minus), zero)

    # faces normal to y-direction
    sqrt_gh_plus = _nplike.sqrt(gravity*states.face.y.plus.h)
    sqrt_gh_minus = _nplike.sqrt(gravity*states.face.y.minus.h)

    states.face.y.plus.a = _nplike.maximum(_nplike.maximum(
        states.face.y.plus.v+sqrt_gh_plus, states.face.y.minus.v+sqrt_gh_minus), zero)

    states.face.y.minus.a = _nplike.minimum(_nplike.minimum(
        states.face.y.plus.v-sqrt_gh_plus, states.face.y.minus.v-sqrt_gh_minus), zero)

    return states


def write_states(states: _States, fname: str):
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

    _nplike.savez(fname, **data)


def remove_rounding_errors(whuhv: _WHUHVModel, tol: float) -> _WHUHVModel:
    """Removing rounding errors from states.

    Arguments
    ---------
    whuhv : torchswe.utils.data.WHUHVModel
        Any instance of WHUHVModel data model.
    tol : float
        Rounding error.
    """

    # remove rounding errors
    zero_ji = _nplike.nonzero(_nplike.logical_and(whuhv.w > -tol, whuhv.w < tol))
    whuhv.w[zero_ji] = 0.

    zero_ji = _nplike.nonzero(_nplike.logical_and(whuhv.hu > -tol, whuhv.hu < tol))
    whuhv.hu[zero_ji] = 0.

    zero_ji = _nplike.nonzero(_nplike.logical_and(whuhv.hv > -tol, whuhv.hv < tol))
    whuhv.hv[zero_ji] = 0.

    return whuhv


def states_assertions(states: _States):
    """A naive assertions for NaN."""

    for item in ["q", "src", "rhs"]:
        for k in ["w", "hu", "hv"]:
            msg = "NaN found in {}.{}".format(item, k)
            assert not _nplike.any(_nplike.isnan(states[item][k])), msg

    for axis in ["x", "y"]:
        for k in ["w", "hu", "hv"]:
            msg = "NaN found in slp.{}.{}".format(axis, k)
            assert not _nplike.any(_nplike.isnan(states.slp[axis][k])), msg

            msg = "NaN found in face.{}.num_flux.{}".format(axis, k)
            assert not _nplike.any(_nplike.isnan(states.face[axis].num_flux[k])), msg

    for axis in ["x", "y"]:
        for side in ["minus", "plus"]:
            for k in ["w", "hu", "hv", "h", "u", "v", "a"]:
                msg = "NaN found in face.{}.{}.{}".format(axis, side, k)
                assert not _nplike.any(_nplike.isnan(states.face[axis][side][k])), msg

    for axis in ["x", "y"]:
        for side in ["minus", "plus"]:
            for k in ["w", "hu", "hv"]:
                msg = "NaN found in face.{}.{}.flux.{}".format(axis, side, k)
                assert not _nplike.any(_nplike.isnan(states.face[axis][side].flux[k])), msg
