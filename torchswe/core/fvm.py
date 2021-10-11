#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Finite-volume scheme from Kurganov and Petrova, 2007.
"""
from torchswe import nplike as _nplike
from torchswe.utils.config import Config as _Config
from torchswe.utils.data import States as _States
from torchswe.utils.misc import DummyDict as _DummyDict
from torchswe.core.sources import topography_gradient as _topography_gradient
from torchswe.core.sources import point_mass_source as _point_mass_source
from torchswe.core.reconstruction import get_discontinuous_cnsrv_q as _get_discontinuous_cnsrv_q
from torchswe.core.reconstruction import correct_negative_depth as _correct_negative_depth
from torchswe.core.flux import get_discontinuous_flux as _get_discontinuous_flux
from torchswe.core.limiters import minmod_slope as _minmod_slope
from torchswe.core.numerical_flux import central_scheme as _central_scheme
from torchswe.core.misc import decompose_variables as _decompose_variables
from torchswe.core.misc import get_local_speed as _get_local_speed
from torchswe.core.misc import remove_rounding_errors as _remove_rounding_errors


def topo_only(states: _States, runtime: _DummyDict, config: _Config):
    """Get the right-hand-side of a time-marching step for SWE with only topography.

    Arguments
    ---------
    states : torchswe.utils.data.States
    runtime : torchswe.utils.misc.DummyDict
    config : torchswe.utils.config.Config

    Returns:
    --------
    states : torchswe.utils.data.States
        The same object as the input. Updated in-place. Returning it just for coding style.
    max_dt : float
        A scalar indicating the maximum safe time-step size.
    """

    # calculate source term contributed from topography gradients
    states = _topography_gradient(states, runtime.topo, config.params.gravity)

    # calculate slopes of piecewise linear approximation
    states = _minmod_slope(states, config.params.theta, runtime.tol)

    # interpolate to get discontinuous conservative quantities at cell faces
    states = _get_discontinuous_cnsrv_q(states)

    # fix non-physical negative depth
    states = _correct_negative_depth(states, runtime.topo)

    # get non-conservative variables at cell faces
    states = _decompose_variables(states, runtime.topo, runtime.epsilon)

    # get local speed at cell faces
    states = _get_local_speed(states, config.params.gravity)

    # get discontinuous PDE flux at cell faces
    states = _get_discontinuous_flux(states, runtime.topo, config.params.gravity)

    # get common/continuous numerical flux at cell faces
    states = _central_scheme(states, runtime.tol)

    # aliases
    dx, dy = states.domain.x.delta, states.domain.y.delta

    # get final right hand side
    states.rhs.w = \
        (states.face.x.num_flux.w[:, :-1] - states.face.x.num_flux.w[:, 1:]) / dx + \
        (states.face.y.num_flux.w[:-1, :] - states.face.y.num_flux.w[1:, :]) / dy + \
        states.src.w

    states.rhs.hu = \
        (states.face.x.num_flux.hu[:, :-1] - states.face.x.num_flux.hu[:, 1:]) / dx + \
        (states.face.y.num_flux.hu[:-1, :] - states.face.y.num_flux.hu[1:, :]) / dy + \
        states.src.hu

    states.rhs.hv = \
        (states.face.x.num_flux.hv[:, :-1] - states.face.x.num_flux.hv[:, 1:]) / dx + \
        (states.face.y.num_flux.hv[:-1, :] - states.face.y.num_flux.hv[1:, :]) / dy + \
        states.src.hv

    # remove rounding errors
    states.rhs = _remove_rounding_errors(states.rhs, runtime.tol)

    # obtain the maximum safe dt
    amax = _nplike.max(_nplike.maximum(states.face.x.plus.a, -states.face.x.minus.a))
    bmax = _nplike.max(_nplike.maximum(states.face.y.plus.a, -states.face.y.minus.a))
    max_dt = min(0.25*dx/amax, 0.25*dy/bmax)

    return states, max_dt


def topo_ptsource(states: _States, runtime: _DummyDict, config: _Config):
    """Get the right-hand-side of a time-marching step with both topo and a point source.

    Arguments
    ---------
    states : torchswe.utils.data.States
    runtime : torchswe.utils.misc.DummyDict
    config : torchswe.utils.config.Config

    Returns:
    --------
    states : torchswe.utils.data.States
        The same object as the input. Updated in-place. Returning it just for coding style.
    max_dt : float
        A scalar indicating the maximum safe time-step size.
    """

    # calculate source term contributed from topography gradients
    states = _topography_gradient(states, runtime.topo, config.params.gravity)

    # add point source term
    states = _point_mass_source(states, runtime.ptsource, runtime.cur_t)

    # calculate slopes of piecewise linear approximation
    states = _minmod_slope(states, config.params.theta, runtime.tol)

    # interpolate to get discontinuous conservative quantities at cell faces
    states = _get_discontinuous_cnsrv_q(states)

    # fix non-physical negative depth
    states = _correct_negative_depth(states, runtime.topo)

    # get non-conservative variables at cell faces
    states = _decompose_variables(states, runtime.topo, runtime.epsilon)

    # get local speed at cell faces
    states = _get_local_speed(states, config.params.gravity)

    # get discontinuous PDE flux at cell faces
    states = _get_discontinuous_flux(states, runtime.topo, config.params.gravity)

    # get common/continuous numerical flux at cell faces
    states = _central_scheme(states, runtime.tol)

    # aliases
    dx, dy = states.domain.x.delta, states.domain.y.delta

    # get final right hand side
    states.rhs.w = \
        (states.face.x.num_flux.w[:, :-1] - states.face.x.num_flux.w[:, 1:]) / dx + \
        (states.face.y.num_flux.w[:-1, :] - states.face.y.num_flux.w[1:, :]) / dy + \
        states.src.w

    states.rhs.hu = \
        (states.face.x.num_flux.hu[:, :-1] - states.face.x.num_flux.hu[:, 1:]) / dx + \
        (states.face.y.num_flux.hu[:-1, :] - states.face.y.num_flux.hu[1:, :]) / dy + \
        states.src.hu

    states.rhs.hv = \
        (states.face.x.num_flux.hv[:, :-1] - states.face.x.num_flux.hv[:, 1:]) / dx + \
        (states.face.y.num_flux.hv[:-1, :] - states.face.y.num_flux.hv[1:, :]) / dy + \
        states.src.hv

    # remove rounding errors
    states.rhs = _remove_rounding_errors(states.rhs, runtime.tol)

    # obtain the maximum safe dt
    amax = _nplike.max(_nplike.maximum(states.face.x.plus.a, -states.face.x.minus.a))
    bmax = _nplike.max(_nplike.maximum(states.face.y.plus.a, -states.face.y.minus.a))

    try:
        max_dt = _nplike.nan_to_num(min(0.25*dx/amax, 0.25*dy/bmax), False, float("NaN"), runtime.dt)
    except TypeError as err:  # TODO: mitigater for issue cupy/cupy#5866; remove this after fix
        if "Wrong number of arguments" not in str(err):
            raise
        max_dt = min(0.25*dx/amax, 0.25*dy/bmax)
        max_dt = runtime.dt if float(max_dt) == float("inf") else max_dt

    return states, max_dt
