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
from torchswe.kernels import get_discontinuous_flux as _get_discontinuous_flux
from torchswe.kernels import central_scheme as _central_scheme
from torchswe.kernels import get_local_speed as _get_local_speed
from torchswe.kernels import reconstruct as _reconstruct


def prepare_rhs(states: _States, runtime: _DummyDict, config: _Config):
    """Get the right-hand-side of a time-marching step for SWE.

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
        A scalar indicating the maximum time-step size if we consider CFL to be one. Note, it
        does not mean this time-step size is safe. Whether it's safe or not depending on the
        allowed CFL of the implemented scheme.
    """

    # reconstruct conservative and non-conservative quantities at cell interfaces
    states = _reconstruct(states, runtime, config)

    # update boundary faces' values based on boundary conditions
    states = runtime.gh_updater(states)

    # get local speed at cell faces
    states = _get_local_speed(states, config.params.gravity)

    # get discontinuous PDE flux at cell faces
    states = _get_discontinuous_flux(states, config.params.gravity)

    # get common/continuous numerical flux at cell faces
    states = _central_scheme(states)

    # aliases
    dy, dx = states.domain.delta

    # get right-hand-side contributed by spatial derivatives
    states.S = \
        (states.face.x.H[:, :, :-1] - states.face.x.H[:, :, 1:]) / dx + \
        (states.face.y.H[:, :-1, :] - states.face.y.H[:, 1:, :]) / dy

    # add explicit source terms in-place to states.S
    for func in runtime.sources:
        states = func(states, runtime, config)

    # add stiff source terms to states.SS (including reset it to zero first)
    for func in runtime.stiff_sources:
        states = func(states, runtime, config)

    # obtain the maximum safe dt
    amax = _nplike.max(_nplike.maximum(states.face.x.plus.a, -states.face.x.minus.a))
    bmax = _nplike.max(_nplike.maximum(states.face.y.plus.a, -states.face.y.minus.a))

    with _nplike.errstate(divide="ignore"):
        max_dt = min(dx/amax, dy/bmax)  # may be a `inf` (but never `NaN`)

    return states, max_dt
