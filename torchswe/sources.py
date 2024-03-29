#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Source terms.
"""
from __future__ import annotations as _annotations  # allows us not using quotation marks for hints
from typing import TYPE_CHECKING as _TYPE_CHECKING  # indicates if we have type checking right now
if _TYPE_CHECKING:  # if we are having type checking, then we import corresponding classes/types
    from torchswe.utils.misc import DummyDict
    from torchswe.utils.config import Config
    from torchswe.utils.data import States

# pylint: disable=wrong-import-position, ungrouped-imports
import logging as _logging
from torchswe import nplike as _nplike


_logger = _logging.getLogger("torchswe.sources")


def topography_gradient(states: States, runtime: DummyDict, config: Config) -> States:
    """Adds topographic forces to `states.s[1]` and `states.s[2]` in-place.

    Arguments
    ---------
    states : torchswe.utils.data.States
        Data model instance holding conservative quantities at cell centers with ghost cells.
    runtime : torchswe.utils.misc.DummyDict
        A DummyDict that we can access a Topography instance through `runtime.topo`.
    config : torchswe.utils.config.Config
        A `Config` instance. We use the gravity paramater (in m/s^2) through config.params.gravity.

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Changes are done in-place. Returning it just for coding style.
    """
    # auto-broadcasting; add to rhs in-place
    states.s[1:, ...] -= (
        config.params.gravity * states.p[(0,)+states.domain.nonhalo_c] * runtime.topo.grad
    )
    return states


def point_mass_source(states: States, runtime: DummyDict, *args, **kwargs) -> States:
    """Adds point source values to `states.s[0]` in-place.

    Arguments
    ---------
    states : torchswe.utils.data.States
        Data model instance holding conservative quantities at cell centers with ghost cells.
    runtime : torchswe.utils.misc.DummyDict
        A DummyDict. We need to access `runtime.ptsource`, `runtime.cur_t` and
        `runtime.dt_constraint`. `runtime.ptsource` is an instance of
        `torchswe.utils.data.PointSource`, and `runtime.cur_t` is a float indicating the current
        simulation time. `runtime.dt_constraint` is a float of current time-step size constraint
        due to non-stability-related issues.
    *args, **kwargs :
        To absorb unused arguments to make all source term calculations using the same signature.

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Changes are done in-place. Returning it just for coding style.

    Notes
    -----
    runtime.dt will be changed to a value so that runtime.cur_t + runtime.dt won't exceed the
    switch time point for the next point source profile.
    """

    if runtime.ptsource is None:
        return states

    # alias
    ptsource = runtime.ptsource

    if runtime.counter == 0:
        runtime.dt_constraint = min(runtime.dt_constraint, ptsource.init_dt)

    # silently assume t is already >= ptsource.times[ptsource.irate-1]
    if ptsource.active:  # that is, if `allowed_dt` is not None
        if runtime.cur_t >= ptsource.times[ptsource.irate]:
            ptsource.irate += 1
            _logger.info(
                "Point source has switched to the next rate %e at T=%e",
                ptsource.rates[ptsource.irate], ptsource.times[ptsource.irate-1])

        # update allowed_dt
        try:
            runtime.dt_constraint = min(
                ptsource.times[ptsource.irate]-runtime.cur_t, runtime.dt_constraint)
        except IndexError as err:  # when reaching the final stage of the point source profile
            if "index out of range" not in str(err):  # unexpected error
                raise
            ptsource.active = False  # otherwise, reach the final rate
            _logger.debug("Point source `allowed_dt` has switched to None")

    states.s[0, ptsource.j, ptsource.i] += ptsource.rates[ptsource.irate]

    return states


def friction(states: States, runtime: DummyDict, config: Config) -> States:
    """Add the friction forces to the stiff source term `states.ss[1]` and `states.ss[2]`.

    Arguments
    ---------
    states : torchswe.utils.data.States
        Data model instance holding conservative quantities at cell centers with ghost cells.
    runtime : torchswe.utils.misc.DummyDict
        A DummyDict that we can access a Topography instance through `runtime.topo`.
    config : torchswe.utils.config.Config
        A `Config` instance. We use the gravity paramater (in m/s^2) through config.params.gravity.

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Changes are done in-place. Returning it just for coding style.
    """
    loc = _nplike.nonzero(states.p[(0,)+states.domain.nonhalo_c] > 0.)

    # views
    h = states.p[(0,)+states.domain.nonhalo_c][loc]
    hu = states.q[(1,)+states.domain.nonhalo_c][loc]
    hv = states.q[(2,)+states.domain.nonhalo_c][loc]
    roughness = runtime.friction.roughness[loc]

    coef = runtime.friction.model(h, hu, hv, config.props.nu, roughness)

    states.ss[1:, loc[0], loc[1]] += (
        - coef * _nplike.sqrt(_nplike.power(hu, 2)+_nplike.power(hv, 2)) /
        (8. * _nplike.power(h, 2))
    )

    return states


def zero_stiff_terms(states: States, *args, **kwargs):
    """Push zeros to states.ss.

    Arguments
    ---------
    states : torchswe.utils.data.States
        Data model instance holding conservative quantities at cell centers with ghost cells.
    *args, **kwargs :
        To absorb unused to make all source term calculations using the same signature.

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Changes are done in-place. Returning it just for coding style.
    """
    states.ss[...] = 0.
    return states
