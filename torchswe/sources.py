#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Source terms.
"""
import logging as _logging
from torchswe import nplike as _nplike
from torchswe.utils.misc import DummyDict as _DummyDict
from torchswe.utils.config import Config as _Config
from torchswe.utils.data import States as _States


_logger = _logging.getLogger("torchswe.sources")


def topography_gradient(states: _States, runtime: _DummyDict, config: _Config) -> _States:
    """Adds topographic forces to `states.S[1]` and `states.S[2]` in-place.

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
    states.S[1:, ...] -= (config.params.gravity * states.H * runtime.topo.grad)
    return states


def point_mass_source(states: _States, runtime: _DummyDict, *args, **kwargs) -> _States:
    """Adds point source values to `states.S[0]` in-place.

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

    states.S[0, ptsource.j, ptsource.i] += ptsource.rates[ptsource.irate]

    return states


def friction(states: _States, runtime: _DummyDict, config: _Config) -> _States:
    """Add the friction forces to the stiff source term `states.SS[1]` and `states.SS[2]`.

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
    slc = slice(states.ngh, -states.ngh)
    loc = _nplike.nonzero(states.H > 0.)

    # views
    h = states.H[loc]
    hu = states.Q[1, slc, slc][loc]  # hu & hv has ghost cells
    hv = states.Q[2, slc, slc][loc]  # hu & hv has ghost cells

    coef = runtime.fc_model(h, hu, hv, config.props.nu, runtime.roughness)

    states.SS[1:, loc[0], loc[1]] += \
        - coef * _nplike.sqrt(_nplike.power(hu, 2)+_nplike.power(hv, 2)) \
        / (8. * _nplike.power(h, 2))

    return states


def zero_stiff_terms(states: _States, *args, **kwargs):
    """Push zeros to states.SS.

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
    states.SS[...] = 0.
    return states
