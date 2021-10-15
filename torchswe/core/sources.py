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
from torchswe.utils.misc import DummyDict as _DummyDict
from torchswe.utils.config import Config as _Config
from torchswe.utils.data import States as _States


_logger = _logging.getLogger("torchswe.core.sources")


def topography_gradient(states: _States, runtime: _DummyDict, config: _Config) -> _States:
    """Adds topographic forces to `states.rhs.hu` and `states.rhs.hv` in-place.

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

    internal = slice(states.ngh, -states.ngh)
    grav_depth = - config.params.gravity * (states.q.w[internal, internal] - runtime.topo.centers)

    states.rhs.hu += runtime.topo.xgrad * grav_depth  # add to rhs in-place
    states.rhs.hv += runtime.topo.ygrad * grav_depth  # add to rhs in-place

    return states


def point_mass_source(states: _States, runtime: _DummyDict, *args, **kwargs) -> _States:
    """Adds point source values to `states.rhs.w` in-place.

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
        To absorb unused provided arguments to make all source term calculations using the same
        signature.

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

    states.rhs.w[ptsource.j, ptsource.i] += ptsource.rates[ptsource.irate]

    return states
