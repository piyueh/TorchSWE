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
from torchswe.utils.data import States as _States
from torchswe.utils.data import Topography as _Topography
from torchswe.utils.data import PointSource as _PointSource


_logger = _logging.getLogger("torchswe.utils.init")


def topography_gradient(states: _States, topo: _Topography, gravity: float) -> _States:
    """Assigns topographic forces to states.src.hu and states.src.hv.

    Arguments
    ---------
    states : torchswe.utils.data.States
        Data model instance holding conservative quantities at cell centers with ghost cells.
    topo : torchswe.utils.data.Topography
        Topography data model instance.
    gravity : float
        Gravity in m/s^2.

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Changes are done in-place. Returning it just for coding style.

    Notes
    -----
    This function does not add data to existing arrays. Instead, it re-initialize w with zeros, and
    re-assign the hu and hv with newly calculated topographic gradients.
    """

    internal = slice(states.ngh, -states.ngh)
    gravity_depth = - gravity * (states.q.w[internal, internal] - topo.centers)

    states.src.w[...] = 0.  # update in-place instead of assigning a new object to w
    states.src.hu = topo.xgrad * gravity_depth  # assign a new object rather than update in-place
    states.src.hv = topo.ygrad * gravity_depth  # assign a new object rather than update in-place
    # we could also do states.src.hu[...] = ..., but I'm not sure about the performance

    return states


def point_mass_source(states: _States, ptsource: _PointSource, t: float) -> _States:
    """Adds a point source term to the continuity equation in states.src.w.
    """

    if ptsource is None:
        return states

    # silently assume t is already >= ptsource.times[ptsource.irate-1]
    if ptsource.active and t >= ptsource.times[ptsource.irate]:
        ptsource.irate += 1

        try:
            assert t < ptsource.times[ptsource.irate]
        except IndexError as err:
            if "list index out of range" not in str(err):  # unexpected error
                raise
            ptsource.active = False  # otherwise, reach the final rate
            _logger.debug("Point source `active` has switched to False")

    states.src.w[ptsource.j, ptsource.i] += ptsource.rates[ptsource.irate]
    return states
