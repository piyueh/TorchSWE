#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Source terms.
"""
from torchswe.utils.data import States as _States
from torchswe.utils.data import Topography as _Topography


def topography_gradient(states: _States, topo: _Topography, gravity: float) -> _States:
    """Momentim sources due to topographic changes.

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
    """

    internal = slice(states.ngh, -states.ngh)
    gravity_depth = - gravity * (states.q.w[internal, internal] - topo.centers)

    states.src.w[...] = 0.  # update in-place instead of assigning a new object to w
    states.src.hu = topo.xgrad * gravity_depth  # assign a new object rather than update in-place
    states.src.hv = topo.ygrad * gravity_depth  # assign a new object rather than update in-place
    # we could also do states.src.hu[...] = ..., but I'm not sure about the performance

    return states
