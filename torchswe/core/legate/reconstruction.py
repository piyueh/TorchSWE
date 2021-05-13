#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Linear reconstruction. (Legate version)
"""
import logging
from torchswe import nplike
from torchswe.utils.data import States, Gridlines, Topography

logger = logging.getLogger("torchswe.core.legate.reconstruction")


# TODO: in legate's version, check if using for-loops is slower or faster than using `where`. The
#       rationale behind using a for-loop is that the number of negative depths may be very small


def correct_negative_depth(states: States, topo: Topography) -> States:
    """Legate's version of fixing negative depth on the both sides of cell faces. (Legate version)

    Arguments
    ---------
    states : torchswe.utils.data.States
    topo : torchswe.utils.data.Topography

    Returns:
    --------
    states : torchswe.utils.data.States
        The same object as the input. Changed inplace. Returning it just for coding style.
    """

    # alias; shorten the code
    ngh = states.ngh

    # aliases
    qwslc = states.q.w[ngh:-ngh, ngh:-ngh]  # hopefully this is just a view in legate
    x, y = states.face.x, states.face.y

    # fix the case when the left depth of an interface is negative
    neg = (x.minus.w < topo.xface)
    x.minus.w = nplike.where(neg, topo.xface, x.minus.w)
    x.plus.w[:, :-1] = nplike.where(neg[:, 1:], 2*qwslc-topo.xface[:, 1:], x.plus.w[:, :-1])

    # fix the case when the right depth of an interface is negative
    neg = (x.plus.w < topo.xface)
    x.plus.w = nplike.where(neg, topo.xface, x.plus.w)
    x.minus.w[:, 1:] = nplike.where(neg[:, :-1], 2*qwslc-topo.xface[:, :-1], x.minus.w[:, 1:])

    # fix rounding errors in x.minus.w caused by the last calculation above
    x.minus.w = nplike.where(x.minus.w < topo.xface, topo.xface, x.minus.w)

    # fix the case when the bottom depth of an interface is negative
    neg = (y.minus.w < topo.yface)
    y.minus.w = nplike.where(neg, topo.yface, y.minus.w)
    y.plus.w[:-1, :] = nplike.where(neg[1:, :], 2*qwslc-topo.yface[1:, :], y.plus.w[:-1, :])

    # fix the case when the top depth of an interface is negative
    neg = (y.plus.w < topo.yface)
    y.plus.w = nplike.where(neg, topo.yface, y.plus.w)
    y.minus.w[1:, :] = nplike.where(neg[:-1, :], 2*qwslc-topo.yface[:-1, :], y.minus.w[1:, :])

    # fix rounding errors in y.minus.w caused by the last calculation above
    y.minus.w = nplike.where(y.minus.w < topo.yface, topo.yface, y.minus.w)

    return states
