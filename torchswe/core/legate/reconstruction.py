#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Linear reconstruction. (Legate version)
"""
from torchswe import nplike
from torchswe.utils.data import States, Gridlines, Topography


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

    # fix the case when the left depth of an interface is negative
    neg_j, neg_i = nplike.nonzero(states.face.x.minus.w < topo.xface)

    # a workaround for Legate; let's hope neg_j and neg_i are not long
    for j, i in zip(neg_j, neg_i):
        j, i =  int(j), int(i)  # legate.numpy does not like numpy.int64
        states.face.x.minus.w[j, i] = topo.xface[j, i]
        if i > 0:
            states.face.x.plus.w[j, i-1] = 2 * states.q.w[j+ngh, i-1+ngh] - topo.xface[j, i]

    # fix the case when the right depth of an interface is negative
    neg_j, neg_i = nplike.nonzero(states.face.x.plus.w < topo.xface)

    for j, i in zip(neg_j, neg_i):
        j, i =  int(j), int(i)  # legate.numpy does not like numpy.int64
        states.face.x.plus.w[j, i] = topo.xface[j, i]
        if i < states.nx:
            states.face.x.minus.w[j, i+1] = 2 * states.q.w[j+ngh, i+ngh] - topo.xface[j, i]

    # fix the case when the bottom depth of an interface is negative
    neg_j, neg_i = nplike.nonzero(states.face.y.minus.w < topo.yface)

    for j, i in zip(neg_j, neg_i):
        j, i =  int(j), int(i)  # legate.numpy does not like numpy.int64
        states.face.y.minus.w[j, i] = topo.yface[j, i]
        if j > 1:
            states.face.y.plus.w[j-1, i] = 2 * states.q.w[j-1+ngh, i+ngh] - topo.yface[j, i]

    # fix the case when the top depth of an interface is negative
    neg_j, neg_i = nplike.nonzero(states.face.y.plus.w < topo.yface)

    for j, i in zip(neg_j, neg_i):
        j, i =  int(j), int(i)  # legate.numpy does not like numpy.int64
        states.face.y.plus.w[j, i] = topo.yface[j, i]
        if j < states.ny:
            states.face.y.minus.w[j+1, i] = 2 * states.q.w[j+ngh, i+ngh] - topo.yface[j, i]

    # ignoring rounding error for now

    return states
