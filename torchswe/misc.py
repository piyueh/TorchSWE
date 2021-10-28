#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Miscellaneous functions.
"""
from typing import List as _List
from torchswe import nplike as _nplike
from torchswe.utils.data import States as _States


class CFLWarning(Warning):
    """A category of Warning for custome controll of warning action."""
    pass  # pylint: disable=unnecessary-pass


def minmod_slope(states: _States, theta: float, tol: float) -> _List[_nplike.ndarray]:
    """Calculate minmod slopes in x- and y-direction of quantity.

    Arguments
    ---------
    states : torchswe.utils.data.States
        Solution object.
    theta: float
        Parameter to controll oscillation and dispassion. 1 <= theta <= 2.
    tol : float
        To control how small can be treat as zero.

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Updated in-place. Returning it just for coding style.
    """

    # aliases
    dxs = (states.domain.x.delta, states.domain.y.delta)
    ngh = states.ngh

    # a list to save slopes in both x and y direction
    slp: _List[_nplike.ndarray] = []

    # correspond to i, i+1, and i-1
    self = [
        (slice(None), slice(ngh, -ngh), slice(ngh-1, -ngh+1)),
        (slice(None), slice(ngh-1, -ngh+1), slice(ngh, -ngh)),
    ]

    selfp1 = [
        (slice(None), slice(ngh, -ngh), slice(ngh, states.Q.shape[2])),  # can't use -ngh+2. It's 0!
        (slice(None), slice(ngh, states.Q.shape[1]), slice(ngh, -ngh)),
    ]

    selfm1 = [
        (slice(None), slice(ngh, -ngh), slice(ngh-2, -ngh)),
        (slice(None), slice(ngh-2, -ngh), slice(ngh, -ngh)),
    ]

    for i in range(2):
        denominator = states.Q[selfp1[i]] - states.Q[self[i]]  # q[i+1] - q[i]
        zeros = _nplike.nonzero(_nplike.logical_and(denominator > -tol, denominator < tol))

        with _nplike.errstate(divide="ignore", invalid="ignore"):
            slp.append((states.Q[self[i]] - states.Q[selfm1[i]]) / denominator)

        slp[i][zeros] = 0.  # where q_[i+1] - q_[i] = 0

        slp[i] = _nplike.maximum(
            _nplike.minimum(_nplike.minimum(theta*slp[i], (1.+slp[i])/2.), _nplike.array(theta)),
            _nplike.array(0.)
        )

        slp[i] *= denominator
        slp[i] /= dxs[i]

    return slp


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

    # for convenience
    zero = _nplike.array(0.)

    # faces normal to x-direction
    sqrt_gh_plus = _nplike.sqrt(gravity*states.face.x.plus.U[0])
    sqrt_gh_minus = _nplike.sqrt(gravity*states.face.x.minus.U[0])

    states.face.x.plus.a = _nplike.maximum(_nplike.maximum(
        states.face.x.plus.U[1]+sqrt_gh_plus, states.face.x.minus.U[1]+sqrt_gh_minus), zero)

    states.face.x.minus.a = _nplike.minimum(_nplike.minimum(
        states.face.x.plus.U[1]-sqrt_gh_plus, states.face.x.minus.U[1]-sqrt_gh_minus), zero)

    # faces normal to y-direction
    sqrt_gh_plus = _nplike.sqrt(gravity*states.face.y.plus.U[0])
    sqrt_gh_minus = _nplike.sqrt(gravity*states.face.y.minus.U[0])

    states.face.y.plus.a = _nplike.maximum(_nplike.maximum(
        states.face.y.plus.U[2]+sqrt_gh_plus, states.face.y.minus.U[2]+sqrt_gh_minus), zero)

    states.face.y.minus.a = _nplike.minimum(_nplike.minimum(
        states.face.y.plus.U[2]-sqrt_gh_plus, states.face.y.minus.U[2]-sqrt_gh_minus), zero)

    return states
