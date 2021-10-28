#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Miscellaneous functions.
"""
from torchswe import nplike as _nplike
from torchswe.utils.data import States as _States


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
