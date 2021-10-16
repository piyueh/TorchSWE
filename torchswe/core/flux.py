#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Functions for calculating discontinuous flux.
"""
from torchswe import nplike as _nplike
from torchswe.utils.data import States as _States


def get_discontinuous_flux(states: _States, gravity: float) -> _States:
    """Calculting the discontinuous fluxes on the both sides at cell faces.

    Arguments
    ---------
    states : torchswe.utils.data.States
    topo : torchswe.utils.data.Topography
    gravity : float

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Changed inplace. Returning it just for coding style.

    Notes
    -----
    When calculating (w-z)^2, it seems using w*w-w*z-z*w+z*z has smaller rounding errors. Not sure
    why. But it worth more investigation. This is apparently slower, though with smaller errors.
    """

    grav2 = gravity / 2.

    # face normal to x-direction: [hu, hu^2 + g(h^2)/2, huv]
    for sign in ["minus", "plus"]:

        states.face.x[sign].F[0] = states.face.x[sign].Q[1]

        states.face.x[sign].F[1] = \
            states.face.x[sign].Q[1] * states.face.x[sign].U[1] + \
            grav2 * (states.face.x[sign].U[0]**2)  # h = U[0] = Q[0] - xfcenter from `reconstruct`

        states.face.x[sign].F[2] = states.face.x[sign].Q[1] * states.face.x[sign].U[2]

    # face normal to y-direction: [hv, huv, hv^2+g(h^2)/2]
    for sign in ["minus", "plus"]:

        states.face.y[sign].F[0] = states.face.y[sign].Q[2]

        states.face.y[sign].F[1] = states.face.y[sign].U[1] * states.face.y[sign].Q[2]

        states.face.y[sign].F[2] = \
            states.face.y[sign].Q[2] * states.face.y[sign].U[2] + \
            grav2 * (states.face.y[sign].U[0]**2)

    return states


def central_scheme(states: _States, tol: float = 1e-12) -> _States:
    """A central scheme to calculate numerical flux at interfaces.

    Arguments
    ---------
    states : torchswe.utils.data.States
    tol : float
        The tolerance that can be considered as zero.

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Updated in-place. Returning it just for coding style.
    """

    for axis in ["x", "y"]:

        denominator = states.face[axis].plus.a - states.face[axis].minus.a

        # NOTE =====================================================================================
        # If `demoninator` is zero, then both `states.face[axis].plus.a` and
        # `states.face[axis].minus.a` should also be zero.
        # ==========================================================================================

        coeff = states.face[axis].plus.a * states.face[axis].minus.a

        # if denominator == 0, the division result will just be the zeros
        j, i = _nplike.nonzero(_nplike.logical_and(denominator > -tol, denominator < tol))

        with _nplike.errstate(divide="ignore", invalid="ignore"):
            states.face[axis].H = (
                states.face[axis].plus.a * states.face[axis].minus.F -
                states.face[axis].minus.a * states.face[axis].plus.F +
                coeff * (states.face[axis].plus.Q - states.face[axis].minus.Q)
            ) / denominator

        states.face[axis].H[:, j, i] = 0.

    return states
