#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Functions to calculate numerical/common flux.
"""
import numpy
from ..utils.data import States


def central_scheme(states: States, tol: float = 1e-12) -> States:
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
    # TODO: check if denominator == 0, ap and am should also be zeros

    for axis in ["x", "y"]:
        denominator = states.face[axis].plus.a - states.face[axis].minus.a
        coeff = states.face[axis].plus.a * states.face[axis].minus.a

        # if denominator == 0, the division result will just be the zeros
        zero_ji = numpy.nonzero(numpy.logical_and(denominator > -tol, denominator < tol))

        for key in ["w", "hu", "hv"]:
            with numpy.errstate(divide="ignore", invalid="ignore"):
                states.face[axis].num_flux[key] = (
                    states.face[axis].plus.a * states.face[axis].minus.flux[key] -
                    states.face[axis].minus.a * states.face[axis].plus.flux[key] +
                    coeff * (states.face[axis].plus[key] - states.face[axis].minus[key])
                ) / denominator

            states.face[axis].num_flux[key][zero_ji] = 0.

    return states
