#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""A collection of Darcy-Weisbach friction coefficient models.
"""
from torchswe import nplike


def approx_lambert_w(x):  # pylint: disable=invalid-name
    """Approximation to the Lambert-W function.

    Arguments
    ---------
    x : float or ndarray of floats
        Inputs.

    Returns
    -------
    float or ndarray of floats.
    """
    lnx = nplike.log(x)
    lnlnx = nplike.log(lnx)
    return lnx - lnlnx + lnlnx / lnx + (lnlnx * lnlnx - 2. * lnlnx) / (2. * lnx * lnx)


def bellos_et_al_2018(velocity, depth, viscosity, roughness):
    """Bellow et al., 2018

    Note that this implementation slightly differs from paper because we recalculated the
    coefficients to have a better precision after the floating point.

    Arguments
    ---------
    velocity : nplike.ndarray or float
        Velocity (magnitude).
    depth : nplike.ndarray or float
        Flow depth.
    viscosity : nplike.ndarray or float
        Flow kinematic viscosity.
    roughness : nplike.ndarray or float
        Absolute roughness with dimension of length.

    Returns
    -------
    coefficients : nplike.ndarray or float
        Darcy-Weisbach friction coefficients.
    """
    # pylint: disable=invalid-name

    # reynolds number defined by depth
    re_h = velocity * depth / viscosity

    # probability
    alpha = nplike.reciprocal(nplike.power(re_h/678., 8.4)+1.)
    beta = nplike.reciprocal(nplike.power(re_h*roughness/(150.*depth), 1.8) + 1.)

    # coefficient of laminar regime
    C1 = nplike.power(
        24. / re_h,
        alpha
    )

    # coefficient of smmoth turbulence regime
    C2 = nplike.power(
        0.7396119392950918 * nplike.exp(2.*approx_lambert_w(1.3484253036698046*re_h)) / re_h**2,
        (1. - alpha) * beta
    )

    # in case roughness == 0; beta is also 0 in this case, so it becomes inf^0 -> 1
    try:
        with nplike.errstate(divide="ignore"):  # numpy won't complain
            # coefficient of fully-rough turbulence regime
            C3 = nplike.power(
                1.3447999999999998 / nplike.log(12.213597446891887 * depth / roughness)**2,
                (1. - alpha) * (1. - beta)
            )
    except ZeroDivisionError:  # when depth and roughness are pure python floats
        C3 = 1.

    return C1 * C2 * C3
