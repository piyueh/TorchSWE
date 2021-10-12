#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""A collection of Darcy-Weisbach friction coefficient models.
"""
from torchswe import nplike as _nplike


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
    lnx = _nplike.log(x)
    lnlnx = _nplike.log(lnx)
    return lnx - lnlnx + lnlnx / lnx + (lnlnx * lnlnx - 2. * lnlnx) / (2. * lnx * lnx)


def bellos_et_al_2018(h, hu, hv, viscosity, roughness):
    """Bellow et al., 2018

    Note that this implementation slightly differs from paper because we recalculated the
    coefficients to have a better precision after the floating point.

    Arguments
    ---------
    h, hu, hv : _nplike.ndarray or float
        Depth and the conservative quantities in x and y directions.
    viscosity : _nplike.ndarray or float
        Flow kinematic viscosity.
    roughness : _nplike.ndarray or float
        Absolute roughness with dimension of length.

    Returns
    -------
    coefficients : _nplike.ndarray or float
        Darcy-Weisbach friction coefficients.
    """
    # pylint: disable=invalid-name

    # reynolds number defined by depth
    re_h = _nplike.sqrt(hu**2 + hv**2) / viscosity

    # probability
    alpha = _nplike.reciprocal(_nplike.power(re_h/678., 8.4)+1.)
    beta = _nplike.reciprocal(_nplike.power(re_h*roughness/(150.*h), 1.8) + 1.)

    # coefficient of laminar regime
    C1 = _nplike.power(24./re_h, alpha)

    # coefficient of smmoth turbulence regime
    C2 = _nplike.power(
        0.7396119392950918 * _nplike.exp(2.*approx_lambert_w(1.3484253036698046*re_h)) / re_h**2,
        (1. - alpha) * beta
    )

    try:
        # if roughness == 0, (1. - beta) is also 0, so C3 becomes inf^0 -> 1
        with _nplike.errstate(divide="ignore"):  # numpy/cupy won't complain division by 0
            # coefficient of fully-rough turbulence regime
            C3 = _nplike.power(
                1.3447999999999998 / _nplike.log(12.213597446891887 * h / roughness)**2,
                (1. - alpha) * (1. - beta)
            )
    except ZeroDivisionError:  # this err was raised when h and roughness are native python floats
        C3 = 1.

    return C1 * C2 * C3
