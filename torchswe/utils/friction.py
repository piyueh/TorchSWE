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

    Arguments
    ---------
    h, hu, hv : _nplike.ndarray
        Depth and the conservative quantities in x and y directions.
    viscosity : _nplike.ndarray
        Flow kinematic viscosity.
    roughness : _nplike.ndarray
        Absolute roughness with dimension of length.

    Returns
    -------
    coefficients : _nplike.ndarray or float
        Darcy-Weisbach friction coefficients.

    Notes
    -----
    1. Note that this implementation slightly differs from paper because we recalculated the
       coefficients to have a better precision after the floating point.
    2. This function does not handle the case when h = 0 but hu != 0 (or hv != 0). This case should
       not happen from the begining. However, some numerical methods may still result in this
       situation.
    """
    # pylint: disable=invalid-name

    # currently we don't expect this function to be used with scalars
    assert isinstance(h, _nplike.ndarray)
    assert isinstance(hu, _nplike.ndarray)
    assert isinstance(hv, _nplike.ndarray)

    # velocity (actually, it is velocity x depth)
    velocity = _nplike.sqrt(hu**2+hv**2)

    # initialize coefficient array
    coeff = _nplike.zeros_like(h)

    # get locations where Re != 0
    loc = velocity.astype(bool)

    # reynolds number defined by depth
    re_h = velocity[loc] / viscosity  # length scale (i.e., h) already included in `velocity`

    # probability
    alpha = _nplike.reciprocal(_nplike.power(re_h/678., 8.4)+1.)
    beta = _nplike.reciprocal(_nplike.power(re_h*roughness/(150.*h[loc]), 1.8) + 1.)

    # coefficient of laminar regime
    coeff[loc] += _nplike.power(24./re_h, alpha)

    # locations of where Re > 1 in re_h, alpha, and beta
    loc = (re_h > 1.)
    re_h = re_h[loc]
    alpha = alpha[loc]
    beta = beta[loc]

    # location of where Re > 1 in the coefficient array
    loc = (velocity > viscosity)

    # coefficient of smmoth turbulence regime
    coeff[loc] *= _nplike.power(
        0.7396119392950918 * _nplike.exp(2.*approx_lambert_w(1.3484253036698046*re_h)) / re_h**2,
        (1. - alpha) * beta
    )

    # coefficient of fully-rough turbulence regime
    if roughness != 0.:  # to avoid division by zero (when roughness = 0, 1- beta is also 0)
        coeff[loc] *= _nplike.power(
            1.3447999999999998 / _nplike.log(12.213597446891887 * h[loc] / roughness)**2,
            (1. - alpha) * (1. - beta)
        )

    return coeff
