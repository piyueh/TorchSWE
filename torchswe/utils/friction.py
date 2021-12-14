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


def friction_model_selector(name: str):
    """Return a friction coefficient model using the corresponding string name.

    Arguments
    ---------
    name : str
        The name of the model.

    Returns
    -------
    A callable.
    """

    if name == "bellos_et_al_2018":
        return bellos_et_al_2018

    # no matching
    raise ValueError(f"Unrecognized model name: {name}")


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

    def laminar(_re, _alpha):
        return _nplike.power(24./_re, _alpha)

    def smooth_turbulence(_re, _alpha, _beta):
        return _nplike.power(
            0.7396119392950918 * _nplike.exp(2.*approx_lambert_w(1.3484253036698046*_re)) / _re**2,
            (1. - _alpha) * _beta
        )

    def fully_rough_turbulence(_h, _r, _alpha, _beta):
        return _nplike.power(
            1.3447999999999998 / _nplike.log(12.213597446891887 * _h / _r)**2,
            (1. - _alpha) * (1. - _beta)
        )

    # velocity (actually, it is velocity x depth)
    velocity = _nplike.sqrt(hu**2+hv**2)

    # initialize coefficient array
    coeff = _nplike.zeros_like(h)

    # reynolds number defined by depth
    re_h = velocity / viscosity  # length scale (i.e., h) already included in `velocity`

    # exponents
    alpha = _nplike.reciprocal(_nplike.power(re_h/678., 8.4)+1.)
    beta = _nplike.reciprocal(_nplike.power(re_h*roughness/(150.*h), 1.8) + 1.)

    # get locations where Re != 0
    loc1 = velocity.astype(bool)

    # laminar regime
    coeff[loc1] += laminar(re_h[loc1], alpha[loc1])

    # locations where Re > 1
    loc2 = (re_h > 1.0)

    # smooth turbulence regime
    coeff[loc2] *= smooth_turbulence(re_h[loc2], alpha[loc2], beta[loc2])

    # locations where Re > 0 and roughness != 0
    loc2 = _nplike.logical_and(loc1, roughness.astype(bool))

    # coefficient of fully-rough turbulence regime
    coeff[loc2] *= fully_rough_turbulence(h[loc2], roughness[loc2], alpha[loc2], beta[loc2])

    return coeff
