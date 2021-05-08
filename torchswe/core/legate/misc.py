#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Miscellaneous functions.
"""
from torchswe import nplike
from torchswe.utils.data import WHUHVModel


def remove_rounding_errors(whuhv: WHUHVModel, tol: float):
    """Removing rounding errors from states.

    Arguments
    ---------
    whuhv : torchswe.utils.data.WHUHVModel
        Any instance of WHUHVModel data model.
    tol : float
        Rounding error.
    """

    zero_loc = (whuhv.w > -tol).astype(int) * (whuhv.w < tol).astype(int)
    whuhv.w = nplike.where(zero_loc.astype(bool), 0, whuhv.w)

    zero_loc = (whuhv.hu > -tol).astype(int) * (whuhv.hu < tol).astype(int)
    whuhv.hu = nplike.where(zero_loc.astype(bool), 0, whuhv.hu)

    zero_loc = (whuhv.hv > -tol).astype(int) * (whuhv.hv < tol).astype(int)
    whuhv.hv = nplike.where(zero_loc.astype(bool), 0, whuhv.hv)

    return whuhv
