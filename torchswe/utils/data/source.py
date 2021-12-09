#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Data models for source-term-related thing.
"""
from typing import Tuple as _Tuple
from pydantic import conint as _conint
from pydantic import confloat as _confloat
from pydantic import validator as _validator
from torchswe.utils.config import BaseConfig as _BaseConfig


class PointSource(_BaseConfig):
    """An object representing a point source and its flow rate profile.

    Attributes
    ----------
    x, y : floats
        The x and y coordinates of the point source.
    i, j : int
        The local cell indices in the current rank's domain.
    times : a tuple of floats
        Times to change flow rates.
    rates : a tiple of floats
        Depth increment rates during given time intervals. Unit: m / sec.
    irate : int
        The index of the current flow rate among those in `rates`.
    """
    x: _confloat(strict=True)
    y: _confloat(strict=True)
    i: _conint(strict=True, ge=0)
    j: _conint(strict=True, ge=0)
    times: _Tuple[_confloat(strict=True), ...]
    rates: _Tuple[_confloat(strict=True, ge=0.), ...]
    irate: _conint(strict=True, ge=0)
    active: bool = True
    init_dt: _confloat(strict=True, gt=0.)

    @_validator("irate")
    def _val_irate(cls, val, values):  # pylint: disable=no-self-argument, no-self-use
        """Validate irate."""
        try:
            target = values["rates"]
        except KeyError as err:
            raise AssertionError("Correct `rates` first.") from err

        assert val < len(target), f"`irate` (={val}) should be smaller than {len(target)}"
        return val
