#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Data models for source-term-related thing.
"""
from logging import getLogger as _getLogger
from typing import Tuple as _Tuple
from mpi4py import MPI as _MPI
from pydantic import conint as _conint
from pydantic import confloat as _confloat
from pydantic import validator as _validator
from torchswe.utils.config import BaseConfig as _BaseConfig
from torchswe.utils.config import Config as _Config
from torchswe.utils.misc import DummyDict as _DummyDict
from torchswe.utils.misc import find_cell_index as _find_cell_index
from torchswe.utils.data.grid import Domain as _Domain
from torchswe.utils.data.grid import get_domain as _get_domain


_logger = _getLogger("torchswe.utils.data.source")


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


def get_pointsource(
    config: _Config, irate: int = 0, domain: _Domain = None, comm: _MPI.Comm = None
):
    """Get a PointSource instance.

    Arguments
    ---------
    config : torchswe.utils.config.Config
        The configuration of a case.
    irate : int
        The index of the current flow rate in the list of `rates`.
    domain : torchswe.utils.data.Domain
        The object describing grids and domain decomposition.
    comm : mpi4py.MPI.Comm
        The communicator to be used if domain is None.

    Returns
    -------
    `None` if the current MPI rank does not own this point source, otherwise an instance of
    torchswe.utils.data.PointSource.

    Notes
    -----
    The returned PointSource object will store depth increment rates, rather than volumetric flow
    rates.
    """

    # to hold data for initializing a Domain instance
    data = _DummyDict()

    # if domain is not provided, get a new one
    if domain is None:
        comm = _MPI.COMM_WORLD if comm is None else comm
        domain = _get_domain(comm, config)

    # aliases
    extent = domain.bounds  # south, north, west, east
    dy, dx = domain.delta

    # location and times to change rates
    data.x = config.ptsource.loc[0]
    data.y = config.ptsource.loc[1]
    data.times=config.ptsource.times

    # which stage in the rate profile
    data.irate = irate

    # a constraint to limit the initial time step size
    data.init_dt = config.ptsource.init_dt

    # the indices of the cell containing the point
    data.i = _find_cell_index(config.ptsource.loc[0], *extent[2:], dx)
    data.j = _find_cell_index(config.ptsource.loc[1], *extent[:2], dy)

    # this MPI process does not own the point
    if data.i is None or data.j is None:
        return None

    # convert volumetric flow rates to depth increment rates; assuming constant/uniform dx & dy
    data.rates = [rate / dx / dy for rate in config.ptsource.rates]

    # determine if the point source is active or not
    data.active = (not irate == len(config.ptsource.times))
    _logger.debug("Point source initial `active`: %s", data.active)

    return PointSource(**data)
