#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Data models for source-term-related thing.
"""
from __future__ import annotations as _annotations  # allows us not using quotation marks for hints
from typing import TYPE_CHECKING as _TYPE_CHECKING  # indicates if we have type checking right now
if _TYPE_CHECKING:  # if we are having type checking, then we import corresponding classes/types
    from mpi4py import MPI
    from torchswe.utils.config import Config
    from torchswe.utils.data.grid import Domain

# pylint: disable=wrong-import-position, ungrouped-imports
from logging import getLogger as _getLogger
from typing import Tuple as _Tuple
from typing import Callable as _Callable
from mpi4py import MPI as _MPI
from pydantic import conint as _conint
from pydantic import confloat as _confloat
from pydantic import validator as _validator
from torchswe import nplike as _nplike
from torchswe.utils.config import BaseConfig as _BaseConfig
from torchswe.utils.io import read_block as _read_block
from torchswe.utils.misc import DummyDict as _DummyDict
from torchswe.utils.misc import find_cell_index as _find_cell_index
from torchswe.utils.misc import interpolate as _interpolate
from torchswe.utils.friction import friction_model_selector as _friction_model_selector
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


class FrictionModel(_BaseConfig):
    """An object holding required data/info for friction.
    """

    domain: _Domain
    roughness: _nplike.ndarray
    model: _Callable  # pydantic does not check the signature, so useless to specify signature

    @_validator("roughness")
    def _val_roughness(cls, val, values):  # pylint: disable=no-self-argument, no-self-use
        """Validate the shape and dtype of roughtness"""
        assert val.shape == values["domain"].shape, "roughness: shapes mismatched"
        assert val.dtype == values["domain"].dtype, "roughness: dtype mismatched"
        return val


def get_pointsource(config: Config, irate: int = 0, domain: Domain = None, comm: MPI.Comm = None):
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


def get_frictionmodel(config: Config, domain: Domain = None, comm: MPI.Comm = None):
    """Get a FrictionModel instance.

    Arguments
    ---------
    config : torchswe.utils.config.Config
        The configuration of a case.
    domain : torchswe.utils.data.Domain
        The object describing grids and domain decomposition.
    comm : mpi4py.MPI.Comm
        The communicator to be used if domain is None.

    Returns
    -------
    torchswe.utils.data.source.FrictionModel
    """

    # to hold data for initializing a Domain instance
    data = _DummyDict()

    # if domain is not provided, get a new one
    if domain is None:
        comm = _MPI.COMM_WORLD if comm is None else comm
        data.domain = domain = _get_domain(comm, config)
    else:
        data.domain = domain

    # set the model
    data.model = _friction_model_selector(config.friction.model)

    # set roughness if a constant value is provided
    if config.friction.value is not None:
        data.roughness = _nplike.full(data.domain.shape, config.friction.value)

    # otherwise, get roughness from a file
    data = _read_block(config.friction.file, config.friction.xykeys, config.friction.key, domain)

    # see if we need to do interpolation
    try:
        interp = not (_nplike.allclose(domain.x.c, data.x) and _nplike.allclose(domain.y.c, data.y))
    except ValueError:  # assume thie excpetion means a shape mismatch
        interp = True

    if interp:  # unfortunately, we need to do interpolation in such a situation
        _logger.warning("Grids do not match. Doing spline interpolation.")
        data.roughness = _nplike.array(
            _interpolate(data.x, data.y, data[config.friction.key].T, domain.x.c, domain.y.c).T
        ).astype(domain.dtype)
    else:  # no need for interpolation
        data.roughness = data[config.friction.key].astype(domain.dtype)

    return FrictionModel(**data)
