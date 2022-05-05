#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Data model for grid-related data.
"""
from __future__ import annotations as _annotations  # allows us not using quotation marks for hints
from typing import TYPE_CHECKING as _TYPE_CHECKING  # indicates if we have type checking right now
if _TYPE_CHECKING:  # if we are having type checking, then we import corresponding classes/types
    from mpi4py import MPI
    from torchswe.utils.config import Config

# pylint: disable=wrong-import-position, ungrouped-imports
from copy import deepcopy as _deepcopy
from operator import itemgetter as _itemgetter
from typing import Literal as _Literal
from typing import Tuple as _Tuple
from typing import Union as _Union

from mpi4py import MPI as _MPI
from mpi4py.util.dtlib import from_numpy_dtype as _from_numpy_dtype
from pydantic import validator as _validator
from pydantic import conint as _conint
from pydantic import confloat as _confloat
from pydantic import root_validator as _root_validator
from torchswe import nplike as _nplike
from torchswe.utils.config import BaseConfig as _BaseConfig
from torchswe.utils.misc import DummyDtype as _DummyDtype
from torchswe.utils.misc import DummyDict as _DummyDict
from torchswe.utils.misc import cal_num_procs as _cal_num_procs
from torchswe.utils.misc import cal_local_gridline_range as _cal_local_gridline_range


class Gridline(_BaseConfig):
    """Local gridline data model.

    Attributes
    ----------
    axis : str
        Either "x" or "y".
    gn : int
        Number of global cells.
    glower, gupper : float
        The global lower and the global higher bounds (coordinates) of this axis.
    n : int
        Number of cells.
    lower, upper : float
        The local lower and the local higher bounds (coordinates) of this gridline.
    ibegin, iend : int
        The lower cell index and upper cell index plus one of this gridline.
    delta : float
        Cell size.
    dtype : str, nplike.float32, or nplike64.
        The type of floating numbers. If a string, it should be either "float32" or "float64".
        If not a string, it should be either `nplike.float32` or `nplike.float64`.
    v: 1D array of length n+1
        Coordinates at vertices.
    c: 1D array of length n
        Coordinates at cell centers.
    xf: 1D array of langth n+1 or n
        Coordinates at the centers of the cell faces normal to x-axis.
    yf: 1D array of langth n or n+1
        Coordinates at the centers of the cell faces normal to y-axis.

    Notes
    -----
    The lengths of xfcenters and yfcenters depend on the direction.
    """

    dtype: _DummyDtype
    axis: _Literal["x", "y"]  # noqa: F821
    gn: _conint(strict=True, gt=0)
    glower: float
    gupper: float
    n: _conint(strict=True, gt=0)
    lower: float
    upper: float
    ibegin: _conint(strict=True, ge=0)
    iend: _conint(strict=True, gt=0)
    delta: _confloat(gt=0.)
    v: _nplike.ndarray
    c: _nplike.ndarray
    xf: _nplike.ndarray
    yf: _nplike.ndarray

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_all(cls, values):  # pylint: disable=no-self-argument, no-self-use
        """Validations that rely the existence of other fields."""

        _tol = 1e-10 if values["v"].dtype == _nplike.double else 1e-7

        # coordinate ranges
        gbg, ged, lbg, led = _itemgetter("glower", "gupper", "lower", "upper")(values)
        assert gbg < ged, f"Global lower bound >= global upper bound: {gbg}, {ged}"
        assert lbg < led, f"Local lower bound >= local upper bound: {lbg}, {led}"
        assert lbg >= gbg, f"Local lower bound < global lower bound: {lbg}, {gbg}"
        assert led <= ged, f"Local upper bound > global upper bound: {led}, {ged}"
        assert abs(lbg-values["v"][0]) < _tol, "lower != vertives[0]"
        assert abs(led-values["v"][-1]) < _tol, "upper != vertives[-1]"

        # index range
        gn, n, ibg, ied = _itemgetter("gn", "n", "ibegin", "iend")(values)
        assert n <= gn, f"Local cell number > global cell number: {gn}, {n}"
        assert n == (ied - ibg), "Local cell number != index difference"
        assert ibg < ied, f"Begining index >= end index: {ibg}, {ied}"

        # check dtype and increment
        for v in _itemgetter("v", "c", "xf", "yf")(values):
            diffs = v[1:] - v[:-1]
            assert all(diff > 0 for diff in diffs), "Not in monotonically increasing order."
            assert all(abs(diff-values["delta"]) <= _tol for diff in diffs), "Delta doesn't match."
            assert v.dtype == values["dtype"], "Floating-number types mismatch"

        # check vertices
        assert values["v"].shape == (values["n"]+1,), "The number of vertices doesn't match."

        # check cell centers
        assert values["c"].shape == (values["n"],), "The number of centers doesn't match."
        assert _nplike.allclose(
            values["c"], (values["v"][:-1]+values["v"][1:])/2.), \
            "Centers are not at the mid-points between neighboring vertices."

        # check the centers of faces
        if values["axis"] == "x":
            assert _nplike.allclose(values["xf"], values["v"])
            assert _nplike.allclose(values["yf"], values["c"])
        else:
            assert _nplike.allclose(values["xf"], values["c"])
            assert _nplike.allclose(values["yf"], values["v"])

        return values


class Timeline(_BaseConfig):
    """An object holding information of times for snapshots.

    This object supports using square brackets and slicing to get value(s). Just like a list.

    Attributes
    ----------
    values : a tuple of floats
        The actual values of times.
    save : bool
        Whether the times are for saving solutions.
    """
    values: _Tuple[_confloat(ge=0.), ...]
    save: bool

    @_validator("values")
    def _val_values(cls, val):  # pylint: disable=no-self-argument, no-self-use
        assert len(val) >= 2, "The length of values should >= 2"
        pos = [(v2-v1) > 0. for v1, v2 in zip(val[:-1], val[1:])]
        assert all(pos), "Times are not in a monotonically increasing order."
        return val

    def __getitem__(self, key):
        return self.values.__getitem__(key)

    def __len__(self):
        return self.values.__len__()


class Domain(_BaseConfig):
    """A base class containing the info of a rank in a 2D Cartesian topology.

    Attributes
    ----------
    comm : mpi4py.MPI.Cartcomm
        The object holding MPI communicator (in a Cartesian topology).
    e, w, s, n : int
        The ranks of the neighbors. If the neighbor does not exist (e.g., a boundary rank), then
        its value will be `mpi4py.MPI.PROC_NULL`. These characters stand for east, west, south, and
        north.
    x, y : Gridline object
        x and y grindline coordinates.
    """

    # mpi communicator
    comm: _MPI.Cartcomm

    # neighbors
    e: _Union[_conint(ge=0), _Literal[_MPI.PROC_NULL]]
    w: _Union[_conint(ge=0), _Literal[_MPI.PROC_NULL]]
    s: _Union[_conint(ge=0), _Literal[_MPI.PROC_NULL]]
    n: _Union[_conint(ge=0), _Literal[_MPI.PROC_NULL]]

    # gridlines
    x: Gridline
    y: Gridline

    # number of halo-ring layers (currently only supports 2)
    nhalo: _Literal[2]

    # internal ranges (i.e., ranges of non-halo cells)
    effxbg: _Literal[2]
    effxed: _conint(gt=2)
    effybg: _Literal[2]
    effyed: _conint(gt=2)

    @_validator("effxed")
    def _val_effxed(cls, val, values):  # pylint: disable=no-self-argument, no-self-use
        """Validate effxed."""
        assert val - values["effxbg"] == values["x"].n, "effxed - effxbg != x.n"
        return val

    @_validator("effyed")
    def _val_effyed(cls, val, values):  # pylint: disable=no-self-argument, no-self-use
        """Validate effxed and effyed."""
        assert val - values["effybg"] == values["y"].n, "effyed - effybg != y.n"
        return val

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_indices(cls, values):  # pylint: disable=no-self-argument, no-self-use

        # aliases
        jbg, jed = values["y"].ibegin, values["y"].iend
        ibg, ied = values["x"].ibegin, values["x"].iend

        # send-recv indices range from neighbors
        sendbuf = _nplike.tile(_nplike.array((jbg, jed, ibg, ied), dtype=int), (4,))
        recvbuf = _nplike.full(16, -999, dtype=int)
        mpitype = _from_numpy_dtype(sendbuf.dtype)
        _nplike.sync()

        try:
            values["comm"].Neighbor_alltoall([sendbuf, mpitype], [recvbuf, mpitype])
        except TypeError as err:
            if _nplike.__name__ == "cunumeric" and "__dlpack_device__" in str(err):
                return values
            raise

        # answers
        inds = {"s": (1, 2, 3), "n": (0, 2, 3), "w": (0, 1, 3), "e": (0, 1, 2), }
        ans = {
            "s": _nplike.array((jbg, ibg, ied), dtype=int),
            "n": _nplike.array((jed, ibg, ied), dtype=int),
            "w": _nplike.array((jbg, jed, ibg), dtype=int),
            "e": _nplike.array((jbg, jed, ied), dtype=int),
        }

        # check the values
        for i, ornt in enumerate(("s", "n", "w", "e")):
            if values[ornt] == _MPI.PROC_NULL:
                assert all(recvbuf[i*4:(i+1)*4] == -999), f"{ornt}, {recvbuf[i*4:(i+1)*4]}"
            else:
                if values["comm"].periods[i//2] == 0:  # regular bc
                    assert all(recvbuf[i*4:(i+1)*4].take(inds[ornt]) == ans[ornt]), \
                        f"{ornt}, {recvbuf[i*4:(i+1)*4].take(inds[ornt])}, {ans[ornt]}"
                else:
                    pass  # periodic bc; haven't come up w/ a good way to check

        return values

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_bounds(cls, values):  # pylint: disable=no-self-argument, no-self-use

        # aliases
        jbg, jed = values["y"].lower, values["y"].upper
        ibg, ied = values["x"].lower, values["x"].upper
        dtype = values["x"].c.dtype

        # send-recv indices range from neighbors
        sendbuf = _nplike.tile(_nplike.array((jbg, jed, ibg, ied), dtype=dtype), (4,))
        recvbuf = _nplike.full(16, float("NaN"), dtype=dtype)
        mpitype = _from_numpy_dtype(dtype)
        _nplike.sync()

        try:
            values["comm"].Neighbor_alltoall([sendbuf, mpitype], [recvbuf, mpitype])
        except TypeError as err:
            if _nplike.__name__ == "cunumeric" and "__dlpack_device__" in str(err):
                return values
            raise

        # answers
        inds = {"s": (1, 2, 3), "n": (0, 2, 3), "w": (0, 1, 3), "e": (0, 1, 2), }
        ans = {
            "s": _nplike.array((jbg, ibg, ied), dtype=dtype),
            "n": _nplike.array((jed, ibg, ied), dtype=dtype),
            "w": _nplike.array((jbg, jed, ibg), dtype=dtype),
            "e": _nplike.array((jbg, jed, ied), dtype=dtype),
        }

        # check the values
        for i, ornt in enumerate(("s", "n", "w", "e")):
            if values[ornt] == _MPI.PROC_NULL:
                assert all(recvbuf[i*4:(i+1)*4] != recvbuf[i*4:(i+1)*4]), \
                    f"{ornt}, {recvbuf[i*4:(i+1)*4]}"  # for nan, self != self
            else:
                if values["comm"].periods[i//2] == 0:  # regular bc
                    assert all(recvbuf[i*4:(i+1)*4].take(inds[ornt]) == ans[ornt]), \
                        f"{ornt}, {recvbuf[i*4:(i+1)*4].take(inds[ornt])}, {ans[ornt]}"
                else:
                    pass  # periodic bc; haven't come up w/ a good way to check

        return values

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_delta(cls, values):  # pylint: disable=no-self-argument, no-self-use

        # check dx
        dxs = values["comm"].allgather(values["x"].delta)
        assert all(dx == values["x"].delta for dx in dxs), "Not all ranks have the same dx."

        # check dy
        dys = values["comm"].allgather(values["y"].delta)
        assert all(dy == values["y"].delta for dy in dys), "Not all ranks have the same dy."

        return values

    @property
    def dtype(self):
        """The dtype of arrays defined on this domain."""
        return self.x.dtype

    @property
    def shape(self):
        """The shape of local grid w/o halo/ghost cells"""
        return self.y.n, self.x.n

    @property
    def hshape(self):
        """The shape of local grid w/ halo/ghost cells"""
        return self.y.n+2*self.nhalo, self.x.n+2*self.nhalo

    @property
    def gshape(self):
        """The shape of the global computational grid."""
        return self.y.gn, self.x.gn

    @property
    def lextent(self):
        """The extent of the local grid in the order of west, east, south, and north"""
        return self.x.lower, self.x.upper, self.y.lower, self.y.upper

    @property
    def gextent(self):
        """The extent of the global grid in the order of west, east, south, and north"""
        return self.x.glower, self.x.gupper, self.y.glower, self.y.gupper

    @property
    def lextent_c(self):
        """The extent defined with the centers of boundary cells."""
        return self.x.c[0], self.x.c[-1], self.y.c[0], self.y.c[-1]

    @property
    def lextent_v(self):
        """The extent defined with the coords of boundary vertices."""
        return self.x.v[0], self.x.v[-1], self.y.v[0], self.y.v[-1]

    @property
    def delta(self):
        """The cell sizes in y and x."""
        return self.y.delta, self.x.delta

    @property
    def nonhalo_c(self):
        """The slicing of internal (non halo) region."""
        return (slice(self.effybg, self.effyed), slice(self.effxbg, self.effxed))

    @property
    def nonhalo_v(self):
        """The slicing of internal (non halo) region."""
        return (slice(self.effybg, self.effyed+1), slice(self.effxbg, self.effxed+1))

    @property
    def nonhalo_xf(self):
        """The slicing of internal (non halo) region."""
        return (slice(self.effybg, self.effyed), slice(self.effxbg, self.effxed+1))

    @property
    def nonhalo_yf(self):
        """The slicing of internal (non halo) region."""
        return (slice(self.effybg, self.effyed+1), slice(self.effxbg, self.effxed))

    @property
    def global_c(self):
        """The slices of cell-centered arrays corresponding to this rank."""
        return (slice(self.y.ibegin, self.y.iend), slice(self.x.ibegin, self.x.iend))

    @property
    def global_v(self):
        """The slices of cell-centered arrays corresponding to this rank."""
        return (slice(self.y.ibegin, self.y.iend+1), slice(self.x.ibegin, self.x.iend+1))

    @property
    def global_xf(self):
        """The slices of cell-centered arrays corresponding to this rank."""
        return (slice(self.y.ibegin, self.y.iend), slice(self.x.ibegin, self.x.iend+1))

    @property
    def global_yf(self):
        """The slices of cell-centered arrays corresponding to this rank."""
        return (slice(self.y.ibegin, self.y.iend+1), slice(self.x.ibegin, self.x.iend))


def get_gridline_x(comm: MPI.Cartcomm, config: Config):
    """Get a Gridline instance in x direction.

    Arguments
    ---------
    comm : mpi4py.MPI.Cartcomm
        The Cartesian communicator describing the topology.
    config : torchswe.utils.config.Config
        The configuration of a case.

    Returns
    -------
    gridline : torchswe.utils.data.grid.Gridline
    """

    assert isinstance(comm, _MPI.Cartcomm), "The communicator must be a Cartesian communicator."

    arg         = _DummyDict()
    arg.axis    = "x"
    arg.gn      = config.spatial.discretization[0]
    arg.glower  = config.spatial.domain[0]
    arg.gupper  = config.spatial.domain[1]
    arg.delta   = (arg.gupper - arg.glower) / arg.gn
    arg.dtype   = _DummyDtype.validator(config.params.dtype)

    arg.n, arg.ibegin, arg.iend = _cal_local_gridline_range(comm.dims[1], comm.coords[1], arg.gn)
    arg.lower                   = arg.ibegin * arg.delta + arg.glower
    arg.upper                   = arg.iend * arg.delta + arg.glower

    arg.v    = _nplike.linspace(arg.lower, arg.upper, arg.n+1, dtype=arg.dtype)
    arg.c    = (arg.v[1:] + arg.v[:-1]) / 2.0
    arg.xf   = _deepcopy(arg.v)
    arg.yf   = _deepcopy(arg.c)

    return Gridline(**arg)


def get_gridline_y(comm: MPI.Cartcomm, config: Config):
    """Get a Gridline instance in y direction.

    Arguments
    ---------
    comm : mpi4py.MPI.Cartcomm
        The Cartesian communicator describing the topology (order: ny and then nx).
    config : torchswe.utils.config.Config
        The configuration of a case.

    Returns
    -------
    gridline : torchswe.utils.data.grid.Gridline
    """

    assert isinstance(comm, _MPI.Cartcomm), "The communicator must be a Cartesian communicator."

    arg         = _DummyDict()
    arg.axis    = "y"
    arg.gn      = config.spatial.discretization[1]
    arg.glower  = config.spatial.domain[2]
    arg.gupper  = config.spatial.domain[3]
    arg.delta   = (arg.gupper - arg.glower) / arg.gn
    arg.dtype   = _DummyDtype.validator(config.params.dtype)

    arg.n, arg.ibegin, arg.iend = _cal_local_gridline_range(comm.dims[0], comm.coords[0], arg.gn)
    arg.lower                   = arg.ibegin * arg.delta + arg.glower
    arg.upper                   = arg.iend * arg.delta + arg.glower

    arg.v    = _nplike.linspace(arg.lower, arg.upper, arg.n+1, dtype=arg.dtype)
    arg.c    = (arg.v[1:] + arg.v[:-1]) / 2.0
    arg.xf   = _deepcopy(arg.c)
    arg.yf   = _deepcopy(arg.v)

    return Gridline(**arg)


def get_timeline(config: Config):
    """Generate a list of times when the solver should output solution snapshots.

    Arguments
    ---------
    config : torchswe.utils.config.Config

    Returns
    -------
    t : torchswe.utils.data.Timeline
    """

    save = True  # default
    output_type = config.temporal.output[0]
    params = config.temporal.output[1:]

    # write solutions to a file at give times
    if output_type == "at":
        t = params[0]

    # output every `every_seconds` seconds `multiple` times from `t_start`
    elif output_type == "t_start every_seconds multiple":
        begin, delta, n = params
        t = (_nplike.arange(0, n+1) * delta + begin).tolist()  # including saving t_start

    # output every `every_steps` constant-size steps for `multiple` times from t=`t_start`
    elif output_type == "t_start every_steps multiple":  # including saving t_start
        begin, steps, n = params
        t = (_nplike.arange(0, n+1) * config.temporal.dt * steps + begin).tolist()

    # from `t_start` to `t_end` evenly outputs `n_saves` times (including both ends)
    elif output_type == "t_start t_end n_saves":
        begin, end, n = params
        t = _nplike.linspace(begin, end, n+1).tolist()  # including saving t_start

    # run simulation from `t_start` to `t_end` but not saving solutions at all
    elif output_type == "t_start t_end no save":
        t = params
        save = False

    # run simulation from `t_start` with `n_steps` iterations but not saving solutions at all
    elif output_type == "t_start n_steps no save":
        t = [params[0], params[0] + params[1] * config.temporal.dt]
        save = False

    # should never reach this branch because pydantic has detected any invalid arguments
    else:
        raise ValueError(f"{output_type} is not an allowed output method.")

    return Timeline(values=t, save=save)


def get_domain(comm: MPI.Comm, config: Config):
    """Get an instance of Domain for the current MPI rank.

    Arguments
    ---------
    comm : mpi4py.MPI.Comm
        The communicator. Should just be a general communicator. And a Cartcomm will be created
        automatically in this function from the provided general Comm.
    config : torchswe.utils.config.Config
        The configuration of a case.

    Returns
    -------
    An instance of torchswe.utils.data.Domain.
    """

    # see if we need periodic bc
    period = (config.bc.west.types[0] == "periodic", config.bc.south.types[0] == "period")

    # to hold data for initializing a Domain instance
    data = _DummyDict()

    # only when the provided communicator is not a Cartesian communicator
    if not isinstance(comm, _MPI.Cartcomm):
        # evaluate the number of ranks in x/y direction and get a Cartesian topology communicator
        pnx, pny = _cal_num_procs(comm.Get_size(), *config.spatial.discretization)
        data.comm = comm.Create_cart((pny, pnx), period, True)
    else:
        data.comm = comm

    # find the rank of neighbors
    data.s, data.n = data.comm.Shift(0, 1)
    data.w, data.e = data.comm.Shift(1, 1)

    # get local gridline
    data.x = get_gridline_x(data.comm, config)
    data.y = get_gridline_y(data.comm, config)

    # halo-ring related
    data.nhalo = 2
    data.effxbg = 2
    data.effybg = 2
    data.effxed = data.effxbg + data.x.n
    data.effyed = data.effybg + data.y.n

    return Domain(**data)
