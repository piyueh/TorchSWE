#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Data model for grid-related data.
"""
from operator import itemgetter as _itemgetter
from typing import Literal as _Literal
from typing import Tuple as _Tuple
from typing import Union as _Union

from mpi4py import MPI as _MPI
from mpi4py.util.dtlib import from_numpy_dtype
from pydantic import validator as _validator
from pydantic import conint as _conint
from pydantic import confloat as _confloat
from pydantic import root_validator as _root_validator
from torchswe import nplike as _nplike
from torchswe.utils.config import BaseConfig as _BaseConfig
from torchswe.utils.misc import DummyDtype as _DummyDtype


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
    vertices: 1D array of length n+1
        Coordinates at vertices.
    centers: 1D array of length n
        Coordinates at cell centers.
    xfcenters: 1D array of langth n+1 or n
        Coordinates at the centers of the cell faces normal to x-axis.
    yfcenters: 1D array of langth n or n+1
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
    vertices: _nplike.ndarray
    centers: _nplike.ndarray
    xfcenters: _nplike.ndarray
    yfcenters: _nplike.ndarray

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_all(cls, values):  # pylint: disable=no-self-argument, no-self-use
        """Validations that rely the existence of other fields."""

        _tol = 1e-10 if values["vertices"].dtype == _nplike.double else 1e-7

        # coordinate ranges
        gbg, ged, lbg, led = _itemgetter("glower", "gupper", "lower", "upper")(values)
        assert gbg < ged, f"Global lower bound >= global upper bound: {gbg}, {ged}"
        assert lbg < led, f"Local lower bound >= local upper bound: {lbg}, {led}"
        assert lbg >= gbg, f"Local lower bound < global lower bound: {lbg}, {gbg}"
        assert led <= ged, f"Local upper bound > global upper bound: {led}, {ged}"
        assert abs(lbg-values["vertices"][0]) < _tol, "lower != vertives[0]"
        assert abs(led-values["vertices"][-1]) < _tol, "upper != vertives[-1]"

        # index range
        gn, n, ibg, ied = _itemgetter("gn", "n", "ibegin", "iend")(values)
        assert n <= gn, f"Local cell number > global cell number: {gn}, {n}"
        assert n == (ied - ibg), "Local cell number != index difference"
        assert ibg < ied, f"Begining index >= end index: {ibg}, {ied}"

        # check dtype and increment
        for v in _itemgetter("vertices", "centers", "xfcenters", "yfcenters")(values):
            diffs = v[1:] - v[:-1]
            assert all(diff > 0 for diff in diffs), "Not in monotonically increasing order."
            assert all(abs(diff-values["delta"]) <= _tol for diff in diffs), "Delta doesn't match."
            assert v.dtype == values["dtype"], "Floating-number types mismatch"

        # check vertices
        assert values["vertices"].shape == (values["n"]+1,), "The number of vertices doesn't match."

        # check cell centers
        assert values["centers"].shape == (values["n"],), "The number of centers doesn't match."
        assert _nplike.allclose(
            values["centers"], (values["vertices"][:-1]+values["vertices"][1:])/2.), \
            "Centers are not at the mid-points between neighboring vertices."

        # check the centers of faces
        if values["axis"] == "x":
            assert _nplike.allclose(values["xfcenters"], values["vertices"])
            assert _nplike.allclose(values["yfcenters"], values["centers"])
        else:
            assert _nplike.allclose(values["xfcenters"], values["centers"])
            assert _nplike.allclose(values["yfcenters"], values["vertices"])

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
        mpitype = from_numpy_dtype(sendbuf.dtype)
        _nplike.sync()
        values["comm"].Neighbor_alltoall([sendbuf, mpitype], [recvbuf, mpitype])

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
        dtype = values["x"].centers.dtype

        # send-recv indices range from neighbors
        sendbuf = _nplike.tile(_nplike.array((jbg, jed, ibg, ied), dtype=dtype), (4,))
        recvbuf = _nplike.full(16, float("NaN"), dtype=dtype)
        mpitype = from_numpy_dtype(dtype)
        _nplike.sync()
        values["comm"].Neighbor_alltoall([sendbuf, mpitype], [recvbuf, mpitype])

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
    def bounds(self):
        """The bounds of the local domain in the order of south, north, west, & east"""
        return self.y.lower, self.y.upper, self.x.lower, self.x.upper

    @property
    def gbounds(self):
        """The bounds of the global domain in the order of south, north, west, & east"""
        return self.y.glower, self.y.gupper, self.x.glower, self.x.gupper

    @property
    def delta(self):
        """The cell sizes in y and x."""
        return self.y.delta, self.x.delta

    @property
    def internal(self):
        """The slicing of internal (non halo) region."""
        return (slice(self.effybg, self.effyed), slice(self.effxbg, self.effxed))
