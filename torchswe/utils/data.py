#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Data models.
"""
# pylint: disable=too-few-public-methods, no-self-argument, invalid-name, no-self-use
import logging as _logging
from operator import itemgetter as _itemgetter
from typing import Optional as _Optional
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

_logger = _logging.getLogger("torchswe.utils.data")


def _pydantic_val_nan_inf(val, field):
    """Validates if any elements are NaN or inf."""
    assert not _nplike.any(_nplike.isnan(val)), f"Got NaN in {field.name}"
    assert not _nplike.any(_nplike.isinf(val)), f"Got Inf in {field.name}"
    return val


def _shape_val_factory(shift: _Union[_Tuple[int, int], int]):
    """A function factory creating a function to validate shapes of arrays."""

    def _core_func(val, values):
        """A function to validate the shape."""
        try:
            if isinstance(shift, int):
                target = (values["n"]+shift,)
            else:
                target = (values["ny"]+shift[0], values["nx"]+shift[1])
        except KeyError as err:
            raise AssertionError("Validation failed due to other validation failures.") from err

        assert val.shape == target, f"Shape mismatch. Should be {target}, got {val.shape}"
        return val

    return _core_func


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
    def _val_all(cls, values):
        """Validations that rely the existence of other fields."""

        _tol = 1e-10 if values["vertices"].dtype == _nplike.double else 1e-7

        # coordinate ranges
        gbg, ged, bg, ed = _itemgetter("glower", "gupper", "lower", "upper")(values)
        assert gbg < ged, f"Global lower bound >= global upper bound: {gbg}, {ged}"
        assert bg < ed, f"Local lower bound >= local upper bound: {bg}, {ed}"
        assert bg >= gbg, f"Local lower bound < global lower bound: {bg}, {gbg}"
        assert ed <= ged, f"Local upper bound > global upper bound: {ed}, {ged}"
        assert abs(bg-values["vertices"][0]) < _tol, "lower != vertives[0]"
        assert abs(ed-values["vertices"][-1]) < _tol, "upper != vertives[-1]"

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
    def _val_values(cls, val):
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

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_indices(cls, values):

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
    def _val_bounds(cls, values):

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
    def _val_delta(cls, values):

        # check dx
        dxs = values["comm"].allgather(values["x"].delta)
        assert all(dx == values["x"].delta for dx in dxs), "Not all ranks have the same dx."

        # check dy
        dys = values["comm"].allgather(values["y"].delta)
        assert all(dy == values["y"].delta for dy in dys), "Not all ranks have the same dy."

        return values


class Topography(_BaseConfig):
    """Data model for digital elevation.

    Attributes
    ----------
    domain : torchswe.utils.data.Domain
        The Domain instance associated with this Topography instance.
    vertices : (ny+1, nx+1) array
        Elevation at vertices.
    centers : (ny, nx) array
        Elevation at cell centers.
    xfcenters : (ny, nx+1) array
        Elevation at cell faces normal to x-axis.
    yfcenters : (ny+1, nx) array
        Elevation at cell faces normal to y-axis.
    grad : (2, ny, nx) array
        Derivatives w.r.t. x and y at cell centers.
    """
    domain: Domain
    vertices: _nplike.ndarray
    centers: _nplike.ndarray
    xfcenters: _nplike.ndarray
    yfcenters: _nplike.ndarray
    grad: _nplike.ndarray

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_arrays(cls, values):
        """Validations that rely on other fields' correctness."""

        # check dtype
        arrays = ["vertices", "centers", "xfcenters", "yfcenters", "grad"]
        target = values["domain"].x.dtype
        for k, v in zip(arrays, _itemgetter(*arrays)(values)):
            assert v.dtype == target, f"{k}: dtype does not match"

        # check shapes
        msg = "shape does not match."
        nx, ny = values["domain"].x.n, values["domain"].y.n
        assert values["vertices"].shape == (ny+1, nx+1), "vertices: " + msg
        assert values["centers"].shape == (ny, nx), "centers: " + msg
        assert values["xfcenters"].shape == (ny, nx+1), "xfcenters: " + msg
        assert values["yfcenters"].shape == (ny+1, nx), "yfcenters: " + msg
        assert values["grad"].shape == (2, ny, nx), "grad: " + msg

        # check linear interpolation
        v = values["vertices"]
        msg = "linear interpolation"
        assert _nplike.allclose(values["xfcenters"], (v[1:, :]+v[:-1, :])/2.), "xfcenters: " + msg
        assert _nplike.allclose(values["yfcenters"], (v[:, 1:]+v[:, :-1])/2.), "yfcenters: " + msg
        assert _nplike.allclose(values["centers"], (v[:-1, :-1]+v[:-1, 1:]+v[1:, :-1]+v[1:, 1:])/4.)

        v = values["xfcenters"]
        assert _nplike.allclose(values["centers"], (v[:, 1:]+v[:, :-1])/2.), "centers vs xfcenters"

        v = values["yfcenters"]
        assert _nplike.allclose(values["centers"], (v[1:, :]+v[:-1, :])/2.), "centers vs yfcenters"

        # check central difference
        v = values["xfcenters"]
        dx = (values["domain"].x.vertices[1:] - values["domain"].x.vertices[:-1])[None, :]
        assert _nplike.allclose(values["grad"][0], (v[:, 1:]-v[:, :-1])/dx), "grad[0] vs xfcenters"

        v = values["yfcenters"]
        dy = (values["domain"].y.vertices[1:] - values["domain"].y.vertices[:-1])[:, None]
        assert _nplike.allclose(values["grad"][1], (v[1:, :]-v[:-1, :])/dy), "grad[1] vs yfcenters"

        return values


class FaceOneSideModel(_BaseConfig):
    """Data model holding quantities on one side of cell faces normal to one direction.

    Attributes
    ----------
    U : nplike.ndarray of shape (3, ny+1, nx) or (3, ny, nx+1)
        The fluid depth, u-velocity, and depth-v-velocity.
    a : nplike.ndarray of shape (ny+1, nx) or (3, ny, nx+1)
        The local speed.
    F : nplike.ndarray of shape (3, ny+1, nx) or (3, ny, nx+1)
        An array holding discontinuous fluxes.
    """
    Q: _nplike.ndarray
    U: _nplike.ndarray
    a: _nplike.ndarray
    F: _nplike.ndarray

    # validator
    _val_valid_numbers = _validator("Q", "U", "a", "F", allow_reuse=True)(_pydantic_val_nan_inf)

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_arrays(cls, values):
        """Validate the consistency of the arrays shapes and dtypes."""
        try:
            Q, U, a, F = values["Q"], values["U"], values["a"], values["F"]
        except KeyError as err:
            raise AssertionError("Fix other fields first.") from err

        n1, n2 = a.shape
        assert Q.shape == (3, n1, n2), f"U shape mismatch. Should be {(3, n1, n2)}. Got {U.shape}."
        assert U.shape == (3, n1, n2), f"U shape mismatch. Should be {(3, n1, n2)}. Got {U.shape}."
        assert F.shape == (3, n1, n2), f"F shape mismatch. Should be {(3, n1, n2)}. Got {F.shape}."
        assert Q.dtype == a.dtype, f"dtype mismatch. Should be {a.dtype}. Got {U.dtype}."
        assert U.dtype == a.dtype, f"dtype mismatch. Should be {a.dtype}. Got {U.dtype}."
        assert F.dtype == a.dtype, f"dtype mismatch. Should be {a.dtype}. Got {U.dtype}."
        return values


class FaceTwoSideModel(_BaseConfig):
    """Date model holding quantities on both sides of cell faces normal to one direction.

    Attributes
    ----------
    plus, minus : FaceOneSideModel
        Objects holding data on one side of each face.
    H : nplike.ndarray of shape (3, ny+1, nx) or (3, ny, nx+1)
        An object holding common (i.e., continuous or numerical) flux
    """
    plus: FaceOneSideModel
    minus: FaceOneSideModel
    H: _nplike.ndarray

    # validator
    _val_valid_numbers = _validator("H", allow_reuse=True)(_pydantic_val_nan_inf)

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_arrays(cls, values):
        """Validate shapes and dtypes."""
        try:
            plus, minus, H = values["plus"], values["minus"], values["H"]
        except KeyError as err:
            raise AssertionError("Fix other fields first.") from err

        assert plus.U.shape == minus.U.shape, f"Shape mismatch: {plus.U.shape} and {minus.U.shape}."
        assert plus.U.dtype == minus.U.dtype, f"dtype mismatch: {plus.U.dtype} and {minus.U.dtype}."
        assert plus.U.shape == H.shape, f"Shape mismatch: {plus.U.shape} and {H.shape}."
        assert plus.U.dtype == H.dtype, f"dtype mismatch: {plus.U.dtype} and {H.dtype}."
        return values


class FaceQuantityModel(_BaseConfig):
    """Data model holding quantities on both sides of cell faces in both x and y directions.

    Attributes
    ----------
    x, y : FaceTwoSideModel
        Objects holding data on faces facing x and y directions.
    """
    x: FaceTwoSideModel
    y: FaceTwoSideModel

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_arrays(cls, values):
        """Validate shapes and dtypes."""
        try:
            x, y = values["x"], values["y"]
        except KeyError as err:
            raise AssertionError("Fix other fields first.") from err

        assert (x.plus.a.shape[1] - y.plus.a.shape[1]) == 1, "Incorrect nx size."
        assert (y.plus.a.shape[0] - x.plus.a.shape[0]) == 1, "Incorrect ny size."
        assert x.plus.a.dtype == y.plus.a.dtype, "Mismatched dtype."
        return values


class States(_BaseConfig):
    """A jumbo data model of all arrays on a mesh patch.

    A brief overview of the structure in this jumbo model (ignoring scalars):
    State: {
        Q: ndarray                                          # shape: (3, ny+2*ngh, nx+2*ngh)
        H: ndarray                                          # shape: (ny, nx)
        S: ndarray                                          # shape: (3, ny, nx)
        SS: ndarray                                         # shape: (3, ny, nx)
        face: {
            x: {
                plus: {
                    Q: ndarray                              # shape: (3, ny, nx+1)
                    U: ndarray                              # shape: (3, ny, nx+1)
                    a: ndarray                              # shape: (ny, nx+1)
                    F: ndarray                              # shape: (3, ny, nx+1)
                },
                minus: {
                    Q: ndarray                              # shape: (3, ny, nx+1)
                    U: ndarray                              # shape: (3, ny, nx+1)
                    a: ndarray                              # shape: (ny, nx+1)
                    F: ndarray                              # shape: (3, ny, nx+1)
                },
                H: ndarray                                  # shape: (3, ny, nx+1)
            },
            y: {                                            # shape: (ny+1, nx)
                plus: {
                    Q: ndarray                              # shape: (3, ny+1, nx)
                    U: ndarray                              # shape: (3, ny+1, nx)
                    a: ndarray                              # shape: (ny+1, nx)
                    F: ndarray                              # shape: (3, ny+1, nx)
                },
                minus: {
                    U: ndarray                              # shape: (3, ny+1, nx)
                    U: ndarray                              # shape: (3, ny+1, nx)
                    a: ndarray                              # shape: (ny+1, nx)
                    F: ndarray                              # shape: (3, ny+1, nx)
                },
                H: ndarray                                  # shape: (3, ny+1, nx)
            }
        }
    }

    Attributes
    ----------
    domain : torchswe.utils.data.Domain
        The domain associated to this state object.
    ngh : int
        Number of ghost cell layers.
    Q : nplike.ndarray of shape (3, ny+2*ngh, nx+2*ngh)
        The conservative quantities defined at cell centers.
    U : nplike.ndarray of shape (3, ny+2*ngh, nx+2*ngh)
        The non-conservative quantities defined at cell centers.
    S : nplike.ndarray of shape (3, ny, nx)
        The explicit right-hand-side terms when during time integration. Defined at cell centers.
    SS : nplike.ndarray of shape (3, ny, nx)
        The stiff right-hand-side term that require semi-implicit handling. Defined at cell centers.
    face: torchswe.utils.data.FaceQuantityModel
        Holding quantites defined at cell faces, including continuous and discontinuous ones.
    estype, wstype, sstype, nstype : mpi4py.MPI.Datatype
        A derived MPI datatype used for sending conservative quantities to neighbors.
    ertype, wrtype, srtype, nrtype : mpi4py.MPI.Datatype
        A derived MPI datatype used for neighbors to receive conservative quantities .
    """

    # associated domain
    domain: Domain

    # associated one-sided communication window
    win: _MPI.Win

    # number of ghost cell layers
    ngh: _conint(strict=True, ge=0)

    # quantities defined at cell centers and faces
    Q: _nplike.ndarray
    H: _nplike.ndarray
    S: _nplike.ndarray
    SS: _Optional[_nplike.ndarray]
    face: FaceQuantityModel

    # send datatype for conservative quantities
    estype: _MPI.Datatype
    wstype: _MPI.Datatype
    sstype: _MPI.Datatype
    nstype: _MPI.Datatype

    # recv datatype for conservative quantities
    ertype: _MPI.Datatype
    wrtype: _MPI.Datatype
    srtype: _MPI.Datatype
    nrtype: _MPI.Datatype

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_all(cls, values):
        nx = values["domain"].x.n
        ny = values["domain"].y.n
        ngh = values["ngh"]
        dtype = values["domain"].x.dtype

        assert values["Q"].shape == (3, ny+2*ngh, nx+2*ngh), "Q: incorrect shape"
        assert values["Q"].dtype == dtype, "Q: incorrect dtype"

        assert values["H"].shape == (ny, nx), "H: incorrect shape"
        assert values["H"].dtype == dtype, "H: incorrect dtype"

        assert values["S"].shape == (3, ny, nx), "S: incorrect shape"
        assert values["S"].dtype == dtype, "S: incorrect dtype"

        assert values["face"].x.plus.U.shape == (3, ny, nx+1), "face.x: incorrect shape"
        assert values["face"].x.plus.U.dtype == dtype, "face.x: incorrect dtype"

        assert values["face"].y.plus.U.shape == (3, ny+1, nx), "face.y: incorrect shape"
        assert values["face"].y.plus.U.dtype == dtype, "face.y: incorrect dtype"

        if values["SS"] is not None:
            assert values["SS"].shape == (3, ny, nx), "SS: incorrect shape"
            assert values["SS"].dtype == dtype, "SS: incorrect dtype"

        return values

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_q_subarray_types(cls, values):

        comm = values["domain"].comm
        nx = values["domain"].x.n
        ny = values["domain"].y.n
        ngh = values["ngh"]

        data = _nplike.zeros((3, ny+2*ngh, nx+2*ngh), dtype=values["Q"].dtype)
        data[0, ngh:-ngh, ngh:-ngh] = comm.rank * 1000
        data[1, ngh:-ngh, ngh:-ngh] = comm.rank * 1000 + 100
        data[2, ngh:-ngh, ngh:-ngh] = comm.rank * 1000 + 200
        _nplike.sync()

        win = _MPI.Win.Create(data, comm=comm)
        win.Fence()
        for k in ("s", "n", "w", "e"):
            win.Put([data, values[f"{k}stype"]], values["domain"][k], [0, 1, values[f"{k}rtype"]])
        win.Fence()

        # check if correct neighbors put correct data to this rank
        if values["domain"]["s"] != _MPI.PROC_NULL:
            assert all(data[0, :ngh, ngh:-ngh].flatten() == values["domain"]["s"]*1000)
            assert all(data[1, :ngh, ngh:-ngh].flatten() == values["domain"]["s"]*1000+100)
            assert all(data[2, :ngh, ngh:-ngh].flatten() == values["domain"]["s"]*1000+200)

        if values["domain"]["n"] != _MPI.PROC_NULL:
            assert all(data[0, -ngh:, ngh:-ngh].flatten() == values["domain"]["n"]*1000)
            assert all(data[1, -ngh:, ngh:-ngh].flatten() == values["domain"]["n"]*1000+100)
            assert all(data[2, -ngh:, ngh:-ngh].flatten() == values["domain"]["n"]*1000+200)

        if values["domain"]["w"] != _MPI.PROC_NULL:
            assert all(data[0, ngh:-ngh, :ngh].flatten() == values["domain"]["w"]*1000)
            assert all(data[1, ngh:-ngh, :ngh].flatten() == values["domain"]["w"]*1000+100)
            assert all(data[2, ngh:-ngh, :ngh].flatten() == values["domain"]["w"]*1000+200)

        if values["domain"]["e"] != _MPI.PROC_NULL:
            assert all(data[0, ngh:-ngh, -ngh:].flatten() == values["domain"]["e"]*1000)
            assert all(data[1, ngh:-ngh, -ngh:].flatten() == values["domain"]["e"]*1000+100)
            assert all(data[2, ngh:-ngh, -ngh:].flatten() == values["domain"]["e"]*1000+200)
        win.Free()

        return values


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
    def _val_irate(cls, val, values):
        """Validate irate."""
        try:
            target = values["rates"]
        except KeyError as err:
            raise AssertionError("Correct `rates` first.") from err

        assert val < len(target), f"`irate` (={val}) should be smaller than {len(target)}"
        return val
