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
import time as _time
import logging as _logging
from operator import itemgetter as _itemgetter
from typing import Optional as _Optional
from typing import Literal as _Literal
from typing import Tuple as _Tuple
from typing import Union as _Union

from mpi4py import MPI as _MPI
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


class Process(_BaseConfig):
    """A base class containing the info of an MPI process in a 2D Cartesian topology.

    Attributes
    ----------
    comm : MPI.Comm
        The communicator.
    pnx, pny : int
        The numbers of processes in the 2D Cartesian process topology.
    pi, pj : int
        The location of this process in the 2D Cartesian topology.
    west, east, south, north : int or None
        The ranks of the neighbors. If None, means the border is on domain boundary.
    """

    # mpi related
    comm: _MPI.Comm
    pnx: _conint(strict=True, gt=0)
    pny: _conint(strict=True, gt=0)
    pi: _conint(strict=True, ge=0)
    pj: _conint(strict=True, ge=0)
    west: _Optional[_conint(strict=True, ge=0)] = ...
    east: _Optional[_conint(strict=True, ge=0)] = ...
    south: _Optional[_conint(strict=True, ge=0)] = ...
    north: _Optional[_conint(strict=True, ge=0)] = ...

    @_validator("comm")
    def _val_comm(cls, val):
        recvbuf = val.allgather(val.Get_rank())
        assert recvbuf == list(range(val.Get_size())), "Communicator not working."
        return val

    @_root_validator
    def _val_topology(cls, values):
        rank = values["comm"].Get_rank()
        size = values["comm"].Get_size()
        pnx, pny, pi, pj = _itemgetter("pnx", "pny", "pi", "pj")(values)
        assert pnx * pny == size, "MPI world size does not equal to pnx * pny."
        assert pj * pnx + pi == rank, "MPI rank does not equal to pj * pnx + pi."
        return values

    @_root_validator
    def _val_neighbors(cls, values):
        # try to communicate with neighbors
        buff = _itemgetter("pnx", "pny", "pi", "pj")(values)
        send_tags = {"west": 31, "east": 41, "south": 51, "north": 61}
        recv_tags = {"west": 41, "east": 31, "south": 61, "north": 51}
        reqs, ready, reply = {}, {}, {}

        for key in ["west", "east", "south", "north"]:
            if values[key] is not None:
                values["comm"].isend(buff, values[key], send_tags[key])
                reqs[key] = values["comm"].irecv(source=values[key], tag=recv_tags[key])
                ready[key], reply[key] = reqs[key].test()

        counter = 0
        while (not all(ready.values())) and counter < 10:
            for key, val in reqs.items():
                if ready[key]:  # already ready some time in the previous iterations
                    continue
                ready[key], reply[key] = val.test()  # test again if this one is not ready

            counter += 1
            _time.sleep(1)

        for key, state, ans in zip(ready.keys(), ready.values(), reply.values()):
            # first make sure we got message from this neighbor
            assert state, f"Neighbor in {key} (rank {values[key]}) did not answer."

            # second make sure the neighbor has the same topology as we do
            assert ans[0] == buff[0], f"Err: pnx, {key}, {ans[0]}, {buff[0]}"
            assert ans[1] == buff[1], f"Err: pny, {key}, {ans[1]}, {buff[1]}"

            # lastly, check this neighbor's pi and pj
            if key == "west":
                assert ans[2] == buff[2] - 1, f"West's pi: {ans[2]}, my pi: {buff[2]}"
                assert ans[3] == buff[3], f"West's pj: {ans[3]}, my pj: {buff[3]}"
            elif key == "east":
                assert ans[2] == buff[2] + 1, f"East's pi: {ans[2]}, my pi: {buff[2]}"
                assert ans[3] == buff[3], f"East's pj: {ans[3]}, my pj: {buff[3]}"
            elif key == "south":
                assert ans[2] == buff[2], f"South's pi: {ans[2]}, my pi: {buff[2]}"
                assert ans[3] == buff[3] - 1, f"South's pj: {ans[3]}, my pj: {buff[3]}"
            elif key == "north":
                assert ans[2] == buff[2], f"North's pi: {ans[2]}, my pi: {buff[2]}"
                assert ans[3] == buff[3] + 1, f"North's pj: {ans[3]}, my pj: {buff[3]}"
            else:  # should be redundant, but I prefer keep it
                raise ValueError(f"Unrecoganized key: {key}")

        return values

    @property
    def proc_shape(self):
        """Returns the shape of the 2D Cartesian process topology."""
        return (self.pny, self.pnx)

    @property
    def proc_loc(self):
        """Returns the location of the current process in the 2D Cartesian process topology."""
        return (self.pj, self.pi)


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
    """A base class containing the info of a process in a 2D Cartesian topology.

    Attributes
    ----------
    process : Process
        The object holding MPI information.
    x, y : Gridline object
        x and y grindline coordinates.
    """

    # mpi process
    process: Process

    # gridlines
    x: Gridline
    y: Gridline

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_range(cls, values):

        buff = [
            values["process"].comm.Get_rank(),
            values["x"].ibegin, values["x"].iend, values["y"].ibegin, values["y"].iend
        ]

        send_tags = {"west": 31, "east": 41, "south": 51, "north": 61}
        recv_tags = {"west": 41, "east": 31, "south": 61, "north": 51}
        reqs, ready, reply = {}, {}, {}

        for key, val in values["process"].dict(include={"west", "east", "south", "north"}).items():
            if val is not None:
                values["process"].comm.isend(buff, val, send_tags[key])
                reqs[key] = values["process"].comm.irecv(source=val, tag=recv_tags[key])
                ready[key], reply[key] = reqs[key].test()

        counter = 0
        while (not all(ready.values())) and counter < 10:
            for key, val in reqs.items():
                if ready[key]:  # already ready some time in the previous iterations
                    continue
                ready[key], reply[key] = val.test()

            counter += 1
            _time.sleep(1)

        for key, state, ans in zip(ready.keys(), ready.values(), reply.values()):
            # first make sure we got message from this neighbor
            assert state, f"Neighbor in {key} (rank {values[key]}) did not answer."
            # second make sure this is the correct neighbor (is this redundant?)
            assert ans[0] == values["process"][key], f"{key}: {ans[0]}, {values['process'][key]}"

            # check the indices from the neighbor in the west
            if key == "west":
                assert ans[2] == buff[1], f"West's ied != my ibg: {ans[2]}, {buff[1]}"
                assert ans[3] == buff[3], f"West's jbg != my jbg: {ans[3]}, {buff[3]}"
                assert ans[4] == buff[4], f"West's jed != my jed: {ans[4]}, {buff[4]}"
            elif key == "east":
                assert ans[1] == buff[2], f"East's ibg != my ied: {ans[1]}, {buff[2]}"
                assert ans[3] == buff[3], f"East's jbg != my jbg: {ans[3]}, {buff[3]}"
                assert ans[4] == buff[4], f"East's jed != my jed: {ans[4]}, {buff[4]}"
            elif key == "south":
                assert ans[1] == buff[1], f"South's ibg != my ibg: {ans[1]}, {buff[1]}"
                assert ans[2] == buff[2], f"South's ied != my ied: {ans[2]}, {buff[2]}"
                assert ans[4] == buff[3], f"South's jed != my jbg: {ans[4]}, {buff[3]}"
            elif key == "north":
                assert ans[1] == buff[1], f"North's ibg != my ibg: {ans[1]}, {buff[1]}"
                assert ans[2] == buff[2], f"North's ied != my ied: {ans[2]}, {buff[2]}"
                assert ans[3] == buff[4], f"North's jbg != my jed: {ans[3]}, {buff[4]}"
            else:  # should be redundant, but I prefer keep it
                raise ValueError(f"Unrecoganized key: {key}")

        return values

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_delta(cls, values):

        # check dx
        dxs = values["process"].comm.allgather(values["x"].delta)
        assert all(dx == values["x"].delta for dx in dxs), "Not all processes have the same dx."

        # check dy
        dys = values["process"].comm.allgather(values["y"].delta)
        assert all(dy == values["y"].delta for dy in dys), "Not all processes have the same dy."

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
    """

    # associated domain
    domain: Domain

    # number of ghost cell layers
    ngh: _conint(strict=True, ge=0)

    # quantities defined at cell centers and faces
    Q: _nplike.ndarray
    H: _nplike.ndarray
    S: _nplike.ndarray
    SS: _Optional[_nplike.ndarray]
    face: FaceQuantityModel

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

    def exchange_data(self):
        """Exchange data with neighbor MPI process to update overlapped slices."""
        # pylint: disable=too-many-locals

        sbuf, sreq, rbuf, rreq = {}, {}, {}, {}

        stags = {"west": 31, "east": 41, "south": 51, "north": 61}
        rtags = {"west": 41, "east": 31, "south": 61, "north": 51}

        sslcs = {
            "west": (slice(None), slice(self.ngh, -self.ngh), slice(self.ngh, 2*self.ngh)),
            "east": (slice(None), slice(self.ngh, -self.ngh), slice(-2*self.ngh, -self.ngh)),
            "south": (slice(None), slice(self.ngh, 2*self.ngh), slice(self.ngh, -self.ngh)),
            "north": (slice(None), slice(-2*self.ngh, -self.ngh), slice(self.ngh, -self.ngh)),
        }

        rslcs = {
            "west": (slice(None), slice(self.ngh, -self.ngh), slice(None, self.ngh)),
            "east": (slice(None), slice(self.ngh, -self.ngh), slice(-self.ngh, None)),
            "south": (slice(None), slice(None, self.ngh), slice(self.ngh, -self.ngh)),
            "north": (slice(None), slice(-self.ngh, None), slice(self.ngh, -self.ngh)),
        }

        # make an alias for convenience
        proc = self.domain.process

        ans = 0
        for ornt in ["west", "east", "south", "north"]:
            if proc[ornt] is not None:

                sbuf[ornt] = self.Q[sslcs[ornt]].copy()
                _nplike.sync()  # make sure the copy is done before sending the data

                sreq[ornt] = proc.comm.Isend(sbuf[ornt], proc[ornt], stags[ornt])

                rbuf[ornt] = _nplike.zeros_like(self.Q[rslcs[ornt]])
                _nplike.sync()  # make sure the buffer is ready before receiving the data

                rreq[ornt] = proc.comm.Irecv(rbuf[ornt], proc[ornt], rtags[ornt])

                ans += 1

        # make sure send requests are done
        tstart = _time.perf_counter()
        done = 0
        while done != ans and _time.perf_counter()-tstart < 5.:
            for ornt, req in sreq.items():
                if ornt in sbuf and req.Test():
                    del sbuf[ornt]
                    done += 1

        # make sure whether the while-loop exited because of done == ans
        if done != ans:
            raise RuntimeError(f"Sending data to neighbor timeout: {sbuf.keys()}")

        # receive data from neighbors
        tstart = _time.perf_counter()
        done = 0
        while done != ans and _time.perf_counter()-tstart < 5.:
            for ornt, req in rreq.items():
                if ornt in rbuf and req.Test():
                    self.Q[rslcs[ornt]] = rbuf[ornt]
                    del rbuf[ornt]
                    done += 1

        # make sure if the while loop exited because of done == ans
        if done != ans:
            raise RuntimeError(f"Receiving data from neighbor timeout: {rbuf.keys()}")


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
