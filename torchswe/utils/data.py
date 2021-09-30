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


def _pydantic_val_dtype(val: _nplike.ndarray, values: dict) -> _nplike.ndarray:
    """Validates that a given ndarray has a matching dtype; used by pydantic."""
    try:
        assert val.dtype == values["dtype"], \
            f"float number type mismatch. Should be {values['dtype']}, got {val.dtype}"
    except KeyError as err:
        raise AssertionError("Validation failed due to other validation failures.") from err
    return val


def _pydantic_val_arrays(val, values):
    """Validates arrays under the same data model, i.e., sharing the same shape and dtype."""
    try:
        shape = (values["ny"], values["nx"])
        dtype = values["dtype"]
    except KeyError as err:
        raise AssertionError("Validation failed due to other validation failures.") from err

    assert val.dtype == dtype, f"Dtype mismatch. Should be {dtype}, got {val.dtype}"
    assert val.shape == shape, f"Shape mismatch. Should be {shape}, got {val.shape}"

    return val


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


class DummyDataModel:
    """A dummy class as a base for those needs the property `shape`."""

    @property
    def shape(self):
        "Shape of the arrays in this object."
        return (self.ny, self.nx)  # pylint: disable=no-member


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
            assert all(abs(diff-values["delta"])<=1e-10 for diff in diffs), "Delta does not match."
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
    values: _Tuple[float, ...]
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
    xgrad : (ny, nx) array
        Derivatives w.r.t. x at cell centers.
    ygrad : (ny, nx) array
        Derivatives w.r.t. y at cell centers.
    """
    domain: Domain
    vertices: _nplike.ndarray
    centers: _nplike.ndarray
    xfcenters: _nplike.ndarray
    yfcenters: _nplike.ndarray
    xgrad: _nplike.ndarray
    ygrad: _nplike.ndarray

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_arrays(cls, values):
        """Validations that rely on other fields' correctness."""

        # check dtype
        arrays = ["vertices", "centers", "xfcenters", "yfcenters", "xgrad", "ygrad"]
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
        assert values["xgrad"].shape == (ny, nx), "xgrad: " + msg
        assert values["ygrad"].shape == (ny, nx), "ygrad: " + msg

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
        assert _nplike.allclose(values["xgrad"], (v[:, 1:]-v[:, :-1])/dx), "xgrad vs xfcenters"

        v = values["yfcenters"]
        dy = (values["domain"].y.vertices[1:] - values["domain"].y.vertices[:-1])[:, None]
        assert _nplike.allclose(values["ygrad"], (v[1:, :]-v[:-1, :])/dy), "ygrad vs yfcenters"

        return values


class WHUHVModel(_BaseConfig, DummyDataModel):
    """Data model with keys w, hu, and v.

    Attributes
    ----------
    nx, ny : int
        They describe the shape of the arrays (not necessarily the shape of the mesh).
    dtype : str, nplike.float32, or nplike64.
        The type of floating numbers. If a string, it should be either "float32" or "float64".
    w, hu, hv : _nplike.ndarray of shape (ny, nx)
        The fluid elevation (depth + topography elevation), depth-u-velocity, and depth-v-velocity.
    """
    nx: _conint(strict=True, gt=0)
    ny: _conint(strict=True, gt=0)
    dtype: _DummyDtype
    w: _nplike.ndarray
    hu: _nplike.ndarray
    hv: _nplike.ndarray

    # validators
    _val_arrays = _validator("w", "hu", "hv", allow_reuse=True)(_pydantic_val_arrays)
    _val_valid_numbers = _validator("w", "hu", "hv", allow_reuse=True)(_pydantic_val_nan_inf)


class HUVModel(_BaseConfig, DummyDataModel):
    """Data model with keys h, u, and v.

    Attributes
    ----------
    nx, ny : int
        They describe the shape of the arrays (not necessarily the shape of the mesh).
    dtype : str, nplike.float32, or nplike64.
        The type of floating numbers. If a string, it should be either "float32" or "float64".
    h, u, v : _nplike.ndarray of shape (ny, nx)
        The fluid depth, u-velocity, and depth-v-velocity.
    """
    nx: _conint(strict=True, gt=0)
    ny: _conint(strict=True, gt=0)
    dtype: _DummyDtype
    h: _nplike.ndarray
    u: _nplike.ndarray
    v: _nplike.ndarray

    # validators
    _val_arrays = _validator("h", "u", "v", allow_reuse=True)(_pydantic_val_arrays)
    _val_valid_numbers = _validator("h", "u", "v", allow_reuse=True)(_pydantic_val_nan_inf)


class FaceOneSideModel(_BaseConfig, DummyDataModel):
    """Data model holding quantities on one side of cell faces normal to one direction.

    Attributes
    ----------
    nx, ny : int
        They describe the shape of the arrays (not necessarily the shape of the mesh).
    dtype : str, nplike.float32, or nplike64.
        The type of floating numbers. If a string, it should be either "float32" or "float64".
    w, hu, hv : _nplike.ndarray of shape (ny, nx)
        The fluid elevation (depth + topography elevation), depth-u-velocity, and depth-v-velocity.
    h, u, v : _nplike.ndarray of shape (ny, nx)
        The fluid depth, u-velocity, and depth-v-velocity.
    a : _nplike.ndarray of shape (ny, nx)
        The local speed.
    flux : WHUHVModel
        An object holding discontinuous fluxes.
    """
    nx: _conint(strict=True, gt=0)
    ny: _conint(strict=True, gt=0)
    dtype: _DummyDtype
    w: _nplike.ndarray
    hu: _nplike.ndarray
    hv: _nplike.ndarray
    h: _nplike.ndarray
    u: _nplike.ndarray
    v: _nplike.ndarray
    a: _nplike.ndarray
    flux: WHUHVModel

    # validator
    _val_arrays = _validator(
        "w", "hu", "hv", "h", "u", "v", "a", "flux", allow_reuse=True)(_pydantic_val_arrays)
    _val_valid_numbers = _validator(
        "w", "hu", "hv", "h", "u", "v", "a", allow_reuse=True)(_pydantic_val_nan_inf)


class FaceTwoSideModel(_BaseConfig):
    """Date model holding quantities on both sides of cell faces normal to one direction.

    Attributes
    ----------
    plus, minus : FaceOneSideModel
        Objects holding data on one side of each face.
    num_flux : WHUHVModel
        An object holding common/continuous flux
    """
    plus: FaceOneSideModel
    minus: FaceOneSideModel
    num_flux: WHUHVModel

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_arrays(cls, values):
        assert values["plus"].nx == values["minus"].nx, "incorrect nx size"
        assert values["plus"].nx == values["num_flux"].nx, "incorrect nx size"
        assert values["plus"].ny == values["minus"].ny, "incorrect ny size"
        assert values["plus"].ny == values["num_flux"].ny, "incorrect nx size"
        assert values["plus"].dtype == values["minus"].dtype, "incorrect dtype"
        assert values["plus"].dtype == values["num_flux"].dtype, "incorrect dtype"
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
        assert (values["x"].plus.nx-values["y"].plus.nx) == 1, "incorrect nx size"
        assert (values["y"].plus.ny-values["x"].plus.ny) == 1, "incorrect ny size"
        assert values["x"].plus.dtype == values["y"].plus.dtype, "mismatched dtype"
        return values


class Slopes(_BaseConfig):
    """Data model for linear extrapolating slopes at cell centers.

    Attributes
    ----------
    x, y : WHUHVModel
        Slopes in x and y directions.
    """
    x: WHUHVModel
    y: WHUHVModel

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_arrays(cls, values):
        assert (values["x"].nx-values["y"].nx) == 2, "incorrect nx size"
        assert (values["y"].ny-values["x"].ny) == 2, "incorrect ny size"
        assert values["x"].dtype == values["y"].dtype, "dtypes do nat match"
        return values


class States(_BaseConfig):
    """A jumbo data model of all arrays on a mesh patch.

    A brief overview of the structure in this jumbo model (ignoring scalars):
    State: {
        q: {w: ndarray hu: ndarray hv: ndarray},            # shape: (ny+2*ngh, nx+2*ngh)
        src: {w: ndarray hu: ndarray hv: ndarray},          # shape: (ny, nx)
        slp: {
            x: {w: ndarray hu: ndarray hv: ndarray},        # shape: (ny, nx+2)
            y: {w: ndarray hu: ndarray hv: ndarray},        # shape: (ny+2, nx)
        },
        rhs: {w: ndarray hu: ndarray hv: ndarray},          # shape: (ny, nx)
        face: {
            x: {                                            # shape: (ny, nx+1)
                plus: {
                    w: ndarray, hu: ndarray, hv: ndarray, h: ndarray u: ndarray v: ndarray,
                    a: ndarray,
                    flux: {w: ndarray, hu: ndarray, hv: ndarray}
                },
                minus: {
                    w: ndarray, hu: ndarray, hv: ndarray, h: ndarray u: ndarray v: ndarray,
                    a: ndarray,
                    flux: {w: ndarray, hu: ndarray, hv: ndarray}
                },
                num_flux: {w: ndarray, hu: ndarray, hv: ndarray},
            },
            y: {                                            # shape: (ny+1, nx)
                plus: {
                    w: ndarray, hu: ndarray, hv: ndarray, h: ndarray u: ndarray v: ndarray,
                    a: ndarray,
                    flux: {w: ndarray, hu: ndarray, hv: ndarray}
                },
                minus: {
                    w: ndarray, hu: ndarray, hv: ndarray, h: ndarray u: ndarray v: ndarray,
                    a: ndarray,
                    flux: {w: ndarray, hu: ndarray, hv: ndarray}
                },
                num_flux: {w: ndarray, hu: ndarray, hv: ndarray},
            }
        }
    }

    Attributes
    ----------
    domain : torchswe.utils.data.Domain
        The domain associated to this state object.
    ngh : int
        Number of ghost cell layers.
    q, src, rhs : torchswe.utils.data.WHUHVModel
        The conservative quantities, source terms, and the right-hand-side terms. Defined at cell
        centers.
    slp: torchswe.utils.data.Slopes
        The slopes for extrapolating cell-centered quantities to cell faces.
    face: torchswe.utils.data.FaceQuantityModel
        Holding quantites defined at cell faces, including continuous and discontinuous ones.
    """

    # associated domain
    domain: Domain

    # number of ghost cell layers
    ngh: _conint(strict=True, ge=0)

    # quantities defined at cell centers and faces
    q: WHUHVModel
    src: WHUHVModel
    slp: Slopes
    rhs: WHUHVModel
    face: FaceQuantityModel

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_all(cls, values):
        nx = values["domain"].x.n
        ny = values["domain"].y.n
        ngh = values["ngh"]
        dtype = values["domain"].x.dtype

        assert values["q"].shape == (ny+2*ngh, nx+2*ngh), "q: incorrect shape"
        assert values["src"].shape == (ny, nx), "src: incorrect shape"
        assert values["slp"].x.shape == (ny, nx+2), "slp.x: incorrect shape"
        assert values["slp"].y.shape == (ny+2, nx), "slp.y: incorrect shape"
        assert values["rhs"].shape == (ny, nx), "slp.y: incorrect shape"
        assert values["face"].x.plus.shape == (ny, nx+1), "face.x: incorrect shape"
        assert values["face"].y.plus.shape == (ny+1, nx), "face.y: incorrect shape"

        assert values["q"].dtype == dtype, "q: incorrect dtype"
        assert values["src"].dtype == dtype, "src: incorrect dtype"
        assert values["slp"].x.dtype == dtype, "slp.x: incorrect dtype"
        assert values["slp"].y.dtype == dtype, "slp.y: incorrect dtype"
        assert values["rhs"].dtype == dtype, "slp.y: incorrect dtype"
        assert values["face"].x.plus.dtype == dtype, "face.x: incorrect dtype"
        assert values["face"].y.plus.dtype == dtype, "face.y: incorrect dtype"

        return values

    def exchange_data(self):
        """Exchange data with neighbor MPI process to update overlapped slices."""
        # pylint: disable=too-many-locals

        sbuf, sreq, rbuf, rreq = {}, {}, {}, {}

        stags = {
            "west": {"w": 31, "hu": 32, "hv": 33}, "east": {"w": 41, "hu": 42, "hv": 43},
            "south": {"w": 51, "hu": 52, "hv": 53}, "north": {"w": 61, "hu": 62, "hv": 63},
        }

        rtags = {
            "west": {"w": 41, "hu": 42, "hv": 43}, "east": {"w": 31, "hu": 32, "hv": 33},
            "south": {"w": 61, "hu": 62, "hv": 63}, "north": {"w": 51, "hu": 52, "hv": 53},
        }

        sslcs = {
            "west": (slice(self.ngh, -self.ngh), slice(self.ngh, 2*self.ngh)),
            "east": (slice(self.ngh, -self.ngh), slice(-2*self.ngh, -self.ngh)),
            "south": (slice(self.ngh, 2*self.ngh), slice(self.ngh, -self.ngh)),
            "north": (slice(-2*self.ngh, -self.ngh), slice(self.ngh, -self.ngh)),
        }

        rslcs = {
            "west": (slice(self.ngh, -self.ngh), slice(None, self.ngh)),
            "east": (slice(self.ngh, -self.ngh), slice(-self.ngh, None)),
            "south": (slice(None, self.ngh), slice(self.ngh, -self.ngh)),
            "north": (slice(-self.ngh, None), slice(self.ngh, -self.ngh)),
        }

        # make an alias for convenience
        proc = self.domain.process

        ans = 0
        for ornt in ["west", "east", "south", "north"]:
            if proc[ornt] is not None:
                for var in ["w", "hu", "hv"]:
                    key = (ornt, var)
                    sbuf[key] = self.q[var][sslcs[ornt]].copy()
                    sreq[key] = proc.comm.Isend(sbuf[key], proc[ornt], stags[ornt][var])
                    rbuf[key] = _nplike.zeros_like(self.q[var][rslcs[ornt]])
                    rreq[key] = proc.comm.Irecv(rbuf[key], proc[ornt], rtags[ornt][var])
                    ans += 1

        # make sure send requests are done
        tstart = _time.perf_counter()
        done = 0
        while done != ans and _time.perf_counter()-tstart < 5.:
            for key, req in sreq.items():
                if key in sbuf and req.Test():
                    del sbuf[key]
                    done += 1

        # make sure if the while loop exited because of done == ans
        if done != ans:
            raise RuntimeError(f"Sending data to neighbor timeout: {sbuf.keys()}")

        # receive data from neighbors
        tstart = _time.perf_counter()
        done = 0
        while done != ans and _time.perf_counter()-tstart < 5.:
            for key, req in rreq.items():
                if key in rbuf and req.Test():
                    self.q[key[1]][rslcs[key[0]]] = rbuf[key]
                    del rbuf[key]
                    done += 1

        # make sure if the while loop exited because of done == ans
        if done != ans:
            raise RuntimeError(f"Receiving data from neighbor timeout: {rbuf.keys()}")
