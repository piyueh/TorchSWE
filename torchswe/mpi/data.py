#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Data models used for MPI runs.
"""
# pylint: disable=too-few-public-methods, no-self-argument, no-self-use, unnecessary-pass
import time
import logging as _logging
from typing import Optional as _Optional, Tuple as _Tuple
from mpi4py import MPI as _MPI
from pydantic import root_validator as _root_validator, validator as _validator, conint as _conint
from torchswe import nplike as _nplike
from torchswe.utils.config import BaseConfig as _BaseConfig
from torchswe.utils.misc import DummyDtype as _DummyDtype, interpolate as _interpolate
from torchswe.utils.data import Gridlines as _Gridlines, States as _States
from torchswe.utils.data import Topography as _Topography, DummyDataModel as _DummyDataModel
from torchswe.utils.data import get_empty_states as _get_empty_states, get_gridline as _get_gridline
from torchswe.utils.netcdf import read as _ncread

_logger = _logging.getLogger("torchswe.mpi.data")


class Block(_BaseConfig, _DummyDataModel):
    """A base class containing the info of a process in a 2D Cartesian topology.

    Attributes
    ----------
    comm : MPI.Comm
        The communicator.
    proc_shape : (int, int)
        The shape of the 2D Cartesian topology of the processes. y-direction first.
    proc_loc : (int, int)
        The location of this process in the 2D Cartesian topology. y-direction first.
    west, east, south, north : int or None
        The ranks of the neighbors. If None, means the border is on domain boundary.
    gnx, gny : int
        The global numbers of cells.
    nx, ny : int
        The local numbers of cells.
    ibg, ied : int
        The global indices of the first and last cells owned by this process in x direction.
    jbg, jed : int
        The global indices of the first and last cells owned by this process in y direction.
    ngh : int
        The number of ghost cells outside boundary. The "boundary" also includes the internal
        boundaries between two blocks. Required when exchanging data between blocks.
    """

    # mpi related
    comm: _MPI.Comm
    proc_shape: _Tuple[_conint(strict=True, ge=0), _conint(strict=True, ge=0)]
    proc_loc: _Tuple[_conint(strict=True, ge=0), _conint(strict=True, ge=0)]
    west: _Optional[_conint(strict=True, ge=0)] = ...
    east: _Optional[_conint(strict=True, ge=0)] = ...
    south: _Optional[_conint(strict=True, ge=0)] = ...
    north: _Optional[_conint(strict=True, ge=0)] = ...

    # global-grid related
    gnx: _conint(strict=True, gt=0)
    gny: _conint(strict=True, gt=0)

    # local-grid related
    nx: _conint(strict=True, gt=0)
    ny: _conint(strict=True, gt=0)
    ibg: _conint(strict=True, ge=0)
    ied: _conint(strict=True, gt=0)
    jbg: _conint(strict=True, ge=0)
    jed: _conint(strict=True, gt=0)

    # ghost cells (the same for internal boundaries and domain boundaries)
    ngh: _conint(strict=True, ge=0)

    def get_block(self):
        """Get a copy of this block.

        Usually used by derived classes.
        """
        return Block(
            comm=self.comm, proc_shape=self.proc_shape, proc_loc=self.proc_loc, west=self.west,
            east=self.east, south=self.south, north=self.north, gnx=self.gnx, gny=self.gny,
            nx=self.nx, ny=self.ny, ibg=self.ibg, ied=self.ied, jbg=self.jbg, jed=self.jed,
            ngh=self.ngh)

    @_validator("comm")
    def _val_comm(cls, val):
        recvbuf = val.allgather(val.Get_rank())
        assert recvbuf == list(range(val.Get_size())), "Communicator not working."
        return val

    @_validator("proc_shape")
    def _val_proc_shape(cls, val, values):
        if "comm" not in values:  # comm did not pass the validation
            return val
        world_size = values["comm"].Get_size()
        assert val[0] * val[1] == world_size, "shape: {}, world_size: {}".format(val, world_size)
        return val

    @_validator("proc_loc")
    def _val_proc_loc(cls, val, values):
        if "proc_shape" not in values:  # proc_shape didn't pass the validation
            return val
        shp = values["proc_shape"]
        assert val[0] < shp[0], "proc_loc[0]: {}, proc_shape[0]: {}".format(val[0], shp[0])
        assert val[1] < shp[1], "proc_loc[1]: {}, proc_shape[1]: {}".format(val[1], shp[1])
        return val

    @_validator("nx", "ny")
    def _val_sum_n(cls, val, values, field):
        if "proc_shape" not in values:  # proc_shape didn't pass the validation
            return val

        tgt = "gnx" if field.name == "nx" else "gny"
        key = 0 if field.name == "nx" else 1
        total = values["comm"].allreduce(val)
        assert total == values[tgt] * values["proc_shape"][key], "Sum({}) != {}: {}, {}".format(
            field.name, tgt, total//values["proc_shape"][key], values[tgt])
        return val

    @_validator("ied", "jed")
    def _val_end(cls, val, values, field):
        # pylint: disable=invalid-name
        other = "ibg" if field.name == "ied" else "jbg"
        if other not in values:  # ibg or jbg didn't pass the validation
            return val
        assert val > values[other], \
            "{} should > {}: {}, {}".format(field.name, other, val, values[other])
        return val

    @_root_validator
    def _val_neighbors(cls, values):

        # try to communicate with neighbors
        buff = [values["comm"].Get_rank()] + [values[k] for k in ["ibg", "ied", "jbg", "jed"]]
        send_tags = {"west": 31, "east": 41, "south": 51, "north": 61}
        recv_tags = {"west": 41, "east": 31, "south": 61, "north": 51}
        reqs = {}
        ready = {}
        reply = {}

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
                ready[key], reply[key] = val.test()

            counter += 1
            time.sleep(1)

        for key, state, ans in zip(ready.keys(), ready.values(), reply.values()):
            # first make sure we got message from this neighbor
            assert state, "Neighbor in {} (rank {}) did not answer.".format(key, values[key])
            # second make sure this is the correct neighbor (is this redundant?)
            assert ans[0] == values[key], "{}: {}, {}".format(key, ans[0], values[key])

            # check the indices from the neighbor in the west
            if key == "west":
                assert ans[2] == buff[1], "West's ied != my ibg: {}, {}".format(ans[2], buff[1])
                assert ans[3] == buff[3], "West's jbg != my jbg: {}, {}".format(ans[3], buff[3])
                assert ans[4] == buff[4], "West's jed != my jed: {}, {}".format(ans[4], buff[4])
            elif key == "east":
                assert ans[1] == buff[2], "East's ibg != my ied: {}, {}".format(ans[1], buff[2])
                assert ans[3] == buff[3], "East's jbg != my jbg: {}, {}".format(ans[3], buff[3])
                assert ans[4] == buff[4], "East's jed != my jed: {}, {}".format(ans[4], buff[4])
            elif key == "south":
                assert ans[1] == buff[1], "South's ibg != my ibg: {}, {}".format(ans[1], buff[1])
                assert ans[2] == buff[2], "South's ied != my ied: {}, {}".format(ans[2], buff[2])
                assert ans[4] == buff[3], "South's jed != my jbg: {}, {}".format(ans[4], buff[3])
            elif key == "north":
                assert ans[1] == buff[1], "North's ibg != my ibg: {}, {}".format(ans[1], buff[1])
                assert ans[2] == buff[2], "North's ied != my ied: {}, {}".format(ans[2], buff[2])
                assert ans[3] == buff[4], "North's jbg != my jed: {}, {}".format(ans[3], buff[4])
            else:  # should be redundant, but I prefer keep it
                raise ValueError("Unrecoganized key: {}".format(key))

        return values


class States(_States, Block):
    """MPI version of the States data model.

    Attributes
    ----------
    The following attributes are inherented from torchswe.mpi.data.Block:

    comm : MPI.Comm
    proc_shape, proc_loc : (int, int)
    west, east, south, north : int or None
    gnx, gny, nx, ny, ngh : int
    ibg, ied, jbg, jed : int

    The following attributes are inherented from torchswe.utils.data.States:

    dtype : torchswe.utils.misc.DummyDtype
    q, src, rhs : torchswe.utils.data.WHUHVModel
    slp: torchswe.utils.data.Slopes
    face: torchswe.utils.data.FaceQuantityModel
    """

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
            "west": (slice(None), slice(self.ngh, 2*self.ngh)),
            "east": (slice(None), slice(-2*self.ngh, -self.ngh)),
            "south": (slice(self.ngh, 2*self.ngh), slice(None)),
            "north": (slice(-2*self.ngh, -self.ngh), slice(None)),
        }

        rslcs = {
            "west": (slice(None), slice(None, self.ngh)),
            "east": (slice(None), slice(-self.ngh, None)),
            "south": (slice(None, self.ngh), slice(None)),
            "north": (slice(-self.ngh, None), slice(None)),
        }

        ans = 0
        for ornt in ["west", "east", "south", "north"]:
            if self[ornt] is not None:
                for var in ["w", "hu", "hv"]:
                    key = (ornt, var)
                    sbuf[key] = self.q[var][sslcs[ornt]].copy()
                    sreq[key] = self.comm.Isend(sbuf[key], self[ornt], stags[ornt][var])
                    rbuf[key] = _nplike.zeros_like(self.q[var][rslcs[ornt]])
                    rreq[key] = self.comm.Irecv(rbuf[key], self[ornt], rtags[ornt][var])
                    ans += 1

        tstart = time.perf_counter()
        done = 0
        while done != ans and time.perf_counter()-tstart < 5.:
            for key, req in rreq.items():
                if key in rbuf and req.Test():
                    self.q[key[1]][rslcs[key[0]]] = rbuf[key]
                    del rbuf[key]
                    done += 1

        # make usre if the while loop exited because of done == ans
        if done != ans:
            raise RuntimeError("Receiving data from neighbor timeout: {}".format(rbuf.keys()))

        # only leave this functio when send requests are also done
        tstart = time.perf_counter()
        done = 0
        while done != ans and time.perf_counter()-tstart < 5.:
            for key, req in sreq.items():
                if key in sbuf and req.Test():
                    del sbuf[key]
                    done += 1

        # make usre if the while loop exited because of done == ans
        if done != ans:
            raise RuntimeError("Sending data from neighbor timeout: {}".format(sbuf.keys()))


class Gridlines(_Gridlines, Block):
    """MPI version of the Gridlines data model.

    Attributes
    ----------
    gxbg, gxed, gybg, gyed : float
        The global bounds in x and y directions.

    The following attributes are inherented from torchswe.mpi.data.Block:

    comm : MPI.Comm
    proc_shape, proc_loc : (int, int)
    west, east, south, north : int or None
    gnx, gny, nx, ny, ngh : int
    ibg, ied, jbg, jed : int

    The following attributes are inherented from torchswe.utils.data.Gridlines:

    x, y : Gridline
    y : Gridline
    t : List[float]
    """
    gxbg: float
    gxed: float
    gybg: float
    gyed: float


def cal_num_procs(world_size: int, gnx: int, gny: int):
    """Calculate the number of MPI processes in x and y directions based on the number of cells.

    Arguments
    ---------
    world_size : int
        Total number of MPI processes.
    gnx, gny : int
        Number of cells globally.

    Retunrs
    -------
    pnx, pny : int
        Number of MPI processes in x and y directions. Note the order of x and y processes in the
        return.

    Notes
    -----
    Based on the following desired conditions (for perfect situation):

    (1) pnx * pny = world_size
    (2) pnx / gnx = pny / gny

    From (2), we get pny = pnx * gny / gnx. Substitute it into (1), we get
    pnx * pnx * gny / gnx = world_size. Then, finally, we have pnx = sqrt(
    world_size * gnx / gny). Round pnx to get an integer.

    If the rounded pnx is 0, then we set it to 1.

    Finally, when determining pny, we decrease pnx until we find a pnx that can
    exactly divide world_size.
    """

    # start with this number for pnx
    pnx = max(int(0.5+(gnx*world_size/gny)**0.5), 1)

    # decrease pnx until it can exactly divide world_size
    while world_size % pnx != 0:
        pnx -= 1

    # calculate pny
    pny = world_size // pnx
    assert world_size == pnx * pny  # sanity check

    if gnx > gny and pnx < pny:
        pnx, pny = pny, pnx  # swap

    return pnx, pny


def cal_proc_loc_from_rank(pnx: int, rank: int):
    """Calculate the location of a rank in a 2D Cartesian topology.

    A very simple assignment method. Using this function to unify the way of assignment throughout
    an application.

    Arguments
    ---------
    pnx : int
        Number of MPI processes in x directions.
    rank : int
        The rank of the process of which we want to calculate local cell numbers.

    Returns
    -------
    pi, pj : int
        The indices of the rank in the 2D MPI topology in x and y directions.
    """
    return rank % pnx, rank // pnx


def cal_rank_from_proc_loc(pnx: int, pi: int, pj: int):
    """Given (pj, pi), calculate the rank.

    Arguments
    ---------
    pnx : int
        Number of MPI processes in x directions.
    pi, pj : int
        The location indices of this process in x and y direction in the 2D process topology.

    Returns
    -------
    rank : int
    """
    # pylint: disable=invalid-name
    return pj * pnx + pi


def cal_local_cell_range(pnx: int, pny: int, pi: int, pj: int, gnx: int, gny: int):
    """Calculate the range of local cells on a target MPI process.

    Arguments
    ---------
    pnx, pny : int
        Number of MPI processes in x and y directions.
    pi, pj : int
        The indices of a rank in the 2D MPI topology in x and y directions.
    gnx, gny : int
        Number of global cells in x and y directions.

    Returns
    -------
    local_ibg, local_ied : int
        The global indices of the first and the last cells in x directions.
    local_jbg, local_jed : int
        The global indices of the first and the last cells in y directions.

    Notes
    -----
    Though we say ied and jed are the indices of the last cells, they are actually the indices of
    the last cells plus 1, so that we can directly use them in slicing, range, iterations, etc.
    without manually adding one in these use case.
    """
    # pylint: disable=invalid-name

    assert pi < pnx
    assert pj < pny

    # x direction
    base = gnx // pnx
    remainder = gnx % pnx
    local_ibg = base * pi + min(pi, remainder)
    local_ied = base * (pi+1) + min(pi+1, remainder)

    # x direction
    base = gny // pny
    remainder = gny % pny
    local_jbg = base * pj + min(pj, remainder)
    local_jed = base * (pj+1) + min(pj+1, remainder)

    return local_ibg, local_ied, local_jbg, local_jed


def cal_neighbors(pnx: int, pny: int, pi: int, pj: int, rank: int):
    """Calculate neighbors' rank.

    Arguments
    ---------
    pnx, pny : int
        Number of MPI processes in x and y directions.
    pi, pj : int
        The indices of a rank in the 2D MPI topology in x and y directions.
    rank : int
        The rank of the process of which we want to calculate local cell numbers.

    Returns
    -------
    west, east, south, north : int or None
        The ranks of neighbors in these direction. If None, it means the current rank is on the
        domain boundary.
    """
    # pylint: disable=invalid-name
    west = rank - 1 if pi != 0 else None
    east = rank + 1 if pi != pnx-1 else None
    south = rank - pnx if pj != 0 else None
    north = rank + pnx if pj != pny-1 else None
    return west, east, south, north


def get_block(comm: _MPI.Comm, gnx: int, gny: int, ngh: int):
    """Get an instance of BloclMPI for the current MPI process.

    Arguments
    ---------
    comm : MPI.Comm
        The communicator.
    gnx, gny : int
        The global numbers of cells.
    ngh : int
        The number of ghost cells outside boundary. The "boundary" also includes the internal
        boundaries between two blocks. Required when exchanging data between blocks.

    Returns
    -------
    An instance of Block.
    """
    # pylint: disable=invalid-name

    data = {"ngh": ngh, "gnx": gnx, "gny": gny, "comm": comm}

    pnx, pny = cal_num_procs(comm.Get_size(), gnx, gny)

    pi, pj = cal_proc_loc_from_rank(pnx, comm.Get_rank())

    data["west"], data["east"], data["south"], data["north"] = \
        cal_neighbors(pnx, pny, pi, pj, comm.Get_rank())

    data["ibg"], data["ied"], data["jbg"], data["jed"] = \
        cal_local_cell_range(pnx, pny, pi, pj, gnx, gny)

    data["nx"], data["ny"] = data["ied"] - data["ibg"], data["jed"] - data["jbg"]

    data["proc_shape"] = (pny, pnx)
    data["proc_loc"] = (pj, pi)

    return Block(**data)


def get_empty_states(comm: _MPI.Comm, gnx: int, gny: int, ngh: int, dtype: str):
    """Get an instance of States for the current process with all-zero data arrays.

    This overloads the serial version of `get_empty_states`.

    Arguments
    ---------
    comm : MPI.Comm
        The communicator.
    gnx, gny : int
        The global numbers of cells.
    ngh : int
        The number of ghost cells outside boundary. The "boundary" also includes the internal
        boundaries between two blocks. Required when exchanging data between blocks.
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    An instance of States.
    """
    block = get_block(comm, gnx, gny, ngh)
    states = _get_empty_states(block.nx, block.ny, block.ngh, dtype)
    kwargs = {**states.__dict__, **block.__dict__}  # to resolve duplicated attrs, e.g., nx, ny, etc
    return States(**kwargs)


def get_gridlines(comm, gnx, gny, gxbg, gxed, gybg, gyed, t, dtype):
    """Get an MPI-version of Gridlines.

    It's the same object as the one used for serial code, but now it stores local grid info owned
    by the current process.

    Arguments
    ---------
    comm : MPI.Comm
    gnx, gny : int
    gxbg, gxed : float
    gybg, gyed : float
    t : list/tuple of floats
    dtype : str, nplike.float32, or nplike.float64

    Returns
    -------
    An instance of torchswe.mpi.data.Gridlines.
    """
    # pylint: disable=too-many-arguments, invalid-name, too-many-locals

    block = get_block(comm, gnx, gny, 0)

    delta = [(gxed - gxbg) / gnx, (gyed - gybg) / gny]
    xbg, xed = block.ibg * delta[0] + gxbg, block.ied * delta[0] + gxbg
    ybg, yed = block.jbg * delta[1] + gybg, block.jed * delta[1] + gybg
    assert abs(block.nx*delta[0]-xed+xbg) <= 1e-10
    assert abs(block.ny*delta[1]-yed+ybg) <= 1e-10

    data = {"gxbg": gxbg, "gxed": gxed, "gybg": gybg, "gyed": gyed}
    data["x"] = _get_gridline("x", block.nx, xbg, xed, dtype)
    data["y"] = _get_gridline("y", block.ny, ybg, yed, dtype)
    data["t"] = t
    data.update(block.__dict__)

    return Gridlines(**data)


def get_topography(comm, topofile, key, grid_xv, grid_yv, dtype):
    """Get a Topography object from a config object.

    Arguments
    ---------
    comm : MPI.Comm
    topofile : str or PathLike
    key : str
    grid_xv, grid_yv : nplike.ndarray
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    topo : torchswe.utils.data.Topography
    """
    # pylint: disable=too-many-locals
    dtype = _DummyDtype.validator(dtype)
    assert dtype == grid_xv.dtype
    assert dtype == grid_yv.dtype

    dem, _ = _ncread(
        topofile, [key], [grid_xv[0], grid_xv[-1], grid_yv[0], grid_yv[-1]],
        parallel=True, comm=comm)

    vert = dem[key]

    # see if we need to do interpolation
    try:
        interp = not (_nplike.allclose(grid_xv, dem["x"]) and _nplike.allclose(grid_yv, dem["y"]))
    except ValueError:  # assume thie excpetion means a shape mismatch
        interp = True

    # unfortunately, we need to do interpolation in such a situation
    if interp:
        _logger.warning("Grids do not match. Doing spline interpolation.")
        vert = _nplike.array(_interpolate(dem["x"], dem["y"], vert.T, grid_xv, grid_yv).T)

    # cast to desired float type
    vert = vert.astype(dtype)

    # topography elevation at cell centers through linear interpolation
    cntr = vert[:-1, :-1] + vert[:-1, 1:] + vert[1:, :-1] + vert[1:, 1:]
    cntr /= 4

    # topography elevation at cell faces' midpoints through linear interpolation
    xface = (vert[:-1, :] + vert[1:, :]) / 2.
    yface = (vert[:, :-1] + vert[:, 1:]) / 2.

    # gradient at cell centers through central difference; here allows nonuniform grids
    # this function does not assume constant cell sizes
    xgrad = (xface[:, 1:] - xface[:, :-1]) / (grid_xv[1:] - grid_xv[:-1])[None, :]
    ygrad = (yface[1:, :] - yface[:-1, :]) / (grid_yv[1:] - grid_yv[:-1])[:, None]

    # initialize DataModel and let pydantic validates data
    return _Topography(
        nx=len(grid_xv)-1, ny=len(grid_yv)-1, dtype=dtype, vert=vert, cntr=cntr,
        xface=xface, yface=yface, xgrad=xgrad, ygrad=ygrad)
