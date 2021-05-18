#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Functions and classes related to MPI runs.
"""
# pylint: disable=too-few-public-methods, no-self-argument, no-self-use, unnecessary-pass
import time
from typing import Optional, Tuple
from mpi4py import MPI
from pydantic import root_validator, validator, conint
from torchswe.utils.config import BaseConfig
from torchswe.utils.data import States, DummyDataModel, get_empty_states


class BlockMPI(BaseConfig, DummyDataModel):
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
    comm: MPI.Comm
    proc_shape: Tuple[conint(strict=True, ge=0), conint(strict=True, ge=0)]
    proc_loc: Tuple[conint(strict=True, ge=0), conint(strict=True, ge=0)]
    west: Optional[conint(strict=True, ge=0)] = ...
    east: Optional[conint(strict=True, ge=0)] = ...
    south: Optional[conint(strict=True, ge=0)] = ...
    north: Optional[conint(strict=True, ge=0)] = ...

    # global-grid related
    gnx: conint(strict=True, gt=0)
    gny: conint(strict=True, gt=0)

    # local-grid related
    nx: conint(strict=True, gt=0)
    ny: conint(strict=True, gt=0)
    ibg: conint(strict=True, ge=0)
    ied: conint(strict=True, gt=0)
    jbg: conint(strict=True, ge=0)
    jed: conint(strict=True, gt=0)

    # ghost cells (the same for internal boundaries and domain boundaries)
    ngh: conint(strict=True, ge=2)

    @validator("comm")
    def _val_comm(cls, val):
        recvbuf = val.allgather(val.Get_rank())
        assert recvbuf == list(range(val.Get_size())), "Communicator not working."
        return val

    @validator("proc_shape")
    def _val_proc_shape(cls, val, values):
        if "comm" not in values:  # comm did not pass the validation
            return val
        world_size = values["comm"].Get_size()
        assert val[0] * val[1] == world_size, "shape: {}, world_size: {}".format(val, world_size)
        return val

    @validator("proc_loc")
    def _val_proc_loc(cls, val, values):
        if "proc_shape" not in values:  # proc_shape didn't pass the validation
            return val
        shp = values["proc_shape"]
        assert val[0] < shp[0], "proc_loc[0]: {}, proc_shape[0]: {}".format(val[0], shp[0])
        assert val[1] < shp[1], "proc_loc[1]: {}, proc_shape[1]: {}".format(val[1], shp[1])
        return val

    @validator("nx", "ny")
    def _val_sum_n(cls, val, values, field):
        if "proc_shape" not in values:  # proc_shape didn't pass the validation
            return val

        tgt = "gnx" if field.name == "nx" else "gny"
        key = 0 if field.name == "nx" else 1
        total = values["comm"].allreduce(val)
        assert total == values[tgt] * values["proc_shape"][key], "Sum({}) != {}: {}, {}".format(
            field.name, tgt, total//values["proc_shape"][key], values[tgt])
        return val

    @validator("ied", "jed")
    def _val_end(cls, val, values, field):
        # pylint: disable=invalid-name
        other = "ibg" if field.name == "ied" else "jbg"
        if other not in values:  # ibg or jbg didn't pass the validation
            return val
        assert val > values[other], \
            "{} should > {}: {}, {}".format(field.name, other, val, values[other])
        return val

    @root_validator
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


class StatesMPI(States, BlockMPI):
    """MPI version of the States data model.

    Attributes
    ----------
    The following attributes are inherented from torchswe.utils.mpi.BlockMPI:

    comm : MPI.Comm
    proc_shape : (int, int)
    proc_loc : (int, int)
    west, east, south, north : int or None
    gnx, gny : int
    nx, ny : int
    ibg, ied : int
    jbg, jed : int
    ngh : int

    The following attributes are inherented from torchswe.utils.data.States:

    dtype: torchswe.utils.dummy.DummyDtype
    q: torchswe.utils.data.WHUHVModel
    src: torchswe.utils.data.WHUHVModel
    slp: torchswe.utils.data.Slopes
    rhs: torchswe.utils.data.WHUHVModel
    face: torchswe.utils.data.FaceQuantityModel
    """
    pass


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


def cal_proc_loc(pnx: int, rank: int):
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


def get_blockmpi(comm: MPI.Comm, gnx: int, gny: int, ngh: int):
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
    An instance of BlockMPI.
    """
    # pylint: disable=invalid-name

    data = {"ngh": ngh, "gnx": gnx, "gny": gny, "comm": comm}

    pnx, pny = cal_num_procs(comm.Get_size(), gnx, gny)

    pi, pj = cal_proc_loc(pnx, comm.Get_rank())

    data["west"], data["east"], data["south"], data["north"] = \
        cal_neighbors(pnx, pny, pi, pj, comm.Get_rank())

    data["ibg"], data["ied"], data["jbg"], data["jed"] = \
        cal_local_cell_range(pnx, pny, pi, pj, gnx, gny)

    data["nx"], data["ny"] = data["ied"] - data["ibg"], data["jed"] - data["jbg"]

    data["proc_shape"] = (pny, pnx)
    data["proc_loc"] = (pj, pi)

    return BlockMPI(**data)


def get_empty_statesmpi(comm: MPI.Comm, gnx: int, gny: int, ngh: int, dtype: str):
    """Get an instance of StatesMPI for the current process with all-zero data arrays.

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
    An instance of StatesMPI.
    """
    block = get_blockmpi(comm, gnx, gny, ngh)
    states = get_empty_states(block.nx, block.ny, block.ngh, dtype)
    kwargs = {**states.__dict__, **block.__dict__}  # to resolve duplicated attrs, e.g., nx, ny, etc
    return StatesMPI(**kwargs)
