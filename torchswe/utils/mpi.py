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


def cal_num_procs(world_size: int, n_cell_x: int, n_cell_y: int):
    """Calculate the number of MPI processes in x and y directions based on the number of cells.

    Arguments
    ---------
    world_size : int
        Total number of MPI processes.
    n_cell_x, n_cell_y : int
        Number of cells globally.

    Retunrs
    -------
    n_proc_x, n_proc_y : int
        Number of MPI processes in x and y directions.

    Notes
    -----
    Based on the following desired conditions (for perfect situation):

    (1) n_proc_x * n_proc_y = world_size
    (2) n_proc_x / n_cell_x = n_proc_y / n_cell_y

    From (2), we get n_proc_y = n_proc_x * n_cell_y / n_cell_x. Substitute it into (1), we get
    n_proc_x * n_proc_x * n_cell_y / n_cell_x = world_size. Then, finally, we have n_proc_x = sqrt(
    world_size * n_cell_x / n_cell_y). Round n_proc_x to get an integer.

    If the rounded n_proc_x is 0, then we set it to 1.

    Finally, when determining n_proc_y, we decrease n_proc_x until we find a n_proc_x that can
    exactly divide world_size.
    """

    # start with this number for n_proc_x
    n_proc_x = max(int(0.5+(n_cell_x*world_size/n_cell_y)**0.5), 1)

    # decrease n_proc_x until it can exactly divide world_size
    while world_size % n_proc_x != 0:
        n_proc_x -= 1

    # calculate n_proc_y
    n_proc_y = world_size // n_proc_x
    assert world_size == n_proc_x * n_proc_y  # sanity check

    if n_cell_x > n_cell_y and n_proc_x < n_proc_y:
        n_proc_x, n_proc_y = n_proc_y, n_proc_x  # swap

    return n_proc_x, n_proc_y


def cal_local_cell_range(n_proc_x: int, n_proc_y: int, n_cell_x: int, n_cell_y: int, rank: int):
    """Calculate the range of local cells on a target MPI process.

    Arguments
    ---------
    n_proc_x, n_proc_y : int
        Number of MPI processes in x and y directions.
    n_cell_x, n_cell_y : int
        Number of global cells in x and y directions.
    rank : int
        The rank of the process of which we want to calculate local cell numbers.

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

    assert rank < (n_proc_x * n_proc_y)

    # identify the location of this rank in the Cartesian topology
    rank_x = rank % n_proc_x
    rank_y = rank // n_proc_x

    # x direction
    base = n_cell_x // n_proc_x
    remainder = n_cell_x % n_proc_x
    local_ibg = base * rank_x + min(rank_x, remainder)
    local_ied = base * (rank_x+1) + min(rank_x+1, remainder)

    # x direction
    base = n_cell_y // n_proc_y
    remainder = n_cell_y % n_proc_y
    local_jbg = base * rank_y + min(rank_y, remainder)
    local_jed = base * (rank_y+1) + min(rank_y+1, remainder)

    return local_ibg, local_ied, local_jbg, local_jed


class BlockMPI(BaseConfig):
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
    ngh: conint(strict=True, gt=0)

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
