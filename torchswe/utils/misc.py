#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""A collection of some misc stuff.
"""
import logging as _logging
import collections as _collections

from scipy.interpolate import RectBivariateSpline as _RectBivariateSpline
from mpi4py import MPI as _MPI
from torchswe import nplike as _nplike


_logger = _logging.getLogger("torchswe.utils.misc")


class DummyDict(_collections.UserDict):  # pylint: disable=too-many-ancestors
    """A dummy dict of which the data can be accessed as attributes.
    """

    def __init__(self, init_attrs=None, /, **kwargs):
        # pylint: disable=super-init-not-called
        object.__setattr__(self, "data", {})

        if init_attrs is not None:
            self.data.update(init_attrs)

        if kwargs:
            self.data.update(kwargs)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __delattr__(self, key):
        self.__delitem__(key)


class DummyDtype:  # pylint: disable=too-few-public-methods
    """A dummy dtype to make all NumPy, Legate, CuPy and PyTorch happy.

    PyTorch is the least numpy-compatible. This class is actually prepared for PyTorch!
    """
    @classmethod
    def __get_validators__(cls):
        """Iteratorate throuh available validators for pydantic's data model"""
        yield cls.validator

    @classmethod
    def validator(cls, v):  # pylint: disable=invalid-name
        """validator."""

        msg = "Either nplike.float32/nplike.float64 or their str representations."

        if isinstance(v, str):
            try:
                return {"float32": _nplike.float32, "float64": _nplike.float64}[v]
            except KeyError as err:
                raise ValueError(msg) from err
        elif v not in (_nplike.float32, _nplike.float64):
            raise ValueError(msg)

        return v


def interpolate(x_in, y_in, data_in, x_out, y_out):
    """A wrapper to interpolation with scipy.interpolate.RectBivariateSpline.

    scipy.interpolate.RectBivariateSpline only accpets vanilla NumPy array. Different np-like
    backends use different method to convert to vanilla numpy.ndarray. This function unifies them
    and the interpolation.

    The return is always vanilla numpy.ndarray.

    Arguments
    ---------
    x_in, y_in, data_in : nplike.ndarray
        The first three inputs to scipy.interpolate.RectBivariateSpline.
    x_out, y_out : nplike.ndarray
        The first two inputs to scipy.interpolate.RectBivariateSpline.__call__.

    Returns
    -------
    data_out : numpy.ndarray
        The output of scipy.interpolate.RectBivariateSpline.__call__.
    """

    try:
        func = _RectBivariateSpline(x_in, y_in, data_in, kx=1, ky=1)
    except TypeError as err:
        if str(err).startswith("Implicit conversion to a NumPy array is not allowe"):
            func = _RectBivariateSpline(x_in.get(), y_in.get(), data_in.get())  # cupy
            x_out = x_out.get()
            y_out = y_out.get()
        elif str(err).startswith("can't convert cuda:"):
            func = _RectBivariateSpline(
                x_in.cpu().numpy(), y_in.cpu().numpy(), data_in.cpu().numpy())  # pytorch
            x_out = x_out.cpu().numpy()
            y_out = y_out.cpu().numpy()
        else:
            raise

    return func(x_out, y_out)


def cal_num_procs(world_size: int, gnx: int, gny: int):
    """Calculate the number of MPI ranks in x and y directions based on the number of cells.

    Arguments
    ---------
    world_size : int
        Total number of MPI ranks.
    gnx, gny : int
        Number of cells globally.

    Retunrs
    -------
    pnx, pny : int
        Number of MPI ranks in x and y directions. Note the order of pnx and pny.

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

    Arguments
    ---------
    pnx : int
        Number of MPI ranks in x directions.
    rank : int
        The rank of which we want to calculate local cell numbers.

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
        Number of MPI ranks in x directions.
    pi, pj : int
        The location indices of this rank in x and y direction in the 2D Cartesian topology.

    Returns
    -------
    rank : int
    """
    # pylint: disable=invalid-name
    return pj * pnx + pi


def cal_local_gridline_range(pn: int, pi: int, gn: int):
    """Calculate the range of local cells on a target MPI rank.

    Arguments
    ---------
    pn : int
        Number of MPI ranks the target direction.
    pi : int
        The indices of this rank in the target direction.
    gn : int
        Number of global cells in the target direction.

    Returns
    -------
    local_n : int
        Number of cells in this local gridline.
    local_ibg, local_ied : int
        The global indices of the first and the last cells in the target direction.

    Notes
    -----
    Though we say local_ied is the indices of the last cells, they are actually the indices of
    the last cells plus 1, so that we can directly use them in slicing, range, iterations, etc.
    """
    # pylint: disable=invalid-name
    assert pi < pn
    base = gn // pn
    remainder = gn % pn
    local_ibg = base * pi + min(pi, remainder)
    local_ied = base * (pi + 1) + min(pi+1, remainder)
    return local_ied - local_ibg, local_ibg, local_ied


def cal_neighbors(pnx: int, pny: int, pi: int, pj: int, rank: int):
    """Calculate neighbors' rank.

    Arguments
    ---------
    pnx, pny : int
        Number of MPI ranks in x and y directions.
    pi, pj : int
        The indices of a rank in the 2D MPI topology in x and y directions.
    rank : int
        The rank which we want to calculate local cell numbers.

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


def set_device(comm: _MPI.Comm):
    """Set the GPU device ID for this rank when using CuPy backend.

    We try our best to make neighber ranks use the same GPU.

    Arguments
    ---------
    comm : mpi4py.MPI.Comm
        The communicator.
    """

    assert _nplike.__name__ == "cupy", "This function only works with CuPy backend."

    # get number of GPUs on this particular compute node
    n_gpus = _nplike.cuda.runtime.getDeviceCount()

    # get the info of this rank on this local compute node
    local_name = _MPI.Get_processor_name()
    local_comm = comm.Split_type(_MPI.COMM_TYPE_SHARED)
    local_size = local_comm.Get_size()
    local_rank = local_comm.Get_rank()

    # set the corresponding gpu id for this rank
    group_size = local_size // n_gpus
    remains = local_size % n_gpus

    if local_rank < (group_size + 1) * remains:  # groups having 1 more rank 'cause of the remainder
        my_gpu = local_rank // (group_size + 1)
    else:
        my_gpu = (local_rank - remains) // group_size

    _nplike.cuda.runtime.setDevice(my_gpu)

    _logger.debug(
        "node name: %s; local size:%d; local rank: %d; gpu id: %d",
        local_name, local_size, local_rank, my_gpu)


def find_cell_index(x, lower, upper, delta):
    """Find the local index in 1D so that lower + i * delta <= x < lower + (i + 1) * delta.

    Arguments
    ---------
    x : float
        The target coordinate.

    """
    if x < lower or x >= upper:
        return None
    return int((x-lower)//delta)  # cupy backend may return scalar cupy.ndarray, so we use `int`


def find_index_bound(x, y, extent):
    """Given x and y gridlines, find the index bounds covering the desired extent.

    Arguments
    ---------
    x, y : nplike.ndarray
        The coordinates.
    extenr : tuple of 4 floats
        The queried extent in format (xmin, xmax, ymin, ymax) or (west, east, south, north).

    Returns
    -------
    ibg, ied, jbg, jed : int
        Slice [ibg:ied] and [jbg:jed] cover desired extent.
    """

    dxl, dxr = (x[1] - x[0]) / 2.0, (x[-1] - x[-2]) / 2.0
    dyl, dyr = (y[1] - y[0]) / 2.0, (y[-1] - y[-2]) / 2.0
    assert x[0] - dxl <= extent[0], f"{x[0]-dxl} is not smaller than or equal to {extent[0]}"
    assert x[-1] + dxr >= extent[1], f"{x[-1]+dyr} is not greater than or equal to {extent[1]}"
    assert y[0] - dyl <= extent[2], f"{y[0]-dyl} is not smaller than or equal to {extent[2]}"
    assert y[-1] + dyr >= extent[3], f"{y[-1]+dyr} is not greater than or equal to {extent[3]}"

    # tol (single precision and double precision will use different tolerance)
    try:
        tol = 1e-12 if x.dtype == "float64" else 1e-6
    except AttributeError:  # not a numpy or cupy datatype -> Python's native floating point
        tol = 1e-12

    # left-search the start/end indices containing the provided domain (with rounding errors)
    ibg, ied = _nplike.searchsorted(x+tol, _nplike.asarray(extent[:2]))
    jbg, jed = _nplike.searchsorted(y+tol, _nplike.asarray(extent[2:]))

    # torch's searchsorted signature differs, so no right search; manual adjustment instead
    ied = min(len(x) - 1, ied)
    jed = min(len(y) - 1, jed)

    # make sure the target domain is big enough for interpolation, except for edge cases
    ibg = int(ibg-1) if x[ibg]-tol > extent[0] else int(ibg)
    ied = int(ied+1) if x[ied]+tol < extent[1] else int(ied)
    jbg = int(jbg-1) if y[jbg]-tol > extent[2] else int(jbg)
    jed = int(jed+1) if y[jed]+tol < extent[3] else int(jed)

    # make sure indices are valid; will do extrapolation if ibg, jbg < 0 and ied, jed >= length
    ibg = max(0, ibg)
    ied = min(len(x)-1, ied)
    jbg = max(0, jbg)
    jed = min(len(y)-1, jed)

    # the end has to shift one for slicing
    ied += 1
    jed += 1

    return ibg, ied, jbg, jed


def exchange_states(states):
    """Exchange w, hu, and hv with neighbors to fill in the cells in halo rings

    Arguments
    ---------
    states : torchswe.utils.data.States

    Returns
    -------
    states : torchswe.utols.data.States
        The same object as the one in the input arguments.
    """

    # alias
    domain = states.domain
    cnsrv = states.q
    osc = states.osc

    # make sure all calculations updating states.q are done
    _nplike.sync()

    # start one-sided communication
    for k in ("s", "n", "w", "e"):
        osc.q.win.Put([cnsrv, osc.q[f"{k}s"]], domain[k], [0, 1, osc.q[f"{k}r"]])
    osc.q.win.Fence()

    # theroretically, this sync should not be neeeded, but just in case
    _nplike.sync()

    return states
