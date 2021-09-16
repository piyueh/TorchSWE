#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Functions for initializing objects.
"""
# pylint: disable=invalid-name
import logging as _logging
import copy as _copy
from typing import List as _List
from typing import Union as _Union
from typing import Optional as _Optional

from mpi4py import MPI as _MPI
from torchswe import nplike as _nplike
from torchswe.utils.data import Process as _Process
from torchswe.utils.data import Gridline as _Gridline
from torchswe.utils.data import Timeline as _Timeline
from torchswe.utils.data import Domain as _Domain
from torchswe.utils.data import Topography as _Topography
from torchswe.utils.data import WHUHVModel as _WHUHVModel
from torchswe.utils.data import HUVModel as _HUVModel
from torchswe.utils.data import FaceOneSideModel as _FaceOneSideModel
from torchswe.utils.data import FaceTwoSideModel as _FaceTwoSideModel
from torchswe.utils.data import FaceQuantityModel as _FaceQuantityModel
from torchswe.utils.data import Slopes as _Slopes
from torchswe.utils.data import States as _States
from torchswe.utils.misc import DummyDtype as _DummyDtype
from torchswe.utils.misc import cal_num_procs as _cal_num_procs
from torchswe.utils.misc import cal_proc_loc_from_rank as _cal_proc_loc_from_rank
from torchswe.utils.misc import cal_neighbors as _cal_neighbors
from torchswe.utils.misc import cal_local_gridline_range as _cal_local_gridline_range
from torchswe.utils.misc import interpolate as _interpolate


_logger = _logging.getLogger("torchswe.utils.init")


def get_process(comm: _MPI.Comm, gnx: int, gny: int):
    """Get an instance of Process for the current MPI process.

    Arguments
    ---------
    comm : mpi4py.MPI.Comm
        The communicator.
    gnx, gny : int
        The global numbers of cells.

    Returns
    -------
    An instance of `torchswe.utils.data.Process`.
    """
    # pylint: disable=invalid-name
    data = {"comm": comm}
    data["pnx"], data["pny"] = _cal_num_procs(comm.Get_size(), gnx, gny)
    data["pi"], data["pj"] = _cal_proc_loc_from_rank(data["pnx"], comm.Get_rank())
    data["west"], data["east"], data["south"], data["north"] = _cal_neighbors(
        data["pnx"], data["pny"], data["pi"], data["pj"], comm.Get_rank())
    return _Process(**data)


def get_gridline(axis: str, pn: int, pi: int, gn: int, glower: float, gupper: float, dtype: str):
    """Get a Gridline instance.

    Arguments
    ---------
    axis : str
        Spatial axis. Either "x" or "y".
    pn, pi : int
        Total number of MPI processes and the index of the current process in this axis.
    gn : int
        Global number of cells on this axis.
    glower, gupper : float
        Global lower and upper bounds of this axis.
    dtype : str, nplike.float32, or nplike.float64
        Floating-point number types.

    Returns
    -------
    gridline : Gridline
    """

    data = {
        "axis": axis,
        "gn": gn,
        "glower": glower,
        "gupper": gupper,
        "delta": (gupper - glower) / gn,
        "dtype": _DummyDtype.validator(dtype)
    }

    data["n"], data["ibegin"], data["iend"] = _cal_local_gridline_range(pn, pi, gn)
    data["lower"] = data["ibegin"] * data["delta"] + data["glower"]
    data["upper"] = data["iend"] * data["delta"] + data["glower"]

    data["vertices"] = _nplike.linspace(
        data["lower"], data["upper"],
        data["n"]+1, dtype=data["dtype"])

    data["centers"] = _nplike.linspace(
        data["lower"]+data["delta"]/2., data["upper"]-data["delta"]/2.,
        data["n"], dtype=data["dtype"])

    if axis == "x":
        data["xfcenters"] = _copy.deepcopy(data["vertices"])
        data["yfcenters"] = _copy.deepcopy(data["centers"])
    else:  # implying axis == "y". If it is not "y", pydantic will raise an error
        data["xfcenters"] = _copy.deepcopy(data["centers"])
        data["yfcenters"] = _copy.deepcopy(data["vertices"])

    return _Gridline(**data)


def get_timeline(output_type: str, params: _List[_Union[int, float]], dt: _Optional[float] = None):
    """Generate a list of times when the solver should output solution snapshots.

    Arguments
    ---------
    output_type : str
        The type of outputting. See the docstring of torchswe.utils.TemporalConfig.
    params : a list/tuple
        The parameters associated with the particular `output_type`.
    dt : float or None
        Needed when output_type is "t_start every_steps multiple" or "t_start n_steps no save".

    Returns
    -------
    t : torchswe.utils.data.Timeline
    """

    save = True  # default

    # write solutions to a file at give times
    if output_type == "at":
        t = list(params)

    # output every `every_seconds` seconds `multiple` times from `t_start`
    elif output_type == "t_start every_seconds multiple":
        bg, delta, n = params
        t = (_nplike.arange(0, n+1) * delta + bg).tolist()  # including saving t_start

    # output every `every_steps` constant-size steps for `multiple` times from t=`t_start`
    elif output_type == "t_start every_steps multiple":
        assert dt is not None, "dt must be provided for \"t_start every_steps multiple\""
        bg, steps, n = params
        t = (_nplike.arange(0, n+1) * dt * steps + bg).tolist()  # including saving t_start

    # from `t_start` to `t_end` evenly outputs `n_saves` times (including both ends)
    elif output_type == "t_start t_end n_saves":
        bg, ed, n = params
        t = _nplike.linspace(bg, ed, n+1).tolist()  # including saving t_start

    # run simulation from `t_start` to `t_end` but not saving solutions at all
    elif output_type == "t_start t_end no save":
        t = params
        save = False

    # run simulation from `t_start` with `n_steps` iterations but not saving solutions at all
    elif output_type == "t_start n_steps no save":
        assert dt is not None, "dt must be provided for \"t_start n_steps no save\""
        t = [params[0], params[0] + params[1] * dt]
        save = False

    # should never reach this branch because pydantic has detected any invalid arguments
    else:
        raise ValueError("{} is not an allowed output method.".format(output_type))

    return _Timeline(values=t, save=save)


def get_domain(process: _Process, x: _Gridline, y: _Gridline):
    """A dummy function to get an instance of Domain for the current MPI process.

    This function does nothing. It is the same as initializing a Domain instance directly with
    `Domain(process=process, x=x, y=y, t=t, ngh=ngh)`.

    Arguments
    ---------
    process : torchswe.utils.data.Process
        An instance of Process for this MPI process.
    x, y : torchswe.utils.data.Gridline
        The gridline objects for x and y axes.

    Returns
    -------
    An instance of torchswe.utils.data.Domain.
    """

    return _Domain(process=process, x=x, y=y)


def get_topography(domain, elev, demx, demy):
    """Get a Topography object from a config object.
    """

    # alias
    dtype = domain.x.dtype

    # see if we need to do interpolation
    try:
        interp = not (
            _nplike.allclose(domain.x.vertices, demx) and
            _nplike.allclose(domain.y.vertices, demy)
        )
    except ValueError:  # assume thie excpetion means a shape mismatch
        interp = True

    if interp:  # unfortunately, we need to do interpolation in such a situation
        _logger.warning("Grids do not match. Doing spline interpolation.")
        vert = _nplike.array(
            _interpolate(demx, demy, elev.T, domain.x.vertices, domain.y.vertices).T).astype(dtype)
    else:  # no need for interpolation
        vert = elev.astype(dtype)

    # topography elevation at cell centers through linear interpolation
    cntr = (vert[:-1, :-1] + vert[:-1, 1:] + vert[1:, :-1] + vert[1:, 1:]) / 4.

    # topography elevation at cell faces' midpoints through linear interpolation
    xface = (vert[:-1, :] + vert[1:, :]) / 2.
    yface = (vert[:, :-1] + vert[:, 1:]) / 2.

    # gradient at cell centers through central difference; here allows nonuniform grids
    dx = (domain.x.vertices[1:] - domain.x.vertices[:-1])[None, :]
    xgrad = (xface[:, 1:] - xface[:, :-1]) / dx
    dy = (domain.y.vertices[1:] - domain.y.vertices[:-1])[:, None]
    ygrad = (yface[1:, :] - yface[:-1, :]) / dy

    # initialize DataModel and let pydantic validates data
    return _Topography(
        domain=domain, vertices=vert, centers=cntr, xfcenters=xface,
        yfcenters=yface, xgrad=xgrad, ygrad=ygrad)


def get_empty_whuhvmodel(nx: int, ny: int, dtype: str):
    """Get an empty (i.e., zero arrays) WHUHVModel.

    Arguments
    ---------
    nx, ny : int
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    A WHUHVModel with zero arrays.
    """
    dtype = _DummyDtype.validator(dtype)
    w = _nplike.zeros((ny, nx), dtype=dtype)
    hu = _nplike.zeros((ny, nx), dtype=dtype)
    hv = _nplike.zeros((ny, nx), dtype=dtype)
    return _WHUHVModel(nx=nx, ny=ny, dtype=dtype, w=w, hu=hu, hv=hv)


def get_empty_huvmodel(nx: int, ny: int, dtype: str):
    """Get an empty (i.e., zero arrays) HUVModel.

    Arguments
    ---------
    nx, ny : int
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    A HUVModel with zero arrays.
    """
    dtype = _DummyDtype.validator(dtype)
    h = _nplike.zeros((ny, nx), dtype=dtype)
    u = _nplike.zeros((ny, nx), dtype=dtype)
    v = _nplike.zeros((ny, nx), dtype=dtype)
    return _HUVModel(nx=nx, ny=ny, dtype=dtype, h=h, u=u, v=v)


def get_empty_faceonesidemodel(nx: int, ny: int, dtype: str):
    """Get an empty (i.e., zero arrays) FaceOneSideModel.

    Arguments
    ---------
    nx, ny : int
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    A FaceOneSideModel with zero arrays.
    """
    dtype = _DummyDtype.validator(dtype)
    return _FaceOneSideModel(
        nx=nx, ny=ny, dtype=dtype, w=_nplike.zeros((ny, nx), dtype=dtype),
        hu=_nplike.zeros((ny, nx), dtype=dtype), hv=_nplike.zeros((ny, nx), dtype=dtype),
        h=_nplike.zeros((ny, nx), dtype=dtype), u=_nplike.zeros((ny, nx), dtype=dtype),
        v=_nplike.zeros((ny, nx), dtype=dtype), a=_nplike.zeros((ny, nx), dtype=dtype),
        flux=get_empty_whuhvmodel(nx, ny, dtype)
    )


def get_empty_facetwosidemodel(nx: int, ny: int, dtype: str):
    """Get an empty (i.e., zero arrays) FaceTwoSideModel.

    Arguments
    ---------
    nx, ny : int
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    A FaceTwoSideModel with zero arrays.
    """
    return _FaceTwoSideModel(
        plus=get_empty_faceonesidemodel(nx, ny, dtype),
        minus=get_empty_faceonesidemodel(nx, ny, dtype),
        num_flux=get_empty_whuhvmodel(nx, ny, dtype)
    )


def get_empty_facequantitymodel(nx: int, ny: int, dtype: str):
    """Get an empty (i.e., zero arrays) FaceQuantityModel.

    Arguments
    ---------
    nx, ny : int
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    A FaceQuantityModel with zero arrays.
    """
    return _FaceQuantityModel(
        x=get_empty_facetwosidemodel(nx+1, ny, dtype),
        y=get_empty_facetwosidemodel(nx, ny+1, dtype),
    )


def get_empty_slopes(nx: int, ny: int, dtype: str):
    """Get an empty (i.e., zero arrays) Slopes.

    Arguments
    ---------
    nx, ny : int
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    A Slopes with zero arrays.
    """
    return _Slopes(
        x=get_empty_whuhvmodel(nx+2, ny, dtype),
        y=get_empty_whuhvmodel(nx, ny+2, dtype),
    )


def get_empty_states(domain: _Domain, ngh: int):
    """Get an empty (i.e., zero arrays) States.

    Arguments
    ---------
    domain : torchswe.utils.data.Domain
    ngh : int

    Returns
    -------
    A States with zero arrays.
    """
    nx = domain.x.n
    ny = domain.x.n
    dtype = domain.x.dtype
    return _States(
        domain=domain, ngh=ngh,
        q=get_empty_whuhvmodel(nx+2*ngh, ny+2*ngh, dtype),
        src=get_empty_whuhvmodel(nx, ny, dtype),
        slp=get_empty_slopes(nx, ny, dtype),
        rhs=get_empty_whuhvmodel(nx, ny, dtype),
        face=get_empty_facequantitymodel(nx, ny, dtype)
    )
