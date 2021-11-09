#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Functions for initializing objects.
"""
import logging as _logging
import copy as _copy
import argparse as _argparse
import pathlib as _pathlib
from typing import List as _List
from typing import Tuple as _Tuple
from typing import Optional as _Optional

import yaml as _yaml
from mpi4py import MPI as _MPI
from torchswe import nplike as _nplike
from torchswe.utils.config import ICConfig as _ICConfig
from torchswe.utils.config import FrictionConfig as _FrictionConfig
from torchswe.utils.config import Config as _Config
from torchswe.utils.config import OutputTypeHint as _OutputTypeHint
from torchswe.utils.config import PointSourceConfig as _PointSourceConfig
from torchswe.utils.data import Gridline as _Gridline
from torchswe.utils.data import Timeline as _Timeline
from torchswe.utils.data import Domain as _Domain
from torchswe.utils.data import Topography as _Topography
from torchswe.utils.data import FaceOneSideModel as _FaceOneSideModel
from torchswe.utils.data import FaceTwoSideModel as _FaceTwoSideModel
from torchswe.utils.data import FaceQuantityModel as _FaceQuantityModel
from torchswe.utils.data import States as _States
from torchswe.utils.data import PointSource as _PointSource
from torchswe.utils.netcdf import read as _ncread
from torchswe.utils.misc import DummyDtype as _DummyDtype
from torchswe.utils.misc import cal_num_procs as _cal_num_procs
from torchswe.utils.misc import cal_local_gridline_range as _cal_local_gridline_range
from torchswe.utils.misc import interpolate as _interpolate
from torchswe.utils.misc import find_cell_index as _find_cell_index


_logger = _logging.getLogger("torchswe.utils.init")


def get_gridline(axis: str, pn: int, pi: int, gn: int, glower: float, gupper: float, dtype: str):
    """Get a Gridline instance.

    Arguments
    ---------
    axis : str
        Spatial axis. Either "x" or "y".
    pn, pi : int
        Total number of MPI ranks and the index of the current ranlk on this axis.
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

    data["vertices"], _ = _nplike.linspace(
        data["lower"], data["upper"],
        data["n"]+1, retstep=True, dtype=data["dtype"])

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


def get_timeline(temporal_config: _OutputTypeHint, dt: _Optional[float] = None):
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
    output_type = temporal_config[0]
    params = temporal_config[1:]

    # write solutions to a file at give times
    if output_type == "at":
        t = params[0]

    # output every `every_seconds` seconds `multiple` times from `t_start`
    elif output_type == "t_start every_seconds multiple":
        begin, delta, n = params
        t = (_nplike.arange(0, n+1) * delta + begin).tolist()  # including saving t_start

    # output every `every_steps` constant-size steps for `multiple` times from t=`t_start`
    elif output_type == "t_start every_steps multiple":
        assert dt is not None, "dt must be provided for \"t_start every_steps multiple\""
        begin, steps, n = params
        t = (_nplike.arange(0, n+1) * dt * steps + begin).tolist()  # including saving t_start

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
        assert dt is not None, "dt must be provided for \"t_start n_steps no save\""
        t = [params[0], params[0] + params[1] * dt]
        save = False

    # should never reach this branch because pydantic has detected any invalid arguments
    else:
        raise ValueError(f"{output_type} is not an allowed output method.")

    return _Timeline(values=t, save=save)


def get_domain(comm: _MPI.Comm, config: _Config):
    """Get an instance of Domain for the current MPI rank.

    Arguments
    ---------
    comm : mpi4py.MPI.Comm
        The communicator. Should just be a general communicator. And a Cartcomm will be created
        automatically in this function from the provided general Comm.
    x, y : torchswe.utils.data.Gridline
        The gridline objects for x and y axes.

    Returns
    -------
    An instance of torchswe.utils.data.Domain.
    """

    # see if we need periodic bc
    period = (config.bc.west.types[0] == "periodic", config.bc.south.types[0] == "period")

    # evaluate the number of ranks in x and y direction
    pnx, pny = _cal_num_procs(comm.Get_size(), *config.spatial.discretization)

    # get a Cartesian topology communicator
    comm = comm.Create_cart((pny, pnx), period, True)

    # find the rank of neighbors
    south, north = comm.Shift(0, 1)
    west, east = comm.Shift(1, 1)

    # get local gridline
    x = get_gridline(
        "x", comm.dims[1], comm.coords[1], config.spatial.discretization[0],
        config.spatial.domain[0], config.spatial.domain[1], config.params.dtype)

    y = get_gridline(
        "y", comm.dims[0], comm.coords[0], config.spatial.discretization[1],
        config.spatial.domain[2], config.spatial.domain[3], config.params.dtype)

    return _Domain(comm=comm, east=east, west=west, south=south, north=north, x=x, y=y)


def get_topography(domain, elev, demx, demy):
    """Get a Topography object.
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
    grad = _nplike.zeros((2,)+cntr.shape, dtype=cntr.dtype)
    dx = (domain.x.vertices[1:] - domain.x.vertices[:-1])[None, :]
    grad[0, ...] = (xface[:, 1:] - xface[:, :-1]) / dx
    dy = (domain.y.vertices[1:] - domain.y.vertices[:-1])[:, None]
    grad[1, ...] = (yface[1:, :] - yface[:-1, :]) / dy

    # initialize DataModel and let pydantic validates data
    return _Topography(
        domain=domain, vertices=vert, centers=cntr, xfcenters=xface, yfcenters=yface, grad=grad)


def get_topography_from_file(file: _pathlib.Path, key: str, domain: _Domain):
    """Read in CF-compliant NetCDF file for topography.
    """

    # get dem (digital elevation model); assume dem values defined at cell centers
    dem, _ = _ncread(
        fpath=file, data_keys=[key],
        extent=(domain.x.lower, domain.x.upper, domain.y.lower, domain.y.upper),
        parallel=True, comm=domain.comm
    )

    assert dem[key].shape == (len(dem["y"]), len(dem["x"]))

    topo = get_topography(domain, dem[key], dem["x"], dem["y"])
    return topo


def get_empty_faceonesidemodel(shape: _Tuple[int, int], dtype: str):
    """Get an empty (i.e., zero arrays) FaceOneSideModel.

    Arguments
    ---------
    shape : a tuple of two int
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    A FaceOneSideModel with zero arrays. The shapes are: Q.shape = (3, n1, n2), a.shape = (n1, n2),
    and F.shape = (3, n1, n2).
    """
    dtype = _DummyDtype.validator(dtype)
    return _FaceOneSideModel(
        Q=_nplike.zeros((3,)+shape, dtype=dtype), U=_nplike.zeros((3,)+shape, dtype=dtype),
        a=_nplike.zeros(shape, dtype=dtype), F=_nplike.zeros((3,)+shape, dtype))


def get_empty_facetwosidemodel(shape: _Tuple[int, int], dtype: str):
    """Get an empty (i.e., zero arrays) FaceTwoSideModel.

    Arguments
    ---------
    shape : a tuple of two int
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    A FaceTwoSideModel with zero arrays.
    """
    dtype = _DummyDtype.validator(dtype)
    return _FaceTwoSideModel(plus=get_empty_faceonesidemodel(shape, dtype),
        minus=get_empty_faceonesidemodel(shape, dtype), H=_nplike.zeros((3,)+shape, dtype))


def get_empty_facequantitymodel(nx: int, ny: int, dtype: str):
    """Get an empty (i.e., zero arrays) FaceQuantityModel.

    Arguments
    ---------
    nx, ny : int
        Number of grid cells.
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    A FaceQuantityModel with zero arrays.
    """
    return _FaceQuantityModel(
        x=get_empty_facetwosidemodel((ny, nx+1), dtype),
        y=get_empty_facetwosidemodel((ny+1, nx), dtype),
    )


def get_empty_states(domain: _Domain, ngh: int, use_stiff: bool):
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
    ny = domain.y.n
    dtype = domain.x.dtype
    return _States(
        domain=domain, ngh=ngh,
        Q=_nplike.zeros((3, ny+2*ngh, nx+2*ngh), dtype=dtype),
        H=_nplike.zeros((ny, nx), dtype=dtype),
        S=_nplike.zeros((3, ny, nx), dtype=dtype),
        SS=(_nplike.zeros((3, ny, nx), dtype=dtype) if use_stiff else None),
        face=get_empty_facequantitymodel(nx, ny, dtype),
    )


def get_initial_states(domain: _Domain, ic: _ICConfig, ngh: int, use_stiff: bool):
    """Get a States instance filled with initial conditions.

    Arguments
    ---------

    Returns
    -------
    torchswe.utils.data.States

    Notes
    -----
    When x and y axes have different resolutions from the x and y in the NetCDF file, an bi-cubic
    spline interpolation will take place.
    """

    # get an empty states
    states = get_empty_states(domain, ngh, use_stiff)

    # special case: constant I.C.
    if ic.values is not None:
        for i in range(3):
            states.Q[i, ngh:-ngh, ngh:-ngh] = ic.values[i]
        states.check()
        return states

    # otherwise, read data from a NetCDF file
    icdata, _ = _ncread(
        ic.file, ic.keys,
        [domain.x.centers[0], domain.x.centers[-1], domain.y.centers[0], domain.y.centers[-1]],
        parallel=True, comm=domain.comm
    )

    # see if we need to do interpolation
    try:
        interp = not (
            _nplike.allclose(domain.x.centers, icdata["x"]) and
            _nplike.allclose(domain.y.centers, icdata["y"])
        )
    except ValueError:  # assume thie excpetion means a shape mismatch
        interp = True

    # unfortunately, we need to do interpolation in such a situation
    if interp:
        _logger.warning("Grids do not match. Doing spline interpolation.")
        for i in range(3):
            states.Q[i, ngh:-ngh, ngh:-ngh] = _nplike.array(
                _interpolate(
                    icdata["x"], icdata["y"], icdata[ic.keys[i]].T,
                    domain.x.centers, domain.y.centers).T
            )
    else:
        for i in range(3):
            states.Q[i, ngh:-ngh, ngh:-ngh] = icdata[ic.keys[i]]

    states.check()
    return states


def get_initial_states_from_config(comm: _MPI.Comm, config: _Config):
    """Get an initial states based on a configuration object.
    """

    # get parallel domain
    domain = get_domain(comm, config)

    # get states
    states = get_initial_states(domain, config.ic, config.params.ngh, (config.friction is not None))
    return states


def get_initial_states_from_snapshot(fpath: str, tidx: int, states):
    """Read a snapshot from a solution file created by TorchSWE.

    Returns
    -------
    states : torchswe.utils.data.States
    """

    data, _ = _ncread(
        fpath, ["w", "hu", "hv"],
        [
            states.domain.x.centers[0], states.domain.x.centers[-1],
            states.domain.y.centers[0], states.domain.y.centers[-1]
        ],
        parallel=True, comm=states.domain.comm
    )

    for i, key in enumerate(["w", "hu", "hv"]):
        states.Q[i, states.ngh:-states.ngh, states.ngh:-states.ngh] = data[key][tidx]

    states.check()
    return states


def get_cmd_arguments(argv: _Optional[_List[str]] = None):
    """Parse and get CMD arguments.

    Attributes
    ----------
    argv : list or None
        By default, None means using `sys.argv`. Only explicitly use this argument for debug.

    Returns
    -------
    args : argparse.Namespace
        CMD arguments.
    """

    # parse command-line arguments
    parser = _argparse.ArgumentParser(
        prog="TorchSWE",
        description="GPU shallow-water equation solver utilizing Legate",
        epilog="Website: https://github.com/piyueh/TorchSWE",
        formatter_class=_argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False
    )

    parser.add_argument(
        "case_folder", metavar="PATH", action="store", type=_pathlib.Path,
        help="The path to a case folder."
    )

    parser.add_argument(
        "--continue", action="store", type=float, default=None, metavar="TIME", dest="cont",
        help="Indicate this run should continue from this time point."
    )

    parser.add_argument(
        "--sp", action="store_true", dest="sp",
        help="Use single precision instead of double precision floating numbers"
    )

    parser.add_argument(
        "--tm", action="store", type=str, choices=["SSP-RK2", "SSP-RK3", "Euler"], default=None,
        help="Overwrite the time-marching scheme. Default is to respect the setting in config.yaml."
    )

    parser.add_argument(
        "--log-steps", action="store", type=int, default=None, metavar="STEPS",
        help="How many steps to output a log message to stdout. Default is to respect config.yaml."
    )

    parser.add_argument(
        "--log-level", action="store", type=str, default="normal", metavar="LEVEL",
        choices=["debug", "normal", "quiet"],
        help="Enabling logging debug messages."
    )

    parser.add_argument(
        "--log-file", action="store", type=_pathlib.Path, default=None, metavar="FILE",
        help="Saving log messages to a file instead of stdout."
    )

    args = parser.parse_args(argv)

    # make sure the case folder path is absolute
    if args.case_folder is not None:
        args.case_folder = args.case_folder.expanduser().resolve()

    # convert log level from string to corresponding Python type
    level_options = {"quiet": _logging.ERROR, "normal": _logging.INFO, "debug": _logging.DEBUG}
    args.log_level = level_options[args.log_level]

    # make sure the file path is absolute
    if args.log_file is not None:
        args.log_file = args.log_file.expanduser().resolve()

    return args


def get_config(args: _argparse.Namespace):
    """Get a Config object.

    Arguments
    ---------
    args : argparse.Namespace
        The result of parsing command-line arguments.

    Returns
    -------
    config : torchswe.utils.config.Config
    """

    args.case_folder = args.case_folder.expanduser().resolve()
    args.yaml = args.case_folder.joinpath("config.yaml")

    # read yaml config file
    with open(args.yaml, "r", encoding="utf-8") as fobj:
        config = _yaml.load(fobj, _yaml.Loader)

    assert isinstance(config, _Config), \
        f"Failed to parse {args.yaml} as an Config object. " + \
        "Check if `--- !Config` appears in the header of the YAML"

    # add args to config
    config.case = args.case_folder

    config.params.dtype = "float32" if args.sp else config.params.dtype  # overwrite dtype if needed

    if args.log_steps is not None:  # overwrite log_steps if needed
        config.params.log_steps = args.log_steps

    if args.tm is not None:  # overwrite the setting in config.yaml
        config.temporal.scheme = args.tm

    # if topo filepath is relative, change to abs path
    config.topo.file = config.topo.file.expanduser()
    if not config.topo.file.is_absolute():
        config.topo.file = config.case.joinpath(config.topo.file).resolve()

    # if ic filepath is relative, change to abs path
    if config.ic.file is not None:
        config.ic.file = config.ic.file.expanduser()
        if not config.ic.file.is_absolute():
            config.ic.file = config.case.joinpath(config.ic.file).resolve()

    # if filepath of the prehook script is relative, change to abs path
    if config.prehook is not None:
        config.prehook = config.prehook.expanduser()
        if not config.prehook.is_absolute():
            config.prehook = config.case.joinpath(config.prehook).resolve()

    # validate data again
    config.check()

    return config


def get_pointsource(ptconfig: _PointSourceConfig, domain: _Domain, irate: int = 0):
    """Get a PointSource instance.

    Arguments
    ---------
    ptconfig : torchswe.utils.config.PointSourceConfig
        The configuration of a point source.
    domain : torchswe.utils.data.Domain
        The object describing grids and domain decomposition.
    irate : int
        The index of the current flow rate in the list of `rates`.

    Returns
    -------
    `None` if the current MPI rank does not own this point source, otherwise an instance of
    torchswe.utils.data.PointSource.

    Notes
    -----
    The returned PointSource object will store depth increment rates, rather than volumetric flow
    rates.
    """
    i = _find_cell_index(ptconfig.loc[0], domain.x.lower, domain.x.upper, domain.x.delta)
    j = _find_cell_index(ptconfig.loc[1], domain.y.lower, domain.y.upper, domain.y.delta)

    if i is None or j is None:
        return None

    # convert volumetric flow rates to depth increment rates; assuming constant/uniform dx & dy
    rates = [rate / domain.x.delta / domain.y.delta for rate in ptconfig.rates]

    # determined from provide irate
    active = (not irate == len(ptconfig.times))
    _logger.debug("Point source initial `active`: %s", active)

    return _PointSource(
        x=ptconfig.loc[0], y=ptconfig.loc[1], i=i, j=j, times=ptconfig.times, rates=rates,
        irate=irate, active=active, init_dt=ptconfig.init_dt
    )


def get_friction_roughness(domain: _Domain, friction: _FrictionConfig):
    """Get an array or a scalar holding the surface roughness.

    Arguments
    ---------
    domain : torchswe.utils.data.Domain
        The object describing grids and domain decomposition.
    friction : torchswe.utils.config.FrictionConfig
        The friction configuration.

    Returns
    -------
    If constant roughness, returns a constant scalar. (Will rely on auto-broadcasting mechanism in
    array operations.) Otherwise, returns an array by reading a file and interpolaing the values if
    needed.
    """

    if friction.value is not None:
        return friction.value

    data, _ = _ncread(
        fpath=friction.file, data_keys=[friction.key],
        extent=(domain.x.lower, domain.x.upper, domain.y.lower, domain.y.upper),
        parallel=True, comm=domain.comm
    )

    assert data[friction.key].shape == (len(data["y"]), len(data["x"]))

    # see if we need to do interpolation
    try:
        interp = not (
            _nplike.allclose(domain.x.centers, data["x"]) and
            _nplike.allclose(domain.y.centers, data["y"])
        )
    except ValueError:  # assume thie excpetion means a shape mismatch
        interp = True

    if interp:  # unfortunately, we need to do interpolation in such a situation
        _logger.warning("Grids do not match. Doing spline interpolation.")
        cntr = _nplike.array(_interpolate(
            data["x"], data["y"], data[friction.key].T,
            domain.x.centers, domain.y.centers).T).astype(domain.x.dtype)
    else:  # no need for interpolation
        cntr = data[friction.key].astype(domain.x.dtype)

    return cntr
