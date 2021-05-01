#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Things relating to initializatio of a simulation.
"""
import pathlib
import argparse
import yaml
import numpy
from scipy.interpolate import RectBivariateSpline
from ..utils.dummydict import DummyDict
from ..utils.netcdf import read_cf
from ..utils.config import Config, TemporalScheme, OutoutType
from .temporal import euler, RK2, RK4


def init():
    """Initialize a simulation and read configuration.

    Returns:
    --------
    config: a torchswe.utils.config.Config
        A Config instance holding a case's simulation configurations. All paths are converted to
        absolute paths. The temporal scheme is replaced with the corresponding function.
    data: torch.utils.dummydict.DummyDict
        Runtime data including gridline coordinates, topography elevations, and initial values.
    """

    # get cmd arguments
    args = get_cmd_arguments()
    args.case_folder = args.case_folder.expanduser().resolve()
    args.yaml = args.case_folder.joinpath("config.yaml")

    # read yaml config file
    with open(args.yaml, "r") as fobj:
        config = yaml.load(fobj, yaml.Loader)

    assert isinstance(config, Config)

    # add args to config
    config.case = args.case_folder
    config.ftype = "float32" if args.sp else "float64"

    if args.tm is not None:  # overwrite the setting in config.yaml
        config.temporal.scheme = TemporalScheme(args.tm)

    if config.temporal.schema == TemporalScheme.EULER:
        config.temporal.schema = euler
    elif config.temporal.schema == TemporalScheme.RK2:
        config.temporal.schema = RK2
    elif config.temporal.schema == TemporalScheme.RK4:
        config.temporal.schema = RK4

    # if topo filepath is relative, change to abs path
    config.topo.file = config.topo.file.expanduser()
    if not config.topo.file.is_absolute():
        config.topo.file = config.case.joinpath(config.topo.file).resolve()

    # if ic filepath is relative, change to abs path
    if config.ic.file is not None:
        config.ic.file = config.ic.file.expanduser()
        if not config.ic.file.is_absolute():
            config.ic.file = config.case.joinpath(config.ic.file).resolve()

    # if ic filepath is relative, change to abs path
    if config.prehook is not None:
        config.prehook = config.prehook.expanduser()
        if not config.prehook.is_absolute():
            config.prehook = config.case.joinpath(config.prehook).resolve()

    # create data
    data = DummyDict()

    # spatial discretization + output time values
    data.update(create_gridlines(config.spatial, config.temporal, config.ftype))

    # topography
    data.topo = create_topography(config.topo, data.x.vert, data.y.vert, config.ftype)

    # initial conditions
    data.conserv_q_ic = create_ic(config.ic, data.x.cntr, data.y.cntr, data.topo.cntr, config.ftype)

    return config, data


def get_cmd_arguments():
    """Parse and get CMD arguments.

    Returns
    -------
    args : argparse.Namespace
        CMD arguments.
    """

    # parse command-line arguments
    parser = argparse.ArgumentParser(
        prog="TorchSWE",
        description="GPU shallow-water equation solver utilizing Legate",
        epilog="Website: https://github.com/piyueh/TorchSWE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False
    )

    parser.add_argument(
        "case_folder", metavar="PATH", action="store", type=pathlib.Path,
        help="The path to a case folder."
    )

    parser.add_argument(
        "--sp", action="store_true", dest="sp",
        help="Use single precision instead of double precision floating numbers"
    )

    parser.add_argument(
        "--tm", action="store", type=str, choices=["RK2", "RK4", "Euler"], default=None,
        help="Overwrite the time-marching scheme. Default is to respect the setting in config.yaml."
    )

    args = parser.parse_args()
    return args


def create_gridlines(spatial, temporal, dtype):
    """Create spatial and temporal gridlines.

    Arguments
    ---------
    spatial : torchswe.utils.config.SpatialConfig
        Spatial configuration in config.yaml.
    temporal: torchswe.utils.config.TemporalConfig
        Temporal control configuration in config.yaml.
    dtype : str
        Either "float32" or "float64"

    Returns
    -------
    A DummyDict with keys "x", "y", and "t". And
        x, y: a DummyDict with the following keys
            - vert: 1D array of length Nx+1 or Ny+1; coordinates at vertices.
            - cntr: 1D array of length Nx or Ny; coordinates at cell centers.
            - xface: 1D array of langth Nx+1 or Ny; coordinates at the cell faces normal to x-axis.
            - yface: 1D array of langth Nx or Ny+1; coordinates at the cell faces normal to y-axis.
            - delta: float; cell size in the corresponding direction.
        t: a list
            Time values for outputing solutions. Time values are not used in numerical calculation,
            so we use native list.
    """
    # initialize
    x, y = DummyDict(), DummyDict()
    nx, ny = spatial.discretization  # aliases # pylint: disable=invalid-name
    west, east, south, north = spatial.domain  # aliases

    # cell sizes
    x.delta = (east - west) / nx
    y.delta = (north - south) / ny

    if abs(x.delta-y.delta) > 1e-10:
        raise NotImplementedError("Currently only support dx = dy.")

    # coordinates at vertices
    x.vert = numpy.linspace(west, east, nx+1, dtype=dtype)
    y.vert = numpy.linspace(spatial.domain[2], north, ny+1, dtype=dtype)

    # coordinates at cell centers
    x.cntr = numpy.linspace(west+x.delta/2., east-x.delta/2., nx, dtype=dtype)
    y.cntr = numpy.linspace(south+y.delta/2., north-y.delta/2., ny, dtype=dtype)

    # coordinates at cell faces
    x.xface = x.vert.copy()
    x.yface = x.cntr.copy()
    y.xface = y.cntr.copy()
    y.yface = y.vert.copy()

    # temporal gridline: not used in computation, so use native list here
    if temporal.output is None:  # no
        t = []
    elif temporal.output[0] == OutoutType.EVERY:  # output every dt
        dt = temporal.output[1]  # alias # pylint: disable=invalid-name
        t = numpy.arange(temporal.start, temporal.end+dt/2., dt).tolist()
    elif temporal.output[0] == OutoutType.AT:  # output at the given times
        t = temporal.output[1]
    else:
        raise RuntimeError("`temporal.output` does not have a valied value.")

    return DummyDict({"x": x, "y": y, "t": t})


def create_topography(topo_config, x_vert, y_vert, dtype):
    """Create required topography information from a NetCDF file.

    The data in the NetCDF DEM file is assumed to be defined at cell vertices.

    Also note, when the xy and yv have different resolutions from the x and y in the DEM file, an
    bi-cubic spline interpolation will take place, which introduces rounding errors to the
    elevation values. The rounding error may be crucial to well-balanced property tests. But
    usually it's fine to real- world applications.

    Arguments
    ---------
    topo_config : torchswe.utils.config.TopoConfig
        Configuration of topography in config.yaml.
    x_vert : 1D numpy.ndarray with length Nx+1
        Vertex gridline of the computational domain.
    y_vert : 1D numpy.ndarray with length Ny+1
        Vertex gridline of the computational domain.
    dtype : str
        Either "float32" or "float64"

    Returns
    -------
    topo : DummyDict
        Contains the following keys
        - vert : (Ny+1, Nx+1) array; elevation at vertices.
        - cntr : (Ny, Nx) array; elevation at cell centers.
        - xface : (Ny, Nx+1) array; elevation at cell faces normal to x-axis.
        - yface : (Ny+1, Nx) array; elevation at cell faces normal to y-axis.
        - dx : (Ny, Nx) array; elevation derivative w.r.t. x coordinates at cell centers.
        - dy : (Ny, Nx) array; elevation derivative w.r.t. y coordinates at cell centers.
    """
    # initialize
    topo = DummyDict()

    # read DEM
    dem, _ = read_cf(topo_config.file, topo_config.key)

    # copy to a numpy.ndarray
    topo.vert = dem[topo_config["key"]][:].copy()

    # see if we need to do interpolation
    try:
        interp = not (numpy.allclose(x_vert, dem["x"]) and numpy.allclose(y_vert, dem["y"]))
    except ValueError:  # assume thie excpetion means a shape mismatch
        interp = True

    # unfortunately, we need to do interpolation in such a situation
    if interp:
        interpolator = RectBivariateSpline(dem["x"], dem["y"], topo.vert.T)
        topo.vert = interpolator(x_vert, y_vert).T

    # cast to desired float type
    topo.cert = topo.vert.astype(dtype)

    # topography elevation at cell centers through linear interpolation
    topo.cntr = topo.vert[:-1, :-1] + topo.vert[:-1, 1:] + topo.vert[1:, :-1] + topo.vert[1:, 1:]
    topo.cntr /= 4

    # topography elevation at cell faces' midpoints through linear interpolation
    topo.xface = (topo.vert[:-1, :] + topo.vert[1:, :]) / 2.
    topo.yface = (topo.vert[:, :-1] + topo.vert[:, 1:]) / 2.

    # gradient at cell centers through central difference
    topo.dx = (topo.xface[:, 1:] - topo.xface[:, :-1]) / (x_vert[1:] - x_vert[:-1])
    topo.dy = (topo.yface[1:, :] - topo.yface[:-1, :]) / (y_vert[1:] - y_vert[:-1])

    # sanity checks
    assert numpy.allclose(topo.cntr, (topo.xface[:, :-1]+topo.xface[:, 1:])/2.)
    assert numpy.allclose(topo.cntr, (topo.yface[:-1, :]+topo.yface[1:, :])/2.)
    assert topo.dx.dtype == dtype
    assert topo.dy.dtype == dtype

    return topo


def create_ic(ic_config, x_cntr, y_cntr, topo_cntr, dtype):
    """Create initial conditions.

    When the x_cntr and y_cntr have different resolutions from the x and y in the NetCDF file, an
    bi-cubic spline interpolation will take place.

    Arguments
    ---------
    ic_config : torchswe.utils.config.ICConfig
        The IC configuration from config.yaml.
    x_cntr : length-Nx 1D array
        x-coordinates at cell centers.
    y_cntr : length-Ny 1D array
        y-coordinates at cell centers.
    topo_cntr : shape-(Ny, Nx) 2D array
        Topography elevations at cell centers.
    dtype : str
        Either "float32" or "float64"

    Returns
    -------
    conserv_q_ic: a (3, Ny, Nx) array representing w, hu, and hv.
    """
    # initialize variable
    conserv_q_ic = numpy.zeros((3, len(y_cntr), len(x_cntr)), dtype=dtype)

    # special case: constant I.C.
    if ic_config.values is not None:
        conserv_q_ic[0, ...] = numpy.maximum(topo_cntr, ic_config.values[0])
        conserv_q_ic[1, ...] = ic_config.values[1]
        conserv_q_ic[2, ...] = ic_config.values[2]
        return conserv_q_ic

    # otherwise, read data from a NetCDF file
    icdata, _ = read_cf(ic_config.file, ic_config.keys)

    # see if we need to do interpolation
    try:
        interp = not (numpy.allclose(x_cntr, icdata["x"]) and numpy.allclose(y_cntr, icdata["y"]))
    except ValueError:  # assume thie excpetion means a shape mismatch
        interp = True

    # unfortunately, we need to do interpolation in such a situation
    if interp:
        interpolator = RectBivariateSpline(icdata["x"], icdata["y"], icdata[ic_config.keys[0]][:].T)
        conserv_q_ic[0, ...] = interpolator(x_cntr, y_cntr).T

        # get an interpolator for conserv_q_ic[1], use the default 3rd order spline
        interpolator = RectBivariateSpline(icdata["x"], icdata["y"], icdata[ic_config.keys[1]][:].T)
        conserv_q_ic[1, ...] = interpolator(x_cntr, y_cntr).T

        # get an interpolator for conserv_q_ic[2], use the default 3rd order spline
        interpolator = RectBivariateSpline(icdata["x"], icdata["y"], icdata[ic_config.keys[2]][:].T)
        conserv_q_ic[2, ...] = interpolator(x_cntr, y_cntr).T
    else:
        conserv_q_ic[0, ...] = icdata[ic_config.keys[0]][:].copy()
        conserv_q_ic[1, ...] = icdata[ic_config.keys[1]][:].copy()
        conserv_q_ic[2, ...] = icdata[ic_config.keys[2]][:].copy()

    # make sure the w can not be smaller than topopgraphy elevation
    conserv_q_ic[0] = numpy.maximum(conserv_q_ic[0], topo_cntr)

    return conserv_q_ic
