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
            An config instance holding a case's simulation configurations plus the following
            additional parameters. All paths are converted to absolute paths. Temporal scheme
            is replaced with the corresponding function.

        data: dict
            Contains all returns from create_gridlines, create_topography, and create_ic.
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

    # topography-related information
    data.update(create_topography(config.topo, data["xv"], data["yv"]))

    # initial conditions
    data["U0"] = create_ic(config.ic, data["xc"], data["yc"], data["Bc"])

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
    temporal: torchswe.utils.config.TemporalConfig
    dtype : str
        Either "float32" or "float64"

    Returns
    -------
    A dictionary with the following keys and values:
        xv: 1D torch.tensor of length Nx+1; x coordinates at vertices.
        yv: 1D torch.tensor of length Ny+1; y coordinates at vertices.
        xc: 1D torch.tensor of length Nx; x coordinates at cell centers.
        yc: 1D torch.tensor of length Ny; y coordinates at cell centers.
        xf: a dictionary of the following key-array pairs
            x: 1D torch.tensor of langth Nx+1; x coordinates at the midpoints
                of cell faces normal to x-direction.
            y: 1D torch.tensor of langth Nx; x coordinates at the midpoints of
                cell faces normal to y-direction.
        yf: a dictionary of the following key-array pairs
            x: 1D torch.tensor of langth Ny; y coordinates at the midpoints
                of cell faces normal to x-direction.
            y: 1D torch.tensor of langth Ny+1; y coordinates at the midpoints of
                cell faces normal to y-direction.
        dx: cell size in x-direction.
        dy: cell size in y-direction; assumed to be the same as dx.
        t: a list of time values to output solutions.
    """
    # pylint: disable=invalid-name

    # for clearer/more readible code
    Nx, Ny = spatial.discretization

    dx = (spatial.domain[1] - spatial.domain[0]) / Nx
    dy = (spatial.domain[3] - spatial.domain[2]) / Ny

    if abs(dx-dy) > 1e-10:
        raise NotImplementedError("Currently only support dx = dy.")

    # coordinates of vertices
    xv = numpy.linspace(spatial.domain[0], spatial.domain[1], Nx+1, dtype=dtype)
    yv = numpy.linspace(spatial.domain[2], spatial.domain[3], Ny+1, dtype=dtype)

    # coordinates of cell centers
    xc = numpy.linspace(spatial.domain[0]+dx/2., spatial.domain[1]-dx/2., Nx, dtype=dtype)
    yc = numpy.linspace(spatial.domain[2]+dy/2., spatial.domain[3]-dy/2., Ny, dtype=dtype)

    # coordinates of midpoints of cell interfaces
    xf = DummyDict({"x": xv, "y": xc})
    yf = DummyDict({"x": yc, "y": yv})

    if temporal.output is None:
        return DummyDict(
            {"xv": xv, "yv": yv, "xc": xc, "yc": yc,
             "xf": xf, "yf": yf, "dx": dx, "dy": dy, "t": None})

    # temporal gridline: not used in computation, so use native list here
    if temporal.output[0] == OutoutType.EVERY:
        Nt = int((temporal.end - temporal.start)//temporal.output[1]) + 1
        t = [temporal.start + i * temporal.output[1] for i in range(Nt)]
        assert t[-1] <= temporal.end
        if (temporal.end - t[-1]) > 1e-10:
            t.append(temporal.end)
    else:  # OutputType.AT
        t = temporal.output[1]

    return DummyDict(
        {"xv": xv, "yv": yv, "xc": xc, "yc": yc,
         "xf": xf, "yf": yf, "dx": dx, "dy": dy, "t": t})


def create_topography(topo, xv, yv):
    """Create required topography information from a NetCDF file.

    The data in the NetCDF DEM file is assumed to be defined at cell vertices.

    Also not, when the xy and yv have different resolutions from the x and y in
    the DEM file, an bi-cubic spline interpolation will take place, which
    introduces rounding errors to the elevation. The rounding error may be
    crucial to well-balanced property tests. But usually it's fine to real-
    world applications.

    Arguments
    ---------
    topo : torchswe.utils.config.TopoConfig
    xv : 1D numpy.ndarray with length Nx+1
        Vertex gridline of the computational domain.
    yv : 1D numpy.ndarray with length Ny+1
        Vertex gridline of the computational domain.

    Returns
    -------
    A dictionary with the following keys and values:
        Bv: a (Ny+1, Nx+1) torch.tensor; elevation at computational cell vertices.
        Bc: a (Ny, Nx) torch.tensor; elevation at computational cell centers.
        Bf: a dictionary with the following two key-value paits:
            x: a (Ny, Nx+1) torch.tensor; elevation at the midpoints of cell
                faces normal to x-direction.
            y: a (Ny+1, Nx) torch.tensor; elevation at the midpoints of cell
                faces normal to y-direction.
        dBc: a dictionary with the following two key-value paits:
            x: a (Ny, Nx) torch.tensor; x-gradient at cell centers.
            y: a (Ny, Nx) torch.tensor; y-gradient at cell centers.
    """
    # pylint: disable=invalid-name

    # read DEM
    topodata, _ = read_cf(topo.file, topo.key)

    # copy to a numpy.ndarray
    Bv = topodata[topo["key"]][:].copy()

    # see if we need to do interpolation
    shape_mismatch = not (topodata["x"].shape == xv.shape and topodata["y"].shape == yv.shape)

    if not shape_mismatch:
        value_mismatch = not (
            numpy.allclose(xv, topodata["x"]) and numpy.allclose(yv, topodata["y"]))
    else:
        value_mismatch = True

    # unfortunately, we need to do interpolation in such a situation
    if shape_mismatch or value_mismatch:

        # get an interpolator, use the default 3rd order spline
        interpolator = RectBivariateSpline(topodata["x"], topodata["y"], Bv.T)

        # get the interpolated elevations at vertices and replace Bv
        Bv = interpolator(xv, yv).T

    # topography elevation at cell centers through linear interpolation
    Bc = (Bv[:-1, :-1] + Bv[:-1, 1:] + Bv[1:, :-1] + Bv[1:, 1:]) / 4.

    # topography elevation at cell faces' midpoints through linear interpolation
    Bf = {
        "x": (Bv[:-1, :] + Bv[1:, :]) / 2.,
        "y": (Bv[:, :-1] + Bv[:, 1:]) / 2.,
    }

    # get cell size
    dx = xv[1] - xv[0]
    dy = yv[1] - yv[0]
    if numpy.abs(dx-dy) >= 1e-10:
        raise NotImplementedError("Currently only support dx = dy.")

    # gradient at cell centers through center difference
    dBc = {
        "x": (Bf["x"][:, 1:] - Bf["x"][:, :-1]) / dx,
        "y": (Bf["y"][1:, :] - Bf["y"][:-1, :]) / dy
    }

    # sanity checks
    assert numpy.allclose(Bc, (Bf["x"][:, :-1]+Bf["x"][:, 1:])/2.)
    assert numpy.allclose(Bc, (Bf["y"][:-1, :]+Bf["y"][1:, :])/2.)

    return {"Bv": Bv, "Bc": Bc, "Bf": Bf, "dBc": dBc}


def create_ic(ic, xc, yc, Bc):
    """Create initial conditions.

    When the xc and yc have different resolutions from the x and y in the NetCDF
    file, an bi-cubic spline interpolation will take place, which introduces
    rounding errors to the elevation. The rounding error may be crucial to
    well-balanced property tests. But usually it's fine to real-world
    applications.

    Arguments
    ---------
    ic : torchswe.utils.config.ICConfig
    xc : 1D numpy.ndarray with length Nx
        Gridline of cell centers.
    yc : 1D numpy.ndarray with length Ny
        Gridline of cell centers.
    Bc : 2D numpy.ndarray with shape (Ny, Nx)
        Elevation at cell centers.

    Returns
    -------
    U0: a (3, Ny, Nx) torch.tensor representing w, hu, and hv.
    """
    # pylint: disable=invalid-name

    # initialize variable
    U0 = numpy.zeros((3, len(yc), len(xc)), dtype=xc.dtype)

    # special case: constant I.C.
    if ic.values is not None:

        U0[0] = ic.values[0]
        U0[1] = ic.values[1]
        U0[2] = ic.values[2]

        # make sure the w can not be smaller than topopgraphy elevation
        U0[0] = numpy.where(U0[0] < Bc, Bc, U0[0])
        return U0

    # for other cases, read data from a NetCDF file
    icdata, _ = read_cf(ic.file, ic.keys)

    # preserve the numpy.ndarray for now, in case we need to do interpolation
    w = icdata[ic["keys"][0]][:].copy()
    hu = icdata[ic["keys"][1]][:].copy()
    hv = icdata[ic["keys"][2]][:].copy()

    # see if we need to do interpolation
    shape_mismatch = not (icdata["x"].shape == xc.shape and icdata["y"].shape == yc.shape)

    if not shape_mismatch:
        value_mismatch = not (numpy.allclose(xc, icdata["x"]) and numpy.allclose(yc, icdata["y"]))
    else:
        value_mismatch = True

    # unfortunately, we need to do interpolation
    if shape_mismatch or value_mismatch:

        # get an interpolator for w (i.e., U0[0]), use the default 3rd order spline
        interpolator = RectBivariateSpline(icdata["x"], icdata["y"], w.T)
        w = interpolator(xc, yc).T

        # get an interpolator for U0[1], use the default 3rd order spline
        interpolator = RectBivariateSpline(icdata["x"], icdata["y"], hu.T)
        hu = interpolator(xc, yc).T

        # get an interpolator for U0[2], use the default 3rd order spline
        interpolator = RectBivariateSpline(icdata["x"], icdata["y"], hv.T)
        hv = interpolator(xc, yc).T

    # convert to torch.tensor
    U0[0] = w
    U0[1] = hu
    U0[2] = hv

    # make sure the w can not be smaller than topopgraphy elevation
    U0[0] = numpy.where(U0[0] < Bc, Bc, U0[0])

    return U0
