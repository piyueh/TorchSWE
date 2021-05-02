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
from ..utils.netcdf import read_cf
from ..utils.config import Config
from ..utils.data import Gridlines, Topography, WHUHVModel


def init():
    """Initialize a simulation and read configuration.

    Returns:
    --------
    config : a torchswe.utils.config.Config
        A Config instance holding a case's simulation configurations. All paths are converted to
        absolute paths. The temporal scheme is replaced with the corresponding function.
    grid : torch.utils.data.Gridlines
        Contains gridline coordinates.
    topo : torch.utils.data.Topography
        Contains topography elevation data.
    state_ic : torchswe.utils.data.WHUHVModel
        Initial confitions.
    """

    # get cmd arguments
    args = get_cmd_arguments()
    args.case_folder = args.case_folder.expanduser().resolve()
    args.yaml = args.case_folder.joinpath("config.yaml")

    # read yaml config file
    with open(args.yaml, "r") as fobj:
        config = yaml.load(fobj, yaml.Loader)

    assert isinstance(config, Config), \
        "Failed to parse {} as an Config object. ".format(args.yaml) + \
        "Check if `--- !Config` appears in the header of the YAML"

    # add args to config
    config.case = args.case_folder
    config.dtype = "float32" if args.sp else "float64"

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

    # spatial discretization + output time values
    grid = Gridlines(config.spatial, config.temporal, config.dtype)

    # topography
    topo = Topography(config.topo, grid, config.dtype)

    # initial conditions
    state_ic = create_ic(config.ic, grid, topo, config.dtype)

    return config, grid, topo, state_ic


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


def create_ic(ic_config, gridlines, topo, dtype):
    """Create initial conditions.

    When the x_cntr and y_cntr have different resolutions from the x and y in the NetCDF file, an
    bi-cubic spline interpolation will take place.

    Arguments
    ---------
    ic_config : torchswe.utils.config.ICConfig
    gridlines : torchswe.utils.data.Gridlines
    topo : torchswe.utils.data.Topography
    dtype : str; either "float32" or "float64"

    Returns
    -------
    torchswe.utils.data.WHUHVModel
    """

    # special case: constant I.C.
    if ic_config.values is not None:
        return WHUHVModel(
            gridlines.x.n, gridlines.y.n, dtype,
            w=numpy.maximum(topo.cntr, ic_config.values[0]),
            hu=numpy.full(topo.cntr.shape, ic_config.values[1], dtype),
            hv=numpy.full(topo.cntr.shape, ic_config.values[2], dtype))

    # otherwise, read data from a NetCDF file
    icdata, _ = read_cf(ic_config.file, ic_config.keys)

    # see if we need to do interpolation
    try:
        interp = not (
            numpy.allclose(gridlines.x.cntr, icdata["x"]) and
            numpy.allclose(gridlines.y.cntr, icdata["y"]))
    except ValueError:  # assume thie excpetion means a shape mismatch
        interp = True

    # unfortunately, we need to do interpolation in such a situation
    if interp:
        interpolator = RectBivariateSpline(icdata["x"], icdata["y"], icdata[ic_config.keys[0]][:].T)
        w = interpolator(gridlines.x.cntr, gridlines.y.cntr).T

        # get an interpolator for conserv_q_ic[1], use the default 3rd order spline
        interpolator = RectBivariateSpline(icdata["x"], icdata["y"], icdata[ic_config.keys[1]][:].T)
        hu = interpolator(gridlines.x.cntr, gridlines.y.cntr).T

        # get an interpolator for conserv_q_ic[2], use the default 3rd order spline
        interpolator = RectBivariateSpline(icdata["x"], icdata["y"], icdata[ic_config.keys[2]][:].T)
        hv = interpolator(gridlines.x.cntr, gridlines.y.cntr).T
    else:
        w = icdata[ic_config.keys[0]][:].copy()
        hu = icdata[ic_config.keys[1]][:].copy()
        hv = icdata[ic_config.keys[2]][:].copy()

    # make sure the w can not be smaller than topopgraphy elevation
    w = numpy.maximum(w, topo.cntr)

    return WHUHVModel(gridlines.x.n, gridlines.y.n, dtype, w=w, hu=hu, hv=hv)
