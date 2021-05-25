#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Things relating to initializatio of a simulation with MPI.
"""
import yaml as _yaml
from torchswe import nplike as _nplike
from torchswe.core.initializer import get_cmd_arguments as _get_cmd_arguments
from torchswe.utils.config import Config as _Config
from torchswe.utils.data import WHUHVModel as _WHUHVModel
from torchswe.utils.netcdf import read as _ncread
from torchswe.utils.data import get_snapshot_times as _get_snapshot_times
from torchswe.utils.misc import interpolate as _interpolate
from torchswe.mpi.data import get_gridlines as _get_gridlines
from torchswe.mpi.data import get_topography as _get_topography


def init(comm, args=None):
    """Initialize a simulation and read configuration.

    Attributes
    ----------
    args : None or argparse.Namespace
        By default, None means getting arguments from command-line. If not None, it should be the
        return from ArgumentParser.parse().

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
    if args is None:
        args = _get_cmd_arguments()
    args.case_folder = args.case_folder.expanduser().resolve()
    args.yaml = args.case_folder.joinpath("config.yaml")

    # read yaml config file
    with open(args.yaml, "r") as fobj:
        config = _yaml.load(fobj, _yaml.Loader)

    assert isinstance(config, _Config), \
        "Failed to parse {} as an Config object. ".format(args.yaml) + \
        "Check if `--- !Config` appears in the header of the YAML"

    # add args to config
    config.case = args.case_folder
    config.dtype = "float32" if args.sp else "float64"

    if args.log_steps is not None:
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

    # spatial discretization + output time values
    grid = _get_gridlines(
        comm,
        *config.spatial.discretization,
        *config.spatial.domain,
        _get_snapshot_times(
            config.temporal.output[0], config.temporal.output[1:],
            config.temporal.dt
        ),
        config.dtype
    )

    # topography
    topo = _get_topography(
        comm, config.topo.file, config.topo.key, grid.x.vert, grid.y.vert, config.dtype)

    # initial conditions
    state_ic = create_ic(comm, config.ic, grid, topo, config.dtype)

    return config, grid, topo, state_ic


def create_ic(comm, ic_config, grid, topo, dtype):
    """Create initial conditions.

    When the x_cntr and y_cntr have different resolutions from the x and y in the NetCDF file, an
    bi-cubic spline interpolation will take place.

    Arguments
    ---------
    ic_config : torchswe.utils.config.ICConfig
    grid : torchswe.utils.data.Gridlines
    topo : torchswe.utils.data.Topography
    dtype : str; either "float32" or "float64"

    Returns
    -------
    torchswe.utils.data.WHUHVModel
    """

    # special case: constant I.C.
    if ic_config.values is not None:
        return _WHUHVModel(
            nx=grid.x.n, ny=grid.y.n, dtype=dtype,
            w=_nplike.maximum(topo.cntr, _nplike.array(ic_config.values[0])),
            hu=_nplike.full(topo.cntr.shape, ic_config.values[1], dtype=topo.dtype),
            hv=_nplike.full(topo.cntr.shape, ic_config.values[2], dtype=topo.dtype))

    # otherwise, read data from a NetCDF file
    icdata, _ = _ncread(
        ic_config.file, ic_config.keys,
        [grid.x.cntr[0], grid.x.cntr[-1], grid.y.cntr[0], grid.y.cntr[-1]],
        parallel=True, comm=comm
    )

    # see if we need to do interpolation
    try:
        interp = not (
            _nplike.allclose(grid.x.cntr, icdata["x"]) and
            _nplike.allclose(grid.y.cntr, icdata["y"]))
    except ValueError:  # assume thie excpetion means a shape mismatch
        interp = True

    # unfortunately, we need to do interpolation in such a situation
    if interp:
        w = _nplike.array(
            _interpolate(
                icdata["x"], icdata["y"], icdata[ic_config.keys[0]].T, grid.x.cntr, grid.y.cntr
            ).T
        )

        hu = _nplike.array(
            _interpolate(
                icdata["x"], icdata["y"], icdata[ic_config.keys[1]].T, grid.x.cntr, grid.y.cntr
            ).T
        )

        hv = _nplike.array(
            _interpolate(
                icdata["x"], icdata["y"], icdata[ic_config.keys[2]].T, grid.x.cntr, grid.y.cntr
            ).T
        )
    else:
        w = icdata[ic_config.keys[0]]
        hu = icdata[ic_config.keys[1]]
        hv = icdata[ic_config.keys[2]]

    # make sure the w can not be smaller than topopgraphy elevation
    w = _nplike.maximum(w, topo.cntr)

    return _WHUHVModel(nx=grid.x.n, ny=grid.y.n, dtype=dtype, w=w, hu=hu, hv=hv)
