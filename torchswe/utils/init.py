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
import argparse as _argparse
import pathlib as _pathlib
from typing import List as _List
from typing import Optional as _Optional

import yaml as _yaml
from torchswe import nplike as _nplike
from torchswe.utils.config import FrictionConfig as _FrictionConfig
from torchswe.utils.config import Config as _Config
from torchswe.utils.data import Domain as _Domain
from torchswe.utils.netcdf import read as _ncread
from torchswe.utils.misc import interpolate as _interpolate


_logger = _logging.getLogger("torchswe.utils.init")


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
        states.Q[(i,) + states.domain.internal] = data[key][tidx]

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
            domain.x.centers, domain.y.centers).T).astype(domain.dtype)
    else:  # no need for interpolation
        cntr = data[friction.key].astype(domain.dtype)

    return cntr
