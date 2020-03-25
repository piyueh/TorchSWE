#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""
Things relating to initializatio of a simulation.
"""
import os
import argparse
import yaml
import torch

def init():
    """Initialize a simulation and read configuration.

    Args:
    -----
        N/A.

    Returns:
    --------
        A dictionary of configurations in the case's config.yaml with the
        following additional parameters:
            path: absolute path to the case folder
            yaml: absolute path to the config.yaml
            device: the device used.
            dtype: either torch.float32 or torch.float64.

        And all paths are converted to absolute paths.
    """

    # parse command-line arguments
    parser = argparse.ArgumentParser(
        "TorchSWE", None,
        "GPU shallow-water equation solver utilizing PyTorch",
        "Website: https://github.com/piyueh/TorchSWE")
    parser.add_argument(
        "case_folder", metavar="PATH", action="store", type=str,
        help="The path to a case folder.")
    parser.add_argument(
        "-cpu", action="store_true", dest="cpu",
        help="Use CPU instead of GPU")
    parser.add_argument(
        "-sp", action="store_true", dest="sp",
        help="Use single precision instead of double precision floating numbers")
    args = parser.parse_args()

    # select device and precision
    device = "cpu" if args.cpu else "cuda:0"
    dtype = torch.float64 if args.cpu else torch.float32

    # paths use absolute paths
    case_folder = os.path.abspath(args.case_folder)
    yaml_path = os.path.join(args.case_folder, "config.yaml")

    # read yaml config file
    with open(yaml_path, "r") as f:
        config = yaml.load(f, yaml.CLoader)

    # add args to config
    config["path"] = case_folder
    config["yaml"] = yaml_path
    config["device"] = device
    config["dtype"] = dtype

    # move all path to absolute path
    config["topography"]["file"] = os.path.abspath(
        os.path.join(case_folder, config["topography"]["file"]))

    if config["ic"]["file"] is not None:
        config["ic"]["file"] = os.path.abspath(
            os.path.join(case_folder, config["ic"]["file"]))

    if config["prehook"] is not None:
        config["prehook"] = os.path.abspath(
            os.path.join(case_folder, config["prehook"]))

    return config

def create_gridlines(dbox, discrtz, trange, device="cuda", dtype=torch.float64):
    """Create spatial and temporal gridlines.

    Args:
    -----
        dbox: the YAML node of "domain" in the config.
        discrtz: the YAML node of "discretization" in the config.
        tbox: the YAML node of "output time" in the config.
        device: a steing indicating the devcie used.
        dtype: either torch.float32 or torchh.float64.

    Returns:
    --------
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

    # for clearer/more readible code
    west, east, north, south = dbox["west"], dbox["east"], dbox["north"], dbox["south"]
    Nx, Ny = discrtz["Nx"], discrtz["Ny"]
    tbg, ted, step = trange["bg"], trange["ed"], trange["step"]

    dx = (east - west) / Nx
    dy = (north - south) / Ny

    if abs(dx-dy) > 1e-10:
        raise NotImplementedError("Currently only support dx = dy.")

    # coordinates of vertices
    xv = torch.linspace(west, east, Nx+1, dtype=dtype, device=device)
    yv = torch.linspace(south, north, Ny+1, dtype=dtype, device=device)

    # coordinates of cell centers
    xc = torch.linspace(west+dx/2., east-dx/2., Nx, dtype=dtype, device=device)
    yc = torch.linspace(south+dy/2., north-dy/2., Ny, dtype=dtype, device=device)

    # coordinates of midpoints of cell interfaces
    xf = {"x": xv, "y": xc}
    yf = {"x": yc, "y": yv}

    # temporal gridline: not used in computation, so use native list here
    Nt = (ted - tbg) // step + 1
    t = [tbg+i*dt for i in range(NT)]
    assert t[-1] <= ted
    if t[-1] != ted:
        t.append(ted)

    return xv, yv, xc, yc, xf, yf, dx, dy, t
