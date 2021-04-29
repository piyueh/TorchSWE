#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Things relating to initializatio of a simulation.

TODO: use schema for config.yaml
"""
import os
import argparse
import yaml
import torch
from ..utils.netcdf import read_cf


def init():
    """Initialize a simulation and read configuration.

    Args:
    -----
        N/A.

    Returns:
    --------
        config: a dictionary of configurations in the case's config.yaml with the
            following additional parameters:
                path: absolute path to the case folder
                yaml: absolute path to the config.yaml
                device: the device used.
                dtype: either torch.float32 or torch.float64.

            And all paths are converted to absolute paths.

        data: a dictionary containing all returns from create_gridlines,
            create_topography, and create_ic.
    """

    # parse command-line arguments
    parser = argparse.ArgumentParser(
        prog="TorchSWE",
        description="GPU shallow-water equation solver utilizing PyTorch",
        epilog="Website: https://github.com/piyueh/TorchSWE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False
    )
    parser.add_argument(
        "case_folder", metavar="PATH", action="store", type=str,
        help="The path to a case folder."
    )
    parser.add_argument(
        "-cpu", action="store_true", dest="cpu",
        help="Use CPU instead of GPU"
    )
    parser.add_argument(
        "-sp", action="store_true", dest="sp",
        help="Use single precision instead of double precision floating numbers"
    )
    parser.add_argument(
        "-tm", action="store", type=str, choices=["RK2", "RK4", "euler"], default=None,
        help="Time-marching scheme."
    )
    args = parser.parse_args()

    # select device and precision
    device = "cpu" if args.cpu else "cuda:0"
    dtype = torch.float32 if args.sp else torch.float64

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

    # TODO: this code can be simplified with config.yaml schema/template
    if args.tm is not None: # force to overwrite any setting in config.yaml
        config["temporal"] = args.tm
    else: # if not in CMD, check if users have the setting in config,yaml
        if "temporal" not in config: # if no, then fall back to RK2
            config["temporal"] = "RK2"

    # TODO: again should be handled by schema default
    if "theta" not in config:
        config["theta"] = 1.3

    # TODO: again should be handled by schema default
    if "drytol" not in config:
        config["drytol"] = 1e-5

    # move all path to absolute path
    config["topography"]["file"] = os.path.abspath(
        os.path.join(case_folder, config["topography"]["file"]))

    if config["ic"]["file"] is not None:
        config["ic"]["file"] = os.path.abspath(
            os.path.join(case_folder, config["ic"]["file"]))

    if config["prehook"] is not None:
        config["prehook"] = os.path.abspath(
            os.path.join(case_folder, config["prehook"]))

    # create data
    data = {}

    # spatial discretization + output time values
    data.update(create_gridlines(
        config["domain"], config["discretization"], config["output time"],
        config["device"], config["dtype"]))

    # topography-related information
    data.update(create_topography(config["topography"], data["xv"], data["yv"]))

    # initial conditions
    data["U0"] = create_ic(config["ic"], data["xc"], data["yc"], data["Bc"])

    return config, data

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
    Nt = int((ted - tbg)//step) + 1
    t = [tbg+i*step for i in range(Nt)]
    assert t[-1] <= ted
    if t[-1] != ted:
        t.append(ted)

    return {"xv": xv, "yv": yv, "xc": xc, "yc": yc,
            "xf": xf, "yf": yf, "dx": dx, "dy": dy, "t": t}

def create_topography(topo, xv, yv):
    """Create required topography information from a NetCDF file.

    The data in the NetCDF DEM file is assumed to be defined at cell vertices.

    Also not, when the xy and yv have different resolutions from the x and y in
    the DEM file, an bi-cubic spline interpolation will take place, which
    introduces rounding errors to the elevation. The rounding error may be
    crucial to well-balanced property tests. But usually it's fine to real-
    world applications.

    Args:
    -----
        topo: the "topography" node from the config returned by init().
        xv: 1D torch.tensor with length Nx+1; vertex gridline of the computational domain.
        yv: 1D torch.tensor with length Ny+1; vertex gridline of the computational domain.

    Returns:
    --------
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

    # read DEM
    topodata, attrs = read_cf(topo["file"], [topo["key"]])

    # copy to a numpy.ndarray
    Bv = topodata[topo["key"]][:].copy()

    # see if we need to do interpolation
    topox = torch.tensor(topodata["x"], device=xv.device)
    topoy = torch.tensor(topodata["y"], device=yv.device)
    shape_mismatch = not (topox.shape==xv.shape and topoy.shape==yv.shape)

    if not shape_mismatch:
        value_mismatch = not (torch.allclose(xv, topox) and torch.allclose(yv, topoy))
    else:
        value_mismatch = True

    if shape_mismatch or value_mismatch:

        # unfortunately, we need scipy to help with the interpolation
        from scipy.interpolate import RectBivariateSpline

        # get an interpolator, use the default 3rd order spline
        interpolator = RectBivariateSpline(topodata["x"], topodata["y"], Bv.T)

        # get the interpolated elevations at vertices and replace Bv
        Bv = interpolator(xv.cpu().numpy(), yv.cpu().numpy()).T

    # convert to torch.tensor
    Bv = torch.tensor(Bv, dtype=xv.dtype, device=xv.device)

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
    if torch.abs(dx-dy) >= 1e-10:
        raise NotImplementedError("Currently only support dx = dy.")

    # gradient at cell centers through center difference
    dBc = {
        "x": (Bf["x"][:, 1:] - Bf["x"][:, :-1]) / dx,
        "y": (Bf["y"][1:, :] - Bf["y"][:-1, :]) / dy
    }

    # sanity checks
    assert torch.allclose(Bc, (Bf["x"][:, :-1]+Bf["x"][:, 1:])/2.)
    assert torch.allclose(Bc, (Bf["y"][:-1, :]+Bf["y"][1:, :])/2.)

    return {"Bv": Bv, "Bc": Bc, "Bf": Bf, "dBc": dBc}

def create_ic(ic, xc, yc, Bc):
    """Create initial conditions.

    When the xc and yc have different resolutions from the x and y in the NetCDF
    file, an bi-cubic spline interpolation will take place, which introduces
    rounding errors to the elevation. The rounding error may be crucial to
    well-balanced property tests. But usually it's fine to real-world
    applications.

    Args:
    -----
        ic: the "ic" node from the config returned by init().
        xc: 1D torch.tensor with length Nx; gridline of cell centers.
        yc: 1D torch.tensor with length Ny; gridline of cell centers.
        Bc: 2D torch.tensor with shape (Ny, Nx); elevation at cell centers.

    Returns:
    --------
        U0: a (3, Ny, Nx) torch.tensor representing w, hu, and hv.
    """

    # initialize variable
    U0 = torch.zeros((3, len(yc), len(xc)), dtype=xc.dtype, device=xc.device)

    # special case: constant I.C.
    if ic["values"] is not None:

        U0[0] = ic["values"][0]
        U0[1] = ic["values"][1]
        U0[2] = ic["values"][2]

        # make sure the w can not be smaller than topopgraphy elevation
        U0[0] = torch.where(U0[0]<Bc, Bc, U0[0])
        return U0

    # for other cases, read data from a NetCDF file
    icdata, attrs = read_cf(ic["file"], ic["keys"])

    # preserve the numpy.ndarray for now, in case we need to do interpolation
    w = icdata[ic["keys"][0]][:].copy()
    hu = icdata[ic["keys"][1]][:].copy()
    hv = icdata[ic["keys"][2]][:].copy()

    # see if we need to do interpolation
    icx = torch.tensor(icdata["x"], device=xc.device)
    icy = torch.tensor(icdata["y"], device=yc.device)
    shape_mismatch = not (icx.shape==xc.shape and icy.shape==yc.shape)

    if not shape_mismatch:
        value_mismatch = not (torch.allclose(xc, icx) and torch.allclose(yc, icy))
    else:
        value_mismatch = True

    if shape_mismatch or value_mismatch:

        # unfortunately, we need scipy to help with the interpolation
        from scipy.interpolate import RectBivariateSpline

        # get an interpolator for w, use the default 3rd order spline
        interpolator = RectBivariateSpline(icdata["x"], icdata["y"], w.T)
        w = interpolator(xc.cpu().numpy(), yc.cpu().numpy()).T

        # get an interpolator for U0[0], use the default 3rd order spline
        interpolator = RectBivariateSpline(icdata["x"], icdata["y"], hu.T)
        hu = interpolator(xc.cpu().numpy(), yc.cpu().numpy()).T

        # get an interpolator for U0[0], use the default 3rd order spline
        interpolator = RectBivariateSpline(icdata["x"], icdata["y"], hv.T)
        hv = interpolator(xc.cpu().numpy(), yc.cpu().numpy()).T

    # convert to torch.tensor
    U0[0] = torch.from_numpy(w)
    U0[1] = torch.from_numpy(hu)
    U0[2] = torch.from_numpy(hv)

    # make sure the w can not be smaller than topopgraphy elevation
    U0[0] = torch.where(U0[0]<Bc, Bc, U0[0])

    return U0
